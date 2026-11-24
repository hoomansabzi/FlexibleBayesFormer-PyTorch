import torch
import torch.nn.functional as F


class RecallWindTunnel:
    """Associative recall: store N cue -> target pairs, then answer Q probes.

    Layout (length 2N + Q):  c_1 t_1 c_2 t_2 ... c_N t_N  p_1 ... p_Q
    The N cues are distinct, and so are the N targets. Each probe repeats one of the cues, again
    without repetition across probes. The answer to probe p_i is read at p_i's position, so the
    model never sees it.

    Token ids: cues in 1..V, targets in V+1..2V. vocab_size = 2V+2, where ids 0 and 2V+1 are
    reserved and never appear, so that the embedding table has 2V+2 rows (514 for V = 256).
    """

    def __init__(self, vocab_size=256, N_pairs=16, Q_probes=3, device='cpu'):
        assert N_pairs <= vocab_size and Q_probes <= N_pairs
        self.device = device
        self.V_base = vocab_size
        self.vocab_size = 2 * vocab_size + 2
        self.N = N_pairs
        self.Q = Q_probes
        self.cue_offset = 1
        self.target_offset = vocab_size + 1

    def generate_batch(self, batch_size):
        """Returns sequences (B, 2N+Q), y_indices (Q,) and probe_targets (B, Q)."""
        B, N, Q, dev = batch_size, self.N, self.Q, self.device
        cues = torch.rand(B, self.V_base, device=dev).argsort(dim=-1)[:, :N] + self.cue_offset
        targets = torch.rand(B, self.V_base, device=dev).argsort(dim=-1)[:, :N] + self.target_offset

        probe_idx = torch.rand(B, N, device=dev).argsort(dim=-1)[:, :Q]
        probe_cues = torch.gather(cues, 1, probe_idx)
        probe_targets = torch.gather(targets, 1, probe_idx)

        sequences = torch.stack([cues, targets], dim=-1).view(B, 2 * N)
        sequences = torch.cat([sequences, probe_cues], dim=1)
        y_indices = torch.arange(2 * N, 2 * N + Q, device=dev)
        return sequences, y_indices, probe_targets

    def get_bayesian_posterior(self, targets):
        """The posterior is a point mass on the bound target."""
        return F.one_hot(targets, num_classes=self.vocab_size).float()


class OverwriteRecallWindTunnel:
    """Associative recall with key overwrites.

    After the N initial pairs come M overwrite pairs, each rebinding one of the N cues (M distinct
    cues) to a fresh target. A probe on a rebound cue must return the most recent target.

    Layout (length 2N + 2M + Q):  N initial pairs, M overwrite pairs, Q probes.
    Returns, besides the probe targets, was_overwritten (B, Q): whether each probe's cue was rebound.
    """

    def __init__(self, vocab_size=256, N_pairs=16, N_overwrites=8, Q_probes=3, device='cpu'):
        assert 0 <= N_overwrites <= N_pairs, "N_overwrites cannot exceed N_pairs"
        assert N_pairs + N_overwrites <= vocab_size and Q_probes <= N_pairs
        self.device = device
        self.V_base = vocab_size
        self.vocab_size = 2 * vocab_size + 2
        self.N = N_pairs
        self.M = N_overwrites
        self.Q = Q_probes
        self.cue_offset = 1
        self.target_offset = vocab_size + 1

    def generate_batch(self, batch_size):
        """Returns sequences (B, 2N+2M+Q), y_indices (Q,), probe_targets (B, Q), was_overwritten (B, Q)."""
        B, N, M, Q, dev = batch_size, self.N, self.M, self.Q, self.device

        cues = torch.rand(B, self.V_base, device=dev).argsort(dim=-1)[:, :N] + self.cue_offset
        # N + M distinct targets: the first N are the initial bindings, the next M the overwrites
        targets_all = torch.rand(B, self.V_base, device=dev).argsort(dim=-1)[:, :N + M] + self.target_offset
        targets_init, targets_overwrite = targets_all[:, :N], targets_all[:, N:]

        overwrite_slot = torch.rand(B, N, device=dev).argsort(dim=-1)[:, :M]
        overwrite_cues = torch.gather(cues, 1, overwrite_slot)
        final_targets = targets_init.clone()
        final_targets.scatter_(1, overwrite_slot, targets_overwrite)

        probe_slot = torch.rand(B, N, device=dev).argsort(dim=-1)[:, :Q]
        probe_cues = torch.gather(cues, 1, probe_slot)
        probe_targets = torch.gather(final_targets, 1, probe_slot)
        was_overwritten = (probe_slot.unsqueeze(-1) == overwrite_slot.unsqueeze(1)).any(dim=-1)

        sequences = torch.cat([torch.stack([cues, targets_init], dim=-1).view(B, 2 * N),
                               torch.stack([overwrite_cues, targets_overwrite], dim=-1).view(B, 2 * M),
                               probe_cues], dim=1)
        y_indices = torch.arange(2 * N + 2 * M, 2 * N + 2 * M + Q, device=dev)
        return sequences, y_indices, probe_targets, was_overwritten

    def get_bayesian_posterior(self, targets):
        return F.one_hot(targets, num_classes=self.vocab_size).float()


class MultiHopRecallWindTunnel:
    """Associative recall that needs H lookups in a row.

    N chains node_0 -> node_1 -> ... -> node_H are stored as their N*H (parent, child) edges, with
    the edges of all chains shuffled together. A probe on a chain's node_0 must return its node_H.

    Layout (length 2NH + Q):  N*H edges, then Q probes. Node ids are 1..V, vocab_size = V+1.
    """

    def __init__(self, vocab_size=256, N_chains=8, hops=2, Q_probes=3, device='cpu'):
        assert hops >= 1, "hops must be >= 1"
        assert vocab_size >= N_chains * (hops + 1), "vocab_size too small for N_chains*(hops+1) distinct nodes"
        assert Q_probes <= N_chains
        self.device = device
        self.V_base = vocab_size
        self.vocab_size = vocab_size + 1
        self.N = N_chains
        self.H = hops
        self.Q = Q_probes
        self.node_offset = 1

    def generate_batch(self, batch_size):
        """Returns sequences (B, 2NH+Q), y_indices (Q,) and probe_targets (B, Q)."""
        B, N, H, Q, dev = batch_size, self.N, self.H, self.Q, self.device

        nodes = torch.rand(B, self.V_base, device=dev).argsort(dim=-1)[:, :N * (H + 1)] + self.node_offset
        nodes = nodes.view(B, N, H + 1)                  # nodes[:, i, 0] starts chain i, nodes[:, i, H] ends it
        src = nodes[:, :, :H].reshape(B, N * H)
        dst = nodes[:, :, 1:].reshape(B, N * H)

        perm = torch.rand(B, N * H, device=dev).argsort(dim=-1)
        src, dst = torch.gather(src, 1, perm), torch.gather(dst, 1, perm)

        chain = torch.rand(B, N, device=dev).argsort(dim=-1)[:, :Q]
        probe_starts = torch.gather(nodes[:, :, 0], 1, chain)
        probe_targets = torch.gather(nodes[:, :, H], 1, chain)

        sequences = torch.cat([torch.stack([src, dst], dim=-1).view(B, 2 * N * H), probe_starts], dim=1)
        y_indices = torch.arange(2 * N * H, 2 * N * H + Q, device=dev)
        return sequences, y_indices, probe_targets

    def get_bayesian_posterior(self, targets):
        return F.one_hot(targets, num_classes=self.vocab_size).float()
