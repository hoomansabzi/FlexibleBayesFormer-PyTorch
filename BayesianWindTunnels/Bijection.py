import torch
import torch.nn.functional as F


class BijectionWindTunnel:
    """Bijection learning: infer a random one-to-one map x -> y from its revealed pairs.

    Each sequence draws a fresh bijection between V x-values and V y-values and shows its pairs
    one at a time, densely: x_1 y_1 x_2 y_2 ... The first pair, and every pair with probability
    1 - repeat_prob after it, reveals the next unused binding. Otherwise the pair repeats a
    binding already seen, chosen uniformly. Once all V bindings are used, every further pair is
    a repeat. Three settings are used in the thesis:

        plain bijection        repeat_prob=0,   seq_pairs=V
        repetition             repeat_prob=p,   seq_pairs=V
        recall at the end      repeat_prob=0,   seq_pairs=V+3 (the last three pairs repeat)

    Token ids: x in 1..V, y in V+1..2V, so vocab_size = 2V+1 (id 0 is the separator, unused unless
    sep=True). Two options, not used in the thesis (Task 1's shared-vocabulary notebook was removed
    on 2026-11-06; kept for future experiments and tested in tests/test_windtunnels.py):

        shared_vocab=True      x and y take values from the same tokens 1..V (vocab_size = V+1), so
                               a token no longer says whether it is an x or a y
        sep=True               each pair is followed by the separator: x_1 y_1 0 x_2 y_2 0 ...

    y_indices are the positions of the y tokens; they locate the pairs. The prediction for y_k is
    read one position earlier, at x_k (y_indices - 1), so the model never sees the answer it
    predicts (TASKS_1-3.md, Section 0). get_bayesian_posterior conditions on x_k and the earlier
    pairs, which is exactly what the model has seen at x_k.
    """

    def __init__(self, V=16, device='cpu', shared_vocab=False):
        self.V = V
        self.device = device
        self.shared_vocab = shared_vocab
        self.sep_token = 0
        self.x_offset = 1
        self.y_offset = 1 if shared_vocab else V + 1
        self.vocab_size = V + 1 if shared_vocab else 2 * V + 1

    def generate_batch(self, batch_size, seq_pairs=None, repeat_prob=0.0, sep=False):
        """Returns sequences (B, 2 * seq_pairs), or (B, 3 * seq_pairs) with sep, and y_indices (seq_pairs,)."""
        B, V, dev = batch_size, self.V, self.device
        P = V if seq_pairs is None else seq_pairs

        # Binding i of a sequence is (perm_x[i], perm_y[i]); new bindings are revealed in order i = 0, 1, ...
        perm_x = torch.rand(B, V, device=dev).argsort(dim=-1)
        perm_y = torch.rand(B, V, device=dev).argsort(dim=-1)

        n_seen = torch.zeros(B, dtype=torch.long, device=dev)
        slot = torch.empty(B, P, dtype=torch.long, device=dev)
        for t in range(P):
            u, r = torch.rand(B, device=dev), torch.rand(B, device=dev)
            repeat = (n_seen > 0) & ((u < repeat_prob) | (n_seen >= V))
            earlier = (r * n_seen).long()                  # uniform over the n_seen bindings so far
            slot[:, t] = torch.where(repeat, earlier, n_seen)
            n_seen += (~repeat).long()

        x = perm_x.gather(1, slot) + self.x_offset
        y = perm_y.gather(1, slot) + self.y_offset
        parts = [x, y, torch.full_like(x, self.sep_token)] if sep else [x, y]
        sequences = torch.stack(parts, dim=-1).view(B, len(parts) * P)
        y_indices = torch.arange(1, len(parts) * P, len(parts), device=dev)
        return sequences, y_indices

    def get_bayesian_posterior(self, sequences, y_indices):
        """Exact p(y_k | x_k, earlier pairs) at every y position, shape (B, P, vocab_size).

        If x_k appeared in an earlier pair, a point mass on its bound y; otherwise uniform over
        the y-values not used by any earlier pair.
        """
        x = sequences[:, y_indices - 1]
        y = sequences[:, y_indices] - self.y_offset                      # 0..V-1
        B, P = x.shape
        earlier = torch.ones(P, P, device=x.device).tril(-1)              # earlier[k, j] = 1 if j < k

        same_x = (x.unsqueeze(2) == x.unsqueeze(1)) & earlier.bool()      # (B, P, P)
        seen = same_x.any(-1, keepdim=True)                               # (B, P, 1)
        bound = y.gather(1, same_x.float().argmax(-1))                    # y of an earlier pair with x_k
        point_mass = F.one_hot(bound, self.V).float()

        used = (earlier @ F.one_hot(y, self.V).float()) > 0               # (B, P, V)
        free = (~used).float()
        uniform = free / free.sum(-1, keepdim=True).clamp(min=1)

        post = torch.zeros(B, P, self.vocab_size, device=x.device)
        post[..., self.y_offset:] = torch.where(seen, point_mass, uniform)
        return post
