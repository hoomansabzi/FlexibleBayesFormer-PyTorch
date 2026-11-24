"""Unit tests for BayesianWindTunnels. CPU only, under a minute.

    python tests/test_windtunnels.py        (or: python -m pytest tests/)
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from BayesianWindTunnels import (BijectionWindTunnel, HMMWindTunnel,  # noqa: E402
                                 MultiHopRecallWindTunnel, OverwriteRecallWindTunnel,
                                 RecallWindTunnel, RegressionWindTunnel)

V = 16


def entropy_nats(p):
    return -(p * torch.log(p.clamp_min(1e-30))).sum(-1)


def repetition_floor(p, V=V):
    """Closed form for seq_pairs = V: step 1 is new; before step t >= 2, 1 + Binomial(t-2, 1-p)
    bindings are known, and the step is new with probability 1-p."""
    q, total = 1 - p, math.log(V)
    for t in range(2, V + 1):
        total += q * sum(math.comb(t - 2, j) * q ** j * p ** (t - 2 - j) * math.log(V - 1 - j)
                         for j in range(t - 1))
    return total / V


def bijection_floor(seq_pairs=None, repeat_prob=0.0, n=20000, seed=0):
    torch.manual_seed(seed)
    tun = BijectionWindTunnel(V=V)
    seq, yi = tun.generate_batch(n, seq_pairs=seq_pairs, repeat_prob=repeat_prob)
    return entropy_nats(tun.get_bayesian_posterior(seq, yi)).mean().item()


# --------------------------------------------------------------------------- bijection
def test_bijection_plain_is_two_permutations():
    torch.manual_seed(0)
    tun = BijectionWindTunnel(V=V)
    seq, yi = tun.generate_batch(256)
    assert seq.shape == (256, 2 * V) and tun.vocab_size == 2 * V + 1
    assert torch.equal(yi, torch.arange(1, 2 * V, 2))
    assert torch.equal(seq[:, 0::2].sort(-1).values, torch.arange(1, V + 1).expand(256, V))
    assert torch.equal(seq[:, 1::2].sort(-1).values, torch.arange(V + 1, 2 * V + 1).expand(256, V))


def test_bijection_bindings_are_consistent():
    """With repeats, the same x always comes with the same y, and different x with different y."""
    torch.manual_seed(1)
    seq, _ = BijectionWindTunnel(V=V).generate_batch(512, seq_pairs=V + 5, repeat_prob=0.4)
    for row in seq.tolist():
        pairs = set(zip(row[0::2], row[1::2]))
        assert len({x for x, _ in pairs}) == len(pairs) == len({y for _, y in pairs})


def test_bijection_repeat_statistics():
    """Repeats happen with probability p while bindings remain, and pick a uniform earlier binding."""
    torch.manual_seed(2)
    p, n = 0.4, 8000
    seq, _ = BijectionWindTunnel(V=V).generate_batch(n, repeat_prob=p)
    eligible = repeats = 0
    rel_ranks = []
    for row in seq[:, 0::2].tolist():
        order = []                              # x-values in order of first appearance
        for t, x in enumerate(row):
            if t > 0:
                eligible += 1
            if x in order:
                repeats += 1
                rel_ranks.append((order.index(x) + 0.5) / len(order))
            else:
                order.append(x)
    assert abs(repeats / eligible - p) < 0.01, repeats / eligible
    assert abs(sum(rel_ranks) / len(rel_ranks) - 0.5) < 0.01


def test_bijection_forced_repeats():
    """Past V pairs every pair repeats; the first pair never does."""
    torch.manual_seed(3)
    tun = BijectionWindTunnel(V=V)
    seq, yi = tun.generate_batch(256, seq_pairs=V + 3)
    post = tun.get_bayesian_posterior(seq, yi)
    assert torch.equal(post[:, V:].max(-1).values, torch.ones(256, 3))
    assert torch.allclose(post[:, 0, V + 1:], torch.full((256, V), 1 / V))


def test_bijection_posterior():
    """Rows sum to one, the true y always has positive probability, repeats are point masses,
    and a novel step k has 1/(V - number of earlier novel steps) on each unused y."""
    torch.manual_seed(4)
    tun = BijectionWindTunnel(V=V)
    seq, yi = tun.generate_batch(512, repeat_prob=0.3)
    post = tun.get_bayesian_posterior(seq, yi)
    assert post.shape == (512, V, 2 * V + 1)
    assert torch.allclose(post.sum(-1), torch.ones(512, V))
    assert (post.gather(2, seq[:, yi].unsqueeze(-1)) > 0).all()
    for b in range(64):
        seen = {}
        for k in range(V):
            x, y = seq[b, 2 * k].item(), seq[b, 2 * k + 1].item()
            row = post[b, k]
            if x in seen:
                assert row[y] == 1.0
            else:
                free = V - len(seen)
                assert torch.allclose(row[V + 1:].sum(), torch.tensor(1.0))
                assert math.isclose(row[y].item(), 1 / free, rel_tol=1e-6)
                assert all(row[u] == 0 for u in seen.values())
                seen[x] = y


def test_bijection_shared_vocab_and_sep():
    """shared_vocab: x and y both in 1..V, vocab V+1. sep: x y 0 triplets, y read every third
    position. The posterior is that of plain bijection, on the y values 1..V."""
    torch.manual_seed(5)
    tun = BijectionWindTunnel(V=V, shared_vocab=True)
    seq, yi = tun.generate_batch(256, sep=True)
    assert tun.vocab_size == V + 1 and seq.shape == (256, 3 * V)
    assert torch.equal(yi, torch.arange(1, 3 * V, 3))
    assert (seq[:, 2::3] == 0).all() and seq[:, 0::3].min() >= 1 and seq[:, 1::3].max() <= V
    for b in range(32):                                    # x's and y's are each a permutation of 1..V
        assert sorted(seq[b, 0::3].tolist()) == list(range(1, V + 1))
        assert sorted(seq[b, 1::3].tolist()) == list(range(1, V + 1))
    post = tun.get_bayesian_posterior(seq, yi)
    assert post.shape == (256, V, V + 1) and (post[..., 0] == 0).all()
    plain = BijectionWindTunnel(V=V)
    seq2 = torch.stack([seq[:, 0::3], seq[:, 1::3] + V], -1).view(256, 2 * V)   # the same bindings, disjoint ids
    assert torch.allclose(post[..., 1:], plain.get_bayesian_posterior(seq2, torch.arange(1, 2 * V, 2))[..., V + 1:])


def test_bijection_floors():
    """Plain log(V!)/V = 1.91699 and recall at the end log(V!)/(V+3) = 1.61431 exactly;
    repetition 1.4879 (p = 0.4) and 1.6492 (p = 0.3) from the closed form."""
    log_fact = math.lgamma(V + 1)
    assert abs(bijection_floor() - log_fact / V) < 1e-5
    assert abs(bijection_floor(seq_pairs=V + 3) - log_fact / (V + 3)) < 1e-5
    assert abs(repetition_floor(0.4) - 1.4879) < 1e-4 and abs(repetition_floor(0.3) - 1.6492) < 1e-4
    for p in (0.4, 0.3):
        assert abs(bijection_floor(repeat_prob=p, n=100000) - repetition_floor(p)) < 0.003


# ------------------------------------------------------------------------------ recall
def test_recall_layout():
    torch.manual_seed(5)
    tun = RecallWindTunnel(vocab_size=256, N_pairs=16, Q_probes=3)
    seq, yi, tgt = tun.generate_batch(512)
    assert seq.shape == (512, 35) and tun.vocab_size == 514
    assert torch.equal(yi, torch.arange(32, 35))
    cues, targets = seq[:, 0:32:2], seq[:, 1:32:2]
    assert cues.min() >= 1 and cues.max() <= 256 and targets.min() >= 257 and targets.max() <= 512
    for b in range(512):
        assert len(set(cues[b].tolist())) == 16 and len(set(targets[b].tolist())) == 16
        probes = seq[b, 32:].tolist()
        assert len(set(probes)) == 3
        bound = dict(zip(cues[b].tolist(), targets[b].tolist()))
        assert [bound[c] for c in probes] == tgt[b].tolist()
    assert torch.equal(tun.get_bayesian_posterior(tgt).argmax(-1), tgt)


def test_overwrite_returns_most_recent_binding():
    torch.manual_seed(6)
    tun = OverwriteRecallWindTunnel(vocab_size=256, N_pairs=16, N_overwrites=8, Q_probes=3)
    seq, yi, tgt, was = tun.generate_batch(512)
    assert seq.shape == (512, 2 * 16 + 2 * 8 + 3) and torch.equal(yi, torch.arange(48, 51))
    for b in range(512):
        store = seq[b, :48].tolist()
        latest, count = {}, {}
        for c, t in zip(store[0::2], store[1::2]):
            latest[c] = t
            count[c] = count.get(c, 0) + 1
        probes = seq[b, 48:].tolist()
        assert [latest[c] for c in probes] == tgt[b].tolist()
        assert [count[c] == 2 for c in probes] == was[b].tolist()
        assert len(set(store[1::2])) == 24                   # every target is fresh


def test_multihop_composes_lookups():
    torch.manual_seed(7)
    tun = MultiHopRecallWindTunnel(vocab_size=256, N_chains=8, hops=2, Q_probes=3)
    seq, yi, tgt = tun.generate_batch(256)
    assert seq.shape == (256, 2 * 8 * 2 + 3) and torch.equal(yi, torch.arange(32, 35))
    for b in range(256):
        edges = dict(zip(seq[b, 0:32:2].tolist(), seq[b, 1:32:2].tolist()))
        assert len(edges) == 16
        for start, end in zip(seq[b, 32:].tolist(), tgt[b].tolist()):
            assert edges[edges[start]] == end


# --------------------------------------------------------------------------------- HMM
def reference_forward(T, E, pi_0, obs):
    """The textbook loop, one sequence at a time."""
    out = torch.zeros(*obs.shape, T.shape[1])
    for b in range(obs.shape[0]):
        a = pi_0[b]
        for t in range(obs.shape[1]):
            a = a * E[b, :, obs[b, t]]
            a = a / a.sum()
            out[b, t] = a
            a = a @ T[b]
    return out


def test_hmm_sampling_and_forward_algorithm():
    torch.manual_seed(8)
    tun = HMMWindTunnel(S=5, V=5, alpha=1.0)
    T, E, pi_0, obs, states = tun.generate_batch(256, K=20)
    assert T.shape == (256, 5, 5) and E.shape == (256, 5, 5) and pi_0.shape == (256, 5)
    for M in (T, E, pi_0):
        assert torch.allclose(M.sum(-1), torch.ones(M.shape[:-1]), atol=1e-5)
    assert obs.shape == states.shape == (256, 20) and obs.max() < 5 and states.max() < 5
    post = tun.forward_algorithm(T, E, pi_0, obs)
    assert torch.allclose(post.sum(-1), torch.ones(256, 20), atol=1e-5)
    assert torch.allclose(post, reference_forward(T, E, pi_0, obs), atol=1e-5)


def test_hmm_hidden_pi0_error():
    """The docstring's number: a filter started from a uniform pi_0 scores ~0.030 bits at K = 20."""
    torch.manual_seed(9)
    tun = HMMWindTunnel(S=5, V=5, alpha=1.0)
    T, E, pi_0, obs, _ = tun.generate_batch(20000, K=20)
    h = lambda p: -(p * torch.log2(p + 1e-12)).sum(-1)
    exact = tun.forward_algorithm(T, E, pi_0, obs)
    blind = tun.forward_algorithm(T, E, torch.full_like(pi_0, 0.2), obs)
    mae = (h(exact) - h(blind)).abs().mean(0)
    assert abs(mae.mean().item() - 0.030) < 0.003
    assert mae[0] > 0.3 and mae[10:].max() < 0.005


# -------------------------------------------------------------------------- regression
def test_regression_family_posterior_is_exact():
    """Mean, covariance and log marginal likelihood against the direct k x k formulas."""
    tun = RegressionWindTunnel.wakayama()
    g = torch.Generator().manual_seed(10)
    X, y, x_q, y_q, w, fam = tun.sample(64, 7, generator=g)
    X, y = X.double(), y.double()
    for i, name in enumerate(tun.families):
        mean, cov, logml = tun.family_posterior(X, y, i)
        Phi = torch.tanh(X) if name == 'tanh' else X
        mu0 = torch.full((64, 5), tun.prior_means[i], dtype=torch.float64)
        C = tun.sigma2 * torch.eye(7, dtype=torch.float64) + tun.tau2 * Phi @ Phi.transpose(1, 2)
        direct = torch.distributions.MultivariateNormal((Phi @ mu0.unsqueeze(-1)).squeeze(-1), C)
        assert torch.allclose(logml, direct.log_prob(y), atol=1e-6)
        A = torch.eye(5, dtype=torch.float64) / tun.tau2 + Phi.transpose(1, 2) @ Phi / tun.sigma2
        assert torch.allclose(cov, torch.linalg.inv(A), atol=1e-8)
        rhs = mu0 / tun.tau2 + (Phi.transpose(1, 2) @ y.unsqueeze(-1)).squeeze(-1) / tun.sigma2
        assert torch.allclose(mean, torch.linalg.solve(A, rhs), atol=1e-6)


def test_regression_predictive():
    tun = RegressionWindTunnel.wakayama()
    X, y, x_q, *_ = tun.sample(128, 5, generator=torch.Generator().manual_seed(11))
    pi, mu, var = tun.predictive_components(X, y, x_q)
    assert torch.allclose(pi.sum(-1), torch.ones(128))
    mean, v = tun.predictive(X, y, x_q)
    assert torch.allclose(mean, (pi * mu).sum(-1))
    assert torch.allclose(v, (pi * (var + mu ** 2)).sum(-1) - mean ** 2, atol=1e-4)
    binned, mean2, var2 = tun.bayes_target(X, y, x_q)
    assert torch.allclose(binned.sum(-1), torch.ones(128)) and torch.allclose(mean2, mean)


def test_regression_tokens():
    tun = RegressionWindTunnel.agarwal()
    X, y, x_q, *_ = tun.sample(4, 6, generator=torch.Generator().manual_seed(12))
    paired = tun.tokens(X, y, x_q)
    assert paired.shape == (4, 7, 5)
    assert torch.equal(paired[:, :6, :3], X) and torch.equal(paired[:, :6, 3], y)
    assert torch.equal(paired[:, 6, :3], x_q) and (paired[:, 6, 3:] == torch.tensor([0., 1.])).all()
    inter = tun.tokens(X, y, x_q, interleaved=True)
    assert inter.shape == (4, 13, 5)
    assert torch.equal(inter[:, 0:12:2, :3], X) and torch.equal(inter[:, 1:12:2, 3], y)
    assert (inter[:, 1:12:2, 4] == 1).all() and torch.equal(inter[:, 12, :3], x_q)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
