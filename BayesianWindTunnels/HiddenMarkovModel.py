import torch


class HMMWindTunnel:
    """HMM filtering: track p(s_t | o_1..o_t) for a fresh hidden Markov model in every sequence.

    Each sequence draws every row of its transition matrix T (S x S), every row of its emission
    matrix E (S x V), and its initial distribution pi_0, from Dirichlet(alpha * 1). It then runs
    the chain for K steps. The target is the filtering posterior from the forward algorithm.

    As in Agarwal et al., the model's header shows T and E but not pi_0. A model that cannot see
    pi_0 can at best run the forward algorithm from E[pi_0], which is uniform, so the target
    keeps an error that no model can remove: about 0.030 bits of entropy MAE at K = 20 and
    alpha = 1, almost all of it at t = 1 and t = 2. To compute it, pass a uniform pi_0 to
    forward_algorithm and compare.

    Sampling and filtering are batched on `device`; the only Python loop is over time.
    """

    def __init__(self, S=5, V=5, alpha=1.0, device='cpu'):
        self.S = S
        self.V = V
        self.alpha = float(alpha)
        self.vocab_size = max(S, V)
        self.device = device

    def generate_batch(self, batch_size, K=20):
        """Returns T (B,S,S), E (B,S,V), pi_0 (B,S), observations (B,K) and hidden_states (B,K)."""
        B, S, V, dev = batch_size, self.S, self.V, self.device
        dirichlet = lambda n: torch.distributions.Dirichlet(
            torch.full((n,), self.alpha, device=dev, dtype=torch.float32))
        T = dirichlet(S).sample((B, S))
        E = dirichlet(V).sample((B, S))
        pi_0 = dirichlet(S).sample((B,))

        observations = torch.zeros(B, K, dtype=torch.long, device=dev)
        hidden_states = torch.zeros(B, K, dtype=torch.long, device=dev)
        rows = torch.arange(B, device=dev)
        state = torch.multinomial(pi_0, 1).squeeze(-1)
        for t in range(K):
            observations[:, t] = torch.multinomial(E[rows, state], 1).squeeze(-1)
            hidden_states[:, t] = state
            state = torch.multinomial(T[rows, state], 1).squeeze(-1)
        return T, E, pi_0, observations, hidden_states

    def forward_algorithm(self, T, E, pi_0, observations):
        """Filtering posteriors p(s_t | o_1..o_t), shape (B, K, S).

        alpha_1 is proportional to pi_0 * E[:, o_1], and alpha_t to (alpha_{t-1} T) * E[:, o_t].
        """
        B, K = observations.shape
        posteriors = torch.zeros(B, K, self.S, device=T.device)
        alpha = pi_0.clone()
        for t in range(K):
            emission = E.gather(2, observations[:, t].view(B, 1, 1).expand(B, self.S, 1)).squeeze(-1)
            alpha = alpha * emission
            alpha = alpha / (alpha.sum(-1, keepdim=True) + 1e-12)
            posteriors[:, t] = alpha
            alpha = torch.bmm(alpha.unsqueeze(1), T).squeeze(1)
        return posteriors
