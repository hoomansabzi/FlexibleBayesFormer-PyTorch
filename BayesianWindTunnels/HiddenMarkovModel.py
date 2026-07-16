import torch
import torch.nn.functional as F

class HMMWindTunnel:
    """
    Tests Recursive Belief Updating (Transport).
    S=5 hidden states, V=5 observations.
    """
    def __init__(self, S=5, V=5):
        self.S = S
        self.V = V
        # Total vocab: S*S (Transition) + S*V (Emission) + V (Observations)
        # We will flatten T and E to token indices to act as the "Header"
        self.vocab_size = 100 # Safe padding for token IDs

    def generate_batch(self, batch_size, K=20):
        """
        Generates HMM rules (T, E) and sequence of observations.
        """
        # 1. Generate Rules from Dirichlet(1,1,1,1,1)
        # Transition Matrix (T): S x S
        T = torch.distributions.Dirichlet(torch.ones(self.S)).sample((batch_size, self.S))
        # Emission Matrix (E): S x V
        E = torch.distributions.Dirichlet(torch.ones(self.V)).sample((batch_size, self.S))
        # Initial State (pi_0)
        pi_0 = torch.distributions.Dirichlet(torch.ones(self.S)).sample((batch_size,))

        observations = torch.zeros(batch_size, K, dtype=torch.long)
        hidden_states = torch.zeros(batch_size, K, dtype=torch.long)

        # 2. Simulate the HMM
        for b in range(batch_size):
            current_state = torch.multinomial(pi_0[b], 1).item()
            for t in range(K):
                # Emit observation
                obs = torch.multinomial(E[b, current_state], 1).item()
                observations[b, t] = obs
                hidden_states[b, t] = current_state
                # Transition to next state
                current_state = torch.multinomial(T[b, current_state], 1).item()

        return T, E, pi_0, observations, hidden_states

    def forward_algorithm(self, T, E, pi_0, observations):
        """
        The exact Bayesian Ground Truth: Calculates p(s_t | o_{1:t})
        """
        batch_size, K = observations.shape
        posteriors = torch.zeros(batch_size, K, self.S)

        for b in range(batch_size):
            alpha = pi_0[b] # Initial belief
            
            for t in range(K):
                obs = observations[b, t]
                # 1. Update with observation (Multiply by Emission prob)
                alpha = alpha * E[b, :, obs]
                # Normalize
                alpha = alpha / alpha.sum()
                
                # Save posterior for current step
                posteriors[b, t] = alpha
                
                # 2. Transport belief forward (Multiply by Transition matrix)
                # alpha_{t} = sum_{s'} T(s | s') * alpha_{t-1}(s')
                alpha = torch.matmul(alpha, T[b])

        return posteriors
