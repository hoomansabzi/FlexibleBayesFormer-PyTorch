import torch
import torch.nn.functional as F

class RecallWindTunnel:
    """
    Tests the Random-Access Binding Primitive.
    Fully vectorized for high-speed batched generation.
    """
    def __init__(self, vocab_size=256, N_pairs=16, Q_probes=3, use_sep=True, use_probe=True, device='cpu'):
        self.device = device
        self.vocab_size = (vocab_size * 2) + 2 
        self.N = N_pairs
        self.Q = Q_probes
        
        self.use_sep = use_sep
        self.use_probe = use_probe
        
        self.sep_token = 0
        self.cue_offset = 1
        self.target_offset = vocab_size + 1
        self.probe_token = self.vocab_size - 1
        self.V_base = vocab_size
      
    def generate_batch(self, batch_size):
        # 1. Vectorized distinct sampling
        # This draws N distinct cues and N distinct targets for the entire batch instantly
        rand_cues = torch.rand(batch_size, self.V_base, device=self.device)
        rand_targs = torch.rand(batch_size, self.V_base, device=self.device)
        
        cues = rand_cues.argsort(dim=-1)[:, :self.N] + self.cue_offset
        targets = rand_targs.argsort(dim=-1)[:, :self.N] + self.target_offset
        
        # 2. Select Q random probes from the N cues we just generated
        probe_indices = torch.rand(batch_size, self.N, device=self.device).argsort(dim=-1)[:, :self.Q]
        probe_cues = torch.gather(cues, 1, probe_indices)
        probe_targets = torch.gather(targets, 1, probe_indices) 
        
        # 3. Dynamic Sequence Construction
        pair_block = 2 + int(self.use_sep)
        probe_block = 1 + int(self.use_probe) + int(self.use_sep)
        
        storage_len = self.N * pair_block
        seq_len = storage_len + (self.Q * probe_block)
        
        sequences = torch.zeros(batch_size, seq_len, dtype=torch.long, device=self.device)
        
        # ==========================================
        # OPTIMIZATION: y_indices is now a flat 1D tensor!
        # ==========================================
        y_indices = torch.zeros(self.Q, dtype=torch.long, device=self.device)
        
        # Fill Storage Phase
        sequences[:, 0:storage_len:pair_block] = cues
        sequences[:, 1:storage_len:pair_block] = targets
        if self.use_sep:
            sequences[:, 2:storage_len:pair_block] = self.sep_token
            
        # Fill Retrieval Phase
        for i in range(self.Q):
            start_idx = storage_len + (i * probe_block)
            
            if self.use_probe:
                sequences[:, start_idx] = self.probe_token
                cue_idx = start_idx + 1
            else:
                cue_idx = start_idx
                
            sequences[:, cue_idx] = probe_cues[:, i]
            
            # OPTIMIZATION: Just map the static scalar integer
            y_indices[i] = cue_idx
            
            if self.use_sep:
                sequences[:, cue_idx + 1] = self.sep_token # Placeholder for model's guess
                
        return sequences, y_indices, probe_targets

    def get_bayesian_posterior(self, targets):
        return F.one_hot(targets, num_classes=self.vocab_size).float()
