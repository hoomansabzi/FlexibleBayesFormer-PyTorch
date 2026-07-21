import torch
import torch.nn as nn

class ContinuousHMMModel(nn.Module):
    def __init__(self, flexformer_backbone, d_model, matrix_dim=5):
        super().__init__()
        self.backbone = flexformer_backbone
        
        # Linear projection for the continuous rows of T and E
        self.matrix_proj = nn.Linear(matrix_dim, d_model)
        
    def forward(self, T_matrix, E_matrix, observations):
        """
        T_matrix: (B, S, S) continuous transition probabilities
        E_matrix: (B, S, V) continuous emission probabilities
        observations: (B, K) discrete token indices
        """
        # Project the continuous rows (e.g., 5 rows of T, 5 rows of E)
        header_T = self.matrix_proj(T_matrix) # (B, S, d_model)
        header_E = self.matrix_proj(E_matrix) # (B, S, d_model)
        
        # Embed the discrete observation tokens using the backbone
        obs_embeds = self.backbone.token_emb(observations) # (B, K, d_model)
        
        # Concatenate: [T_rows, E_rows, observations]
        # Total sequence length: S + S + K = 10 + K
        inputs_embeds = torch.cat([header_T, header_E, obs_embeds], dim=1) 
        
        # Pass directly to the backbone via inputs_embeds
        logits = self.backbone(inputs_embeds=inputs_embeds)
        
        return logits
