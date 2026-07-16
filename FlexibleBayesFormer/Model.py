import torch
import torch.nn as nn
from .block import FlexTransformerBlock

class FlexFormer(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model=192, d_qk=None, 
                 n_heads=6, d_ffn=768, num_layers=6, dropout=0.1, pe_type="APE",
                 activation=nn.GELU(), freeze_emb=False, freeze_pos=False, 
                 layer_configs=None):
        super().__init__()
        self.d_model = d_model
        self.d_qk = d_qk if d_qk is not None else d_model
        self.pe_type = pe_type
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        if freeze_emb:
            self.token_emb.weight.requires_grad = False
            
        self.dropout = nn.Dropout(dropout)
        
        if self.pe_type == "APE":
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
            if freeze_pos:
                self.pos_emb.weight.requires_grad = False
                
        # Format n_heads into a list for per-layer configuration
        if isinstance(n_heads, int):
            self.n_heads_list = [n_heads] * num_layers
        else:
            assert len(n_heads) == num_layers, "n_heads list must match num_layers"
            self.n_heads_list = n_heads

        # Handle RoPE for potentially different head dimensions per layer
        if self.pe_type == "RoPE":
            self.rope_cos_cache = nn.ParameterDict()
            self.rope_sin_cache = nn.ParameterDict()
            unique_heads = set(self.n_heads_list)
            
            for nh in unique_heads:
                d_qk_head = self.d_qk // nh
                inv_freq = 1.0 / (10000 ** (torch.arange(0, d_qk_head, 2).float() / d_qk_head))
                t = torch.arange(max_seq_len).float()
                freqs = torch.einsum("i,j->ij", t, inv_freq)
                self.rope_cos_cache[str(nh)] = nn.Parameter(freqs.cos().unsqueeze(0).unsqueeze(2), requires_grad=False)
                self.rope_sin_cache[str(nh)] = nn.Parameter(freqs.sin().unsqueeze(0).unsqueeze(2), requires_grad=False)
        
        # Build blocks dynamically based on layer_configs
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            # Default layer config
            cfg = {
                'use_res_attn': True, 'use_res_ffn': True,
                'freeze_q': False, 'freeze_k': False, 'freeze_v': False, 'freeze_o': False
            }
            # Override with user-provided config if available
            if layer_configs and i < len(layer_configs):
                cfg.update(layer_configs[i])
                
            self.blocks.append(
                FlexTransformerBlock(
                    d_model=d_model, n_heads=self.n_heads_list[i], d_ffn=d_ffn, 
                    d_qk=self.d_qk, dropout=dropout, pe_type=pe_type, 
                    activation=activation, **cfg
                )
            )
            
        self.ln_final = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, mask=None, save_cache=False, use_ffn=True):
        B, T = idx.size()
        device = idx.device
        
        if mask is None:
            mask = torch.tril(torch.ones((T, T), device=device)).view(1, 1, T, T)
            
        x = self.token_emb(idx)
        
        if self.pe_type == "APE":
            pos = torch.arange(0, T, dtype=torch.long, device=device)
            x = x + self.pos_emb(pos)
            
        x = self.dropout(x)
        
        for i, block in enumerate(self.blocks):
            layer_use_ffn = use_ffn[i] if isinstance(use_ffn, list) else use_ffn
            
            rope_cos, rope_sin = None, None
            if self.pe_type == "RoPE":
                nh_str = str(self.n_heads_list[i])
                rope_cos = self.rope_cos_cache[nh_str][:, :T, :, :]
                rope_sin = self.rope_sin_cache[nh_str][:, :T, :, :]
                
            x = block(x, mask=mask, save_cache=save_cache, 
                      rope_cos=rope_cos, rope_sin=rope_sin, use_ffn=layer_use_ffn)
            
        x = self.ln_final(x)
        logits = self.unembed(x)
        
        return logits
