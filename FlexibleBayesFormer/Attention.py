import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FlexMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_qk=None, dropout=0.1, pe_type="APE",
                 freeze_q=False, freeze_k=False, freeze_v=False, freeze_o=False):
        super().__init__()
        
        self.d_model = d_model
        self.d_qk = d_qk if d_qk is not None else d_model
        self.n_heads = n_heads
        
        assert self.d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert self.d_qk % n_heads == 0, "d_qk must be divisible by n_heads"
        
        self.d_v_head = self.d_model // n_heads
        self.d_qk_head = self.d_qk // n_heads
        self.pe_type = pe_type
        
        self.w_q = nn.Linear(self.d_model, self.d_qk, bias=False)
        self.w_k = nn.Linear(self.d_model, self.d_qk, bias=False)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)

        # Freeze individual matrices based on configuration
        if freeze_q: self.w_q.weight.requires_grad = False
        if freeze_k: self.w_k.weight.requires_grad = False
        if freeze_v: self.w_v.weight.requires_grad = False
        if freeze_o: self.w_o.weight.requires_grad = False
        
        self.dropout = nn.Dropout(dropout)
        self.cache = {} 
        
        if self.pe_type == "ALiBi":
            self.register_buffer("alibi_slopes", get_alibi_slopes(n_heads).view(1, n_heads, 1, 1))

    def forward(self, x, mask=None, save_cache=False, rope_cos=None, rope_sin=None):
        B, T, C = x.size()
        
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        
        if save_cache:
            self.cache['Q_raw'] = q.detach()
            self.cache['K_raw'] = k.detach()
            self.cache['V_raw'] = v.detach()
            
        q = q.view(B, T, self.n_heads, self.d_qk_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_qk_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_v_head).transpose(1, 2)
        
        if self.pe_type == "RoPE" and rope_cos is not None:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)
        
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_qk_head)
        
        if self.pe_type == "ALiBi":
            position_ids = torch.arange(T, device=x.device)
            rel_distances = position_ids.unsqueeze(1) - position_ids.unsqueeze(0)
            alibi_bias = (rel_distances * self.alibi_slopes).to(scores.device)
            scores = scores + alibi_bias
            
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        if save_cache:
            self.cache['attn_weights'] = attn_weights.detach()
            
        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        
        return self.w_o(out)
