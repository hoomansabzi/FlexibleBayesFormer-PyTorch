import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, cos, sin):
    """Applies Rotary Position Embedding to the query/key tensors."""
    # cos, sin shape from Model.py: (1, T, 1, head_dim // 2)
    # x shape: (B, n_heads, T, head_dim)
    
    # Transpose to align dimensions: (1, 1, T, head_dim // 2)
    cos = cos.transpose(1, 2)
    sin = sin.transpose(1, 2)
    
    # Duplicate cos/sin to match the full head_dim
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    
    # Apply rotation
    return (x * cos) + (rotate_half(x) * sin)

def get_alibi_slopes(n_heads):
    """Calculates ALiBi slopes for attention bias."""
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]
    
    if math.log2(n_heads).is_integer():
        slopes = get_slopes_power_of_2(n_heads)
    else:
        closest_power_of_2 = 2**math.floor(math.log2(n_heads))
        slopes = get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2*closest_power_of_2)[0::2][:n_heads-closest_power_of_2]
    return torch.tensor(slopes)

class FlexMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_qk=None, dropout=0.1, pe_type="APE",
...

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
