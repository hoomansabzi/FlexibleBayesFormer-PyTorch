import math

import torch.nn as nn
import torch.nn.functional as F


class FlexMultiHeadAttention(nn.Module):
    """Multi-head attention with a query/key width d_qk independent of d_model.

    All four projections are bias-free. Each one can be frozen at its
    initialisation. With save_cache=True the forward pass keeps
    Q_raw, K_raw, V_raw (before the head split) and attn_weights (B, H, T, T)
    in self.cache.
    """

    def __init__(self, d_model, n_heads, d_qk=None, dropout=0.0,
                 freeze_q=False, freeze_k=False, freeze_v=False, freeze_o=False):
        super().__init__()
        self.d_model = d_model
        self.d_qk = d_qk if d_qk is not None else d_model
        self.n_heads = n_heads

        assert self.d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert self.d_qk % n_heads == 0, "d_qk must be divisible by n_heads"

        self.d_v_head = self.d_model // n_heads
        self.d_qk_head = self.d_qk // n_heads

        self.w_q = nn.Linear(self.d_model, self.d_qk, bias=False)
        self.w_k = nn.Linear(self.d_model, self.d_qk, bias=False)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)

        if freeze_q: self.w_q.weight.requires_grad = False
        if freeze_k: self.w_k.weight.requires_grad = False
        if freeze_v: self.w_v.weight.requires_grad = False
        if freeze_o: self.w_o.weight.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.cache = {}

    def forward(self, x, mask=None, save_cache=False):
        B, T, _ = x.size()

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

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_qk_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        if save_cache:
            self.cache['attn_weights'] = attn_weights.detach()

        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.w_o(out)
