import torch.nn as nn
from .attention import FlexMultiHeadAttention
from .ffn import FeedForward

class FlexTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ffn, d_qk=None, dropout=0.1, pe_type="APE",
                 activation=nn.GELU(), use_res_attn=True, use_res_ffn=True,
                 freeze_q=False, freeze_k=False, freeze_v=False, freeze_o=False):
        super().__init__()
        
        self.use_res_attn = use_res_attn
        self.use_res_ffn = use_res_ffn
        
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = FlexMultiHeadAttention(
            d_model, n_heads, d_qk, dropout, pe_type, 
            freeze_q, freeze_k, freeze_v, freeze_o
        )
        
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ffn, dropout, activation)
        self.cache = {}

    def forward(self, x, mask=None, save_cache=False, rope_cos=None, rope_sin=None, use_ffn=True):
        if save_cache:
            self.cache['residual_pre_attn'] = x.detach()
            
        attn_out = self.attn(
            self.ln1(x), mask=mask, save_cache=save_cache, 
            rope_cos=rope_cos, rope_sin=rope_sin
        )
        
        # Toggleable Attention Residual
        x = x + attn_out if self.use_res_attn else attn_out
        
        if save_cache:
            self.cache['residual_post_attn'] = x.detach()
            
        if use_ffn:
            ffn_out = self.ffn(self.ln2(x))
            # Toggleable MLP Residual
            x = x + ffn_out if self.use_res_ffn else ffn_out
        
        if save_cache:
            self.cache['residual_post_ffn'] = x.detach()
            
        return x
