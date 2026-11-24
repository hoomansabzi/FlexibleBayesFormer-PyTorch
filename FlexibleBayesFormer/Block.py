import torch.nn as nn

from .Attention import FlexMultiHeadAttention
from .FFN import FeedForward


class FlexTransformerBlock(nn.Module):
    """Pre-LayerNorm block: x + attn(LN1(x)), then optionally + ffn(LN2(x)).

    layer_norm=False puts the identity in place of LN1 and LN2 (no LayerNorm in the block).

    use_res_attn=False drops the skip connection around attention, so the block
    computes attn(LN1(x)) and the input is lost; use_res_ffn=False does the same
    around the FFN.

    With save_cache=True the residual stream is kept in self.cache at three
    points: residual_pre_attn, residual_post_attn and residual_post_ffn. When the
    FFN is skipped, residual_post_ffn equals residual_post_attn.
    """

    def __init__(self, d_model, n_heads, d_ffn, d_qk=None, dropout=0.0, activation=None,
                 freeze_q=False, freeze_k=False, freeze_v=False, freeze_o=False,
                 use_res_attn=True, use_res_ffn=True, layer_norm=True):
        super().__init__()
        self.use_res_attn = use_res_attn
        self.use_res_ffn = use_res_ffn
        # LayerNorm draws no random numbers at initialisation, so layer_norm leaves the other
        # initial weights of a seed unchanged.
        self.ln1 = nn.LayerNorm(d_model) if layer_norm else nn.Identity()
        self.attn = FlexMultiHeadAttention(d_model, n_heads, d_qk, dropout,
                                           freeze_q, freeze_k, freeze_v, freeze_o)
        self.ln2 = nn.LayerNorm(d_model) if layer_norm else nn.Identity()
        self.ffn = FeedForward(d_model, d_ffn, dropout, activation)
        self.cache = {}

    def forward(self, x, mask=None, save_cache=False, use_ffn=True):
        if save_cache:
            self.cache['residual_pre_attn'] = x.detach()

        out = self.attn(self.ln1(x), mask=mask, save_cache=save_cache)
        x = x + out if self.use_res_attn else out

        if save_cache:
            self.cache['residual_post_attn'] = x.detach()

        if use_ffn:
            out = self.ffn(self.ln2(x))
            x = x + out if self.use_res_ffn else out

        if save_cache:
            self.cache['residual_post_ffn'] = x.detach()

        return x
