import torch
import torch.nn as nn

from .Block import FlexTransformerBlock

# Accepted spellings of pe_type. 'off' and 'NoPE' are the names used in the run configs.
PE_TYPES = {"APE": "APE", "none": "none", "off": "none", "NoPE": "none"}
LAYER_KEYS = {"freeze_q", "freeze_k", "freeze_v", "freeze_o", "use_res_attn", "use_res_ffn"}


class FlexFormer(nn.Module):
    """Decoder-only pre-LayerNorm transformer used for every experiment in the thesis.

    Args:
        vocab_size: number of input tokens and of output logits.
        max_seq_len: longest sequence, i.e. the size of the positional table.
        d_model: residual stream width.
        d_qk: total query/key width, split across heads (default d_model).
        n_heads: attention heads per layer.
        d_ffn: hidden width of the feedforward block.
        num_layers: number of blocks.
        dropout: dropout rate (0.0 in all thesis experiments).
        pe_type: "APE" (learned absolute positions) or "none" ("off", "NoPE").
        activation: FFN nonlinearity, as a module (default nn.GELU()).
        freeze_emb, freeze_pos: hold the token / position table at its initialisation.
        layer_norm: False puts the identity in place of every LayerNorm (LN1 and LN2 in each
            block, and the final one); default True. Added 2026-11-20 for the runs without
            LayerNorm; the default model and its initial weights are unchanged.
        layer_configs: optional list with one dict per layer, holding any of
            freeze_q, freeze_k, freeze_v, freeze_o (hold that projection at its
            initialisation) and use_res_attn, use_res_ffn (False drops the skip
            connection around attention / the FFN; default True).

    Attention projections carry no bias and the unembedding is not tied to the embedding.
    """

    def __init__(self, vocab_size, max_seq_len, d_model=192, d_qk=None,
                 n_heads=6, d_ffn=768, num_layers=6, dropout=0.0, pe_type="APE",
                 activation=None, freeze_emb=False, freeze_pos=False,
                 layer_configs=None, layer_norm=True):
        super().__init__()
        if pe_type not in PE_TYPES:
            raise ValueError(f"pe_type must be one of {sorted(PE_TYPES)}, got {pe_type!r}")
        if layer_configs is not None:
            if len(layer_configs) != num_layers:
                raise ValueError("layer_configs needs one dict per layer")
            for cfg in layer_configs:
                if set(cfg) - LAYER_KEYS:
                    raise ValueError(f"unknown layer_configs keys {set(cfg) - LAYER_KEYS}")

        self.d_model = d_model
        self.d_qk = d_qk if d_qk is not None else d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.pe_type = PE_TYPES[pe_type]
        self.layer_norm = layer_norm

        # Parameters are created in a fixed order (token table, position table, blocks,
        # final LayerNorm, unembedding), so a given seed always gives the same initial weights.
        self.token_emb = nn.Embedding(vocab_size, d_model)
        if freeze_emb:
            self.token_emb.weight.requires_grad = False

        self.dropout = nn.Dropout(dropout)

        if self.pe_type == "APE":
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
            if freeze_pos:
                self.pos_emb.weight.requires_grad = False

        self.blocks = nn.ModuleList([
            FlexTransformerBlock(d_model=d_model, n_heads=n_heads, d_ffn=d_ffn,
                                 d_qk=self.d_qk, dropout=dropout, activation=activation,
                                 layer_norm=layer_norm, **(layer_configs[i] if layer_configs else {}))
            for i in range(num_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model) if layer_norm else nn.Identity()
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx=None, inputs_embeds=None, save_cache=False, use_ffn=True, prefix_len=None):
        """Return logits of shape (B, T, vocab_size).

        Args:
            idx: token ids (B, T). Give either idx or inputs_embeds.
            inputs_embeds: input vectors (B, T, d_model), used in place of the token table.
                Positions are still added when pe_type is "APE".
            save_cache: keep queries, keys, values, attention weights and residuals.
            use_ffn: bool, or a list with one bool per layer. False skips that layer's FFN.
            prefix_len: if given, the first prefix_len positions attend to each other
                bidirectionally (the HMM header); everything else stays causal.
        """
        if inputs_embeds is not None:
            x = inputs_embeds
        elif idx is not None:
            x = self.token_emb(idx)
        else:
            raise ValueError("Must specify either idx or inputs_embeds")
        T, device = x.size(1), x.device

        if isinstance(use_ffn, (list, tuple)):
            if len(use_ffn) != self.num_layers:
                raise ValueError("use_ffn list needs one entry per layer")
        else:
            use_ffn = [use_ffn] * self.num_layers

        mask = torch.tril(torch.ones((T, T), device=device))
        if prefix_len is not None:
            mask[:prefix_len, :prefix_len] = 1.0
        mask = mask.view(1, 1, T, T)

        if self.pe_type == "APE":
            x = x + self.pos_emb(torch.arange(T, device=device))
        x = self.dropout(x)

        for block, layer_use_ffn in zip(self.blocks, use_ffn):
            x = block(x, mask=mask, save_cache=save_cache, use_ffn=layer_use_ffn)

        return self.unembed(self.ln_final(x))
