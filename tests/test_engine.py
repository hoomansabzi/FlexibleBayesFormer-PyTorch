"""Unit tests for FlexibleBayesFormer. CPU only, a few seconds.

    python tests/test_engine.py        (or: python -m pytest tests/)
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from FlexibleBayesFormer import FlexFormer  # noqa: E402

V, T, B = 11, 9, 3


def make(seed=0, **kw):
    torch.manual_seed(seed)
    args = dict(vocab_size=V, max_seq_len=16, d_model=16, n_heads=2, d_ffn=32,
                num_layers=2, dropout=0.0, pe_type="APE")
    args.update(kw)
    return FlexFormer(**args).eval()


def tokens(seed=1):
    return torch.randint(0, V, (B, T), generator=torch.Generator().manual_seed(seed))


def test_shapes():
    m = make()
    assert m(tokens()).shape == (B, T, V)
    assert m(inputs_embeds=torch.randn(B, T, 16)).shape == (B, T, V)


def test_checkpoint_format():
    names = list(make(num_layers=1).state_dict())
    assert names == [
        "token_emb.weight", "pos_emb.weight",
        "blocks.0.ln1.weight", "blocks.0.ln1.bias",
        "blocks.0.attn.w_q.weight", "blocks.0.attn.w_k.weight",
        "blocks.0.attn.w_v.weight", "blocks.0.attn.w_o.weight",
        "blocks.0.ln2.weight", "blocks.0.ln2.bias",
        "blocks.0.ffn.net.0.weight", "blocks.0.ffn.net.0.bias",
        "blocks.0.ffn.net.3.weight", "blocks.0.ffn.net.3.bias",
        "ln_final.weight", "ln_final.bias", "unembed.weight"], names
    assert "pos_emb.weight" not in make(pe_type="none").state_dict()


def test_same_seed_same_weights():
    a, b = make(seed=5).state_dict(), make(seed=5).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_pe_type_names():
    for name in ("none", "off", "NoPE"):
        m = make(pe_type=name)
        assert m.pe_type == "none" and not hasattr(m, "pos_emb")
    for bad in ("RoPE", "ALiBi", "ape"):
        try:
            make(pe_type=bad)
        except ValueError:
            continue
        raise AssertionError(f"pe_type={bad!r} should be rejected")


def test_causal():
    """Changing token t leaves every logit before t unchanged."""
    for pe in ("APE", "none"):
        m, x = make(pe_type=pe), tokens()
        y = x.clone()
        y[:, 5] = (y[:, 5] + 1) % V
        a, b = m(x), m(y)
        assert torch.allclose(a[:, :5], b[:, :5], atol=1e-6)
        assert not torch.allclose(a[:, 5:], b[:, 5:], atol=1e-6)


def test_prefix_is_bidirectional():
    """With prefix_len=4, positions 0-3 see each other; later positions stay causal."""
    m, x = make(), tokens()
    y = x.clone()
    y[:, 3] = (y[:, 3] + 1) % V
    a, b = m(x, prefix_len=4), m(y, prefix_len=4)
    assert not torch.allclose(a[:, 0], b[:, 0], atol=1e-6)     # position 0 sees position 3
    y = x.clone()
    y[:, 6] = (y[:, 6] + 1) % V
    a, b = m(x, prefix_len=4), m(y, prefix_len=4)
    assert torch.allclose(a[:, :6], b[:, :6], atol=1e-6)       # nothing sees position 6 early


def test_inputs_embeds_matches_idx():
    m, x = make(), tokens()
    assert torch.allclose(m(x), m(inputs_embeds=m.token_emb(x)), atol=1e-6)


def test_freeze_flags():
    m = make(freeze_emb=True, freeze_pos=True,
             layer_configs=[dict(freeze_q=True, freeze_v=True), dict(freeze_k=True, freeze_o=True)])
    frozen = {n for n, p in m.named_parameters() if not p.requires_grad}
    assert frozen == {"token_emb.weight", "pos_emb.weight",
                      "blocks.0.attn.w_q.weight", "blocks.0.attn.w_v.weight",
                      "blocks.1.attn.w_k.weight", "blocks.1.attn.w_o.weight"}, frozen
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.1)
    nn.functional.cross_entropy(m(tokens()).flatten(0, 1), tokens(2).flatten()).backward()
    opt.step()
    for n, p in m.named_parameters():
        assert torch.equal(before[n], p) == (n in frozen), n


def test_layer_configs_checked():
    for bad in ([dict(freeze_q=True)],                        # one dict for two layers
                [dict(use_res=False), dict()]):               # unknown key
        try:
            make(layer_configs=bad)
        except ValueError:
            continue
        raise AssertionError(f"layer_configs={bad} should be rejected")


def test_residual_switches():
    """use_res_attn=False makes a layer's output attn(LN1(x)) alone; defaults change nothing."""
    m, x = make(), tokens()
    on = make(layer_configs=[dict(use_res_attn=True, use_res_ffn=True)] * 2)
    assert torch.equal(m(x), on(x))
    # One layer, no FFN: the logits are a function of attention's output only.
    off = make(num_layers=1, layer_configs=[dict(use_res_attn=False)])
    blk = off.blocks[0]
    e = off.token_emb(x) + off.pos_emb.weight[:T]
    a = blk.attn(blk.ln1(e), mask=torch.tril(torch.ones(T, T)).view(1, 1, T, T))
    want = off.unembed(off.ln_final(a))
    assert torch.allclose(off(x, use_ffn=False), want, atol=1e-5)
    # The same weights with the skip connection give different logits.
    on1 = make(num_layers=1)
    assert not torch.allclose(on1(x, use_ffn=False), want, atol=1e-3)
    # use_res_ffn=False: the FFN output replaces the stream.
    f = make(num_layers=1, layer_configs=[dict(use_res_ffn=False)])
    blk = f.blocks[0]
    h = f.token_emb(x) + f.pos_emb.weight[:T]
    h = h + blk.attn(blk.ln1(h), mask=torch.tril(torch.ones(T, T)).view(1, 1, T, T))
    assert torch.allclose(f(x), f.unembed(f.ln_final(blk.ffn(blk.ln2(h)))), atol=1e-5)


def test_use_ffn_skips_the_ffn():
    """use_ffn=False gives the same logits as an FFN whose output layer is zero."""
    m, x = make(), tokens()
    zeroed = make()
    for blk in zeroed.blocks:
        nn.init.zeros_(blk.ffn.net[3].weight)
        nn.init.zeros_(blk.ffn.net[3].bias)
    assert torch.allclose(m(x, use_ffn=False), zeroed(x), atol=1e-6)
    half = make()
    nn.init.zeros_(half.blocks[0].ffn.net[3].weight)
    nn.init.zeros_(half.blocks[0].ffn.net[3].bias)
    assert torch.allclose(m(x, use_ffn=[False, True]), half(x), atol=1e-6)


def test_d_qk_is_decoupled():
    m = make(d_qk=8, n_heads=2)
    attn = m.blocks[0].attn
    assert attn.w_q.weight.shape == (8, 16) and attn.w_v.weight.shape == (16, 16)
    assert attn.d_qk_head == 4 and attn.d_v_head == 8
    assert m(tokens()).shape == (B, T, V)


def test_cache():
    m, x = make(), tokens()
    m(x, save_cache=True)
    blk = m.blocks[1]
    assert set(blk.attn.cache) == {"Q_raw", "K_raw", "V_raw", "attn_weights"}
    assert set(blk.cache) == {"residual_pre_attn", "residual_post_attn", "residual_post_ffn"}
    a = blk.attn.cache["attn_weights"]
    assert a.shape == (B, 2, T, T)
    assert torch.allclose(a.sum(-1), torch.ones(B, 2, T), atol=1e-6)
    assert a.triu(1).abs().max() == 0
    assert torch.equal(m.blocks[0].cache["residual_post_ffn"], blk.cache["residual_pre_attn"])
    m(x, save_cache=True, use_ffn=False)
    assert torch.equal(blk.cache["residual_post_attn"], blk.cache["residual_post_ffn"])


def test_zero_frozen_qk_is_mean_pooling():
    """The uniform-attention control (HMM, regression): zero, frozen W_Q and W_K."""
    m = make(layer_configs=[dict(freeze_q=True, freeze_k=True)] * 2)
    for blk in m.blocks:
        nn.init.zeros_(blk.attn.w_q.weight)
        nn.init.zeros_(blk.attn.w_k.weight)
    m(tokens(), save_cache=True)
    expected = torch.tril(torch.ones(T, T)) / torch.arange(1, T + 1).unsqueeze(1)
    for blk in m.blocks:
        assert torch.allclose(blk.attn.cache["attn_weights"], expected.expand(B, 2, T, T), atol=1e-6)


def test_matches_the_written_equation():
    """One attention-only layer reproduces the equation in the Method chapter:
    logits_t = W_U LN_f( x_t + W_O sum_{s<=t} a_ts W_V LN_1(x_s) ),  x_t = e_{z_t} + p_t."""
    m, x = make(num_layers=1, n_heads=1), tokens()
    blk = m.blocks[0]
    with torch.no_grad():
        x0 = m.token_emb(x) + m.pos_emb.weight[:T]
        h = blk.ln1(x0)
        q, k, v = h @ blk.attn.w_q.weight.T, h @ blk.attn.w_k.weight.T, h @ blk.attn.w_v.weight.T
        s = (q @ k.transpose(1, 2)) / 16 ** 0.5
        a = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf")).softmax(-1)
        logits = m.ln_final(x0 + (a @ v) @ blk.attn.w_o.weight.T) @ m.unembed.weight.T
        assert torch.allclose(m(x, use_ffn=False), logits, atol=1e-5)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
