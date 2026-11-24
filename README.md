# FlexibleBayesFormer & BayesianWindTunnels

A small decoder-only transformer in PyTorch in which every component can be switched off or
frozen, and a set of sequence tasks whose exact Bayesian posterior is known in closed form.
Together they answer one kind of question: how little of a transformer is needed to compute a
posterior exactly, and what circuit it builds to do so.

This is the code used in the MSc thesis *The Minimal Machinery of Bayesian Computation in
Transformers* (Hooman Sabzi, University of Padova, 2026).

## FlexibleBayesFormer: the model

`FlexFormer` is a pre-LayerNorm decoder-only transformer. Each part can be changed on its own:

| Switch | What it does |
|---|---|
| `num_layers`, `n_heads`, `d_model`, `d_ffn` | depth, heads per layer, residual width, FFN width |
| `d_qk` | query/key width, independent of `d_model` (split across heads) |
| `pe_type="APE"` or `"none"` | learned absolute positions, or no positional information |
| `layer_norm=False` | the identity in place of every LayerNorm |
| `activation` | the FFN nonlinearity, as a module (default `nn.GELU()`) |
| `use_ffn` (in `forward`) | `False`, or one bool per layer, skips the FFN: attention only |
| `layer_configs` | one dict per layer: `use_res_attn` / `use_res_ffn` (`False` drops the skip connection), `freeze_q`, `freeze_k`, `freeze_v`, `freeze_o` |
| `freeze_emb`, `freeze_pos` | hold the token or position table at its initialisation |
| `save_cache=True` (in `forward`) | keeps queries, keys, values, attention weights and the residual stream at three points of every block |
| `inputs_embeds` (in `forward`) | feed vectors in place of token ids |

Attention projections carry no bias, and the unembedding is not tied to the embedding. Parameters
are created in a fixed order, so a seed always gives the same initial weights, whatever the
switches.

## BayesianWindTunnels: the tasks

Each generator draws fresh sequences and returns the exact posterior a Bayesian observer would
hold at each position. Training with cross-entropy against that posterior, the best possible loss
is its entropy, so a model's gap to that floor says how close it is to exact inference.

| Class | Task |
|---|---|
| `BijectionWindTunnel` | A fresh one-to-one map from V x-values to V y-values, shown one pair at a time. A new x's y is uniform over the y-values not used yet: the model must eliminate. With `repeat_prob > 0` some pairs repeat an earlier binding, whose y must be recalled. |
| `RecallWindTunnel` | Store N cue-target pairs, then return the target of each repeated cue (associative recall, solved by an induction head). |
| `OverwriteRecallWindTunnel`, `MultiHopRecallWindTunnel` | Recall in which some cues are bound again later, or which needs several lookups in a row. |
| `HMMWindTunnel` | Filtering in a fresh hidden Markov model per sequence; the posterior comes from the forward algorithm. |
| `RegressionWindTunnel` | In-context Bayesian regression with a Gaussian prior, over one or a mixture of function families; closed-form posterior predictive. |

## Install

```bash
pip install git+https://github.com/hoomansabzi/FlexibleBayesFormer-PyTorch.git
```

or clone the repository and work next to the two package folders. The only dependency is PyTorch
(tested with Python 3.12 and PyTorch 2.8).

## Example

The smallest model that solves the bijection task in the thesis: one layer, one head, no
positional embedding, no residual connection, no FFN.

```python
import torch
from FlexibleBayesFormer import FlexFormer
from BayesianWindTunnels import BijectionWindTunnel

torch.manual_seed(0)
tunnel = BijectionWindTunnel(V=16)                  # x ids 1..16, y ids 17..32
model = FlexFormer(vocab_size=tunnel.vocab_size, max_seq_len=32, d_model=32, n_heads=1,
                   num_layers=1, pe_type="none", layer_configs=[dict(use_res_attn=False)])
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

for step in range(2000):
    seq, y_idx = tunnel.generate_batch(256)
    post = tunnel.get_bayesian_posterior(seq, y_idx)               # exact p(y_k | x_k, earlier pairs)
    logp = model(seq, use_ffn=False)[:, y_idx - 1].log_softmax(-1)  # y_k is read at x_k
    loss = -(post * logp).sum(-1).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

floor = -(post * post.clamp(min=1e-30).log()).sum(-1).mean()       # the posterior's entropy
print(f"loss {loss.item():.3f}, floor {floor.item():.3f} nats")
```

The thesis trains for 30,000 steps with batches of 1024 and calls a model solved when its gap to
the floor is below 0.001 nats.

## Tests

```bash
python tests/test_engine.py
python tests/test_windtunnels.py
```

## Earlier versions

Earlier commits of this repository also had rotary (RoPE) and ALiBi position encodings and a
continuous-observation HMM. The thesis did not use them, and they were removed; they remain in
the history.

## License

MIT, see `LICENSE`.
