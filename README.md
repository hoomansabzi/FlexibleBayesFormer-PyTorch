# FlexibleBayesFormer & BayesianWindTunnels

A highly modular Transformer architecture and a suite of diagnostic data environments designed for deep theoretical analysis, mechanistic interpretability, and Information Bottleneck (IB) experiments.

Designed for experimental control rather than production, this codebase allows researchers to manipulate individual network components and evaluate them in perfectly controlled, exact-Bayesian testing environments to study theories like *Birth of a Transformer*.

---

## 🧠 FlexibleBayesFormer (The Architecture)

A hyper-granular Transformer model allowing layer-by-layer architectural control.

* **Granular Attention Control:** * Decoupled dimensions for Queries/Keys ($d_{qk}$) and Values ($d_v$).
* Independently freeze $W_Q$, $W_K$, $W_V$, and $W_O$.
* Force $W_Q$ and $W_K$ to act as Identity matrices for pure associative memory lookups.


* **Dynamic & Asymmetric Layers:**
* Define a different number of attention heads per layer.
* Independently toggle residual streams on/off for Attention and MLP blocks.
* Bypass FeedForward layers entirely if desired.


* **Information Plane Ready:** Swappable FFN activation functions (GELU, Tanh, ReLU) for computing mutual information.
* **Positional Encodings:** Supports APE (Absolute), RoPE (Rotary, scaled correctly to decoupled head dims), and ALiBi (Linear Biases).

---

## 🌪️ BayesianWindTunnels (The Environments)

A suite of synthetic data generators that provide both training sequences and exact Bayesian posteriors for evaluating specific cognitive primitives in Transformers.

* **Associative Recall (`associative_recall.py`):** * Tests pure memory capacity and the formation of **Induction Heads**.
* Generates separate Storage and Retrieval phases to isolate exact $W_K, W_Q, W_V$ copying dynamics.


* **Bijection (`bijection.py`):** * Tests **deductive reasoning** and process-of-elimination.
* Models must track a shrinking pool of available mappings to deduce unseen "trivial pairs."
* Supports variable `repeat_prob` and `repeat_mode` ('anywhere' vs. 'last_only') to contrast logic vs. memory.


* **Hidden Markov Model (`hmm.py`):** * Tests **recursive belief updating** and Bayesian state tracking across sequential observations.

---

## 📂 Project Structure

```text
Project_Root/
│
├── FlexibleBayesFormer/
│   ├── __init__.py
│   ├── ffn.py          # FeedForward with custom activations
│   ├── attention.py    # MultiHeadAttention with granular matrix freezing & decoupled dims
│   ├── block.py        # TransformerBlock with independent residual toggles
│   └── model.py        # Master FlexFormer orchestrating dynamic layer configurations
│
└── BayesianWindTunnels/
    ├── __init__.py
    ├── associative_recall.py  # Tests memory & induction heads
    ├── bijection.py           # Tests deduction & process-of-elimination
    └── hmm.py                 # Tests recursive belief updating

```
