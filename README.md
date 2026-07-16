# FlexibleBayesFormer-PyTorch

**FlexibleBayesFormer** is a highly modular, research-oriented Transformer architecture designed specifically for deep theoretical analysis, studying learning dynamics, and conducting Information Bottleneck (IB) / Information Plane experiments. 

Unlike standard production-focused Transformers, this codebase prioritizes hyper-granularity and experimental control. It allows researchers to manipulate individual components of the network—such as freezing specific weight matrices, toggling residual streams, and altering head dimensions layer-by-layer—to empirically test mechanistic interpretability theories (e.g., *Birth of a Transformer: A Memory Viewpoint*).

---

## 🚀 Key Properties & Abilities

### 1. Granular Attention Control
* **Decoupled Dimensions:** Supports separate dimensions for Queries/Keys ($d_{qk}$) and Values ($d_v$). 
* **Independent Matrix Freezing:** You can independently freeze (disable gradient updates for) $W_Q$, $W_K$, $W_V$, and $W_O$.
* **Identity Mapping:** Easily force $W_Q$ and $W_K$ to act as Identity matrices to simulate direct associative memory lookups.

### 2. Information Bottleneck & Plane Analysis
* **Swappable Activations:** The FeedForward Network (FFN) accepts arbitrary PyTorch activation functions (e.g., `nn.GELU()`, `nn.Tanh()`, `nn.ReLU()`). This is critical for computing mutual information and studying how different non-linearities shape the Information Plane during training.

### 3. Dynamic & Asymmetric Architecture
* **Variable Heads per Layer:** You can pass a list of integers to `n_heads` to specify a different number of attention heads for each layer.
* **Toggleable Residual Streams:** Residual connections (skip connections) can be turned on or off *independently* for the Attention block (`use_res_attn`) and the MLP block (`use_res_ffn`) on a per-layer basis.
* **Toggleable MLPs:** Completely bypass the FeedForward layer if desired (`use_ffn=False`).

### 4. Advanced Positional Encodings
Supports multiple state-of-the-art positional encoding schemes out of the box:
* **APE (Absolute Positional Embeddings):** Standard learned positional embeddings.
* **RoPE (Rotary Position Embeddings):** Dynamically applies rotary embeddings correctly scaled to the specific $Q/K$ head dimension of each layer.
* **ALiBi (Attention with Linear Biases):** Applies distance-based penalties to attention scores without modifying the input embeddings.

---

## 📂 Codebase Structure

```text
FlexibleBayesFormer/
│
├── __init__.py
├── ffn.py          # Modular FeedForward with custom activations
├── attention.py    # FlexMultiHeadAttention with granular matrix freezing & decoupled dims
├── block.py        # FlexTransformerBlock with independent residual toggles
└── model.py        # Master FlexFormer class orchestrating dynamic layer configurations
```
