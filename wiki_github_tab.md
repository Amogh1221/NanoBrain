# NanoBrain: The Definitive Guide to Large Language Models (LLMs)

Welcome to the official **NanoBrain** repository documentation and masterclass wiki. This repository contains a clean, high-performance, modular implementation of a 124-Million parameter Generative Pre-trained Transformer (GPT-2 Small architecture) built in PyTorch.

> **Rendering Note:** This document contains mathematical formulas formatted in LaTeX ($...$ and $$...$$). For optimal rendering, view this file on GitHub or in a Markdown previewer with MathJax / KaTeX support enabled.

This document serves as an exhaustive, first-principles textbook and technical guide. It is structured into two main parts:
1. **Theoretical Foundations (Modules 1–5):** Mathematical, architectural, hardware, and algorithmic mechanics powering modern Large Language Models.
2. **Implementation Deep Dive (Module 6):** A file-by-file walkthrough of the NanoBrain codebase.

---

# Table of Contents
- [NanoBrain: The Definitive Guide to Large Language Models (LLMs)](#nanobrain-the-definitive-guide-to-large-language-models-llms)
- [Table of Contents](#table-of-contents)
- [PART 1: THEORETICAL FOUNDATIONS](#part-1-theoretical-foundations)
  - [Module 1: Language Modeling Fundamentals](#module-1-language-modeling-fundamentals)
    - [1.1 Mathematical Formulation of Language Modeling](#11-mathematical-formulation-of-language-modeling)
    - [1.2 Historical Evolution: From N-Grams to Recurrent Neural Networks](#12-historical-evolution-from-n-grams-to-recurrent-neural-networks)
    - [1.3 The Bottlenecks of Sequential Models](#13-the-bottlenecks-of-sequential-models)
  - [Module 2: The Transformer Decoder Architecture](#module-2-the-transformer-decoder-architecture)
    - [2.1 High-Dimensional Vector Spaces and Embedding Representation](#21-high-dimensional-vector-spaces-and-embedding-representation)
    - [2.2 Tokenization Mechanics: Byte-Pair Encoding (BPE)](#22-tokenization-mechanics-byte-pair-encoding-bpe)
    - [2.3 Positional Representations: Absolute vs. Rotary (RoPE)](#23-positional-representations-absolute-vs-rotary-rope)
    - [2.4 Causal Multi-Head Self-Attention (MHSA)](#24-causal-multi-head-self-attention-mhsa)
    - [2.5 Non-Linear Activations: GELU vs. ReLU vs. SwiGLU](#25-non-linear-activations-gelu-vs-relu-vs-swiglu)
    - [2.6 Normalization Topologies and The Residual Highway](#26-normalization-topologies-and-the-residual-highway)
  - [Module 3: Computational and Hardware Optimizations](#module-3-computational-and-hardware-optimizations)
    - [3.1 GPU Memory Architecture: SRAM vs. HBM](#31-gpu-memory-architecture-sram-vs-hbm)
    - [3.2 FlashAttention: Tiling, Online Softmax, and Kernel Fusion](#32-flashattention-tiling-online-softmax-and-kernel-fusion)
    - [3.3 Numerical Precision Formats: FP32, FP16, BF16, TF32](#33-numerical-precision-formats-fp32-fp16-bf16-tf32)
    - [3.4 Mixed-Precision Training (AMP) and Gradient Scaling](#34-mixed-precision-training-amp-and-gradient-scaling)
    - [3.5 Gradient Accumulation and Gradient Clipping](#35-gradient-accumulation-and-gradient-clipping)
    - [3.6 Weight Decay, Fused AdamW, and Learning Rate Scheduling](#36-weight-decay-fused-adamw-and-learning-rate-scheduling)
    - [3.7 Exponential Moving Average (EMA) of Weights](#37-exponential-moving-average-ema-of-weights)
  - [Module 4: Text Generation and Decoding Mechanics](#module-4-text-generation-and-decoding-mechanics)
    - [4.1 Autoregressive Sampling Loop](#41-autoregressive-sampling-loop)
    - [4.2 Softmax Temperature Scaling](#42-softmax-temperature-scaling)
    - [4.3 Truncation Algorithms: Top-K and Top-p (Nucleus) Sampling](#43-truncation-algorithms-top-k-and-top-p-nucleus-sampling)
  - [Module 5: Modern Industry Evolution and Comparative Architecture](#module-5-modern-industry-evolution-and-comparative-architecture)
    - [5.1 Architectural Shift: GPT-2 (NanoBrain) vs. LLaMA 1/2/3 and Mistral](#51-architectural-shift-gpt-2-nanobrain-vs-llama-123-and-mistral)
    - [5.2 Grouped-Query Attention (GQA) and Multi-Query Attention (MQA)](#52-grouped-query-attention-gqa-and-multi-query-attention-mqa)
- [PART 2: CODEBASE ARCHITECTURE AND IMPLEMENTATION](#part-2-codebase-architecture-and-implementation)
  - [Module 6: NanoBrain Codebase Walkthrough](#module-6-nanobrain-codebase-walkthrough)
    - [6.1 Hyperparameter Configuration (config.py / config.json)](#61-hyperparameter-configuration-configpy--configjson)
    - [6.2 Data Sourcing and Pre-Processing (build_dataset.py)](#62-data-sourcing-and-pre-processing-build_datasetpy)
    - [6.3 Binary Packing and Zero-Copy Data Loading (tokenize_dataset.py and dataset.py)](#63-binary-packing-and-zero-copy-data-loading-tokenize_datasetpy-and-datasetpy)
    - [6.4 PyTorch Model Implementation (model.py)](#64-pytorch-model-implementation-modelpy)
    - [6.5 Unified Training Engine (trainer.py and train.py)](#65-unified-training-engine-trainerpy-and-trainpy)
    - [6.6 Text Generation Script (generate.py)](#66-text-generation-script-generatepy)
- [Summary and Best Practices for Experimentation](#summary-and-best-practices-for-experimentation)

---

# PART 1: THEORETICAL FOUNDATIONS

---

## Module 1: Language Modeling Fundamentals

### 1.1 Mathematical Formulation of Language Modeling

> **Intuition:** A Language Model is an advanced word-predictor. Given a sequence of words, its goal is to assign a probability to every possible next word in its dictionary, picking the most plausible option based on context. During training, we show it millions of sentences and penalize it whenever it assigns a low probability to the actual next word.

At its core, formal language modeling frames natural language generation as a joint probability distribution estimation problem over sequence variables. Given a sequence of $N$ discrete tokens $X = (x_1, x_2, \dots, x_N)$, the joint probability $P(X)$ of observing the sequence is factored auto-regressively via the probabilistic chain rule:

$$P(X) = P(x_1, x_2, \dots, x_N) = \prod_{t=1}^N P(x_t \mid x_1, x_2, \dots, x_{t-1})$$

In a causal (or auto-regressive) language model, the network parameterizes the conditional probability distribution $P(x_t = k \mid x_{<t})$ over a finite vocabulary set $V$:

$$P(x_t = k \mid x_{<t}; \theta) = \text{softmax}(z_t)_k = \frac{\exp(z_{t, k})}{\sum_{j=1}^{|V|} \exp(z_{t, j})}$$

where $\theta$ represents the trainable weights of the neural network and $z_t \in \mathbb{R}^{|V|}$ represents the raw unnormalized logit scores predicted at step $t$.

Training an LLM consists of minimizing the empirical cross-entropy loss over a corpus of $T$ tokens:

$$\mathcal{L}_{CE}(\theta) = -\frac{1}{T} \sum_{t=1}^T \log P(x_t^* \mid x_1^*, \dots, x_{t-1}^*; \theta)$$

where $x_t^*$ is the target ground-truth token index at position $t$.

```
Text Sequence: "The quick brown fox jumps"
Step 1: P("The" | <START>)
Step 2: P("quick" | "The")
Step 3: P("brown" | "The", "quick")
Step 4: P("fox" | "The", "quick", "brown")
Step 5: P("jumps" | "The", "quick", "brown", "fox")
```

---

### 1.2 Historical Evolution: From N-Grams to Recurrent Neural Networks

> **Intuition:** Early AI models tried to predict the next word by looking only at the last 2 or 3 words ($N$-grams) or by passing a single running memory vector down a chain of words (RNNs). While RNNs were a big step forward, they forced computers to read words one by one, creating a huge bottleneck.

```
+-------------------------------------------------------------------+
|                            EVOLUTION                              |
+-------------------------------------------------------------------+
| N-Gram Statistical Models                                         |
|   - Markov assumption: P(w_t | w_<t) ≈ P(w_t | w_{t-n+1:t-1})      |
|   - Count tables, zero-probability fallbacks                      |
+-------------------------------------------------------------------+
                                  │
                                  ▼
+-------------------------------------------------------------------+
| Recurrent Neural Networks (RNNs / LSTMs / GRUs)                   |
|   - Hidden state update: h_t = tanh(W_h h_{t-1} + W_x x_t + b)   |
|   - Gated memory cells (LSTM memory cells, forget gates)          |
+-------------------------------------------------------------------+
                                  │
                                  ▼
+-------------------------------------------------------------------+
| Transformer Decoder (Causal Self-Attention)                       |
|   - Direct token-to-token interactions: O(1) path length          |
|   - Parallelizable matrix multiplications across sequence length   |
+-------------------------------------------------------------------+
```

1. **$N$-Gram Language Models:** Relied on the strict Markov assumption that the probability of word $x_t$ depends only on the preceding $N-1$ words:
   $$P(x_t \mid x_{<t}) \approx P(x_t \mid x_{t-N+1}, \dots, x_{t-1})$$
   *Limitations:* Combinatorial explosion of storage tables ($|V|^N$) and zero generalization to unseen sequences without smoothing techniques.

2. **Recurrent Neural Networks (RNNs & LSTMs):** Processed text sequentially by updating an internal hidden vector state $h_t \in \mathbb{R}^d$:
   $$h_t = \text{tanh}(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
   LSTMs added gating mechanisms (Input, Forget, Output gates) to maintain information over longer context windows.

---

### 1.3 The Bottlenecks of Sequential Models

> **Intuition:** Imagine trying to read an entire encyclopedia, but you are only allowed to keep a single sentence in your head to summarize everything you've read so far. That was the LSTM bottleneck. Additionally, because word 100 couldn't be processed until word 99 was finished, modern parallel graphics cards (GPUs) sat mostly idle.

<p align="center">
  <img src="https://raw.githubusercontent.com/Amogh1221/NanoBrain/main/docs/images/rnn_vs_transformer.svg" alt="Sequential RNN Bottleneck vs Parallel Transformer Self-Attention" width="680">
  <br>
  <em>Sequential RNN Bottleneck vs. Parallel Transformer Self-Attention Vector Diagram.</em>
</p>

Despite the architectural improvements of LSTMs and GRUs, three fundamental bottlenecks hindered scaling:

1. **Sequential Compute Bottleneck:** Computing $h_t$ strictly requires $h_{t-1}$. Training cannot be parallelized across temporal sequence length $T$. GPU hardware cores remain underutilized because sequence computation is inherently serial $O(T)$.
2. **Vanishing and Exploding Gradients:** Backpropagation Through Time (BPTT) requires repeatedly multiplying weight matrices $W_{hh}$ over $T$ timesteps. The gradient scale decays or explodes exponentially according to the spectral radius of $W_{hh}$:
   $$\frac{\partial \mathcal{L}}{\partial h_1} = \frac{\partial \mathcal{L}}{\partial h_T} \prod_{k=2}^T \frac{\partial h_k}{\partial h_{k-1}}$$
3. **Information Compression Bottleneck:** Long-range contextual information must be continually compressed into a single fixed-size hidden vector $h_t$, causing progressive loss of early tokens.

---

## Module 2: The Transformer Decoder Architecture

> **Intuition:** The Transformer decoder throws away sequential reading. Instead, it looks at the entire sequence of words at once, allowing every word to form direct connections with every preceding word. This architecture powers modern GPT models.

The Transformer architecture (*Vaswani et al., 2017*) resolved the sequential bottleneck by replacing recurrences entirely with **Self-Attention**. NanoBrain specifically implements the **Decoder-Only Transformer** (pioneered by GPT-* Radford et al.*), where every token can directly attend to all previous tokens simultaneously in parallel operations.

<p align="center">
  <img src="https://raw.githubusercontent.com/Amogh1221/NanoBrain/main/docs/images/gpt_architecture.svg" alt="Full GPT Decoder Architecture Vector SVG" width="400">
  <br>
  <em>Full GPT Decoder Architecture Vector Diagram.</em>
</p>

```
                  +-----------------------------+
                  |       Output Logits         |
                  +-----------------------------+
                                 ▲
                                 │
                  +-----------------------------+
                  |    Final LayerNorm (ln_f)   |
                  +-----------------------------+
                                 ▲
                                 │
                  +-----------------------------+
                  |  Transformer Block N        |
                  |  +-----------------------+  |
                  |  |   MLP / FeedForward   |  |
                  |  +-----------------------+  |
                  |              ▲              |
                  |              │ (Residual)   |
                  |  +-----------------------+  |
                  |  | Causal Self-Attention |  |
                  |  +-----------------------+  |
                  +-----------------------------+
                                 ▲
                                 │  (x N Block Stacks)
                                 ▲
                  +-----------------------------+
                  |   Token + Pos Embeddings    |
                  +-----------------------------+
                                 ▲
                                 │
                  +-----------------------------+
                  |    Input Token Indices      |
                  +-----------------------------+
```

---

### 2.1 High-Dimensional Vector Spaces and Embedding Representation

> **Intuition:** Computers cannot multiply words; they can only multiply numbers. An embedding transforms a discrete word index into a point in a multi-dimensional space (e.g., 768 dimensions), where words with similar meanings (like "king" and "queen") end up close to each other.

Text tokens are discrete symbols. To process them through continuous matrix linear algebra, each discrete token integer ID $x_t \in \{0, 1, \dots, |V|-1\}$ is mapped into a continuous high-dimensional vector space $\mathbb{R}^{d_{embd}}$.

In NanoBrain (GPT-2 Small parameter scale), $d_{embd} = 768$.

The input sequence of token indices $X \in \mathbb{N}^{B \times T}$ is transformed into an input embedding tensor $E_{tok} \in \mathbb{R}^{B \times T \times d_{embd}}$ via an embedding matrix lookup $W_{te} \in \mathbb{R}^{|V| \times d_{embd}}$:

$$E_{tok} = \text{Lookup}(X, W_{te})$$

High-dimensional vector spaces allow the neural network to encode semantic relationships as geometric distances and directional alignments (cosine similarity) in vector space.

---

### 2.2 Tokenization Mechanics: Byte-Pair Encoding (BPE)

> **Intuition:** Instead of splitting text word-by-word (which creates a massive dictionary) or character-by-character (which creates super long sentences), Byte-Pair Encoding (BPE) merges common pairs of letters into subword pieces. For instance, "unbelievable" becomes "un", "believ", and "able".

Tokenization is the mapping layer between string characters and numerical token IDs. NanoBrain utilizes **Byte-Pair Encoding (BPE)** via OpenAI’s `tiktoken` library.

*Note on Vocabulary Sizes:* The standard OpenAI GPT-2 base tokenizer contains $|V| = 50,257$ subword tokens. In NanoBrain's `config.py`, `vocab_size` is configured as `50258`. This $+1$ token allocation is a common optimization pattern that reserves an explicit slot for special control tokens (such as a dedicated `<|endoftext|>` or padding index) or rounds vocabulary sizes to GPU-friendly multiples.

```
Raw String: "unbelievable"
Character Level: ['u', 'n', 'b', 'e', 'l', 'i', 'e', 'v', 'a', 'b', 'l', 'e'] (Length: 12)
BPE Subwords:    ["un", "believ", "able"]                                     (Length: 3)
Token IDs:       [3415, 29194, 1284]
```

#### BPE Algorithm Breakdown
1. **Initialization:** Prepare the vocabulary with all individual base characters (or raw bytes 0–255).
2. **Co-occurrence Counting:** Scan the pre-training dataset to calculate frequencies of adjacent token pairs $(t_i, t_j)$.
3. **Merge Iteration:** Iteratively merge the most frequent adjacent pair into a new unified subword token $t_{new} = t_i \oplus t_j$.
4. **Termination:** Stop when the target vocabulary size $|V|$ is achieved.

*Why BPE over Word/Character level?*
- Word-level tokenization causes massive vocabulary sizes ($>1M$), resulting in giant linear layer memory footprints and Out-Of-Vocabulary (OOV) lookup errors for unseen words.
- Character-level tokenization produces long sequence lengths $T$, increasing attention matrix computational cost ($O(T^2)$).
- BPE balances context compression with complete coverage of rare words and code via subwords.

---

### 2.3 Positional Representations: Absolute vs. Rotary (RoPE)

> **Intuition:** Because attention processes all words simultaneously, the model has no default idea of word order. Without positional embeddings, "dog bites man" and "man bites dog" would look identical. Positional embeddings inject a sense of time and order into each word vector.

<p align="center">
  <img src="https://raw.githubusercontent.com/Amogh1221/NanoBrain/main/docs/images/positional_encoding.svg" alt="Positional Encoding Matrix Vector SVG" width="650">
  <br>
  <em>Vector Heatmap of Sinusoidal Positional Encoding Patterns (Vaswani et al. Fixed Scheme).</em>
</p>

Historically, the original Transformer paper (*Vaswani et al., 2017*) introduced fixed, non-learnable **Sinusoidal Positional Encodings** (visualized above) using alternating sine and cosine functions of varying frequencies to map token positions to vectors. Subsequent architectures evolved this concept into learned parameters and relative rotation matrices:

#### 1. Absolute Learned Positional Embeddings (NanoBrain / GPT-2)
NanoBrain defines a trainable parameter matrix $W_{pe} \in \mathbb{R}^{T_{max} \times d_{embd}}$ where $T_{max} = 1024$. For sequence length $T$, position indices $P = (0, 1, 2, \dots, T-1)$ are mapped to embeddings $E_{pos} \in \mathbb{R}^{B \times T \times d_{embd}}$ and directly summed element-wise with token embeddings:

$$H_0 = E_{tok} + E_{pos} = \text{Lookup}(X, W_{te}) + \text{Lookup}(P, W_{pe})$$

*Limitation:* Absolute embeddings cannot generalize beyond the maximum pre-trained block size $T_{max}$.

#### 2. Rotary Position Embeddings (RoPE - Used in LLaMA / Modern LLMs)
Modern architectures (LLaMA, Mistral) apply a complex rotation matrix to the Query ($Q$) and Key ($K$) representations in 2D vector pairs rather than adding static vectors to the input:

$$R_{\Theta, m}^d = \text{diag}\left( R_{\theta_1, m}, R_{\theta_2, m}, \dots, R_{\theta_{d/2}, m} \right)$$

$$R_{\theta_i, m} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}$$

This formulation guarantees that the inner product $\langle R_m q, R_n k \rangle$ depends solely on relative distance $(m - n)$, enabling better context window extension.

---

### 2.4 Causal Multi-Head Self-Attention (MHSA)

> **Intuition:** Self-attention allows every word to ask a question ("Query"), check matching keys from earlier words ("Keys"), and extract relevant context ("Values"). The word "it" in "The bank approved the loan because it had money" uses self-attention to link "it" back to "bank". The "causal" part means a word can only look at past words, never future ones.

<p align="center">
  <img src="https://raw.githubusercontent.com/Amogh1221/NanoBrain/main/docs/images/multi_head_attention.svg" alt="Multi-Head Attention Vector SVG" width="650">
  <br>
  <em>Causal Multi-Head Self-Attention Architecture Vector Diagram.</em>
</p>

#### Step-by-Step Mathematical Derivation

Given an input tensor $X \in \mathbb{R}^{B \times T \times d_{embd}}$, three distinct linear projections generate Queries ($Q$), Keys ($K$), and Values ($V$):

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

where $W_Q, W_K, W_V \in \mathbb{R}^{d_{embd} \times d_{embd}}$.

In NanoBrain, a single unified linear layer `c_attn` projects $X$ into $3 \cdot d_{embd}$ dimensions for speed, which is then split:

$$\text{qkv} = \text{Linear}_{d_{embd} \to 3d_{embd}}(X) \implies Q, K, V \in \mathbb{R}^{B \times T \times d_{embd}}$$

#### Multi-Head Splitting
The embedding dimension $d_{embd}$ is partitioned into $h$ independent attention heads, each with dimension $d_k = \frac{d_{embd}}{h}$.
For NanoBrain: $h = 12$, $d_{embd} = 768 \implies d_k = 64$.

The tensors are reshaped:
$$Q, K, V \in \mathbb{R}^{B \times h \times T \times d_k}$$

#### The Scaled Dot-Product Attention Equation

$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} + M \right) V$$

```
Query (Q) [B, h, T, d_k]  ──┐
                            ├─► (Q @ K^T) ─► Scale (1/√d_k) ─► Add Mask (M) ─► Softmax ─► (@ V) ─► Output
Key   (K) [B, h, T, d_k]  ──┘
```

#### Why Scale by $\sqrt{d_k}$?
Assuming components of $Q$ and $K$ are independent random variables with mean 0 and variance 1, their dot product $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ has a mean of 0 and variance of $d_k$. For large dimensions (e.g., $d_k = 64$), variance reaches 64, pushing dot products into high-magnitude regions.

Applying $\text{softmax}(z)$ at extreme values pushes activation gradients into saturated zero regions:

$$\frac{\partial \text{softmax}(z)_i}{\partial z_j} \approx 0 \quad \text{for } |z_i| \gg 0$$

Dividing by $\sqrt{d_k}$ scales variance back to 1.0, preserving healthy gradient flow during backpropagation.

#### Causal Masking Mechanics
To enforce causality during language modeling (preventing token $t$ from looking at token $t+1$), an lower-triangular causal attention mask $M \in \mathbb{R}^{T \times T}$ is applied:

$$M_{i, j} = \begin{cases} 0 & \text{if } i \ge j \\ -\infty & \text{if } i < j \end{cases}$$

When added to scaled scores before $\text{softmax}$, $e^{-\infty} = 0$, zeroing out attention probabilities for future positions.

```
Causal Mask Matrix M (T=4):
[  0, -inf, -inf, -inf ]
[  0,   0,  -inf, -inf ]
[  0,   0,    0,  -inf ]
[  0,   0,    0,    0  ]
```

---

### 2.5 Non-Linear Activations: GELU vs. ReLU vs. SwiGLU

> **Intuition:** Neural networks need non-linear functions to learn complex patterns. Without them, stacking 100 layers of neural networks would collapse into a single basic linear equation. GELU acts like a smooth dimmer switch rather than a harsh on/off toggle.

<p align="center">
  <img src="https://raw.githubusercontent.com/Amogh1221/NanoBrain/main/docs/images/activation_functions.svg" alt="Activation Functions Comparison Vector Plot" width="600">
  <br>
  <em>Exact Mathematical Comparison of ReLU, GELU, and Swish/SwiGLU Curves.</em>
</p>

The Multi-Layer Perceptron (MLP) block processes outputs from the attention layer. It expands hidden dimensions by a factor of 4 ($d_{embd} \to 4 d_{embd} \to d_{embd}$):

$$\text{MLP}(H) = \left( \text{Activation}(H W_1 + b_1) \right) W_2 + b_2$$

```
     ReLU Activation                        GELU Activation
   y │                                    y │
     │     /                                │     /
     │    /                                 │    /
     │   /                                  │   /
─────┼──/───────── x                    ────┼──/───────── x
     │ /                                    │ /  (smooth curve
     │/                                     │/    near zero)
```

1. **ReLU (Rectified Linear Unit):** $\text{ReLU}(x) = \max(0, x)$.
   *Drawback:* Hard zero boundary at $x < 0$ creates "dead neurons" where gradients vanish completely ($\frac{d}{dx} = 0$).

2. **GELU (Gaussian Error Linear Unit - NanoBrain):**
   $$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot P(X \le x), \quad X \sim \mathcal{N}(0, 1)$$
   Approximated mathematically in PyTorch as:
   $$\text{GELU}(x) \approx 0.5 x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3\right)\right)\right)$$
   GELU provides a smooth probabilistic non-linearity across all input ranges, retaining small negative gradients that improve convergence stability.

3. **SwiGLU (Swish Gated Linear Unit - LLaMA):**
   $$\text{SwiGLU}(x) = \text{Swish}_1(x W) \otimes (x V) = (x W \cdot \sigma(x W)) \otimes (x V)$$
   SwiGLU uses a dual-weight matrix gating mechanism that improves capacity, though it adds a third projection matrix per MLP block.

---

### 2.6 Normalization Topologies and The Residual Highway

> **Intuition:** As neural networks grow very deep, signals can fade away or blow up out of control. Residual connections create a "superhighway" for signals to bypass layers directly, while Layer Normalization keeps numbers bounded within a healthy range.

Deep networks face vanishing and exploding gradients when stacking multiple layers. Two architectural choices maintain stability across 12+ layers:

#### 1. The Residual Stream (Highway Connection)
Introduced in ResNet (*He et al.*), every sub-layer output is added directly to its input:

$$x_{l+1} = x_l + \text{SubLayer}(\text{Norm}(x_l))$$

This architecture converts the deep stack into a continuous **residual stream**. Gradients can flow backwards directly through addition operations without scaling degradation:

$$\frac{\partial x_{l+1}}{\partial x_l} = I + \frac{\partial \text{SubLayer}(\text{Norm}(x_l))}{\partial x_l}$$

#### 2. Layer Normalization (LayerNorm)
Unlike Batch Normalization (which normalizes across batch dimension $B$), LayerNorm computes mean and variance statistics across feature dimension $C = d_{embd}$ independently for each individual token:

$$\mu = \frac{1}{C} \sum_{i=1}^C x_i, \quad \sigma^2 = \frac{1}{C} \sum_{i=1}^C (x_i - \mu)^2$$

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma_i + \beta_i$$

where $\gamma, \beta \in \mathbb{R}^C$ are learnable gain and bias parameters, and $\epsilon = 10^{-5}$ prevents division by zero.

#### Pre-LN vs. Post-LN Topologies
- **Post-LN (Original Transformer):** LayerNorm is applied *after* residual addition: $x_{l+1} = \text{LN}(x_l + \text{SubLayer}(x_l))$. Requires careful learning rate warmup to avoid early divergence.
- **Pre-LN (NanoBrain / GPT-2):** LayerNorm is applied *before* sub-layer input: $x_{l+1} = x_l + \text{SubLayer}(\text{LN}(x_l))$. Keeps inputs to main residual stream unnormalized, ensuring robust gradient flow.

---

## Module 3: Computational and Hardware Optimizations

---

### 3.1 GPU Memory Architecture: SRAM vs. HBM

> **Intuition:** A GPU has two types of memory: a tiny, ultra-fast scratchpad right next to the processor (SRAM), and a huge, slower storage pool (HBM/VRAM). If your GPU spends all its time moving data back and forth between storage and the processor, training runs slowly regardless of compute power.

```
+-------------------------------------------------------------------+
|                            NVIDIA GPU                             |
|                                                                   |
|  +-------------------------------------------------------------+  |
|  |                 SRAM (On-Chip L1 Cache)                     |  |
|  |   Capacity: ~20-50 MB  |  Bandwidth: ~19,000 GB/s (Ultra)   |  |
|  +-------------------------------------------------------------+  |
|                                ▲                                  |
|                     Memory Access Bottleneck                      |
|                                ▼                                  |
|  +-------------------------------------------------------------+  |
|  |               HBM / VRAM (High Bandwidth Memory)            |  |
|  |   Capacity: 12-80 GB   |  Bandwidth: ~1,500-3,000 GB/s     |  |
|  +-------------------------------------------------------------+  |
+-------------------------------------------------------------------+
```

1. **High Bandwidth Memory (HBM / VRAM):** Off-chip memory (e.g., 12GB on RTX 3060, 80GB on A100). Holds model weights, activations, and optimizer states. Large capacity, but lower bandwidth (~1.5–3.0 TB/sec).
2. **Static RAM (SRAM / L1 Cache):** On-chip execution memory directly adjacent to Streaming Multiprocessors (SMs). Ultra-fast bandwidth (~19 TB/sec), but small capacity (~20–100 MB).

Operations are categorized into:
- **Compute-Bound:** Matrix Multiplications (GEMMs) with high Arithmetic Intensity (FLOPs per byte read). Fully utilizes GPU tensor cores.
- **Memory-Bound:** Element-wise ops (Softmax, LayerNorm, Dropout, Gelu). Low Arithmetic Intensity. Spends most time reading/writing data between HBM and SRAM.

---

### 3.2 FlashAttention: Tiling, Online Softmax, and Kernel Fusion

> **Intuition:** Standard attention writes huge intermediate grids to slow GPU memory. FlashAttention breaks matrices into small tiles that fit entirely inside ultra-fast SRAM, calculates attention in chunks, and writes back only the final answer. This slashes memory usage and speeds up training.

<p align="center">
  <img src="https://raw.githubusercontent.com/Amogh1221/NanoBrain/main/docs/images/flash_attention_tiling.svg" alt="GPU SRAM vs HBM Memory Hierarchy & FlashAttention Tiling Vector SVG" width="650">
  <br>
  <em>GPU SRAM vs HBM Memory Hierarchy &amp; FlashAttention Tiled Execution Strategy Vector Diagram.</em>
</p>

Standard PyTorch self-attention computes intermediate matrices explicitly in HBM:

$$S = Q K^T \in \mathbb{R}^{B \times h \times T \times T} \quad (\text{HBM Read/Write } O(T^2))$$
$$P = \text{softmax}(S) \in \mathbb{R}^{B \times h \times T \times T} \quad (\text{HBM Read/Write } O(T^2))$$
$$O = P V \in \mathbb{R}^{B \times h \times T \times d_k} \quad (\text{HBM Write } O(T \cdot d_k))$$

For sequence length $T=1024$, materializing $S$ and $P$ creates high memory overhead and slows execution due to HBM bandwidth bottlenecks.

**FlashAttention** (*Dao et al., 2022*) solves this memory wall via three techniques:

```
Standard Attention:  Q, K, V (HBM) ---> Write S (HBM) ---> Write P (HBM) ---> Output O (HBM)
FlashAttention:      Q, K, V (HBM) ---> Load Tiles into SRAM ---> Compute Partial Softmax/V ---> Write Final O (HBM)
```

1. **Tiling:** Partition input matrices $Q, K, V$ into smaller blocks (tiles) sized to fit inside on-chip SRAM (~100 KB).
2. **Online Softmax:** Reconstruct global softmax scaling factors dynamically using max-scaling stabilization:
   $$m_{new} = \max(m_{prev}, m_{block}), \quad d_{new} = d_{prev} e^{m_{prev} - m_{new}} + d_{block} e^{m_{block} - m_{new}}$$
3. **Kernel Fusion & Recomputation:** Fuse matrix multiplication, scaling, masking, and softmax operations into a single CUDA kernel. Instead of saving $T \times T$ intermediate probability matrices to HBM for backpropagation, FlashAttention recomputes them on-the-fly in fast SRAM during the backward pass.

*Result:* Reduces memory complexity from $O(T^2)$ to $O(T)$ and speeds up execution by 2–4×.

In NanoBrain ([`model.py`](https://github.com/Amogh1221/NanoBrain/blob/main/model.py)), FlashAttention is invoked natively via PyTorch 2.0's:
`F.scaled_dot_product_attention(q, k, v, is_causal=True)`

---

### 3.3 Numerical Precision Formats: FP32, FP16, BF16, TF32

> **Intuition:** Numbers in computers are stored using bits. Standard float32 uses 32 bits for maximum precision. BF16 (Brain Floating Point) uses only 16 bits while retaining the exact same exponential range as 32-bit floats. This cuts memory usage in half with virtually zero loss in quality.

```
FP32 (Single Precision):
S (1 bit) | Exponent (8 bits) | Mantissa / Fraction (23 bits)

FP16 (Half Precision):
S (1 bit) | Exponent (5 bits) | Mantissa (10 bits)

BF16 (Bfloat16 - Brain Floating Point):
S (1 bit) | Exponent (8 bits) | Mantissa (7 bits)

TF32 (TensorFloat-32 - Ampere internal):
S (1 bit) | Exponent (8 bits) | Mantissa (10 bits)
```

| Format | Total Bits | Exponent Bits | Mantissa Bits | Dynamic Range | Precision |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **FP32** | 32 | 8 | 23 | $10^{-38} \dots 10^{38}$ | High |
| **FP16** | 16 | 5 | 10 | $6 \times 10^{-5} \dots 65504$ | Low (Prone to underflow) |
| **BF16** | 16 | 8 | 7 | $10^{-38} \dots 10^{38}$ | Same as FP32 (Robust) |
| **TF32** | 19 (internal) | 8 | 10 | Same as FP32 | Medium |

- **Why BF16 is ideal for LLMs:** BF16 preserves FP32's 8-bit exponent field, matching its dynamic range. This prevents underflow (gradients dropping to zero) and overflow (gradients hitting `NaN`) without requiring complex loss scaling.

---

### 3.4 Mixed-Precision Training (AMP) and Gradient Scaling

> **Intuition:** Mixed-Precision training uses fast 16-bit math for heavy matrix calculations, but keeps a high-precision 32-bit copy of master weights in memory so tiny mathematical updates aren't lost to rounding errors.

NanoBrain leverages Automatic Mixed Precision (AMP) via `torch.amp.autocast("cuda", dtype=torch.bfloat16)` during forward passes:

```
           Forward Pass (BF16 / FP16)                   Backward Pass
Model Weights (FP32 Master Copy) ──Cast──► BF16 Activations ──► Loss Computation
            ▲                                                        │
            │                                                        ▼
   Optimizer Step (FP32) ◄── Unscale Grads ◄── Scaled Grads (FP16) ◄─┘
```

1. **Forward Pass:** Linear layer operations compute in fast BF16/FP16 tensor cores. Activations are stored in 16-bit precision, saving 50% VRAM.
2. **Master Weights:** High-precision FP32 master copies of parameters are preserved in optimizer states to accumulate small gradient updates correctly without rounding loss.
3. **GradScaler (for FP16 training):** When training in float16, small gradients risk underflowing to 0. `torch.amp.GradScaler` multiplies loss by a scale factor $S$ (e.g., $2^{16}$) before backprop, preserving small gradient values:
   $$g_{scaled} = \frac{\partial (S \cdot \mathcal{L})}{\partial \theta} = S \cdot g$$
   Before updating optimizer parameters, gradients are unscaled: $g = \frac{g_{scaled}}{S}$.
   *Note:* While BF16's wide dynamic range avoids underflow (making scaling unnecessary), `trainer.py` retains `GradScaler` to maintain seamless backward compatibility for FP16 fallback runs (where it acts as a harmless no-op under BF16).

---

### 3.5 Gradient Accumulation and Gradient Clipping

> **Intuition:** Large batch sizes make AI training stable, but large batches might crash a small GPU. Gradient Accumulation processes data in small micro-batches, adds up the learning updates quietly, and applies them all at once. Gradient Clipping prevents sudden massive updates from breaking the model.

#### 1. Gradient Accumulation
Training stability requires large effective batch sizes (e.g., $B_{eff} = 64$ sequences $\times 1024$ tokens $= 65,536$ tokens/step). If a GPU's VRAM can only fit micro-batch $B_{micro} = 8$, Gradient Accumulation splits the target step across $K = 8$ micro-steps:

```
Micro-Step 1: Forward(B=8)  ──► Backward() [Accumulate Grad += g1]
Micro-Step 2: Forward(B=8)  ──► Backward() [Accumulate Grad += g2]
...
Micro-Step 8: Forward(B=8)  ──► Backward() [Accumulate Grad += g8]
Optimizer Step: Update Weights using Accumulated Grad / 8 ──► zero_grad()
```

Mathematically, accumulated gradient $\bar{g}$ matches a single large batch pass:

$$\bar{g} = \frac{1}{K} \sum_{k=1}^K \nabla_\theta \mathcal{L}(X_k; \theta)$$

#### 2. Gradient Clipping
During training, sudden loss spikes can generate large gradients $\|g\|_2$. To prevent parameters from diverging, NanoBrain applies $L_2$ norm gradient clipping:

$$\text{If } \|g\|_2 > c_{max}, \quad g \leftarrow g \cdot \frac{c_{max}}{\|g\|_2}$$

where $\|g\|_2 = \sqrt{\sum_{i} g_i^2}$ and $c_{max} = 1.0$ (`grad_clip`).

---

### 3.6 Weight Decay, Fused AdamW, and Learning Rate Scheduling

> **Intuition:** Optimizers adjust model weights to minimize errors. AdamW adds a gentle penalty (weight decay) to keep weights small and prevent overfitting. A Cosine Learning Rate Schedule starts with a gentle warmup, accelerates, and then slowly ramps down step sizes as the model nears peak performance.

<p align="center">
  <img src="https://raw.githubusercontent.com/Amogh1221/NanoBrain/main/docs/images/learning_rate_schedule.svg" alt="Cosine Annealing Learning Rate Schedule Vector Plot" width="600">
  <br>
  <em>Linear Warmup + Cosine Annealing Learning Rate Schedule.</em>
</p>

#### 1. Decoupled Weight Decay (AdamW)
Standard $L_2$ regularization adds weight magnitude directly to loss: $\mathcal{L}_{reg} = \mathcal{L} + \frac{\lambda}{2} \|\theta\|^2$. In adaptive optimizers like Adam, this divides weight decay by moving gradient variances $\sqrt{v_t}$, weakening regularizing effects on parameters with large gradients.

**AdamW** (*Loshchilov & Hutter*) decouples weight decay, applying decay factor $\lambda$ directly to parameter updates:

$$\theta_t = \theta_{t-1} - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$

In NanoBrain ([`model.py`](https://github.com/Amogh1221/NanoBrain/blob/main/model.py)), weight decay ($\lambda = 0.1$) is applied exclusively to 2D linear weight matrices, while 1D LayerNorm weights and bias vectors are explicitly excluded.

#### 2. Fused AdamW
Standard PyTorch AdamW launches separate CUDA kernels for each parameter tensor to update optimizer states. **Fused AdamW** combines parameter tensor updates into a single CUDA kernel call, minimizing launch overhead and HBM roundtrips.

#### 3. Cosine Annealing Learning Rate Schedule with Warmup

```
Learning Rate η(t)
 ^
 |          / \
 |         /   \  Cosine Decay
 |        /     \
 |       /       ` - . _
 |      /                ` - . _
 |     /                         ` - . _  min_lr
 +----+----------------------------------+----> Iterations
     Warmup
```

NanoBrain executes a two-phase schedule:
1. **Linear Warmup:** For $t < T_{warmup}$ ($2,000$ steps), $\eta_t$ increases linearly from 0 to peak $\eta_{max} = 6 \times 10^{-4}$:
   $$\eta_t = \eta_{max} \frac{t + 1}{T_{warmup} + 1}$$
2. **Cosine Decay:** For $T_{warmup} \le t \le T_{decay}$ ($100,000$ steps), $\eta_t$ decays following a cosine curve down to $\eta_{min} = 6 \times 10^{-5}$:
   $$\alpha = \frac{t - T_{warmup}}{T_{decay} - T_{warmup}}$$
   $$\eta_t = \eta_{min} + 0.5 (1 + \cos(\pi \alpha)) (\eta_{max} - \eta_{min})$$

---

### 3.7 Exponential Moving Average (EMA) of Weights

> **Intuition:** During training, model weights jump around with every step. EMA keeps a running "smooth average" of all past weights. This smoothed model often performs better on test data.

Training stochasticity causes model parameters $\theta_t$ to oscillate around local minima. NanoBrain includes an **Exponential Moving Average (EMA)** module ([`model.py`](https://github.com/Amogh1221/NanoBrain/blob/main/model.py)) that maintains a smoothed shadow copy of parameters $\theta_{EMA}$:

$$\theta_{EMA}^{(t)} = \beta_{ema} \theta_{EMA}^{(t-1)} + (1 - \beta_{ema}) \theta_t$$

where $\beta_{ema} = 0.999$.

During validation evaluation and sample generation, EMA shadow weights are applied to the model. EMA parameter smoothing reduces evaluation loss variance and improves out-of-domain generalization.

---

## Module 4: Text Generation and Decoding Mechanics

---

### 4.1 Autoregressive Sampling Loop

> **Intuition:** Generating text is a step-by-step process: the model predicts one word, appends it to the prompt, feeds the expanded prompt back into itself, and repeats.

After predicting logits $z_T \in \mathbb{R}^{|V|}$ for sequence $x_{1:T}$, generating text requires appending predicted token $x_{T+1} \sim P(x \mid x_{\le T})$ to the context window and repeating prediction iteratively.

```
Prompt: "Artificial Intelligence is"
Step 1: Predict "a"         ---> New Sequence: "Artificial Intelligence is a"
Step 2: Predict "field"     ---> New Sequence: "Artificial Intelligence is a field"
Step 3: Predict "of"        ---> New Sequence: "Artificial Intelligence is a field of"
```

---

### 4.2 Softmax Temperature Scaling

> **Intuition:** Temperature controls creativity. Low temperature makes the AI pick only high-probability words (safe, predictable, robotic). High temperature flattens probabilities, making the output creative or chaotic.

Before applying sampling functions, logits $z$ are scaled by temperature scalar $T > 0$:

$$P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

```
Low Temp (T = 0.2):     [0.01,  0.01,  0.96,  0.01,  0.01]  (Peak distribution, deterministic)
Standard Temp (T = 1.0): [0.05,  0.10,  0.70,  0.10,  0.05]  (Balanced probability)
High Temp (T = 1.5):    [0.15,  0.20,  0.30,  0.20,  0.15]  (Flattened, uniform/random)
```

- **$T \to 0$ (Greedy Decoding):** Distribution approaches a one-hot vector at $\text{argmax}(z)$. Results in repetitive text.
- **$T = 1.0$:** Preserves true model probabilities.
- **$T > 1.0$:** Flattens probability distribution, increasing output randomness.

---

### 4.3 Truncation Algorithms: Top-K and Top-p (Nucleus) Sampling

> **Intuition:** Without truncation, a model might occasionally pick a bizarre, low-probability token. Top-K forces the model to choose only from the top $K$ candidates (e.g., top 50). Top-p dynamically expands or shrinks the pool of candidates until their combined probability reaches $p$ (e.g., 95%).

<p align="center">
  <img src="https://raw.githubusercontent.com/Amogh1221/NanoBrain/main/docs/images/decoding_sampling.svg" alt="Temperature Scaling & Truncation Sampling Vector SVG" width="680">
  <br>
  <em>Temperature Scaling Distributions and Top-K / Top-p (Nucleus) Truncation Sampling.</em>
</p>

```
Original Sorted Probabilities:
Token:      ["cat", "dog", "mat", "car", "banana", "galaxy"]
Prob:       [0.45,  0.25,  0.15,  0.10,   0.04,     0.01]

Top-K (K=3):
Retained:   ["cat", "dog", "mat"] (Mask remainder to -inf)

Top-p (p=0.80):
Cumulative: [0.45,  0.70,  0.85 (>=0.80 -> cutoff), ...]
Retained:   ["cat", "dog", "mat"]
```

#### 1. Top-K Sampling
Restricts sampling choices strictly to the $K$ highest-probability tokens:

$$V_{Top-K} = \text{TopK}(z, K) \implies \text{Set } z_i = -\infty \quad \forall i \notin V_{Top-K}$$

*Limitation:* Fixed $K$ does not adapt to distribution context variance (e.g., when 1 token is 99% probable vs. when 20 tokens are equally viable).

#### 2. Top-p (Nucleus) Sampling (*Holtzman et al.*)
Dynamically selects the smallest subset of tokens whose cumulative probability exceeds threshold $p$ (e.g., $p = 0.95$):

$$\sum_{i \in V_{Top-p}} P(x_i) \ge p$$

Logits outside this dynamic nucleus set are set to $-\infty$, ensuring adaptive truncation filtering across both narrow and wide probability distributions.

---

## Module 5: Modern Industry Evolution and Comparative Architecture

---

### 5.1 Architectural Shift: GPT-2 (NanoBrain) vs. LLaMA 1/2/3 and Mistral

> **Intuition:** GPT-2 laid the groundwork for modern LLMs in 2019. Modern open models like LLaMA 3 and Mistral use upgraded components—like Rotary Embeddings (RoPE), RMSNorm, and SwiGLU activations—which run faster and scale better across thousands of GPUs.

| Architectural Component | NanoBrain (GPT-2 Base) | Modern Standard (LLaMA 3 / Mistral) | Advantage of Modern Approach |
| :--- | :--- | :--- | :--- |
| **Positional Encoding** | Absolute Learned ($W_{pe}$) | Rotary Positional Embedding (RoPE) | Extrapolates to longer context lengths ($>128\text{K}$) |
| **Normalization Layer** | LayerNorm (Mean + Variance) | RMSNorm (Root Mean Square) | 10–50% faster computation by dropping mean calculation |
| **Normalization Position** | Pre-LN | Pre-LN with RMSNorm | Stable gradient flow in multi-billion parameter stacks |
| **Activation Function** | GELU (`approximate="tanh"`) | SwiGLU Gated Activation | Higher parameter expressivity per FLOP |
| **Attention Mechanism** | Multi-Head Attention (MHA) | Grouped-Query Attention (GQA) | Reduces KV Cache VRAM footprint during multi-user inference |
| **Attention Kernel** | FlashAttention-1 / SDPA | FlashAttention-2 / 3 | Higher GPU FLOPS utilization |
| **Bias Vectors** | Optional Linear Biases | Bias-Free Linear Layers (`bias=False`) | Improves hardware memory alignment and training stability |

---

### 5.2 Grouped-Query Attention (GQA) and Multi-Query Attention (MQA)

> **Intuition:** During chat inference, saving past Keys and Values (KV Cache) takes up massive GPU memory. Standard attention gives every Query its own Key/Value head. GQA groups Queries together to share Key/Value heads, saving huge amounts of memory without hurting accuracy.

<p align="center">
  <img src="https://raw.githubusercontent.com/Amogh1221/NanoBrain/main/docs/images/gqa_mha_mqa.svg" alt="Attention Head Architecture Comparison Vector SVG" width="650">
  <br>
  <em>Structural Comparison of Multi-Head Attention (MHA), Grouped-Query Attention (GQA), and Multi-Query Attention (MQA).</em>
</p>

```
Multi-Head Attention (MHA):     Grouped-Query Attention (GQA):    Multi-Query Attention (MQA):
Q Q Q Q  K K K K  V V V V        Q Q Q Q  K K  V V                Q Q Q Q  K  V
│ │ │ │  │ │ │ │  │ │ │ │        │ │ │ │  │ │  │ │                │ │ │ │  │  │
└─┴─┴─┘  └─┴─┴─┘  └─┴─┴─┘        ├───┼─┘  │ │  │ │                └───┴───┘  │  │
 (1:1 Ratio Q to K/V)             (Grouped Ratio, e.g., 4:2)        (All Q share 1 K/V)
```

- **Multi-Query Attention (MQA):** All Query heads share a single Key and Value head ($h_K = h_V = 1$). Slashing KV Cache memory by $h_Q\times$, but can drop modeling quality.
- **Grouped-Query Attention (GQA):** Partitions Query heads into $G$ groups, where each group shares one $K$ and $V$ head (e.g., $h_Q = 4, h_{K,V} = 2$ as shown in the diagram). Delivers near-MHA quality while matching MQA inference speeds.

---

# PART 2: CODEBASE ARCHITECTURE AND IMPLEMENTATION

---

## Module 6: NanoBrain Codebase Walkthrough

```
GPT/
├── config.py             # GPTConfig dataclass definition & methods
├── config.json           # Active hyperparameter values
├── model.py              # PyTorch modules (LayerNorm, Attention, MLP, GPT, EMA)
├── build_dataset.py      # Streaming raw datasets & corpus compilation
├── tokenize_dataset.py   # Pre-tokenizing text to uint16 train.bin / val.bin
├── dataset.py            # BinDataset (np.memmap) & DataLoader creation
├── trainer.py            # Trainer class (AMP, GradScaler, Logging, Eval, Checkpointing)
├── train.py              # Main entry point script
└── generate.py           # Text generation / inference script
```

---

### 6.1 Hyperparameter Configuration ([`config.py`](https://github.com/Amogh1221/NanoBrain/blob/main/config.py) / [`config.json`](https://github.com/Amogh1221/NanoBrain/blob/main/config.json))

The `GPTConfig` dataclass in [`config.py`](https://github.com/Amogh1221/NanoBrain/blob/main/config.py) encapsulates architectural, training, dataset, and system settings.

```python
@dataclass
class GPTConfig:
    # Architecture Defaults (GPT-2 Small 124M)
    vocab_size: int = 50258
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    block_size: int = 1024
    dropout: float = 0.0
    bias: bool = False

    # Optimization
    batch_size: int = 8
    gradient_accumulation_steps: int = 8
    max_iters: int = 100000
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_iters: int = 2000
    lr_decay_iters: int = 100000
    min_lr: float = 6e-5

    # System & Precision
    device: str = "cuda"
    dtype: str = "bfloat16"
    fused_adam: bool = True
    tf32: bool = True
```

#### Complete Parameter Calculation Breakdown (124,339,200 Parameters)
1. **Token Embeddings Matrix ($W_{te}$):** $V \times d_{embd} = 50,258 \times 768 = 38,598,144$
2. **Positional Embeddings Matrix ($W_{pe}$):** $T_{max} \times d_{embd} = 1,024 \times 768 = 786,432$
3. **Per Transformer Block ($N=12$):**
   - Self-Attention Linear `c_attn`: $d_{embd} \times 3 d_{embd} = 768 \times 2,304 = 1,769,472$
   - Self-Attention Projection `c_proj`: $d_{embd} \times d_{embd} = 768 \times 768 = 589,824$
   - MLP Expansion `c_fc`: $d_{embd} \times 4 d_{embd} = 768 \times 3,072 = 2,359,296$
   - MLP Projection `c_proj`: $4 d_{embd} \times d_{embd} = 3,072 \times 768 = 2,359,296$
   - Two LayerNorm Scale Vectors: $2 \times 768 = 1,536$
   - **Total per Block:** $7,079,424$ parameters.
4. **All 12 Blocks Combined:** $12 \times 7,079,424 = 84,953,088$
5. **Final LayerNorm (`ln_f`):** $2 \times 768 = 1,536$ parameters.
6. **Full Exact Model Total:** 
   $$\text{Wte } (38,598,144) + \text{Wpe } (786,432) + \text{Blocks } (84,953,088) + \text{LN}_f (1,536) = \mathbf{124,339,200 \text{ Parameters}}$$
   *(Note: Weight tying `self.wte.weight = self.lm_head.weight` means the output language head reuses $W_{te}$, avoiding an additional $38.6\text{M}$ parameter duplication).*

---

### 6.2 Data Sourcing and Pre-Processing ([`build_dataset.py`](https://github.com/Amogh1221/NanoBrain/blob/main/build_dataset.py))

[`build_dataset.py`](https://github.com/Amogh1221/NanoBrain/blob/main/build_dataset.py) constructs a high-quality pre-training text corpus by streaming and mixing 5 datasets:

```python
DATA_MIX = [
    Source("FineWeb-Edu", 0.33, hf_path="HuggingFaceFW/fineweb-edu", field="text"),
    Source("Wikipedia",   0.27, hf_path="wikimedia/wikipedia", hf_config="20231101.en", field="text"),
    Source("Code",        0.15, hf_path="transformersbook/codeparrot-train", field="content"),
    Source("Gutenberg",   0.15),
    Source("FineMath",    0.10, hf_path="HuggingFaceTB/finemath", hf_config="finemath-3plus", field="text"),
]
```

- Cleans text via Unicode NFKC normalization (`unicodedata.normalize`).
- Strips invalid control characters (`re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)`).
- Filters out document lengths under 300 characters (`is_valid_document`).
- Appends document separator `<|endoftext|>` between entries, writing the output to `data/corpus.txt`.

---

### 6.3 Binary Packing and Zero-Copy Data Loading ([`tokenize_dataset.py`](https://github.com/Amogh1221/NanoBrain/blob/main/tokenize_dataset.py) and [`dataset.py`](https://github.com/Amogh1221/NanoBrain/blob/main/dataset.py))

#### 1. Binary Pre-Tokenization ([`tokenize_dataset.py`](https://github.com/Amogh1221/NanoBrain/blob/main/tokenize_dataset.py))
Tokenizing a 10GB text file on every training launch is slow and memory-intensive. `tokenize_dataset.py` pre-tokenizes `corpus.txt` into raw 16-bit binary files:

- Uses `tiktoken` to map string tokens to `uint16` integers (since $\max(|V|) = 50,257 < 2^{16} = 65,535$).
- Writes non-overlapping flat arrays directly to `data/train.bin` (90%) and `data/val.bin` (10%).

#### 2. Zero-Copy Memory Mapping ([`dataset.py`](https://github.com/Amogh1221/NanoBrain/blob/main/dataset.py))
[`dataset.py`](https://github.com/Amogh1221/NanoBrain/blob/main/dataset.py) defines `BinDataset`, leveraging `np.memmap` to interface with binary dataset files:

```python
class BinDataset(Dataset):
    def __init__(self, bin_path: str, block_size: int):
        self.block_size = block_size
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.data[idx : idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1 : idx + self.block_size + 1].astype(np.int64))
        return x, y
```

- **Memory Efficiency:** `np.memmap` creates a virtual pointer to disk files. Pages are loaded into RAM on demand by the OS kernel and evicted dynamically.
- **Speed:** Instant dataset startup (sub-millisecond load time) regardless of whether dataset size is 1GB or 100GB.

---

### 6.4 PyTorch Model Implementation ([`model.py`](https://github.com/Amogh1221/NanoBrain/blob/main/model.py))

[`model.py`](https://github.com/Amogh1221/NanoBrain/blob/main/model.py) implements the Transformer architecture components:

#### 1. Layer Normalization
```python
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
```

#### 2. Causal Self-Attention & FlashAttention
```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, layer_past=None):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if layer_past is not None:
            k = torch.cat([layer_past[0], k], dim=2)
            v = torch.cat([layer_past[1], v], dim=2)

        present = torch.stack([k.detach(), v.detach()])

        # Uses PyTorch FlashAttention / SDPA Kernel
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, present
```

#### 3. Weight Initialization Scaling
GPT-2 scales projection weights at initialization to account for growth in residual stream variance at deeper layers:

```python
for pn, p in self.named_parameters():
    if pn.endswith("c_proj.weight"):
        torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
```

Standard weights initialize to $\mathcal{N}(0, 0.02)$. Residual projections (`c_proj`) scale down by $\frac{1}{\sqrt{2 N_{layers}}}$, preserving activation variance bounds across 12 stacked residual layers.

---

### 6.5 Unified Training Engine ([`trainer.py`](https://github.com/Amogh1221/NanoBrain/blob/main/trainer.py) and [`train.py`](https://github.com/Amogh1221/NanoBrain/blob/main/train.py))

[`trainer.py`](https://github.com/Amogh1221/NanoBrain/blob/main/trainer.py) manages the training execution loop:

```python
# Mixed-Precision Autocast Context
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    logits, _ = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    loss = loss / config.gradient_accumulation_steps

# Scaled Gradient Backprop
scaler.scale(loss).backward()

# Step Execution after Gradient Accumulation Micro-Steps
if self.micro_step % config.gradient_accumulation_steps == 0:
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    if self.ema is not None:
        self.ema.update()
```

- **Logging & Monitoring:** Logs step loss, gradient norm, throughput ($\text{tokens/sec}$), allocated VRAM, and estimated completion time (ETA) to stdout, TensorBoard (`runs/`), and a persistent structured file (`logs/training_log.txt`).
- **Checkpoint Handling:** Periodically evaluates validation loss and exports `checkpoints/latest.pt` and `checkpoints/best.pt`.

---

### 6.6 Text Generation Script ([`generate.py`](https://github.com/Amogh1221/NanoBrain/blob/main/generate.py))

[`generate.py`](https://github.com/Amogh1221/NanoBrain/blob/main/generate.py) runs model inference:

```python
# Temperature, Top-K, Top-P Nucleus Sampling Implementation
logits = logits[:, -1, :] / temperature

if top_k is not None:
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = -float("Inf")

if top_p is not None:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = -float("Inf")

probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)
```

---

# Summary and Best Practices for Experimentation

Congratulations! You now have a first-principles understanding of Large Language Models and the complete implementation details of the NanoBrain codebase.

### Quickstart Command Sequence

```bash
# 1. Download and construct the text dataset corpus (Target: 1 GB)
python build_dataset.py

# 2. Pre-tokenize corpus into uint16 train.bin and val.bin files
python tokenize_dataset.py --input data/corpus.txt --split 0.9

# 3. Launch mixed-precision model training
python train.py

# 4. Generate text samples from the trained checkpoint
python generate.py "Artificial Intelligence is"
```

Happy modeling! For questions or contributions, feel free to open an issue or pull request on the repository.
