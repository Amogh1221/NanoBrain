<div align="center">

# 🧠 nano_brain

**A clean GPT-2-class language model trained from scratch in PyTorch.**  
Designed to run on a single consumer GPU (12 GB VRAM) for 2–4 weeks and produce a genuinely capable 124M-parameter model.

[**📖 Read the Masterclass Technical Wiki**](https://github.com/Amogh1221/NanoBrain/wiki)  
*An exhaustive, first-principles textbook covering LLM theory, Transformer math, FlashAttention, CUDA memory hierarchy, and codebase implementation.*

</div>

---

## Overview

nano_brain is an end-to-end LLM pretraining pipeline:

```
build_dataset.py  →  tokenize_dataset.py  →  train.py  →  generate.py
   (collect data)       (binarise data)       (train)       (inference)
```

It covers every step: multi-source dataset collection, pre-tokenisation, training with full AMP/gradient accumulation/EMA/checkpointing, rich terminal + file logging, and interactive text generation.

---

## Architecture — GPT-2 Small (124M parameters)

| Component | Specification |
|---|---|
| Layers (`n_layer`) | 12 |
| Attention heads (`n_head`) | 12 |
| Embedding dim (`n_embd`) | 768 |
| Head dim | 64 |
| Context length (`block_size`) | 1024 tokens |
| Feed-forward | 4× expansion, GELU (tanh approx.) |
| Vocabulary | 50,258 (GPT-2 BPE, via tiktoken) |
| Parameters | ~124M |
| Attention | Flash Attention via `scaled_dot_product_attention` |
| Positional encoding | Learned absolute (wpe) |

This matches the GPT-2 Small architecture exactly. It is the proven sweet spot for a **12 GB VRAM GPU** — it uses ~10–11 GB, leaving headroom for activations.

---

## Hardware & Time Budget

| GPU | Tokens/sec (estimated) | Tokens in 4 weeks |
|---|---|---|
| **RTX 3060 12GB** | 14,000 – 18,000 | **34B – 43B** |
| RTX 4070 12GB | 25,000 – 35,000 | 60B – 85B |

The [Chinchilla scaling law](https://arxiv.org/abs/2203.15556) recommends **~20 tokens per parameter** for compute-optimal training:

- **Chinchilla-optimal for 124M:** ~2.5B tokens
- **Recommended dataset size:** 10–15 GB raw text (~2.6B–4B tokens)
- **RTX 3060 in 4 weeks:** vastly exceeds Chinchilla-optimal → the model will be well over-trained, which is **desirable** for a model this size (better quality at inference)

---

## Complete Setup & Workflow

### 1. Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install tiktoken datasets tqdm requests numpy
```

> **Windows note:** `torch.compile` is not reliable on Windows + PyTorch < 2.4.  
> Keep `"compile": false` in `config.json` unless you're on WSL2 or Linux.

---

### 2. Build the dataset

```bash
python build_dataset.py
# When prompted, enter a target size in GB (recommended: 10)
```

This streams from **5 high-quality open sources** in a curated proportion mix:

| Source | Proportion | Description |
|---|---|---|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 33% | Filtered educational web text |
| [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) | 27% | English Wikipedia (Nov 2023) |
| [CodeParrot](https://huggingface.co/datasets/transformersbook/codeparrot-train) | 15% | Python source code |
| [Project Gutenberg](https://www.gutenberg.org/) | 15% | Classic literature (via gutendex API) |
| [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) | 10% | Mathematical text |

**Output:** `data/corpus.txt` + `data/dataset_stats.json`

The cleaning pipeline:
- Unicode NFKC normalization
- Control character removal
- Whitespace normalisation (max 2 consecutive newlines)
- Hash-based deduplication (per source)
- Minimum document length filter (300+ chars)

---

### 3. Pre-tokenise the dataset *(required before training)*

```bash
python tokenize_dataset.py
```

This converts `data/corpus.txt` into two compact binary files:

```
data/train.bin   –  90% of tokens  (uint16, memory-mapped)
data/val.bin     –  10% of tokens  (uint16, memory-mapped)
```

**Why this step is critical:**

| Method | 10 GB corpus startup | RAM needed |
|---|---|---|
| Load `corpus.txt` at train time | 10–30 minutes | ~20 GB |
| **Load `train.bin` at train time** | **< 1 second** | **~0 GB** |

The binary files are memory-mapped — the OS loads pages on demand, so training starts immediately and RAM usage is negligible.

An `<|endoftext|>` (token id 50256) separator is inserted between every document. This teaches the model where documents end and prevents it from learning spurious cross-document context.

**Options:**

```bash
python tokenize_dataset.py --input data/corpus.txt   # default
python tokenize_dataset.py --split 0.95              # more training data
python tokenize_dataset.py --no-eot                  # skip EOT (not recommended)
```

The default `config.json` already points to `train.bin`, so no manual change is needed if you're using the recommended config.

---

### 4. Configure

Edit `config.json` (created automatically on first `python train.py`, or copy the recommended config below):

```json
{
  "vocab_size": 50258,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12,
  "block_size": 1024,
  "dropout": 0.0,
  "bias": false,
  "batch_size": 8,
  "gradient_accumulation_steps": 8,
  "max_iters": 100000,
  "learning_rate": 6e-4,
  "weight_decay": 0.1,
  "beta1": 0.9,
  "beta2": 0.95,
  "warmup_iters": 2000,
  "lr_decay_iters": 100000,
  "min_lr": 6e-5,
  "eval_interval": 500,
  "eval_iters": 200,
  "log_interval": 10,
  "save_interval": 1000,
  "gen_interval": 5000,
  "max_new_tokens_gen": 256,
  "num_generations": 3,
  "device": "cuda",
  "dtype": "bfloat16",
  "compile": false,
  "fused_adam": true,
  "tf32": true,
  "dataset": "train.bin",
  "data_dir": "data",
  "temperature": 0.8,
  "top_k": 50,
  "top_p": 0.95,
  "ema_decay": 0.999,
  "use_ema": true,
  "grad_clip": 1.0
}
```

> **Note on `dropout: 0.0`:** For pretraining at this scale (tokens/param >> 1), dropout hurts more than it helps. GPT-2 and most modern LLMs train with zero dropout. Only enable it if your dataset is very small and you're seeing severe overfitting.

> **Note on effective batch size:** `batch_size=8` × `gradient_accumulation_steps=8` × `block_size=1024` = **65,536 tokens per update**. This is the effective batch size that drives gradient stability. Never reduce `gradient_accumulation_steps` without also reducing `learning_rate`.

---

### 5. Train

```bash
python train.py
```

Training resumes automatically from `checkpoints/latest.pt` if it exists. To start fresh, delete the checkpoint.

**Interrupt safely at any time with `Ctrl+C`** — the trainer catches `KeyboardInterrupt` and saves a checkpoint before exiting.

---

### 6. Generate text

```bash
python generate.py "The meaning of life is"
```

Generation parameters (via environment variables):

| Variable | Default | Description |
|---|---|---|
| `TEMP` | `0.8` | Temperature (higher = more random) |
| `TOP_K` | `50` | Top-k sampling cutoff |
| `TOP_P` | `0.95` | Nucleus (top-p) sampling threshold |
| `MAX_NEW` | `500` | Maximum tokens to generate |

```bash
# Example: low temperature, longer output
set TEMP=0.6 && set MAX_NEW=1000 && python generate.py "Once upon a time"
```

---

## Monitoring & Logging

### Terminal output

Every `log_interval` steps (default: every 10 steps), the trainer prints:

```
[Step   1000/100000]  loss=3.4521  lr=2.85e-04  grad_norm=0.82  tok/s=16,234  VRAM=10.1/12.0GB  ETA=3d 14h
```

Every `eval_interval` steps (default: every 500 steps), a full evaluation block is printed:

```
════════════════════════════════════════════════════
  EVALUATION @ Step 1000 / 100000   (1.0%)
════════════════════════════════════════════════════
  Train Loss     : 3.4521
  Val Loss       : 3.6102  (best: 3.5901  Δ: +0.020)
  Perplexity     : 37.02
  EMA Val Loss   : 3.5834
  Learning Rate  : 2.85e-04
  Avg Grad Norm  : 0.823
  Tokens/sec     : 16,234
  VRAM           : 10.1 / 12.0 GB
  Elapsed        : 00:14:22
  ETA            : 3d 13h 46m
════════════════════════════════════════════════════
```

### TensorBoard

```bash
tensorboard --logdir runs
```

Logged metrics:

| Tag | Description |
|---|---|
| `train/loss` | Smoothed training loss (every `log_interval` steps) |
| `train/lr` | Current learning rate |
| `train/grad_norm` | Average gradient L2 norm |
| `train/tokens_per_sec` | Training throughput |
| `eval/train_loss` | Train loss from `estimate_loss()` |
| `eval/val_loss` | Validation loss |
| `eval/perplexity` | `exp(val_loss)` — more interpretable than raw loss |
| `eval/ema_val_loss` | EMA model validation loss |

### logs/training_log.txt

A persistent, structured log file is written throughout training. It survives terminal closes, SSH drops, and crashes. Each evaluation block is appended as a readable record:

```
[2026-07-23 01:15:32] STEP 1000 | train=3.4521 | val=3.6102 | ppl=37.02 | lr=2.85e-04 | tok/s=16234 | VRAM=10.1GB | ETA=3d13h
```

---

## Features

| Feature | Details |
|---|---|
| **Flash Attention** | `F.scaled_dot_product_attention` auto-dispatches to FlashAttention on Ampere+ (RTX 30xx/40xx). No extra install needed. |
| **Mixed precision (AMP)** | Trains in `bfloat16` by default. ~1.5–2× faster on Tensor Core GPUs. GradScaler for float16 fallback. |
| **TF32** | Enables TensorFloat-32 matmuls on Ampere+ for free speed at no quality cost. |
| **EMA** | Exponential moving average of weights (`decay=0.999`). EMA model is evaluated separately and used at inference for better generalisation. |
| **Gradient accumulation** | Simulates large batches (65K tokens/step) on limited VRAM. |
| **Gradient clipping** | Clips gradient norm to 1.0. Grad norm is tracked and logged. |
| **Cosine LR schedule** | Warmup 0 → `learning_rate` over `warmup_iters` steps, then cosine decay to `min_lr`. |
| **Fused AdamW** | CUDA-fused AdamW (~5–10% optimizer speedup). |
| **Weight tying** | `wte` (token embedding) and `lm_head` share weights — fewer parameters, better quality. |
| **KV cache** | Keys/values are cached during generation for O(T) per-token cost. |
| **Checkpointing** | Saves `checkpoints/latest.pt` (every `save_interval` steps) and `checkpoints/best.pt` (whenever val loss improves). Both include full optimizer state for seamless resume. |
| **Auto-resume** | Detects `checkpoints/latest.pt` at startup and resumes from saved iteration. |
| **Sample generation** | Generates `num_generations` text samples every `gen_interval` steps into `samples/`. |
| **Memory-mapped dataset** | `train.bin`/`val.bin` load in <1 second via `np.memmap`. No RAM spike at startup. |

---

## Project Structure

```
nano_brain/
├── build_dataset.py      # Multi-source dataset collector (HuggingFace + Gutenberg)
├── tokenize_dataset.py   # Converts corpus.txt → train.bin + val.bin (run once)
├── config.py             # GPTConfig dataclass (all hyperparameters)
├── config.json           # Active hyperparameter values (edit this)
├── tokenizer.py          # Thin wrapper around tiktoken (GPT-2 BPE)
├── dataset.py            # BinDataset (memmap) + TextDataset + DataLoader factory
├── model.py              # GPT model: LayerNorm, CausalSelfAttention, MLP, Block, EMA
├── trainer.py            # Training loop: AMP, grad accum, logging, checkpointing
├── train.py              # Entry point
├── generate.py           # Interactive text generation
│
├── data/
│   ├── corpus.txt        # Raw text corpus (from build_dataset.py)
│   ├── train.bin         # Pre-tokenised training tokens, uint16 (from tokenize_dataset.py)
│   ├── val.bin           # Pre-tokenised validation tokens, uint16
│   ├── dataset_stats.json
│   └── tokenize_stats.json
│
├── checkpoints/
│   ├── latest.pt         # Most recent checkpoint (auto-resumed)
│   └── best.pt           # Best validation loss checkpoint
│
├── logs/
│   └── training_log.txt  # Persistent structured training log
│
├── samples/              # Generated text samples (step_NNNNNNN_i.txt)
└── runs/                 # TensorBoard event files
```

---

## Configuration Reference

All hyperparameters live in `config.json`. Key fields:

### Model Architecture

| Key | Default | Description |
|---|---|---|
| `n_embd` | `768` | Embedding / hidden dimension |
| `n_head` | `12` | Number of attention heads |
| `n_layer` | `12` | Number of transformer blocks |
| `block_size` | `1024` | Context window (tokens) |
| `vocab_size` | `50258` | Set automatically from tokenizer |
| `dropout` | `0.0` | Dropout probability (0 = disabled) |
| `bias` | `false` | Add bias to Linear layers (GPT-2 style: false) |

### Training

| Key | Default | Description |
|---|---|---|
| `batch_size` | `8` | Micro-batch size per GPU step |
| `gradient_accumulation_steps` | `8` | Accumulate before weight update |
| `max_iters` | `100000` | Total optimizer steps |
| `learning_rate` | `6e-4` | Peak learning rate |
| `warmup_iters` | `2000` | LR warmup steps |
| `lr_decay_iters` | `100000` | Steps over which to decay LR |
| `min_lr` | `6e-5` | Minimum LR (end of cosine schedule) |
| `weight_decay` | `0.1` | AdamW weight decay |
| `grad_clip` | `1.0` | Gradient norm clipping threshold |

### Logging & Saving

| Key | Default | Description |
|---|---|---|
| `log_interval` | `10` | Print training metrics every N steps |
| `eval_interval` | `500` | Run full evaluation every N steps |
| `eval_iters` | `200` | Batches to average for loss estimate |
| `save_interval` | `1000` | Save `latest.pt` every N steps |
| `gen_interval` | `5000` | Generate text samples every N steps |

### System

| Key | Default | Description |
|---|---|---|
| `device` | `"cuda"` | `"cuda"` or `"cpu"` |
| `dtype` | `"bfloat16"` | `"bfloat16"` (recommended) or `"float16"` or `"float32"` |
| `compile` | `false` | Enable `torch.compile` (Linux/WSL2 only for reliability) |
| `tf32` | `true` | TensorFloat-32 matmuls (Ampere+ only) |
| `fused_adam` | `true` | CUDA-fused AdamW |

---

## Understanding the Loss Curve

| Validation Loss | Perplexity | What it means |
|---|---|---|
| ~4.5–5.0 | ~90–150 | Early training — random-ish output |
| ~3.5–4.0 | ~33–55 | Mid training — grammatical, coherent short phrases |
| ~3.0–3.5 | ~20–33 | Good — coherent paragraphs, follows topic |
| ~2.8–3.0 | ~16–20 | Strong — GPT-2 Small territory |
| < 2.8 | < 16 | Excellent for a 124M model |

After 4 weeks on an RTX 3060, expect to land around **~3.0–3.3** depending on dataset quality.

---

## References

- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) — Radford et al. 2019
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Karpathy's minimal GPT-2 implementation (primary inspiration)
- [Chinchilla](https://arxiv.org/abs/2203.15556) — Hoffmann et al. 2022 (scaling laws)
- [Flash Attention](https://arxiv.org/abs/2205.14135) — Dao et al. 2022
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) — HuggingFace dataset
- [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) — HuggingFace dataset
- [tiktoken](https://github.com/openai/tiktoken) — OpenAI's fast BPE tokenizer
