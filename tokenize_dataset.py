#!/usr/bin/env python3
"""
tokenize_dataset.py  —  nano_brain pre-tokenizer
================================================
Converts data/corpus.txt (or any plain-text corpus) into memory-mapped
uint16 binary shards that the DataLoader can stream directly from disk.

Why this exists
---------------
Loading a 10 GB text file, then calling tokenizer.encode() on the whole
thing takes ~10–30 minutes and requires ~20 GB of RAM on every training
run.  Pre-tokenising once (this script) produces two compact binary files:

    data/train.bin   – 90 % of tokens, uint16, memory-mapped at load time
    data/val.bin     –  10 % of tokens, uint16, memory-mapped at load time

These files load in milliseconds and use almost no RAM beyond what is
actually being read.  Token ids fit in uint16 (max 65535), which covers
the GPT-2 vocabulary (50,257 ids) without overflow.

Usage
-----
    python tokenize_dataset.py                       # uses defaults
    python tokenize_dataset.py --input data/corpus.txt --split 0.9
    python tokenize_dataset.py --input data/corpus.txt --shard-size 500

Options
-------
    --input        Path to the source text file  [default: data/corpus.txt]
    --out-dir      Directory for .bin files      [default: data]
    --split        Train / val ratio             [default: 0.90]
    --shard-size   Tokens per write shard (M)    [default: 250 M tokens]
    --eot          Encode <|endoftext|> between  [default: True]
                   documents (recommended)

Output
------
    data/train.bin          – training tokens (uint16, little-endian)
    data/val.bin            – validation tokens (uint16, little-endian)
    data/tokenize_stats.json – stats about the tokenisation run
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import tiktoken
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tokenize_dataset")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

BASE = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE / "data" / "corpus.txt"
DEFAULT_OUT = BASE / "data"

# GPT-2 tokenizer (tiktoken)
ENCODING_NAME = "gpt2"

# How many raw characters to batch before calling encode().
# Larger = faster but uses more RAM during tokenisation.
ENCODE_CHUNK_CHARS = 50_000_000  # 50 MB of text at a time

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def human_size(n_bytes: int) -> str:
    """Return a human-readable file-size string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def human_tokens(n: int) -> str:
    """Return a human-readable token count string."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    return f"{n:,}"


def write_bin(tokens: np.ndarray, path: Path) -> None:
    """Append a uint16 token array to a binary file."""
    with open(path, "ab") as f:
        tokens.astype(np.uint16).tofile(f)


# ──────────────────────────────────────────────────────────────────────────────
# Core tokenisation
# ──────────────────────────────────────────────────────────────────────────────


def tokenize(
    input_path: Path,
    out_dir: Path,
    split_ratio: float = 0.90,
    add_eot: bool = True,
    shard_size_M: int = 250,
) -> dict:
    """
    Stream-tokenise *input_path* and write train.bin / val.bin.

    Returns a stats dict suitable for JSON serialisation.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"

    # Remove old outputs so we start clean
    for p in (train_path, val_path):
        if p.exists():
            log.warning("Removing existing %s", p)
            p.unlink()

    enc = tiktoken.get_encoding(ENCODING_NAME)
    eot_id: int = enc.eot_token  # 50256 for GPT-2

    log.info("Tokenizer : %s  (vocab=%d, eot=%d)", ENCODING_NAME, enc.n_vocab, eot_id)
    log.info("Input     : %s  (%s)", input_path, human_size(input_path.stat().st_size))
    log.info("Output    : %s/  (train.bin + val.bin)", out_dir)
    log.info("Split     : %.0f%% train / %.0f%% val", split_ratio * 100, (1 - split_ratio) * 100)
    log.info("EOT token : %s", "yes" if add_eot else "no")
    log.info("")

    input_size = input_path.stat().st_size
    shard_tokens = shard_size_M * 1_000_000

    total_tokens = 0
    total_docs = 0
    doc_lengths: list[int] = []   # tokens per document (sampled, not all)

    start_time = time.time()

    # ── First pass: count total tokens to know the split boundary ──────────
    log.info("Pass 1 / 2 — counting tokens to determine split boundary …")

    # We stream the file in large text chunks.  Documents are separated by
    # blank lines (two consecutive newlines) in the corpus.txt format that
    # build_dataset.py produces.

    chunk_buf = ""
    approx_total = 0

    with open(input_path, "r", encoding="utf-8") as f:
        pbar = tqdm(
            total=input_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Counting",
        )
        for line in f:
            pbar.update(len(line.encode("utf-8")))
            chunk_buf += line
            if len(chunk_buf) >= ENCODE_CHUNK_CHARS:
                ids = enc.encode_ordinary(chunk_buf)
                approx_total += len(ids)
                chunk_buf = ""
        if chunk_buf:
            ids = enc.encode_ordinary(chunk_buf)
            approx_total += len(ids)
        pbar.close()

    # Adjust for EOT tokens (one per document; we'll count docs in pass 2)
    # We don't know doc count yet, so use approximation for the split byte
    split_token_target = int(approx_total * split_ratio)
    log.info(
        "Approx total tokens : %s  →  split target: %s train / %s val",
        human_tokens(approx_total),
        human_tokens(split_token_target),
        human_tokens(approx_total - split_token_target),
    )
    log.info("")

    # ── Second pass: tokenise and write ─────────────────────────────────────
    log.info("Pass 2 / 2 — tokenising and writing binary files …")

    train_tokens_written = 0
    val_tokens_written = 0
    shard_buf: list[int] = []
    in_val = False

    def flush_shard(to_val: bool) -> None:
        nonlocal train_tokens_written, val_tokens_written
        arr = np.array(shard_buf, dtype=np.uint16)
        if to_val:
            write_bin(arr, val_path)
            val_tokens_written += len(arr)
        else:
            write_bin(arr, train_path)
            train_tokens_written += len(arr)
        shard_buf.clear()

    with open(input_path, "r", encoding="utf-8") as f:
        pbar = tqdm(
            total=input_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Tokenising",
        )

        doc_text = []
        for raw_line in f:
            pbar.update(len(raw_line.encode("utf-8")))
            line = raw_line.rstrip("\n")

            # Document boundary: corpus.txt uses "\n\n" between documents
            if line == "" and doc_text and doc_text[-1] == "":
                # We have a blank-line sequence → end of document
                full_doc = "\n".join(doc_text).strip()
                if full_doc:
                    ids = enc.encode_ordinary(full_doc)
                    if add_eot:
                        ids.append(eot_id)

                    # Sample doc lengths (every 100th doc) for stats
                    if total_docs % 100 == 0:
                        doc_lengths.append(len(ids))

                    total_docs += 1
                    total_tokens += len(ids)
                    shard_buf.extend(ids)

                    # Decide whether we've crossed into val territory
                    if not in_val and total_tokens >= split_token_target:
                        # Flush remaining train shard
                        if shard_buf:
                            flush_shard(to_val=False)
                        in_val = True

                    if len(shard_buf) >= shard_tokens:
                        flush_shard(to_val=in_val)

                doc_text = []
            else:
                doc_text.append(line)

        # Handle last document (file may not end with double newline)
        if doc_text:
            full_doc = "\n".join(doc_text).strip()
            if full_doc:
                ids = enc.encode_ordinary(full_doc)
                if add_eot:
                    ids.append(eot_id)
                doc_lengths.append(len(ids))
                total_docs += 1
                total_tokens += len(ids)
                shard_buf.extend(ids)

        # Flush remaining buffer
        if shard_buf:
            flush_shard(to_val=in_val)

        pbar.close()

    # ── Verification ────────────────────────────────────────────────────────
    log.info("")
    log.info("Verifying output files …")

    def verify(p: Path, expected_tokens: int) -> int:
        if not p.exists():
            log.error("  MISSING: %s", p)
            return 0
        actual = p.stat().st_size // 2  # uint16 = 2 bytes per token
        status = "OK" if abs(actual - expected_tokens) < 1000 else "MISMATCH"
        log.info(
            "  %-12s  %s tokens  (%s)  [%s]",
            p.name,
            human_tokens(actual),
            human_size(p.stat().st_size),
            status,
        )
        return actual

    train_actual = verify(train_path, train_tokens_written)
    val_actual = verify(val_path, val_tokens_written)

    elapsed = time.time() - start_time
    tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0

    # ── Stats ────────────────────────────────────────────────────────────────
    avg_doc_len = int(np.mean(doc_lengths)) if doc_lengths else 0
    median_doc_len = int(np.median(doc_lengths)) if doc_lengths else 0

    stats = {
        "encoding": ENCODING_NAME,
        "input_file": str(input_path),
        "input_size_bytes": int(input_path.stat().st_size),
        "total_documents": total_docs,
        "total_tokens": total_tokens,
        "train_tokens": train_actual,
        "val_tokens": val_actual,
        "split_ratio": split_ratio,
        "eot_token_added": add_eot,
        "avg_doc_tokens": avg_doc_len,
        "median_doc_tokens": median_doc_len,
        "elapsed_seconds": round(elapsed, 1),
        "tokens_per_second": round(tok_per_sec, 0),
        "train_bin": str(train_path),
        "val_bin": str(val_path),
    }

    stats_path = out_dir / "tokenize_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Pretty-print summary
# ──────────────────────────────────────────────────────────────────────────────


def print_summary(stats: dict) -> None:
    hr = "=" * 64
    print()
    print(hr)
    print("  TOKENISATION COMPLETE — nano_brain")
    print(hr)
    print(f"  Input file      : {Path(stats['input_file']).name}")
    print(f"  Input size      : {human_size(stats['input_size_bytes'])}")
    print(f"  Total documents : {stats['total_documents']:,}")
    print(f"  Total tokens    : {human_tokens(stats['total_tokens'])}")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  train.bin       : {human_tokens(stats['train_tokens'])} tokens")
    print(f"  val.bin         : {human_tokens(stats['val_tokens'])} tokens")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Avg doc length  : {stats['avg_doc_tokens']:,} tokens")
    print(f"  Median doc len  : {stats['median_doc_tokens']:,} tokens")
    print(f"  EOT separator   : {'yes  (<|endoftext|> = 50256)' if stats['eot_token_added'] else 'no'}")
    print(f"  Elapsed         : {stats['elapsed_seconds']:.0f}s")
    print(f"  Throughput      : {human_tokens(int(stats['tokens_per_second']))} tok/s")
    print(hr)
    print(f"  → Update config.json: set  \"dataset\": \"train.bin\"")
    print(f"  → Stats saved to: {Path(stats['val_bin']).parent / 'tokenize_stats.json'}")
    print(hr)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Chinchilla / time budget estimate
# ──────────────────────────────────────────────────────────────────────────────


def print_training_estimate(total_tokens: int) -> None:
    """
    Print a rough estimate for a 124M-parameter GPT on an RTX 3060 12GB.
    Numbers are approximate — real throughput depends on compile, dtype, etc.
    """
    model_params = 124_000_000
    chinchilla_optimal = model_params * 20      # Chinchilla law
    days_28 = 28 * 24 * 3600                   # 4 weeks in seconds
    # RTX 3060 12GB @ bf16, 124M model, block_size=1024, grad_accum=8
    tok_per_sec_low  = 14_000
    tok_per_sec_high = 18_000

    tokens_4w_low  = tok_per_sec_low  * days_28
    tokens_4w_high = tok_per_sec_high * days_28

    print()
    print("=" * 64)
    print("  TRAINING BUDGET ESTIMATE  (RTX 3060 12GB, 124M model)")
    print("=" * 64)
    print(f"  Dataset tokens          : {human_tokens(total_tokens)}")
    print(f"  Chinchilla optimal      : {human_tokens(chinchilla_optimal)}  (20 × params)")
    print(f"  Tokens in 4 weeks (low) : {human_tokens(tokens_4w_low)}")
    print(f"  Tokens in 4 weeks (high): {human_tokens(tokens_4w_high)}")
    if total_tokens >= chinchilla_optimal:
        ratio = total_tokens / chinchilla_optimal
        print(f"  Your dataset is         : {ratio:.1f}x Chinchilla-optimal ✓")
    else:
        deficit = chinchilla_optimal - total_tokens
        print(f"  Deficit vs Chinchilla   : {human_tokens(deficit)} tokens — consider more data")
    print("=" * 64)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-tokenise corpus.txt → train.bin + val.bin for nano_brain training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the source text corpus (corpus.txt from build_dataset.py)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory for train.bin and val.bin",
    )
    p.add_argument(
        "--split",
        type=float,
        default=0.90,
        metavar="RATIO",
        help="Fraction of tokens to use for training (rest goes to validation)",
    )
    p.add_argument(
        "--shard-size",
        type=int,
        default=250,
        metavar="MILLION_TOKENS",
        help="Tokens per write shard (controls peak RAM during tokenisation)",
    )
    p.add_argument(
        "--no-eot",
        action="store_true",
        help="Do NOT insert <|endoftext|> between documents (not recommended)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    input_path: Path = args.input
    out_dir: Path = args.out_dir
    split_ratio: float = args.split
    shard_size_M: int = args.shard_size
    add_eot: bool = not args.no_eot

    # ── Validate input ───────────────────────────────────────────────────────
    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        log.error("Run  python build_dataset.py  first to generate data/corpus.txt")
        sys.exit(1)

    if not (0.5 <= split_ratio <= 0.99):
        log.error("--split must be between 0.50 and 0.99, got %.2f", split_ratio)
        sys.exit(1)

    # ── Run tokenisation ─────────────────────────────────────────────────────
    log.info("nano_brain — Dataset Pre-Tokeniser")
    log.info("=" * 52)

    stats = tokenize(
        input_path=input_path,
        out_dir=out_dir,
        split_ratio=split_ratio,
        add_eot=add_eot,
        shard_size_M=shard_size_M,
    )

    print_summary(stats)
    print_training_estimate(stats["total_tokens"])

    log.info("Done. Next steps:")
    log.info('  1. Edit config.json  →  set "dataset": "train.bin"')
    log.info("  2. python train.py")


if __name__ == "__main__":
    main()
