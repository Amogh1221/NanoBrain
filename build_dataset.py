#!/usr/bin/env python3
"""
General-purpose LLM pretraining dataset builder.
Streams from 5 high-quality open sources in proportional mix.

Usage:
    python build_dataset.py
    # Enter target size in GB (0.1 - 100)

Output:
    data/corpus.txt          – single UTF-8 text file
    data/dataset_stats.json  – metadata
"""

import gc
import json
import logging
import os
import random
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_dataset")

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

SEED = 42
random.seed(SEED)

BASE = Path(__file__).resolve().parent
OUTPUT_DIR = BASE / "data"
TEMP_DIR = OUTPUT_DIR / "_tmp"
BATCH_SIZE = 1000

HEADERS = {
    "User-Agent": "GPT-DatasetBuilder/1.0 (educational; contact@example.com)"
}

# ──────────────────────────────────────────────────────────────
#  Data mix — 5 sources covering web, Wikipedia, code, books, math
# ──────────────────────────────────────────────────────────────


@dataclass
class Source:
    name: str
    proportion: float
    hf_path: Optional[str] = None
    hf_config: Optional[str] = None
    field: Optional[str] = None


DATA_MIX = [
    Source("FineWeb-Edu", 0.33, hf_path="HuggingFaceFW/fineweb-edu", field="text"),
    Source("Wikipedia",   0.27, hf_path="wikimedia/wikipedia", hf_config="20231101.en", field="text"),
    Source("Code",        0.15, hf_path="transformersbook/codeparrot-train", field="content"),
    Source("Gutenberg",   0.15),
    Source("FineMath",    0.10, hf_path="HuggingFaceTB/finemath", hf_config="finemath-3plus", field="text"),
]

# ──────────────────────────────────────────────────────────────
#  Text cleaning
# ──────────────────────────────────────────────────────────────


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


def is_valid_document(text: str) -> bool:
    return len(text) >= 300


# ──────────────────────────────────────────────────────────────
#  JSONL helpers
# ──────────────────────────────────────────────────────────────


def write_jsonl(path: Path, docs: list[str], append: bool = False):
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)["text"]


# ──────────────────────────────────────────────────────────────
#  Gutenberg builder
# ──────────────────────────────────────────────────────────────

GUTENDEX_URL = "https://gutendex.com/books"
GUTENBERG_DL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

GUTENBERG_TOPICS = [
    ("fiction", 80), ("mystery", 40), ("adventure", 40),
    ("science+fiction", 40), ("history", 40), ("philosophy", 30), ("fantasy", 30),
]

GUT_START = re.compile(
    r"\*{3}\s*(START\s+(OF\s+(THIS|THE)\s+)?PROJECT\s+GUTENBERG)", re.IGNORECASE
)
GUT_END = re.compile(
    r"\*{3}\s*(END\s+(OF\s+(THIS|THE)\s+)?PROJECT\s+GUTENBERG)", re.IGNORECASE
)


def strip_gutenberg_header(text: str) -> str:
    s = GUT_START.search(text)
    if s:
        text = text[s.end():]
    e = GUT_END.search(text)
    if e:
        text = text[:e.start()]
    text = re.sub(r"(?i)end of (the |this )?project gutenberg.*", "", text)
    text = re.sub(r"(?i)this file should be named.*?project gutenberg", "", text, flags=re.DOTALL)
    text = re.sub(r"(?i)produced by .*?project gutenberg", "", text, flags=re.DOTALL)
    return text


def fetch_gutenberg_book(book_id: int) -> Optional[str]:
    url = GUTENBERG_DL.format(id=book_id)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        raw = resp.content.decode("utf-8", errors="replace")
        if len(raw) < 2000:
            return None
        text = strip_gutenberg_header(raw)
        text = clean_text(text)
        return text if is_valid_document(text) else None
    except requests.RequestException:
        return None


def fetch_book_ids_by_topic(topic: str, max_books: int) -> list[int]:
    ids: list[int] = []
    url = f"{GUTENDEX_URL}?topic={topic}&languages=en&sort=popular"
    page = 1
    while len(ids) < max_books:
        try:
            resp = requests.get(f"{url}&page={page}", headers=HEADERS, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                break
            for book in results:
                if len(ids) >= max_books:
                    break
                bid = book["id"]
                if bid not in ids:
                    ids.append(bid)
            if data.get("next") is None:
                break
            page += 1
            time.sleep(0.3)
        except requests.RequestException as e:
            log.warning("  Gutendex page %d failed: %s", page, e)
            break
    return ids


def build_gutenberg(target_bytes: int, out_path: Path) -> int:
    all_ids: list[int] = []
    for topic, count in GUTENBERG_TOPICS:
        ids = fetch_book_ids_by_topic(topic, count)
        all_ids.extend(ids)
        time.sleep(0.3)
    random.shuffle(all_ids)
    log.info("  Found %d unique Gutenberg books to try", len(all_ids))

    seen: set[int] = set()
    current_size = 0
    batch: list[str] = []

    def process_book(book_id: int) -> Optional[str]:
        text = fetch_gutenberg_book(book_id)
        if text is None:
            return None
        h = hash(text)
        if h in seen:
            return None
        seen.add(h)
        return text

    pbar = tqdm(total=min(len(all_ids), 500), desc="Gutenberg", unit=" book",
                postfix=f"0/{target_bytes/1e6:.0f} MB")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_book, bid): bid for bid in all_ids}
        done = 0
        for future in as_completed(futures):
            if current_size >= target_bytes:
                break
            done += 1
            text = future.result()
            pbar.update(1)
            if text is None:
                continue
            current_size += len(text.encode("utf-8"))
            batch.append(text)
            if len(batch) >= BATCH_SIZE:
                write_jsonl(out_path, batch, append=out_path.exists())
                batch = []
            pbar.postfix = f"{current_size/1e6:.0f}/{target_bytes/1e6:.0f} MB"
            if done >= 500:
                break

    pbar.close()
    if batch:
        write_jsonl(out_path, batch, append=out_path.exists())

    log.info("  Gutenberg: %d docs, %.1f MB", len(seen), current_size / 1e6)
    return current_size


# ──────────────────────────────────────────────────────────────
#  HF streaming builder
# ──────────────────────────────────────────────────────────────


def build_from_hf(source: Source, target_bytes: int, out_path: Path) -> int:
    log.info("  Streaming from %s%s ...",
             source.hf_path,
             f" ({source.hf_config})" if source.hf_config else "")

    dataset = load_dataset(
        source.hf_path,
        **(dict(name=source.hf_config) if source.hf_config else {}),
        split="train",
        streaming=True,
    )

    seen: set[int] = set()
    current_size = 0
    batch: list[str] = []
    count = 0

    pbar = tqdm(desc=source.name, unit=" doc",
                postfix=f"0/{target_bytes/1e6:.0f} MB")

    for example in dataset:
        if current_size >= target_bytes:
            break
        raw = example.get(source.field or "text") or ""
        if not raw.strip():
            continue
        text = clean_text(raw)
        if not is_valid_document(text):
            continue
        h = hash(text)
        if h in seen:
            continue
        seen.add(h)

        current_size += len(text.encode("utf-8"))
        batch.append(text)
        count += 1

        if len(batch) >= BATCH_SIZE:
            write_jsonl(out_path, batch, append=out_path.exists())
            batch = []
            pbar.postfix = f"{current_size/1e6:.0f}/{target_bytes/1e6:.0f} MB"
        pbar.update(1)

    if batch:
        write_jsonl(out_path, batch, append=out_path.exists())
    pbar.close()

    log.info("  %s: %d docs, %.1f MB", source.name, count, current_size / 1e6)
    del dataset
    gc.collect()
    return current_size


# ──────────────────────────────────────────────────────────────
#  Build orchestration
# ──────────────────────────────────────────────────────────────


def build_source(source: Source, target_bytes: int, cache_path: Path) -> int:
    if cache_path.exists() and cache_path.stat().st_size >= target_bytes * 0.8:
        log.info("  Cached (%.0f MB >= %.0f MB) — reusing",
                 cache_path.stat().st_size / 1e6, target_bytes / 1e6)
        return cache_path.stat().st_size

    if cache_path.exists():
        cache_path.unlink()

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    if source.name == "Gutenberg":
        return build_gutenberg(target_bytes, cache_path)
    return build_from_hf(source, target_bytes, cache_path)


# ──────────────────────────────────────────────────────────────
#  Statistics
# ──────────────────────────────────────────────────────────────


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4.3))


def compute_statistics(source_files: list[Path], final_path: Path) -> dict:
    source_stats = []
    total_chars = 0
    total_bytes = 0
    total_tokens = 0
    total_docs = 0

    for sf in source_files:
        chars = 0
        bcount = 0
        n = 0
        for doc_text in iter_jsonl(sf):
            chars += len(doc_text)
            bcount += len(doc_text.encode("utf-8"))
            n += 1
        tok = max(1, int(chars / 4.3))
        src_name = sf.stem.replace("_tmp_", "")
        source_stats.append({
            "source": src_name,
            "documents": n,
            "characters": chars,
            "bytes": bcount,
            "estimated_tokens": tok,
        })
        total_docs += n
        total_chars += chars
        total_bytes += bcount
        total_tokens += tok

    for s in source_stats:
        s["percentage"] = round(s["bytes"] / total_bytes * 100, 1) if total_bytes else 0

    final_size = final_path.stat().st_size if final_path.exists() else 0

    return {
        "total_documents": total_docs,
        "total_characters": total_chars,
        "total_bytes": total_bytes,
        "estimated_tokens": total_tokens,
        "final_file_size_bytes": final_size,
        "sources": source_stats,
    }


def print_statistics(stats: dict):
    hr = "=" * 64
    print()
    print(hr)
    print("  DATASET STATISTICS")
    print(hr)
    print(f"  {'Source':<22s} {'Docs':>10s} {'Bytes':>12s} {'Pct':>5s}")
    print(f"  {'─' * 22} {'─' * 10} {'─' * 12} {'─' * 5}")
    for s in stats["sources"]:
        pct = s.get("percentage", 0)
        print(f"  {s['source']:<22s} {s['documents']:>10,} {s['bytes']:>12,} {pct:>4.1f}%")
    print(f"  {'─' * 22} {'─' * 10} {'─' * 12} {'─' * 5}")
    print(f"  {'Total':<22s} {stats['total_documents']:>10,} {stats['total_bytes']:>12,} {100.0:>4.1f}%")
    print()
    print(f"  Characters      : {stats['total_characters']:>12,}")
    print(f"  Estimated tokens: {stats['estimated_tokens']:>12,}")
    print(f"  Final file size : {stats['final_file_size_bytes']:>12,} ({stats['final_file_size_bytes']/1e6:.0f} MB)")
    print(hr)


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────


def main():
    try:
        raw = input("Enter target dataset size in GB (0.1 - 100): ").strip()
        target_gb = float(raw)
        if target_gb < 0.1 or target_gb > 100:
            log.error("Target must be between 0.1 and 100 GB")
            sys.exit(1)
    except (ValueError, EOFError):
        log.error("Invalid input")
        sys.exit(1)

    target_bytes = int(target_gb * 1_000_000_000)
    log.info("Building dataset: %.2f GB target", target_gb)
    log.info("─" * 52)
    for src in DATA_MIX:
        mb = target_bytes * src.proportion / 1e6
        log.info("  %-16s %6.0f MB (%5.1f%%)", src.name, mb, src.proportion * 100)
    log.info("─" * 52)

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    source_files: list[Path] = []
    total_collected = 0

    for source in DATA_MIX:
        src_target = int(target_bytes * source.proportion)
        name_key = source.name.lower().replace(" ", "_")
        cache_path = TEMP_DIR / f"{name_key}.jsonl"

        log.info("")
        log.info("Building %s  (target ~%d MB)", source.name, src_target // 1_000_000)
        try:
            collected = build_source(source, src_target, cache_path)
            if collected > 0 and cache_path.exists() and cache_path.stat().st_size > 0:
                source_files.append(cache_path)
                total_collected += collected
            else:
                log.warning("  ✗ %s produced no output", source.name)
        except Exception as e:
            log.error("  ✗ %s failed: %s", source.name, e)

    if len(source_files) < 2:
        log.error("At least 2 sources required; got %d. Aborting.", len(source_files))
        sys.exit(1)

    pct = total_collected / target_bytes * 100
    log.info("")
    log.info("─" * 52)
    log.info("Collected %.2f GB / %.2f GB target (%.0f%%)",
             total_collected / 1e9, target_gb, pct)
    if total_collected < target_bytes * 0.3:
        log.warning("Low yield (%.0f%% of target). Proceeding anyway.", pct)

    final_path = OUTPUT_DIR / "corpus.txt"
    log.info("")
    log.info("Concatenating sources into %s ...", final_path)
    written = 0
    with open(final_path, "w", encoding="utf-8") as outf:
        for sf in source_files:
            pbar = tqdm(desc=sf.stem, unit=" doc", leave=False)
            for doc_text in iter_jsonl(sf):
                outf.write(doc_text + "\n\n\n")
                written += 1
                pbar.update(1)
            pbar.close()

    log.info("Wrote %d documents to %s (%.0f MB)",
             written, final_path, final_path.stat().st_size / 1e6)

    log.info("Computing statistics ...")
    stats = compute_statistics(source_files, final_path)
    stats_path = OUTPUT_DIR / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print_statistics(stats)

    log.info("Cleaning up temp files ...")
    for sf in TEMP_DIR.glob("*.jsonl"):
        sf.unlink()
    try:
        TEMP_DIR.rmdir()
    except OSError:
        pass

    log.info("Done! Dataset: %s (%.0f MB)", final_path, final_path.stat().st_size / 1e6)
    log.info("Stats: %s", stats_path)


if __name__ == "__main__":
    main()
