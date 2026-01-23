#!/usr/bin/env python3
"""
Generate paper summaries using Ollama's qwen2.5:3b model.
Takes first 28k tokens from each paper and generates a summary.
"""

import json
import asyncio
import aiohttp
import tiktoken
import time
from pathlib import Path
from typing import Optional

# Config
DATASET_DIR = Path("/home/cloud/c3/C3-Context-Cascade-Compression/dataset")
DATASETS = [
    ("arxiv_ai_papers_val.json", "arxiv_summaries_val.json", "VAL"),
    ("arxiv_ai_papers.json", "arxiv_summaries.json", "TRAIN"),
]
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"
MAX_TOKENS = 28000
MAX_CONCURRENT = 4  # Ollama default OLLAMA_NUM_PARALLEL is 1, increase if you set it higher
SAVE_EVERY = 10

enc = tiktoken.get_encoding("cl100k_base")


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to max_tokens."""
    tokens = enc.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


async def get_summary(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    paper_text: str,
    idx: int,
    total: int,
    results: dict,
    counter: dict,
    start_time: float,
    name: str,
) -> None:
    """Get summary with semaphore limiting concurrency."""
    print(f"[{name}] {idx}: waiting for semaphore...", flush=True)
    async with semaphore:
        print(f"[{name}] {idx}: got semaphore, sending request...", flush=True)
        prompt = f"""Summarize the paper:
{paper_text}

Summary:"""

        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 32768,
                "temperature": 0.3,
            }
        }

        try:
            print(f"[{name}] {idx}: posting to {OLLAMA_URL}...", flush=True)
            async with session.post(OLLAMA_URL, json=payload) as resp:
                print(f"[{name}] {idx}: got response status {resp.status}", flush=True)
                if resp.status == 200:
                    print(f"[{name}] {idx}: reading json...", flush=True)
                    data = await resp.json()
                    print(f"[{name}] {idx}: got json, extracting response...", flush=True)
                    summary = data.get("response", "").strip()
                    if summary:
                        results[idx] = (paper_text, summary)
                        counter["done"] += 1
                        elapsed = time.time() - start_time
                        rate = counter["done"] / elapsed if elapsed > 0 else 0
                        print(f"[{name}] {counter['done']}/{total} done ({rate:.2f}/sec) - {len(summary)} chars", flush=True)
                    else:
                        print(f"[{name}] {idx}: empty response, keys: {list(data.keys())}", flush=True)
                else:
                    text = await resp.text()
                    print(f"[{name}] {idx}: HTTP {resp.status}: {text[:200]}", flush=True)
        except Exception as e:
            import traceback
            print(f"[{name}] {idx}: Error {e}", flush=True)
            traceback.print_exc()


def extract_context(paper: dict) -> str:
    """Extract context from paper dict."""
    for conv in paper.get("conversations", []):
        if conv.get("from") == "context":
            return conv.get("value", "")
    return ""


def create_result(context: str, summary: str) -> dict:
    """Create output in training format."""
    return {
        "image": "",
        "conversations": [
            {"from": "context", "value": context},
            {"from": "human", "value": "<image>\nSummarize the paper: "},
            {"from": "gpt", "value": summary}
        ]
    }


async def process_dataset(
    input_path: Path,
    output_path: Path,
    name: str,
    session: aiohttp.ClientSession
) -> tuple[int, int]:
    """Process a single dataset."""
    print(f"\n{'='*60}", flush=True)
    print(f"[{name}] Processing {input_path.name}", flush=True)
    print(f"{'='*60}", flush=True)

    if not input_path.exists():
        print(f"[{name}] Input file not found, skipping")
        return 0, 0

    with open(input_path, "r") as f:
        papers = json.load(f)
    print(f"[{name}] Found {len(papers)} papers", flush=True)

    # Prepare all contexts
    work_items = []
    for i, paper in enumerate(papers):
        context = extract_context(paper)
        if context:
            truncated = truncate_to_tokens(context, MAX_TOKENS)
            work_items.append((i, truncated))

    print(f"[{name}] Prepared {len(work_items)} papers, max {MAX_CONCURRENT} concurrent", flush=True)
    print(f"[{name}] Creating tasks...", flush=True)

    results = {}
    counter = {"done": 0}
    start_time = time.time()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Create all tasks
    tasks = [
        get_summary(session, semaphore, text, idx, len(work_items), results, counter, start_time, name)
        for idx, text in work_items
    ]

    # Run with periodic saves
    save_task = asyncio.create_task(periodic_save(results, output_path, name))
    await asyncio.gather(*tasks)
    save_task.cancel()

    # Final save
    final_results = [create_result(ctx, summ) for idx, (ctx, summ) in sorted(results.items())]
    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2)

    elapsed = time.time() - start_time
    print(f"[{name}] Done: {len(results)}/{len(work_items)} in {elapsed:.1f}s")

    return len(results), len(work_items)


async def periodic_save(results: dict, output_path: Path, name: str):
    """Save results periodically."""
    while True:
        await asyncio.sleep(30)
        if results:
            final_results = [create_result(ctx, summ) for idx, (ctx, summ) in sorted(results.items())]
            with open(output_path, "w") as f:
                json.dump(final_results, f, indent=2)
            print(f"[{name}] Saved {len(results)} results")


async def main():
    print("=" * 60)
    print("Generate Summaries")
    print(f"Model: {MODEL}, Max concurrent: {MAX_CONCURRENT}")
    print("=" * 60)

    total_start = time.time()
    total_success = 0
    total_attempted = 0

    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for input_file, output_file, name in DATASETS:
            input_path = DATASET_DIR / input_file
            output_path = DATASET_DIR / output_file
            success, attempted = await process_dataset(input_path, output_path, name, session)
            total_success += success
            total_attempted += attempted

    total_time = time.time() - total_start
    print(f"\nDone! {total_success}/{total_attempted} in {total_time:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
