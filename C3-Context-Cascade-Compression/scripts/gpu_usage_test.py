#!/usr/bin/env python3
"""
Test GPU usage during Ollama inference.
Runs 5 samples and prints GPU memory usage for each.
"""

import json
import subprocess
import threading
import time
import requests
import tiktoken
from pathlib import Path

# Config
INPUT_PATH = Path("/home/cloud/c3/C3-Context-Cascade-Compression/dataset_v1/arxiv_ai_papers_v1.json")
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"
MAX_TOKENS = 28000
NUM_SAMPLES = 5

enc = tiktoken.get_encoding("cl100k_base")


def get_gpu_usage():
    """Get current GPU memory usage in MB using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            gpus = []
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    gpus.append({
                        "used_mb": int(parts[0]),
                        "total_mb": int(parts[1]),
                        "util_pct": int(parts[2])
                    })
            return gpus
    except Exception as e:
        print(f"Error getting GPU usage: {e}")
    return []


class GPUMonitor:
    """Monitor GPU usage in a background thread."""

    def __init__(self, interval=0.1):
        self.interval = interval
        self.running = False
        self.max_used_mb = 0
        self.max_util_pct = 0
        self.samples = []
        self._thread = None

    def start(self):
        self.running = True
        self.max_used_mb = 0
        self.max_util_pct = 0
        self.samples = []
        self._thread = threading.Thread(target=self._monitor)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join()

    def _monitor(self):
        while self.running:
            gpus = get_gpu_usage()
            if gpus:
                # Sum across all GPUs
                total_used = sum(g["used_mb"] for g in gpus)
                total_util = max(g["util_pct"] for g in gpus)
                self.samples.append(total_used)
                self.max_used_mb = max(self.max_used_mb, total_used)
                self.max_util_pct = max(self.max_util_pct, total_util)
            time.sleep(self.interval)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


def get_summary_with_monitoring(paper_text: str) -> tuple[str, dict]:
    """Get summary from Ollama while monitoring GPU usage."""
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

    monitor = GPUMonitor(interval=0.05)  # Sample every 50ms
    monitor.start()

    start_time = time.time()
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()["response"].strip()
    except Exception as e:
        print(f"Error: {e}")
        result = ""
    finally:
        elapsed = time.time() - start_time
        monitor.stop()

    stats = {
        "max_gpu_mb": monitor.max_used_mb,
        "max_util_pct": monitor.max_util_pct,
        "avg_gpu_mb": sum(monitor.samples) / len(monitor.samples) if monitor.samples else 0,
        "elapsed_sec": elapsed,
        "num_samples": len(monitor.samples)
    }

    return result, stats


def main():
    print("=" * 60)
    print("GPU Usage Test - Ollama Inference")
    print(f"Model: {MODEL}")
    print(f"Context window: 32768")
    print(f"Max input tokens: {MAX_TOKENS}")
    print(f"Samples: {NUM_SAMPLES}")
    print("=" * 60)

    # Get baseline GPU usage
    baseline = get_gpu_usage()
    if baseline:
        print(f"\nBaseline GPU usage: {baseline[0]['used_mb']} MB / {baseline[0]['total_mb']} MB")
    else:
        print("\nCould not get baseline GPU usage")

    # Load papers
    print(f"\nLoading papers from {INPUT_PATH}")
    with open(INPUT_PATH, "r") as f:
        papers = json.load(f)

    print(f"Found {len(papers)} papers, using first {NUM_SAMPLES}")

    all_stats = []
    overall_max_mb = 0

    for i in range(min(NUM_SAMPLES, len(papers))):
        paper = papers[i]

        # Get context
        context = ""
        for conv in paper.get("conversations", []):
            if conv.get("from") == "context":
                context = conv.get("value", "")
                break

        if not context:
            print(f"\nSample {i+1}: No context found, skipping")
            continue

        truncated = truncate_to_tokens(context, MAX_TOKENS)
        token_count = len(enc.encode(truncated))

        print(f"\n{'='*60}")
        print(f"Sample {i+1}/{NUM_SAMPLES}")
        print(f"Input tokens: {token_count}")
        print("-" * 40)

        summary, stats = get_summary_with_monitoring(truncated)
        all_stats.append(stats)
        overall_max_mb = max(overall_max_mb, stats["max_gpu_mb"])

        print(f"Max GPU memory:  {stats['max_gpu_mb']} MB")
        print(f"Avg GPU memory:  {stats['avg_gpu_mb']:.0f} MB")
        print(f"Max GPU util:    {stats['max_util_pct']}%")
        print(f"Time:            {stats['elapsed_sec']:.1f}s")
        print(f"Summary length:  {len(summary)} chars")

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if all_stats:
        avg_max_mb = sum(s["max_gpu_mb"] for s in all_stats) / len(all_stats)
        avg_time = sum(s["elapsed_sec"] for s in all_stats) / len(all_stats)

        print(f"Samples run:        {len(all_stats)}")
        print(f"Overall max GPU:    {overall_max_mb} MB")
        print(f"Average max GPU:    {avg_max_mb:.0f} MB")
        print(f"Average time:       {avg_time:.1f}s")

        if baseline:
            delta = overall_max_mb - baseline[0]["used_mb"]
            print(f"Delta from baseline: +{delta} MB")
    else:
        print("No samples completed")


if __name__ == "__main__":
    main()
