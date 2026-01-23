"""
Build training dataset from AI arxiv papers for context compression training.
Downloads papers, extracts text (~10 pages per sample), creates JSON dataset.
Also creates a validation set of recent papers (last 2 months).

Optimized for speed with parallel downloads and exponential backoff.
"""

import os
import json
import time
import arxiv
import fitz  # PyMuPDF
import tempfile
import random
import threading
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, List, Tuple, Set

# Config
NUM_TRAIN_SAMPLES = 5000
NUM_VAL_SAMPLES = 50
PAGES_PER_SAMPLE = 10
TRAIN_OUTPUT_PATH = "/home/cloud/c3/C3-Context-Cascade-Compression/dataset/arxiv_ai_papers.json"
VAL_OUTPUT_PATH = "/home/cloud/c3/C3-Context-Cascade-Compression/dataset/arxiv_ai_papers_val.json"

# Parallel processing config
MAX_WORKERS = 10  # Number of parallel download threads
MAX_RETRIES = 3   # Retries per paper
BASE_DELAY = 0.5  # Base delay between requests (seconds)
MAX_DELAY = 30    # Max backoff delay (seconds)

# AI/ML arxiv categories
AI_CATEGORIES = [
    "cs.LG",   # Machine Learning
    "cs.CL",   # Computation and Language (NLP)
    "cs.CV",   # Computer Vision
    "cs.AI",   # Artificial Intelligence
    "stat.ML", # Machine Learning (stats)
]

# Thread-safe rate limiter
class RateLimiter:
    """Token bucket rate limiter with exponential backoff"""
    def __init__(self, requests_per_second: float = 3.0):
        self.min_interval = 1.0 / requests_per_second
        self.lock = threading.Lock()
        self.last_request = 0.0
        self.backoff_until = 0.0
        self.consecutive_errors = 0

    def wait(self):
        """Wait for rate limit, respecting backoff"""
        with self.lock:
            now = time.time()

            # Check backoff
            if now < self.backoff_until:
                sleep_time = self.backoff_until - now
                time.sleep(sleep_time)
                now = time.time()

            # Check rate limit
            elapsed = now - self.last_request
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)

            self.last_request = time.time()

    def report_success(self):
        """Reset backoff on success"""
        with self.lock:
            self.consecutive_errors = 0

    def report_error(self):
        """Increase backoff on error"""
        with self.lock:
            self.consecutive_errors += 1
            # Exponential backoff with jitter
            delay = min(BASE_DELAY * (2 ** self.consecutive_errors), MAX_DELAY)
            jitter = random.uniform(0, delay * 0.1)
            self.backoff_until = time.time() + delay + jitter
            return delay + jitter

# Global rate limiter
rate_limiter = RateLimiter(requests_per_second=3.0)


def search_arxiv_papers(num_papers=150, recent_only=False, days_back=60):
    """Search for AI papers on arxiv

    Args:
        num_papers: Number of papers to fetch
        recent_only: If True, only fetch papers from last `days_back` days
        days_back: Number of days to look back for recent papers
    """
    print(f"Searching arxiv for AI papers...")

    # Build query for AI categories
    cat_query = " OR ".join([f"cat:{cat}" for cat in AI_CATEGORIES])

    # Add date filter for recent papers
    if recent_only:
        cutoff_date = datetime.now() - timedelta(days=days_back)
        date_str = cutoff_date.strftime("%Y%m%d")
        query = f"({cat_query}) AND submittedDate:[{date_str} TO 99991231]"
        print(f"  Filtering for papers after {cutoff_date.strftime('%Y-%m-%d')}")
    else:
        query = cat_query

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=num_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers = list(client.results(search))
    print(f"Found {len(papers)} papers")
    return papers


def download_and_extract_pdf(paper, max_pages=PAGES_PER_SAMPLE) -> Optional[str]:
    """Download paper PDF and extract text from first N pages with retries"""
    tmp_path = None

    for attempt in range(MAX_RETRIES):
        try:
            # Wait for rate limit
            rate_limiter.wait()

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = tmp.name

            # Download PDF
            paper.download_pdf(filename=tmp_path)

            # Extract text
            doc = fitz.open(tmp_path)
            text_parts = []

            pages_to_read = min(max_pages, len(doc))
            for page_num in range(pages_to_read):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    text_parts.append(text)

            doc.close()

            # Cleanup
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

            rate_limiter.report_success()
            return "\n\n".join(text_parts)

        except Exception as e:
            backoff = rate_limiter.report_error()
            error_msg = str(e).lower()

            # Check if rate limited
            is_rate_limit = "429" in error_msg or "rate" in error_msg or "too many" in error_msg

            if attempt < MAX_RETRIES - 1:
                if is_rate_limit:
                    print(f"  Rate limited, backing off {backoff:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(backoff)
            else:
                print(f"  Error processing {paper.title[:50]}...: {e}")

            # Cleanup on error
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            tmp_path = None

    return None


def clean_text(text):
    """Clean extracted text"""
    if not text:
        return None

    # Remove excessive whitespace
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line:
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Skip if too short (probably extraction failed)
    if len(text) < 5000:
        return None

    # Truncate if too long (keep ~10 pages worth, roughly 30k chars)
    if len(text) > 35000:
        text = text[:35000]
        # Try to end at a sentence
        last_period = text.rfind(".")
        if last_period > 30000:
            text = text[:last_period + 1]

    return text


def create_sample(text):
    """Create a training sample in C3 format"""
    return {
        "image": "",
        "conversations": [
            {
                "from": "context",
                "value": text
            },
            {
                "from": "human",
                "value": "<image>\nRepeat the text: "
            },
            {
                "from": "gpt",
                "value": text
            }
        ]
    }


def process_single_paper(args: Tuple[int, any, str]) -> Tuple[Optional[dict], str, int]:
    """Process a single paper - designed for parallel execution"""
    idx, paper, dataset_name = args
    try:
        text = download_and_extract_pdf(paper)
        text = clean_text(text)

        if text:
            sample = create_sample(text)
            return (sample, paper.entry_id, idx)
        return (None, paper.entry_id, idx)
    except Exception as e:
        print(f"  [{dataset_name}] Error on paper {idx}: {e}")
        return (None, paper.entry_id, idx)


def process_papers(papers, num_samples, dataset_name):
    """Process papers in parallel and return samples"""
    samples = []
    paper_ids = set()
    lock = threading.Lock()
    completed = [0]  # Use list for mutability in closure
    success = [0]

    print(f"[{dataset_name}] Processing {len(papers)} papers with {MAX_WORKERS} workers...")
    start_time = time.time()

    # Prepare work items
    work_items = [(i, paper, dataset_name) for i, paper in enumerate(papers)]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_paper, item): item for item in work_items}

        for future in as_completed(futures):
            # Check if we have enough samples
            with lock:
                if len(samples) >= num_samples:
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break

            try:
                sample, paper_id, idx = future.result()

                with lock:
                    completed[0] += 1
                    paper_ids.add(paper_id)

                    if sample and len(samples) < num_samples:
                        samples.append(sample)
                        success[0] += 1
                        text_len = len(sample["conversations"][0]["value"])
                        elapsed = time.time() - start_time
                        rate = success[0] / elapsed if elapsed > 0 else 0
                        print(f"[{dataset_name}] {success[0]}/{num_samples} samples "
                              f"({completed[0]} processed, {rate:.1f}/sec, {text_len} chars)")

            except Exception as e:
                with lock:
                    completed[0] += 1
                print(f"  [{dataset_name}] Future error: {e}")

    elapsed = time.time() - start_time
    print(f"[{dataset_name}] Completed: {len(samples)} samples from {completed[0]} papers in {elapsed:.1f}s")
    print(f"[{dataset_name}] Rate: {len(samples)/elapsed:.2f} samples/sec")

    return samples, paper_ids


def save_dataset(samples, output_path, name):
    """Save dataset and print stats"""
    print(f"\nSaving {len(samples)} {name} samples to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    total_chars = sum(len(s["conversations"][0]["value"]) for s in samples)
    avg_chars = total_chars / len(samples) if samples else 0

    print(f"  {name} samples: {len(samples)}")
    print(f"  Total chars: {total_chars:,}")
    print(f"  Avg chars/sample: {avg_chars:,.0f}")


def main():
    print("=" * 60)
    print("Building Arxiv AI Papers Dataset (Parallelized)")
    print(f"Training set: {NUM_TRAIN_SAMPLES} samples")
    print(f"Validation set: {NUM_VAL_SAMPLES} samples (recent papers, last 2 months)")
    print(f"Pages per sample: ~{PAGES_PER_SAMPLE}")
    print(f"Parallel workers: {MAX_WORKERS}")
    print(f"Max retries: {MAX_RETRIES}")
    print("=" * 60)

    total_start = time.time()

    # ===== VALIDATION SET (recent papers from last 2 months) =====
    print("\n" + "=" * 60)
    print("STEP 1: Building VALIDATION set (recent papers)")
    print("=" * 60)

    val_papers = search_arxiv_papers(num_papers=NUM_VAL_SAMPLES + 30, recent_only=True, days_back=60)
    val_samples, val_paper_ids = process_papers(val_papers, NUM_VAL_SAMPLES, "VAL")
    save_dataset(val_samples, VAL_OUTPUT_PATH, "Validation")

    # ===== TRAINING SET (older papers, excluding validation papers) =====
    print("\n" + "=" * 60)
    print("STEP 2: Building TRAINING set (excluding validation papers)")
    print("=" * 60)

    # Get more papers for training (older ones)
    train_papers = search_arxiv_papers(num_papers=NUM_TRAIN_SAMPLES + 500, recent_only=False)

    # Filter out papers that are in validation set
    train_papers = [p for p in train_papers if p.entry_id not in val_paper_ids]
    print(f"Filtered to {len(train_papers)} papers (excluded {len(val_paper_ids)} validation papers)")

    random.shuffle(train_papers)
    train_samples, _ = process_papers(train_papers, NUM_TRAIN_SAMPLES, "TRAIN")
    save_dataset(train_samples, TRAIN_OUTPUT_PATH, "Training")

    # ===== FINAL SUMMARY =====
    total_elapsed = time.time() - total_start
    total_samples = len(train_samples) + len(val_samples)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"  Training set:   {len(train_samples)} samples -> {TRAIN_OUTPUT_PATH}")
    print(f"  Validation set: {len(val_samples)} samples -> {VAL_OUTPUT_PATH}")
    print(f"  Total time:     {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Overall rate:   {total_samples/total_elapsed:.2f} samples/sec")
    print("=" * 60)


if __name__ == "__main__":
    main()
