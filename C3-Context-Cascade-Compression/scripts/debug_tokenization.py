#!/usr/bin/env python3
"""Debug tokenization to find what's causing the image token mismatch error."""

import json
import torch
from transformers import AutoTokenizer

# Token IDs for Qwen2 C3
IM_START_TOKEN = 151857  # <img>
IM_END_TOKEN = 151858    # </img>
IM_PATCH_TOKEN = 151859  # <imgpad>

# Special tokens
DEFAULT_IMAGE_TOKEN = '<image>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'

# Config - match your training
LATENT_TOKEN_LEN = 160  # What you're training with
MODEL_PATH = "liufanfanlff/C3-Context-Cascade-Compression"

# Dataset paths
DATASETS = [
    "/home/cloud/c3/C3-Context-Cascade-Compression/dataset/arxiv_ai_papers.json",
    "/home/cloud/c3/C3-Context-Cascade-Compression/dataset/arxiv_summaries.json",
]


def check_special_tokens(tokenizer):
    """Verify special tokens are in vocabulary"""
    print("=" * 60)
    print("CHECKING SPECIAL TOKENS")
    print("=" * 60)

    tokens_to_check = [
        (DEFAULT_IM_START_TOKEN, IM_START_TOKEN),
        (DEFAULT_IM_END_TOKEN, IM_END_TOKEN),
        (DEFAULT_IMAGE_PATCH_TOKEN, IM_PATCH_TOKEN),
    ]

    for token_str, expected_id in tokens_to_check:
        # Check if token exists
        token_ids = tokenizer.encode(token_str, add_special_tokens=False)
        actual_id = token_ids[0] if len(token_ids) == 1 else None

        print(f"  '{token_str}':")
        print(f"    Expected ID: {expected_id}")
        print(f"    Tokenized as: {token_ids}")
        print(f"    Single token: {len(token_ids) == 1}")
        if actual_id != expected_id:
            print(f"    WARNING: ID mismatch!")
        print()

    # Test the full replacement pattern
    test_pattern = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * 5 + DEFAULT_IM_END_TOKEN
    test_ids = tokenizer.encode(test_pattern, add_special_tokens=False)
    print(f"  Test pattern '{DEFAULT_IM_START_TOKEN}[5x imgpad]{DEFAULT_IM_END_TOKEN}':")
    print(f"    Expected: [151857, 151859, 151859, 151859, 151859, 151859, 151858]")
    print(f"    Got:      {test_ids}")
    print(f"    Length: {len(test_ids)} (expected 7)")
    print()


def process_sample(sample, tokenizer, latent_len, sample_idx, dataset_name):
    """Process a single sample and check for issues"""
    conversations = sample.get("conversations", [])

    context_text = None
    human_text = None

    for conv in conversations:
        if conv["from"] == "context":
            context_text = conv["value"]
        elif conv["from"] == "human":
            human_text = conv["value"]

    issues = []

    # Check context
    if context_text:
        # Add image tokens to end (like processor_context does)
        replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * latent_len + DEFAULT_IM_END_TOKEN
        context_with_tokens = context_text + replace_token

        context_ids = tokenizer.encode(context_with_tokens, add_special_tokens=False)
        context_ids = torch.tensor(context_ids)

        # Find image start token
        start_positions = (context_ids == IM_START_TOKEN).nonzero(as_tuple=True)[0]
        end_positions = (context_ids == IM_END_TOKEN).nonzero(as_tuple=True)[0]
        patch_count = (context_ids == IM_PATCH_TOKEN).sum().item()

        if len(start_positions) == 0:
            issues.append(f"CONTEXT: No <img> token found!")
        elif len(end_positions) == 0:
            issues.append(f"CONTEXT: No </img> token found!")
        else:
            start_pos = start_positions[0].item()
            end_pos = end_positions[0].item()

            # Check: end should be at start + latent_len + 1
            expected_end_pos = start_pos + latent_len + 1
            if end_pos != expected_end_pos:
                issues.append(f"CONTEXT: </img> at wrong position! Expected {expected_end_pos}, got {end_pos}")
                issues.append(f"  start_pos={start_pos}, end_pos={end_pos}, diff={end_pos - start_pos - 1}")
                issues.append(f"  patch_count={patch_count}, expected={latent_len}")

            # Check patch count
            if patch_count != latent_len:
                issues.append(f"CONTEXT: Wrong patch count! Expected {latent_len}, got {patch_count}")

    # Check human/input
    if human_text and DEFAULT_IMAGE_TOKEN in human_text:
        replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * latent_len + DEFAULT_IM_END_TOKEN
        human_with_tokens = human_text.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        input_ids = tokenizer.encode(human_with_tokens, add_special_tokens=False)
        input_ids = torch.tensor(input_ids)

        start_positions = (input_ids == IM_START_TOKEN).nonzero(as_tuple=True)[0]
        end_positions = (input_ids == IM_END_TOKEN).nonzero(as_tuple=True)[0]
        patch_count = (input_ids == IM_PATCH_TOKEN).sum().item()

        if len(start_positions) != len(end_positions):
            issues.append(f"INPUT: Mismatched <img>({len(start_positions)}) and </img>({len(end_positions)}) counts!")

        if len(start_positions) > 0 and len(end_positions) > 0:
            start_pos = start_positions[0].item()
            end_pos = end_positions[0].item()
            expected_end_pos = start_pos + latent_len + 1

            if end_pos != expected_end_pos:
                issues.append(f"INPUT: </img> at wrong position! Expected {expected_end_pos}, got {end_pos}")
                issues.append(f"  patch_count={patch_count}, expected={latent_len}")

    return issues


def check_dataset(path, tokenizer, latent_len, max_samples=None):
    """Check all samples in a dataset"""
    print(f"\n{'=' * 60}")
    print(f"CHECKING: {path}")
    print(f"{'=' * 60}")

    with open(path, 'r') as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    print(f"Loaded {len(data)} samples")

    bad_samples = []
    for i, sample in enumerate(data):
        issues = process_sample(sample, tokenizer, latent_len, i, path)
        if issues:
            bad_samples.append((i, issues))
            if len(bad_samples) <= 10:  # Print first 10
                print(f"\n[SAMPLE {i}] ISSUES:")
                for issue in issues:
                    print(f"  {issue}")

    print(f"\n{'-' * 40}")
    print(f"Total samples: {len(data)}")
    print(f"Bad samples: {len(bad_samples)}")
    if bad_samples:
        print(f"First bad indices: {[b[0] for b in bad_samples[:20]]}")

    return bad_samples


def main():
    print("=" * 60)
    print("C3 TOKENIZATION DEBUG")
    print(f"Model: {MODEL_PATH}")
    print(f"Latent token len: {LATENT_TOKEN_LEN}")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Check special tokens
    check_special_tokens(tokenizer)

    # Check each dataset
    all_bad = {}
    for path in DATASETS:
        try:
            bad = check_dataset(path, tokenizer, LATENT_TOKEN_LEN)
            all_bad[path] = bad
        except Exception as e:
            print(f"ERROR loading {path}: {e}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    for path, bad in all_bad.items():
        name = path.split('/')[-1]
        print(f"  {name}: {len(bad)} bad samples")


if __name__ == "__main__":
    main()
