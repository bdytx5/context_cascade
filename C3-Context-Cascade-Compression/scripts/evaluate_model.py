#!/usr/bin/env python3
"""
Evaluate trained C3 model on validation set using weave evaluation framework.
Supports multiple scorers: ROUGE, BERTScore, Claude, compression, coverage.
"""

import json
import argparse
import torch
from time import sleep
from typing import Dict
from transformers import AutoModel, AutoTokenizer
from litellm import completion

import weave
from weave import EvaluationLogger

# Optional imports - graceful fallback
try:
    import bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    print("Warning: bert_score not installed, skipping BERTScore")

try:
    from rouge_score.rouge_scorer import RougeScorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge_score not installed, skipping ROUGE")

# ========== SCORERS ==========

def bert_scorer(gt_text: str, model_output: str) -> Dict[str, float]:
    """Calculate BERTScore."""
    if not BERT_SCORE_AVAILABLE or not model_output:
        return {'bert_score': 0.0}

    try:
        P, R, F1 = bert_score.score(
            [model_output],
            [gt_text],
            lang='en',
            model_type='microsoft/deberta-xlarge-mnli'
        )
        return {'bert_score': float(F1.mean())}
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        return {'bert_score': 0.0}


def llm_scorer(gt_text: str, model_output: str) -> Dict[str, float]:
    """Evaluate using LLM via LiteLLM."""
    if not model_output:
        return {'llm_score': 0.0}

    prompt = f'''Rate how well this generated text captures the key information from the ground truth on a scale from 1-5:
1: Poor - Missing most key information or seriously misrepresenting the content
2: Fair - Captures some information but misses crucial elements
3: Good - Captures most key points but has some gaps or inaccuracies
4: Very Good - Accurately captures nearly all key information with minor omissions
5: Excellent - Perfectly captures all key information and maintains accuracy

Ground Truth:
{gt_text}

Generated:
{model_output}

Provide your rating as a JSON object: {{"score": <integer 1-5>}}'''

    try:
        response = completion(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt}],
        )
        result = response["choices"][0]["message"]["content"].strip()
        score = json.loads(result)["score"]
        sleep(1)  # Rate limiting
        return {'llm_score': float(score)}
    except Exception as e:
        print(f"Error in LLM evaluation: {e}")
        return {'llm_score': 0.0}


def rouge_scorer_fn(gt_text: str, model_output: str) -> Dict[str, float]:
    """Calculate ROUGE scores."""
    if not ROUGE_AVAILABLE or not model_output:
        return {'rouge1_f': 0.0, 'rouge2_f': 0.0, 'rougeL_f': 0.0}

    try:
        scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(gt_text, model_output)
        return {
            'rouge1_f': float(scores['rouge1'].fmeasure),
            'rouge2_f': float(scores['rouge2'].fmeasure),
            'rougeL_f': float(scores['rougeL'].fmeasure)
        }
    except Exception as e:
        print(f"Error calculating ROUGE: {e}")
        return {'rouge1_f': 0.0, 'rouge2_f': 0.0, 'rougeL_f': 0.0}


def compression_scorer(gt_text: str, model_output: str) -> Dict[str, float]:
    """Calculate compression/length ratio."""
    if not model_output:
        return {'compression_ratio': 0.0, 'length_ratio': 0.0}

    gt_words = len(gt_text.split())
    gen_words = len(model_output.split())

    compression_ratio = min(gt_words, gen_words) / max(gt_words, gen_words) if max(gt_words, gen_words) > 0 else 0
    length_ratio = gen_words / gt_words if gt_words > 0 else 0

    return {
        'compression_ratio': float(compression_ratio),
        'length_ratio': float(length_ratio)
    }


def coverage_scorer(gt_text: str, model_output: str) -> Dict[str, float]:
    """Calculate word overlap (Jaccard similarity)."""
    if not model_output:
        return {'coverage_score': 0.0}

    gt_words = set(gt_text.lower().split())
    gen_words = set(model_output.lower().split())

    intersection = len(gt_words.intersection(gen_words))
    union = len(gt_words.union(gen_words))

    return {'coverage_score': float(intersection / union) if union > 0 else 0.0}


# ========== MODEL INFERENCE ==========

class C3Model:
    """Wrapper for C3 model inference."""

    def __init__(self, model_path: str):
        print(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).eval().cuda()
        print(f"Model loaded. Latent tokens: {self.model.config.latent_token_len}")

    def generate(self, context: str, prompt: str) -> str:
        """Generate output from context and prompt."""
        try:
            output = self.model.chat(self.tokenizer, context, prompt)
            return output
        except Exception as e:
            print(f"Error during generation: {e}")
            return ""


# ========== DATASET LOADING ==========

def load_val_dataset(path: str, max_samples: int = None):
    """Load validation dataset."""
    with open(path, 'r') as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    samples = []
    for item in data:
        context = ""
        prompt = ""
        gt_output = ""

        for conv in item.get("conversations", []):
            if conv["from"] == "context":
                context = conv["value"]
            elif conv["from"] == "human":
                # Extract just the prompt part (after <image>\n)
                prompt = conv["value"].replace("<image>\n", "").strip()
            elif conv["from"] == "gpt":
                gt_output = conv["value"]

        if context and gt_output:
            samples.append({
                "context": context,
                "prompt": prompt or "Summarize the paper: ",
                "ground_truth": gt_output
            })

    return samples


# ========== MAIN EVALUATION ==========

def main():
    parser = argparse.ArgumentParser(description="Evaluate C3 model on validation set")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--val_path", type=str, default=None,
                        help="Path to validation dataset (auto-selected based on task if not specified)")
    parser.add_argument("--task", type=str, default="summarize", choices=["summarize", "repeat"],
                        help="Task to evaluate: 'summarize' or 'repeat'")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to evaluate (default: all)")
    parser.add_argument("--max_context_chars", type=int, default=50000,
                        help="Max context characters to use")
    parser.add_argument("--pct", type=int, default=100,
                        help="Percentage of original context to use (1-100)")
    parser.add_argument("--weave_project", type=str, default="c3-evaluation",
                        help="Weave project name")
    parser.add_argument("--use_llm", action="store_true", default=True,
                        help="Use LLM scorer via LiteLLM (gpt-5)")
    parser.add_argument("--use_bert", action="store_true", default=True,
                        help="Use BERTScore (slow)")
    args = parser.parse_args()

    # Auto-select validation path based on task if not specified
    if args.val_path is None:
        if args.task == "repeat":
            args.val_path = "/home/cloud/c3/C3-Context-Cascade-Compression/dataset/arxiv_ai_papers_val.json"
        else:
            args.val_path = "/home/cloud/c3/C3-Context-Cascade-Compression/dataset/arxiv_summaries_val.json"

    # Initialize weave
    weave.init(args.weave_project)

    # Load model
    model = C3Model(args.model_path)

    # Load dataset
    print(f"\nLoading validation data from: {args.val_path}")
    samples = load_val_dataset(args.val_path, args.max_samples)
    print(f"Loaded {len(samples)} samples")

    # Create evaluation logger
    model_name = args.model_path.split("/")[-1]
    ev = EvaluationLogger(name=f"c3-{args.task}-eval", model=model_name)

    print(f"\n{'='*60}")
    print(f"Starting evaluation: {len(samples)} samples")
    print(f"Task: {args.task}")
    print(f"Context percentage: {args.pct}%")
    print(f"{'='*60}\n")

    for i, sample in enumerate(samples):
        context = sample["context"][:args.max_context_chars]

        # Apply percentage truncation if specified
        if args.pct < 100:
            truncate_len = int(len(context) * args.pct / 100)
            context = context[:truncate_len]

        # Set prompt and ground truth based on task
        if args.task == "repeat":
            prompt = "Repeat the text: "
            gt = context  # For repeat task, ground truth is the input context
        else:
            prompt = sample["prompt"]
            gt = sample["ground_truth"]

        print(f"[{i+1}/{len(samples)}] Running inference (context: {len(context)} chars, {args.pct}%)...")

        # Generate output
        output = model.generate(context, prompt)

        # Calculate scores
        scores = {}

        # Always run these (fast)
        scores.update(rouge_scorer_fn(gt, output))
        scores.update(coverage_scorer(gt, output))

        # Task-specific scorers
        if args.task == "repeat":
            # For repeat task, just use basic metrics (ROUGE + coverage)
            pass
        else:
            # For summarization, add more scorers
            scores.update(compression_scorer(gt, output))
            if args.use_bert and BERT_SCORE_AVAILABLE:
                scores.update(bert_scorer(gt, output))
            if args.use_llm:
                scores.update(llm_scorer(gt, output))

        print(f"  ROUGE-L: {scores.get('rougeL_f', 0):.3f}, Coverage: {scores.get('coverage_score', 0):.3f}")

        # Log to weave
        ev.log_example(
            inputs={
                "context": context,
                "prompt": prompt,
                "grount_truth": gt
            },
            output=output,
            scores=scores,
        )

    # Finish and summarize
    ev.log_summary()

    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"Results: {ev.ui_url}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
