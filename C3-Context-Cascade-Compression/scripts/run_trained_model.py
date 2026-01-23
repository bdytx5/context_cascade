#!/usr/bin/env python3
"""
Run inference with locally trained C3 model on a new paper.
"""

from transformers import AutoModel, AutoTokenizer
import torch
import arxiv
import argparse
import difflib
from collections import Counter

# ========== ACCURACY METRICS ==========

def char_accuracy(original: str, generated: str) -> float:
    """Character-level accuracy"""
    if not original:
        return 0.0
    matches = sum(1 for a, b in zip(original, generated) if a == b)
    return matches / len(original) * 100

def word_accuracy(original: str, generated: str) -> float:
    """Word-level exact match accuracy"""
    orig_words = original.split()
    gen_words = generated.split()
    if not orig_words:
        return 0.0
    matches = sum(1 for a, b in zip(orig_words, gen_words) if a == b)
    return matches / len(orig_words) * 100

def sequence_match_ratio(original: str, generated: str) -> float:
    """SequenceMatcher ratio (similar to fuzzy match)"""
    return difflib.SequenceMatcher(None, original, generated).ratio() * 100

def word_overlap_f1(original: str, generated: str) -> float:
    """F1 score based on word overlap (bag of words)"""
    orig_words = Counter(original.lower().split())
    gen_words = Counter(generated.lower().split())
    common = sum((orig_words & gen_words).values())
    if common == 0:
        return 0.0
    precision = common / sum(gen_words.values()) if sum(gen_words.values()) > 0 else 0
    recall = common / sum(orig_words.values()) if sum(orig_words.values()) > 0 else 0
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall) * 100

def ngram_overlap(original: str, generated: str, n: int = 3) -> float:
    """N-gram overlap percentage"""
    def get_ngrams(text, n):
        return set(text[i:i+n] for i in range(len(text) - n + 1))
    orig_ngrams = get_ngrams(original, n)
    gen_ngrams = get_ngrams(generated, n)
    if not orig_ngrams:
        return 0.0
    overlap = len(orig_ngrams & gen_ngrams)
    return overlap / len(orig_ngrams) * 100

def calculate_all_metrics(original: str, generated: str) -> dict:
    """Calculate all metrics and return as dict"""
    return {
        "char_accuracy": char_accuracy(original, generated),
        "word_accuracy": word_accuracy(original, generated),
        "sequence_match": sequence_match_ratio(original, generated),
        "word_f1": word_overlap_f1(original, generated),
        "trigram_overlap": ngram_overlap(original, generated, n=3),
        "output_len": len(generated),
        "input_len": len(original),
        "len_ratio": len(generated) / len(original) * 100 if original else 0,
    }

# ========== ARXIV DOWNLOAD ==========

def download_arxiv_paper(arxiv_id):
    """Download arxiv paper and extract text content"""
    print(f"Downloading arxiv paper: {arxiv_id}")
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(search.results())
    pdf_path = paper.download_pdf(dirpath="./", filename=f"{arxiv_id}.pdf")
    print(f"Downloaded to: {pdf_path}")

    try:
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        print(f"Extracted {len(text)} characters from PDF")
        return text, paper.title
    except Exception as e:
        print(f"Error extracting text with PyPDF2: {e}")
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text()
            print(f"Extracted {len(text)} characters from PDF using pdfplumber")
            return text, paper.title
        except Exception as e2:
            print(f"Error extracting text with pdfplumber: {e2}")
            return None, None

# ========== MAIN ==========

def main():
    parser = argparse.ArgumentParser(description="Run trained C3 model on a new paper")
    parser.add_argument("--model_path", type=str,
                        default="/home/cloud/c3/output_dir/checkpoint-30",
                        help="Path to trained model")
    parser.add_argument("--arxiv_id", type=str, default="2401.04088",
                        help="Arxiv paper ID to test (default: a paper not in training set)")
    parser.add_argument("--context_length", type=int, default=10000,
                        help="Number of characters to use from paper")
    parser.add_argument("--prompt", type=str, default="Repeat the text: ",
                        help="Prompt to use")
    args = parser.parse_args()

    # Download paper
    full_context, paper_title = download_arxiv_paper(args.arxiv_id)

    if full_context is None:
        print("Failed to extract text from PDF. Exiting.")
        exit(1)

    print(f"\nPaper Title: {paper_title}")
    print(f"Total context length: {len(full_context)} characters")

    # Load model from local path
    print("\n" + "="*80)
    print(f"Loading trained C3 model from: {args.model_path}")
    print("="*80)

    # Load tokenizer and model directly from checkpoint
    # (After retraining with fixed trainer, all weights including llm1 are saved properly)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print("Loading model from checkpoint...")
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    model = model.eval().cuda()

    print(f"Model loaded successfully!")
    print(f"Latent tokens: {model.config.latent_token_len}")

    # Get context slice
    context = full_context[:args.context_length]

    print("\n" + "="*80)
    print(f"RUNNING INFERENCE")
    print(f"Context length: {len(context)} chars")
    print(f"Prompt: {args.prompt}")
    print("="*80)

    print(f"\nInput preview:\n}...")

    # Run model
    print(f"\nGenerating output...")
    output = model.chat(tokenizer, context, args.prompt)

    # Calculate metrics
    metrics = calculate_all_metrics(context, output)

    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Output length: {metrics['output_len']} chars ({metrics['len_ratio']:.1f}% of input)")
    print(f"Char Accuracy:    {metrics['char_accuracy']:.2f}%")
    print(f"Word Accuracy:    {metrics['word_accuracy']:.2f}%")
    print(f"Sequence Match:   {metrics['sequence_match']:.2f}%")
    print(f"Word F1:          {metrics['word_f1']:.2f}%")
    print(f"Trigram Overlap:  {metrics['trigram_overlap']:.2f}%")

    print("\n" + "="*80)
    print("OUTPUT PREVIEW (first 500 chars):")
    print("="*80)
    print(output[:500])

    print("\n" + "="*80)
    print("FULL OUTPUT:")
    print("="*80)
    print(output)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Paper: {paper_title} (arxiv:{args.arxiv_id})")
    print(f"Context: {len(context)} chars -> Output: {len(output)} chars")
    print("="*80)

if __name__ == "__main__":
    main()
