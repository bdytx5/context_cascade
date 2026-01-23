from transformers import AutoModel, AutoTokenizer
import arxiv
import os
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

# Download paper
arxiv_id = "1706.03762"  # "Attention is All You Need" paper
full_context, paper_title = download_arxiv_paper(arxiv_id)

if full_context is None:
    print("Failed to extract text from PDF. Exiting.")
    exit(1)

print(f"\nPaper Title: {paper_title}")
print(f"Total context length: {len(full_context)} characters")

# Load model
print("\n" + "="*80)
print("Loading C3 model...")
print("="*80)
model_name = 'liufanfanlff/C3-Context-Cascade-Compression'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
model = model.eval().cuda()

prompt = 'Repeat the text: '

# Test lengths
test_lengths = [10000]

print("\n" + "="*80)
print("STARTING MULTI-PASS BENCHMARK")
print(f"Testing context lengths: {test_lengths}")
print(f"Latent tokens used: {model.config.latent_token_len}")
print("="*80 + "\n")

results = []

for length in test_lengths:
    print(f"\n{'='*80}")
    print(f"TEST: {length} characters")
    print(f"{'='*80}")

    # Get context slice
    context = full_context[:length]
    print(f"Input preview (first 100 chars): {context}...")

    # Run model
    print(f"\nGenerating output...")
    output = model.chat(tokenizer, context, prompt)

    # Calculate metrics
    metrics = calculate_all_metrics(context, output)
    metrics["context_length"] = length
    results.append(metrics)

    # Print results for this pass
    print(f"\n--- Results for {length} chars ---")
    print(f"Output length: {metrics['output_len']} chars ({metrics['len_ratio']:.1f}% of input)")
    print(f"Char Accuracy:    {metrics['char_accuracy']:.2f}%")
    print(f"Word Accuracy:    {metrics['word_accuracy']:.2f}%")
    print(f"Sequence Match:   {metrics['sequence_match']:.2f}%")
    print(f"Word F1:          {metrics['word_f1']:.2f}%")
    print(f"Trigram Overlap:  {metrics['trigram_overlap']:.2f}%")

    print(f"\nOutput preview \:\n{output}...")

# Final summary
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"\n{'Context':<10} {'Char%':<10} {'Word%':<10} {'SeqMatch%':<12} {'WordF1%':<10} {'Trigram%':<10} {'OutLen':<10}")
print("-" * 72)

for r in results:
    print(f"{r['context_length']:<10} {r['char_accuracy']:<10.2f} {r['word_accuracy']:<10.2f} {r['sequence_match']:<12.2f} {r['word_f1']:<10.2f} {r['trigram_overlap']:<10.2f} {r['output_len']:<10}")

print("\n" + "="*80)
print(f"Model: {model_name}")
print(f"Latent tokens: {model.config.latent_token_len}")
print(f"Paper: {paper_title}")
print("="*80)
