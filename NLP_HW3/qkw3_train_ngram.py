import random
from ngram_skeleton import NgramModel, create_ngram_model

SHAKESPEARE_FILE = "shakespeare_input.txt"
NYT_TEST_FILE = "nytimes_article.txt"
SONNETS_TEST_FILE = "shakespeare_sonnets.txt"
CONTEXT_SIZES = [2, 3, 4, 7]
GENERATED_TEXT_LENGTH = 250

def train_models():
    models = {}

    for c in CONTEXT_SIZES:
        print(f"[{c}] Training {c}-gram model on Shakespeare dataset...")
        models[c] = create_ngram_model(NgramModel, SHAKESPEARE_FILE, c)

    return models

def generate_text(models):
    for c, model in models.items():
        print(f"\n[>] Generated text using {c}-gram model:")
        random.seed(42)  # For reproducibility's sake
        print(f"[_] \n{model.random_text(GENERATED_TEXT_LENGTH)}")

def compute_perplexity(models):
    test_files = {
        "NYTimes Article": NYT_TEST_FILE,
        "Shakespeare Sonnets": SONNETS_TEST_FILE
    }
    
    perplexity_results = {}

    for test_name, test_file in test_files.items():
        print(f"\n[>] Computing perplexity on {test_name}...")
        with open(test_file, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        perplexity_results[test_name] = {}
        
        for c, model in models.items():
            line_perplexities = []
            for line in lines:
                line = line.strip()
                if line:
                    ppl = model.perplexity(line)
                    line_perplexities.append(ppl)

            if line_perplexities:
                average_ppl = sum(line_perplexities) / len(line_perplexities)
            else:
                average_ppl = float('inf')

            perplexity_results[test_name][c] = average_ppl
            print(f"[>] {c}-gram model average perplexity: {average_ppl:.2f}")

    return perplexity_results

if __name__ == "__main__":
    models = train_models()
    generate_text(models)
    perplexity_results = compute_perplexity(models)
