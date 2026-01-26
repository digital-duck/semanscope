"""
Quick test script for Semantic Affinity feature
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from components.semantic_affinity import SemanticAffinityAnalyzer, load_translation_dataset
import numpy as np

# Configuration
CSV_PATH = Path(__file__).parent / "data/input/ICML-elemental-characters-translated.csv"
SELECTED_LANGUAGES = ['English', 'Chinese']
METRIC = 'euclidean'

print("=" * 60)
print("Semantic Affinity Feature Test")
print("=" * 60)

# Step 1: Load dataset
print(f"\n1. Loading dataset from: {CSV_PATH}")
if not CSV_PATH.exists():
    print(f"   ✗ Dataset not found!")
    sys.exit(1)

word_translations, language_codes = load_translation_dataset(
    str(CSV_PATH),
    SELECTED_LANGUAGES
)
print(f"   ✓ Loaded {len(word_translations)} words")
print(f"   ✓ Languages: {language_codes}")
print(f"   ✓ Sample: {word_translations[0]}")

# Step 2: Create mock embeddings for testing
print(f"\n2. Creating mock embeddings (random for testing)")
# In real usage, these would come from an actual embedding model

embeddings_dict = {}
embedding_dim = 384  # Common embedding dimension

for lang_code in language_codes:
    # Create random embeddings for testing
    # In a real scenario, different quality models would produce different alignment
    np.random.seed(42)  # For reproducibility
    embeddings_dict[lang_code] = np.random.randn(len(word_translations), embedding_dim)
    print(f"   ✓ {lang_code}: {embeddings_dict[lang_code].shape}")

# Step 3: Compute Semantic Affinity
print(f"\n3. Computing Semantic Affinity...")
analyzer = SemanticAffinityAnalyzer(collapse_epsilon=1e-6)

result = analyzer.compute_semantic_affinity(
    embeddings_dict=embeddings_dict,
    word_translations=word_translations,
    metric=METRIC
)

print(f"\n4. Results:")
print(f"   Status: {result.status}")
print(f"   SA Score: {result.score:.4f}")
print(f"   Std Dev: {result.std:.4f}")
print(f"   Global Spread: {result.global_spread:.4f}")
print(f"   Words Analyzed: {result.n_words}")
print(f"   Languages: {result.n_languages}")

if result.outliers:
    print(f"\n5. Top 5 Outliers (hardest to align):")
    for i, (word, score) in enumerate(result.outliers[:5], 1):
        print(f"   {i}. {word}: {score:.4f}")

print("\n" + "=" * 60)
print("✓ Test completed successfully!")
print("=" * 60)
