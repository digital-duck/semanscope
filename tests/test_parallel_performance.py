"""
Performance test for parallel semantic affinity computation
"""
import sys
from pathlib import Path
import time
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from components.semantic_affinity import SemanticAffinityAnalyzer, load_translation_dataset

# Configuration
CSV_PATH = Path(__file__).parent / "data/input/ICML-elemental-characters-translated-SA.csv"
SELECTED_LANGUAGES = ['English', 'Chinese', 'Spanish']
METRIC = 'euclidean'

print("=" * 70)
print("Semantic Affinity Parallel Performance Test")
print("=" * 70)

# Load dataset
print(f"\n1. Loading dataset: {CSV_PATH.name}")
word_translations, language_codes = load_translation_dataset(
    str(CSV_PATH),
    SELECTED_LANGUAGES
)
print(f"   ✓ {len(word_translations)} words, {len(language_codes)} languages")

# Create mock embeddings
print(f"\n2. Creating mock embeddings...")
embedding_dim = 384
np.random.seed(42)

embeddings_dict = {}
for lang_code in language_codes:
    embeddings_dict[lang_code] = np.random.randn(len(word_translations), embedding_dim)
    print(f"   ✓ {lang_code}: {embeddings_dict[lang_code].shape}")

# Test 1: Sequential
print(f"\n3. Sequential computation (n_jobs=1)...")
analyzer_seq = SemanticAffinityAnalyzer(n_jobs=1)
start = time.time()
result_seq = analyzer_seq.compute_semantic_affinity(
    embeddings_dict=embeddings_dict,
    word_translations=word_translations,
    metric=METRIC
)
time_seq = time.time() - start
print(f"   ✓ Time: {time_seq:.3f}s")
print(f"   ✓ SA Score: {result_seq.score:.4f}")

# Test 2: Parallel with 2 cores
print(f"\n4. Parallel computation (n_jobs=2)...")
analyzer_par2 = SemanticAffinityAnalyzer(n_jobs=2)
start = time.time()
result_par2 = analyzer_par2.compute_semantic_affinity(
    embeddings_dict=embeddings_dict,
    word_translations=word_translations,
    metric=METRIC
)
time_par2 = time.time() - start
print(f"   ✓ Time: {time_par2:.3f}s")
print(f"   ✓ SA Score: {result_par2.score:.4f}")
print(f"   ✓ Speedup: {time_seq/time_par2:.2f}x")

# Test 3: Parallel with all cores
print(f"\n5. Parallel computation (n_jobs=-1, all cores)...")
analyzer_par = SemanticAffinityAnalyzer(n_jobs=-1)
start = time.time()
result_par = analyzer_par.compute_semantic_affinity(
    embeddings_dict=embeddings_dict,
    word_translations=word_translations,
    metric=METRIC
)
time_par = time.time() - start
print(f"   ✓ Time: {time_par:.3f}s")
print(f"   ✓ SA Score: {result_par.score:.4f}")
print(f"   ✓ Speedup: {time_seq/time_par:.2f}x")
print(f"   ✓ Using {analyzer_par.n_jobs} cores")

# Verify results match
print(f"\n6. Verification...")
score_diff = abs(result_seq.score - result_par.score)
print(f"   ✓ Score difference (seq vs parallel): {score_diff:.10f}")
if score_diff < 1e-10:
    print(f"   ✓ Results identical!")
elif score_diff < 1e-6:
    print(f"   ✓ Results match (negligible floating point differences)")
else:
    print(f"   ⚠️ Results differ (may indicate issue)")

# Summary
print(f"\n" + "=" * 70)
print("Performance Summary")
print("=" * 70)
print(f"Dataset: {len(word_translations)} words × {len(language_codes)} languages")
print(f"Sequential:     {time_seq:.3f}s")
print(f"Parallel (2):   {time_par2:.3f}s ({time_seq/time_par2:.2f}x speedup)")
print(f"Parallel (all): {time_par:.3f}s ({time_seq/time_par:.2f}x speedup)")
print(f"Efficiency:     {(time_seq/time_par) / analyzer_par.n_jobs * 100:.1f}%")
print("=" * 70)
