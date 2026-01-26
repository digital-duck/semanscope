#!/usr/bin/env python3
"""
CLI Batch Benchmark Tool for Relational Affinity Analysis

This tool mirrors the Streamlit RA workflow for batch/headless execution:
- Uses embedding cache (shared with Streamlit app)
- Auto-saves RA scores as JSON
- Logs intermediate calculations for debugging
- Compatible with model_manager.py architecture

Usage Examples:
conda activate semanscope
cd /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope


# Single model, single dataset
$ python cli_batch_benchmark_ra.py run --datasets "NeurIPS-01-family-relations" --models "LaBSE" --metrics "cosine,euclidean"

# One model on all categories
$ python cli_batch_benchmark_ra.py run --datasets "NeurIPS-01-family-relations,NeurIPS-02-royalty-hierarchy,NeurIPS-03-gendered-occupations,NeurIPS-04-animal-gender,NeurIPS-05-opposite-relations,NeurIPS-06-comparative-superlative" --models "LaBSE" --metrics "cosine,euclidean" 2>&1 | tee ./batch_benchmark/cli_batch_benchmark_ra-2025-12-25.txt

# Multiple models and 2 datasets
$ python cli_batch_benchmark_ra.py run --datasets "NeurIPS-01-family-relations,NeurIPS-05-opposite-relations,NeurIPS-04-animal-gender" --models "LaBSE, Universal-Sentence-Encoder-Multilingual, Multilingual-E5-Large-Instruct-v2, Sentence-BERT Multilingual,  Qwen3-Embedding-0.6B, Qwen3-Embedding-8B (OpenRouter), Gemini-Embedding-001 (OpenRouter),  OpenAI Text-Embedding-Ada-002 (OpenRouter), OpenAI Text-Embedding-3-Small (OpenRouter), OpenAI Text-Embedding-3-Large (OpenRouter),   mBERT, XLM-RoBERTa-v2, EmbeddingGemma-300M, DistilBERT Multilingual, BGE-M3 (Ollama), Snowflake-Arctic-Embed2 (Ollama), Qwen3-Embedding-4B (Ollama)" --languages "english,chinese" --metrics "cosine,euclidean" 2>&1 | tee ./batch_benchmark/cli_batch_benchmark_ra-2025-12-27-all-models-2-datasets-EN-ZH.txt

# Multiple models and multiple datasets
$ python cli_batch_benchmark_ra.py run --datasets "NeurIPS-01-family-relations,NeurIPS-02-royalty-hierarchy,NeurIPS-03-gendered-occupations,NeurIPS-04-comparative-superlative,NeurIPS-05-opposite-relations,NeurIPS-06-animal-gender,NeurIPS-07-math-numbers,NeurIPS-07-sequential-mathematical, 
NeurIPS-08-hierarchical-relations,NeurIPS-09-part-whole,NeurIPS-10-size-scale,NeurIPS-11-cause-effect" --models "LaBSE, Universal-Sentence-Encoder-Multilingual, Multilingual-E5-Large-Instruct-v2, Sentence-BERT Multilingual,  Qwen3-Embedding-0.6B, Qwen3-Embedding-8B (OpenRouter), Gemini-Embedding-001 (OpenRouter),  OpenAI Text-Embedding-Ada-002 (OpenRouter), OpenAI Text-Embedding-3-Small (OpenRouter), OpenAI Text-Embedding-3-Large (OpenRouter),   mBERT, XLM-RoBERTa-v2, EmbeddingGemma-300M, DistilBERT Multilingual, BGE-M3 (Ollama), Snowflake-Arctic-Embed2 (Ollama), Qwen3-Embedding-4B (Ollama)"  --languages "english,chinese" --metrics "cosine,euclidean" 2>&1 | tee ./batch_benchmark/cli_batch_benchmark_ra-2025-12-29-all-models-all-datasets-EN-ZH.txt


# RA : all models and all datasets: EN-ZH
$ python cli_batch_benchmark_ra.py run --datasets "NeurIPS-01-family-relations,NeurIPS-02-royalty-hierarchy,NeurIPS-03-gendered-occupations,NeurIPS-04-comparative-superlative,NeurIPS-05-opposite-relations,NeurIPS-06-animal-gender,NeurIPS-07-math-numbers,NeurIPS-07-sequential-mathematical, 
NeurIPS-08-hierarchical-relations,NeurIPS-09-part-whole,NeurIPS-10-size-scale,NeurIPS-11-cause-effect" --models "Voyage-3 (Voyage AI), Voyage-Multilingual-2 (Voyage AI), LaBSE, Universal-Sentence-Encoder-Multilingual, Multilingual-E5-Large-Instruct-v2, Sentence-BERT Multilingual,  Qwen3-Embedding-0.6B, Qwen3-Embedding-8B (OpenRouter), Gemini-Embedding-001 (OpenRouter),  OpenAI Text-Embedding-Ada-002 (OpenRouter), OpenAI Text-Embedding-3-Small (OpenRouter), OpenAI Text-Embedding-3-Large (OpenRouter),   mBERT, XLM-RoBERTa-v2, EmbeddingGemma-300M, DistilBERT Multilingual, BGE-M3 (Ollama), Snowflake-Arctic-Embed2 (Ollama), Qwen3-Embedding-4B (Ollama)"  --languages "english,chinese" --metrics "cosine,euclidean" 2>&1 | tee ./batch_benchmark/cli_batch_benchmark_ra-2025-12-31-all-models-all-datasets-EN-ZH.txt


# RA : all models and all datasets: EN-ES
$ python cli_batch_benchmark_ra.py run --datasets "NeurIPS-01-family-relations,NeurIPS-02-royalty-hierarchy,NeurIPS-03-gendered-occupations,NeurIPS-04-comparative-superlative,NeurIPS-05-opposite-relations,NeurIPS-06-animal-gender,NeurIPS-07-math-numbers,NeurIPS-07-sequential-mathematical, 
NeurIPS-08-hierarchical-relations,NeurIPS-09-part-whole,NeurIPS-10-size-scale,NeurIPS-11-cause-effect" --models "Voyage-3 (Voyage AI), Voyage-Multilingual-2 (Voyage AI), LaBSE, Universal-Sentence-Encoder-Multilingual, Multilingual-E5-Large-Instruct-v2, Sentence-BERT Multilingual,  Qwen3-Embedding-0.6B, Qwen3-Embedding-8B (OpenRouter), Gemini-Embedding-001 (OpenRouter),  OpenAI Text-Embedding-Ada-002 (OpenRouter), OpenAI Text-Embedding-3-Small (OpenRouter), OpenAI Text-Embedding-3-Large (OpenRouter),   mBERT, XLM-RoBERTa-v2, EmbeddingGemma-300M, DistilBERT Multilingual, BGE-M3 (Ollama), Snowflake-Arctic-Embed2 (Ollama), Qwen3-Embedding-4B (Ollama)"  --languages "english,spanish" --metrics "cosine,euclidean" 2>&1 | tee ./batch_benchmark/cli_batch_benchmark_ra-2025-12-31-all-models-all-datasets-EN-ES.txt

cd batch_benchmark

# With pivoted output for cross-language RA

python parse_batch_results_ra.py \
    --log-file cli_batch_benchmark_ra-2025-12-29-all-models-all-datasets-EN-ZH.txt \
    --output cli_batch_benchmark_ra-2025-12-29-all-models-all-datasets-EN-ZH.csv \
    --pivot \
    --pivot-cosine cli_batch_benchmark_ra-2025-12-29-all-models-all-datasets-EN-ZH-cos-cross-pivot.csv \
    --pivot-euclidean cli_batch_benchmark_ra-2025-12-29-all-models-all-datasets-EN-ZH-eucl-cross-pivot.csv


# ./batch_benchmark/cli_batch_benchmark_ra-2025-12-27-all-models-all-datasets-EN-ZH.txt

# RA - 2026-01-02 : all models and all datasets-v2.5: EN-ZH-ES-DE-TR
$ source ~/.bashrc  # set VOYAGE_API_KEY
$ python cli_batch_benchmark_ra.py run --datasets "NeurIPS-01-family-relations-v2.5,NeurIPS-02-royalty-hierarchy-v2.5, NeurIPS-03-gendered-occupations-v2.5, NeurIPS-04-comparative-superlative-v2.5,NeurIPS-05-opposite-relations-v2.5, NeurIPS-06-animal-gender-v2.5, NeurIPS-07-sequential-mathematical-v2.5, 
NeurIPS-08-hierarchical-relations-v2.5, NeurIPS-09-part-whole-v2.5, NeurIPS-10-size-scale-v2.5, NeurIPS-11-cause-effect-v2.5" --models "Voyage-3 (Voyage AI), Voyage-Multilingual-2 (Voyage AI), LaBSE, Universal-Sentence-Encoder-Multilingual, Multilingual-E5-Large-Instruct-v2, Sentence-BERT Multilingual,  Qwen3-Embedding-0.6B, Qwen3-Embedding-8B (OpenRouter), Gemini-Embedding-001 (OpenRouter),  OpenAI Text-Embedding-Ada-002 (OpenRouter), OpenAI Text-Embedding-3-Small (OpenRouter), OpenAI Text-Embedding-3-Large (OpenRouter),   mBERT, XLM-RoBERTa-v2, EmbeddingGemma-300M, DistilBERT Multilingual, BGE-M3 (Ollama), Snowflake-Arctic-Embed2 (Ollama), Qwen3-Embedding-4B (Ollama)"  --languages "english,chinese,spanish,german,turkish" --metrics "cosine" 2>&1 | tee ./batch_benchmark/cli_batch_benchmark_ra-2026-01-02-all-models-all-datasets-EN-ZH-ES-DE-TR.txt





cd batch_benchmark

# With pivoted output for cross-language RA
python parse_batch_results_ra.py \
    --log-file cli_batch_benchmark_ra-2025-12-27-all-models-all-datasets-EN-ZH.txt \
    --output cli_batch_benchmark_ra-2025-12-27-all-models-all-datasets-EN-ZH.csv \
    --pivot \
    --pivot-cosine cli_batch_benchmark_ra-2025-12-27-all-models-all-datasets-EN-ZH-cos-cross-pivot.csv \
    --pivot-euclidean cli_batch_benchmark_ra-2025-12-27-all-models-all-datasets-EN-ZH-eucl-cross-pivot.csv

python parse_batch_results_ra.py \
    --log-file cli_batch_benchmark_ra-2025-12-29-all-models-all-datasets-EN-ZH.txt \
    --output cli_batch_benchmark_ra-2025-12-29-all-models-all-datasets-EN-ZH.csv \
    --pivot \
    --pivot-cosine cli_batch_benchmark_ra-2025-12-29-all-models-all-datasets-EN-ZH-cos-cross-pivot.csv \
    --pivot-euclidean cli_batch_benchmark_ra-2025-12-29-all-models-all-datasets-EN-ZH-eucl-cross-pivot.csv

cd batch_benchmark
python parse_batch_results_ra_v2.5.py \
    --log-file cli_batch_benchmark_ra-2026-01-02-all-models-all-datasets-EN-ZH-ES-DE-TR.txt \
    --output cli_batch_benchmark_ra-2026-01-02-all-models-all-datasets-EN-ZH-ES-DE-TR.csv \
    --pivot \
    --pivot-nway cli_batch_benchmark_ra-2026-01-02-all-models-all-datasets-EN-ZH-ES-DE-TR-nway-pivot.csv 

# to aggregate SA+RA results for your v2.5 data:
python create_sa_ra_aggregated.py \
    --ra-csv cli_batch_benchmark_ra-2026-01-02-all-models-all-datasets-EN-ZH-ES-DE-TR.csv \
    --sa-csv 2026-01-02-all-models-all-datasets-EN-ZH-ES-DE-TR-sa.csv \
    --output sa-ra-aggregated-2026-01-02-v2.5-5lang.csv

    
# generate figures for v2.5 dataset
# Then generate all figures (EN, ZH, ES for main comparisons):
python neurips_figures-v2.py --figures all --languages EN,ZH,ES | tee batch_benchmark-2026-01-02-all-models-v2.5-datasets-EN-ZH-ES.txt


# added Turkish and German languages on 2026-01-02 

# Test with just 1 model, 1 dataset, all 5 languages
python cli_batch_benchmark_ra.py run \
    --datasets "NeurIPS-01-family-relations-v2.5" \
    --models "LaBSE" \
    --languages "english,chinese,spanish,german,turkish" \
    --metrics "cosine" \
    2>&1 | tee ./batch_benchmark/test-v2.5-turkish-support.txt


Author: Claude Sonnet 4.5 (Anthropic)
Date: 2025-12-25
"""

import click
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_PATH
from models.model_manager import get_model
from utils.embedding_cache import get_embedding_cache


# Language name to code mapping
LANG_NAME_TO_CODE = {
    'chinese': 'chn',
    'english': 'enu',
    'spanish': 'spa',
    'french': 'fra',
    'german': 'deu',
    'russian': 'rus',
    'korean': 'kor',
    'arabic': 'ara',
    'turkish': 'tur'
}

CODE_TO_LANG_NAME = {v: k.title() for k, v in LANG_NAME_TO_CODE.items()}

# Language code to CSV column suffix mapping
LANG_CODE_TO_CSV_SUFFIX = {
    'enu': 'en',
    'chn': 'zh',
    'spa': 'es',
    'fra': 'fr',
    'deu': 'de',
    'rus': 'ru',
    'kor': 'ko',
    'ara': 'ar',
    'tur': 'tr'
}


def round_floats(obj, decimals=4):
    """Recursively round all float values to specified decimal places"""
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, dict):
        return {k: round_floats(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(item, decimals) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(round_floats(item, decimals) for item in obj)
    else:
        return obj


def get_embeddings_for_words(
    words: List[str],
    model_name: str,
    lang_code: str = None
) -> np.ndarray:
    """Get embeddings for words using specified model"""
    if not words:
        return np.array([])

    model = get_model(model_name)
    embeddings = model.get_embeddings(words, lang="en", debug_flag=False)

    if embeddings is None:
        raise ValueError(f"Model {model_name} returned None for {len(words)} words. Check model initialization errors above.")

    return np.array(embeddings)


def ra_cosine(rel_vec1: np.ndarray, rel_vec2: np.ndarray) -> float:
    """Compute RA using cosine similarity"""
    dot_product = np.dot(rel_vec1, rel_vec2)
    norm1 = np.linalg.norm(rel_vec1)
    norm2 = np.linalg.norm(rel_vec2)
    return float(dot_product / (norm1 * norm2))


def ra_euclidean(rel_vec1: np.ndarray, rel_vec2: np.ndarray) -> float:
    """Compute RA using normalized Euclidean distance"""
    diff = rel_vec1 - rel_vec2
    dist = np.linalg.norm(diff)
    norm1 = np.linalg.norm(rel_vec1)
    norm2 = np.linalg.norm(rel_vec2)
    return float(2 * dist / (norm1 + norm2))


def pairwise_ra(rel_vecs: List[np.ndarray], metric: str = 'cosine') -> np.ndarray:
    """Compute all pairwise RA scores"""
    N = len(rel_vecs)
    ra_matrix = np.zeros((N, N))

    if metric == 'cosine':
        ra_func = ra_cosine
        diagonal_value = 1.0
    elif metric == 'euclidean':
        ra_func = ra_euclidean
        diagonal_value = 0.0
    else:
        raise ValueError(f"Unknown metric: {metric}")

    for i in range(N):
        for j in range(i, N):
            if i == j:
                ra_matrix[i, j] = diagonal_value
            else:
                score = ra_func(rel_vecs[i], rel_vecs[j])
                ra_matrix[i, j] = score
                ra_matrix[j, i] = score

    return ra_matrix


def aggregate_ra_category(ra_matrix: np.ndarray, metric: str = 'cosine') -> Dict:
    """Aggregate pairwise RA scores"""
    N = ra_matrix.shape[0]
    upper_triangle_indices = np.triu_indices(N, k=1)
    off_diagonal = ra_matrix[upper_triangle_indices]

    result = {
        'mean': float(np.mean(off_diagonal)),
        'std': float(np.std(off_diagonal)),
        'min': float(np.min(off_diagonal)),
        'max': float(np.max(off_diagonal))
    }

    if metric == 'cosine':
        result['negative_count'] = int(np.sum(off_diagonal < 0))

    return result


def compute_ra_single(
    dataset_name: str,
    model_name: str,
    languages: List[str] = ['english', 'chinese'],
    metrics: List[str] = ['cosine', 'euclidean'],
    output_dir: Optional[Path] = None,
    force_recompute_embeddings: bool = False,
    verbose: bool = True,
    log_intermediate: bool = True,
    epsilon: float = 1e-8
) -> Dict[str, Any]:
    """
    Compute Relational Affinity for a single configuration

    This function mirrors the Streamlit RA page workflow exactly:
    1. Load dataset (relation pairs for specified languages)
    2. Get embeddings (cache-aware)
    3. L2 normalize all embeddings
    4. Compute relational vectors
    5. Compute RA for each metric
    6. Auto-save results + intermediate calculations

    Args:
        dataset_name: Dataset name without -RA suffix
        model_name: Model identifier
        languages: List of language names (e.g., ['english', 'chinese', 'spanish'])
        metrics: Distance metrics to compute
        output_dir: Where to save results (default: data/batch_results_ra)
        force_recompute_embeddings: Ignore cache
        verbose: Print progress
        log_intermediate: Log detailed calculations
        epsilon: Zero vector threshold

    Returns:
        Dictionary with results and metadata
    """

    if output_dir is None:
        output_dir = DATA_PATH / "batch_results_ra"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert language names to codes
    lang_codes = []
    for lang_name in languages:
        lang_name_lower = lang_name.lower()
        if lang_name_lower not in LANG_NAME_TO_CODE:
            raise ValueError(f"Unknown language: {lang_name}. Supported: {list(LANG_NAME_TO_CODE.keys())}")
        lang_codes.append(LANG_NAME_TO_CODE[lang_name_lower])

    # Get CSV suffixes for each language
    csv_suffixes = []
    for code in lang_codes:
        if code not in LANG_CODE_TO_CSV_SUFFIX:
            raise ValueError(f"No CSV suffix mapping for language code: {code}")
        csv_suffixes.append(LANG_CODE_TO_CSV_SUFFIX[code])

    if verbose:
        click.echo(f"\n{'='*80}")
        click.echo(f"Dataset: {dataset_name}")
        click.echo(f"Model: {model_name}")
        click.echo(f"Languages: {', '.join(languages)} ({', '.join([c.upper() for c in csv_suffixes])})")
        click.echo(f"Metrics: {', '.join(metrics)}")
        click.echo(f"{'='*80}\n")

    try:
        start_time = time.time()

        # 1. Load dataset
        if verbose:
            click.echo("📂 Loading dataset...")

        dataset_path = DATA_PATH / "input" / f"{dataset_name}-RA.csv"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        df = pd.read_csv(dataset_path, comment='#')

        # Build required columns dynamically based on languages
        required_cols = []
        for suffix in csv_suffixes:
            required_cols.extend([f'word1_{suffix}', f'word2_{suffix}'])

        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}. Available columns: {list(df.columns)}")

        if verbose:
            click.echo(f"  Loaded {len(df)} relation pairs\n")

        # 2. Collect unique words for each language
        all_words_by_lang = {}
        for i, suffix in enumerate(csv_suffixes):
            words = set()
            for _, row in df.iterrows():
                words.add(row[f'word1_{suffix}'])
                words.add(row[f'word2_{suffix}'])
            all_words_by_lang[suffix] = sorted(list(words))

        if verbose:
            word_counts = ', '.join([f"{len(all_words_by_lang[suffix])} {suffix.upper()}" for suffix in csv_suffixes])
            click.echo(f"📊 Unique words: {word_counts}\n")

        # 3. Get embeddings (cache-aware) for all languages
        if verbose:
            click.echo("🔮 Generating embeddings (cache-aware)...")

        embedding_cache = get_embedding_cache()
        embeddings_by_lang = {}
        cache_stats = {}

        for i, (suffix, code) in enumerate(zip(csv_suffixes, lang_codes)):
            words = all_words_by_lang[suffix]
            embeddings, cached, computed = embedding_cache.get_embeddings(
                words=words,
                model_name=model_name,
                lang_code=code,
                embedding_func=lambda w, c=code: get_embeddings_for_words(w, model_name, c),
                force_recompute=force_recompute_embeddings
            )
            embeddings_by_lang[suffix] = embeddings
            cache_stats[suffix] = {'cached': cached, 'computed': computed}

        embedding_cache.save_master_cache()

        if verbose:
            for suffix in csv_suffixes:
                stats = cache_stats[suffix]
                click.echo(f"  ⚡ {suffix.upper()}: {stats['cached']} cached, {stats['computed']} computed")
            click.echo()

        # 4. L2 normalize all embeddings (CRITICAL for RA!)
        embeddings_normalized = {}
        for suffix in csv_suffixes:
            emb = embeddings_by_lang[suffix]
            embeddings_normalized[suffix] = emb / np.linalg.norm(emb, axis=1, keepdims=True)

        if verbose:
            click.echo(f"🔧 Normalized all embeddings to unit length (L2 norm = 1.0)\n")

        # Build word-to-embedding maps for each language
        word_to_emb = {}
        for suffix in csv_suffixes:
            words = all_words_by_lang[suffix]
            emb_norm = embeddings_normalized[suffix]
            word_to_emb[suffix] = {word: emb_norm[i] for i, word in enumerate(words)}

        # 5. Compute relational vectors for all languages
        if verbose:
            click.echo("📐 Computing relational vectors...")

        # Initialize storage for each language
        rel_vecs_by_lang = {suffix: [] for suffix in csv_suffixes}
        pair_labels_by_lang = {suffix: [] for suffix in csv_suffixes}
        excluded_zero_vec = {suffix: 0 for suffix in csv_suffixes}
        excluded_oov = 0

        valid_row_indices = []

        for idx, row in df.iterrows():
            try:
                # Compute relational vector for each language
                rel_vecs_row = {}
                skip_row = False

                for suffix in csv_suffixes:
                    w1 = row[f'word1_{suffix}']
                    w2 = row[f'word2_{suffix}']

                    emb1 = word_to_emb[suffix][w1]
                    emb2 = word_to_emb[suffix][w2]
                    rel_vec = emb2 - emb1

                    norm_rel_vec = np.linalg.norm(rel_vec)
                    if norm_rel_vec < epsilon:
                        excluded_zero_vec[suffix] += 1
                        skip_row = True
                        break

                    rel_vecs_row[suffix] = rel_vec

                if skip_row:
                    continue

                # Valid pair for all languages - store
                for suffix in csv_suffixes:
                    rel_vecs_by_lang[suffix].append(rel_vecs_row[suffix])
                    w1 = row[f'word1_{suffix}']
                    w2 = row[f'word2_{suffix}']
                    pair_labels_by_lang[suffix].append(f"{w1}→{w2}")

                valid_row_indices.append(idx)

            except KeyError as e:
                excluded_oov += 1

        total_pairs = len(df)
        valid_pairs = len(valid_row_indices)
        coverage = valid_pairs / total_pairs if total_pairs > 0 else 0.0

        if verbose:
            click.echo(f"  ✓ Valid pairs: {valid_pairs}/{total_pairs} ({coverage*100:.1f}%)")
            excl_msg = ', '.join([f"{excluded_zero_vec[s]} {s.upper()} zero-vec" for s in csv_suffixes])
            click.echo(f"  ✗ Excluded: {excl_msg}, {excluded_oov} OOV\n")

        # 6. Compute RA for each metric
        if verbose:
            click.echo("📏 Computing Relational Affinity...\n")

        results_dict = {}

        for metric in metrics:
            if verbose:
                click.echo(f"  {metric.title()}:")

            # Within-language RA for each language
            ra_within = {}
            for suffix in csv_suffixes:
                rel_vecs = rel_vecs_by_lang[suffix]
                ra_matrix = pairwise_ra(rel_vecs, metric=metric)
                ra_stats = aggregate_ra_category(ra_matrix, metric=metric)
                ra_within[suffix] = ra_stats

            # Cross-language RA computation (UNIFIED MODE - matches SemanScope default)
            # Mode: Pool all vectors from all languages, compute ALL pairwise comparisons
            # For 2 languages (EN, ZH) with N relations each:
            #   - Total vectors: 2N
            #   - Total comparisons: (2N × (2N-1)) / 2
            #   - Includes: EN-EN (within), ZH-ZH (within), EN-ZH (cross)

            if metric == 'cosine':
                ra_func = ra_cosine
            else:
                ra_func = ra_euclidean

            ra_cross = {}

            # For 2-language case: use unified mode (pool all vectors)
            if len(csv_suffixes) == 2:
                suffix1, suffix2 = csv_suffixes[0], csv_suffixes[1]

                # Pool all relational vectors from both languages
                all_rel_vecs = rel_vecs_by_lang[suffix1] + rel_vecs_by_lang[suffix2]

                # Compute pairwise RA matrix for ALL comparisons
                ra_matrix_unified = pairwise_ra(all_rel_vecs, metric=metric)

                # Aggregate statistics from the unified matrix
                cross_stats_unified = aggregate_ra_category(ra_matrix_unified, metric=metric)

                cross_key = f"{suffix1}_{suffix2}"
                ra_cross[cross_key] = cross_stats_unified

            # For 3+ languages: compute pairwise cross-language RAs
            else:
                for i in range(len(csv_suffixes)):
                    for j in range(i + 1, len(csv_suffixes)):
                        suffix1, suffix2 = csv_suffixes[i], csv_suffixes[j]

                        # Pool vectors from these two languages
                        all_rel_vecs = rel_vecs_by_lang[suffix1] + rel_vecs_by_lang[suffix2]

                        # Compute unified pairwise RA
                        ra_matrix_unified = pairwise_ra(all_rel_vecs, metric=metric)
                        cross_stats_unified = aggregate_ra_category(ra_matrix_unified, metric=metric)

                        cross_key = f"{suffix1}_{suffix2}"
                        ra_cross[cross_key] = cross_stats_unified

            # N-way cross-language RA (for 3+ languages)
            if len(csv_suffixes) >= 3:
                # Compute average of all pairwise cross RAs for each relation
                nway_scores = []
                for pair_idx in range(valid_pairs):
                    # Get all relational vectors for this pair across all languages
                    rel_vecs_this_pair = [rel_vecs_by_lang[suffix][pair_idx] for suffix in csv_suffixes]

                    # Compute all pairwise RAs for this relation
                    pairwise_scores = []
                    for i in range(len(csv_suffixes)):
                        for j in range(i + 1, len(csv_suffixes)):
                            score = ra_func(rel_vecs_this_pair[i], rel_vecs_this_pair[j])
                            pairwise_scores.append(score)

                    # Average pairwise score represents N-way consistency
                    nway_score = float(np.mean(pairwise_scores))
                    nway_scores.append(nway_score)

                nway_key = '_'.join(csv_suffixes)
                ra_cross[nway_key] = {
                    'mean': float(np.mean(nway_scores)),
                    'std': float(np.std(nway_scores)),
                    'min': float(np.min(nway_scores)),
                    'max': float(np.max(nway_scores)),
                    'note': 'N-way RA computed as average of all pairwise cross RAs per relation'
                }

                if metric == 'cosine':
                    ra_cross[nway_key]['negative_count'] = int(np.sum(np.array(nway_scores) < 0))

            # Verbose output
            if verbose:
                for suffix in csv_suffixes:
                    stats = ra_within[suffix]
                    click.echo(f"    {suffix.upper()} (within):  {stats['mean']:.4f} ± {stats['std']:.4f}")

                # Show pairwise cross RAs first
                pairwise_keys = [k for k in ra_cross.keys() if k.count('_') == 1]
                for cross_key in sorted(pairwise_keys):
                    stats = ra_cross[cross_key]
                    lang_pair = '-'.join([s.upper() for s in cross_key.split('_')])
                    click.echo(f"    {lang_pair} (cross):   {stats['mean']:.4f} ± {stats['std']:.4f}")

                # Show N-way cross RA if exists
                nway_keys = [k for k in ra_cross.keys() if k.count('_') >= 2]
                for nway_key in nway_keys:
                    stats = ra_cross[nway_key]
                    lang_nway = '-'.join([s.upper() for s in nway_key.split('_')])
                    click.echo(f"    {lang_nway} (N-way):   {stats['mean']:.4f} ± {stats['std']:.4f}")

            # Store results
            results_dict[metric] = {
                'ra_within': ra_within,
                'ra_cross': ra_cross,
                'pairwise_comparisons': valid_pairs * (valid_pairs - 1) // 2,
                'coverage': coverage,
                'excluded_zero_vec': excluded_zero_vec,
                'excluded_oov': excluded_oov
            }

        elapsed_time = time.time() - start_time

        # 7. Save results as JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = model_name.replace('/', '_').replace(' ', '_')
        safe_dataset_name = dataset_name.replace('/', '_').replace(' ', '_')
        lang_suffix = '-'.join(csv_suffixes)

        json_filename = f"ra-{safe_dataset_name}-{lang_suffix}-{safe_model_name}-{timestamp}.json"
        json_path = output_dir / json_filename

        # Build cache stats dynamically
        cache_stats_dict = {}
        for suffix in csv_suffixes:
            stats = cache_stats[suffix]
            cache_stats_dict[f'{suffix}_cached'] = stats['cached']
            cache_stats_dict[f'{suffix}_computed'] = stats['computed']

        output_data = {
            'metadata': {
                'dataset': dataset_name,
                'model': model_name,
                'languages': languages,
                'language_codes': lang_codes,
                'csv_suffixes': csv_suffixes,
                'metrics': metrics,
                'total_pairs': total_pairs,
                'valid_pairs': valid_pairs,
                'coverage': coverage,
                'timestamp': timestamp,
                'datetime': datetime.now().isoformat(),
                'cache_stats': cache_stats_dict,
                'execution_time_seconds': elapsed_time
            },
            'results': results_dict
        }

        # Round all floats to 4 decimal places
        output_data = round_floats(output_data, decimals=4)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        if verbose:
            click.echo(f"\n💾 Results saved: {json_path}")
            click.echo(f"⏱️  Total time: {elapsed_time:.2f}s\n")

        return output_data

    except Exception as e:
        if verbose:
            click.echo(f"\n❌ Error: {str(e)}", err=True)
        import traceback
        traceback.print_exc()
        return None


@click.group()
def cli():
    """CLI Batch Benchmark Tool for Relational Affinity Analysis"""
    pass


@cli.command()
@click.option('--datasets', required=True, help='Comma-separated dataset names (without -RA suffix)')
@click.option('--models', required=True, help='Comma-separated model names')
@click.option('--languages', default='english,chinese', help='Comma-separated languages (e.g., english,chinese,spanish) [default: english,chinese]')
@click.option('--metrics', multiple=True, default=['cosine', 'euclidean'], help='Distance metrics')
@click.option('--output-dir', type=click.Path(), default=None, help='Output directory (default: data/batch_results_ra)')
@click.option('--force-recompute', is_flag=True, help='Ignore embedding cache')
@click.option('--no-intermediate', is_flag=True, help='Skip intermediate calculations logging')
@click.option('--continue-on-error', is_flag=True, help='Continue processing other models/datasets when an error occurs (default: stop on first error)')
def run(datasets, models, languages, metrics, output_dir, force_recompute, no_intermediate, continue_on_error):
    """Run RA computation with flexible model/dataset/language combinations

    This command runs a cross-product of models × datasets with specified languages.

    Examples:
        # Single model, single dataset, default EN-ZH
        --models LaBSE --datasets NeurIPS-01-family-relations

        # Multiple models, multiple datasets, EN-ZH
        --models "LaBSE,mBERT" --datasets "NeurIPS-01-family-relations,NeurIPS-04-animal-gender"

        # English-Spanish analysis
        --models LaBSE --datasets NeurIPS-01-family-relations --languages english,spanish

        # Three-way comparison (EN-ZH-ES)
        --models LaBSE --datasets NeurIPS-01-family-relations --languages english,chinese,spanish
    """
    # Parse comma-separated inputs
    model_list = [m.strip() for m in models.strip().split(',') if m.strip()]
    dataset_list = [d.strip() for d in datasets.strip().split(',') if d.strip()]
    language_list = [lang.strip() for lang in languages.strip().split(',') if lang.strip()]

    # Handle metrics: can be passed as --metrics cosine,euclidean or --metrics cosine --metrics euclidean
    if len(metrics) == 1 and ',' in metrics[0]:
        # User passed comma-separated: --metrics cosine,euclidean
        metrics_list = [m.strip() for m in metrics[0].split(',') if m.strip()]
    else:
        # User passed multiple flags: --metrics cosine --metrics euclidean
        metrics_list = list(metrics)

    # Calculate total runs
    total_runs = len(model_list) * len(dataset_list)

    click.echo(f"\n{'='*80}")
    click.echo(f"🚀 Batch Run Configuration")
    click.echo(f"{'='*80}")
    click.echo(f"Models ({len(model_list)}): {', '.join(model_list)}")
    click.echo(f"Datasets ({len(dataset_list)}): {', '.join(dataset_list)}")
    click.echo(f"Languages ({len(language_list)}): {', '.join(language_list)}")
    click.echo(f"Total runs: {len(model_list)} models × {len(dataset_list)} datasets = {total_runs}")
    click.echo(f"{'='*80}\n")

    results = []
    completed = 0

    # Outer loop: models × datasets
    for model_name in model_list:
        for dataset_name in dataset_list:
            completed += 1

            click.echo(f"\n{'='*80}")
            click.echo(f"[{completed}/{total_runs}] Progress: {completed/total_runs*100:.1f}%")
            click.echo(f"Model: {model_name}")
            click.echo(f"Dataset: {dataset_name}")
            click.echo(f"{'='*80}")

            result = compute_ra_single(
                dataset_name=dataset_name,
                model_name=model_name,
                languages=language_list,
                metrics=metrics_list,
                output_dir=Path(output_dir) if output_dir else None,
                force_recompute_embeddings=force_recompute,
                verbose=True,
                log_intermediate=not no_intermediate
            )

            if result:
                results.append(result)
            else:
                # Error occurred
                if not continue_on_error:
                    click.echo(f"\n{'='*80}")
                    click.echo(f"❌ Stopping due to error. Completed {len(results)}/{total_runs} computations.")
                    click.echo(f"   Use --continue-on-error to skip failed computations and continue.")
                    click.echo(f"{'='*80}\n")
                    sys.exit(1)
                else:
                    click.echo(f"\n⚠️  Skipping failed computation, continuing with next...\n")

    click.echo(f"\n{'='*80}")
    click.echo(f"✅ Completed {len(results)}/{total_runs} computations")
    click.echo(f"{'='*80}\n")


@cli.command()
def list_datasets():
    """List all available RA datasets"""

    input_dir = DATA_PATH / "input"

    if not input_dir.exists():
        click.echo(f"Error: Input directory not found: {input_dir}", err=True)
        sys.exit(1)

    # Find all *-RA.csv files
    dataset_files = sorted(input_dir.glob("*-RA.csv"))

    if not dataset_files:
        click.echo("No RA datasets found in data/input/", err=True)
        sys.exit(1)

    click.echo("\nAvailable RA datasets:")
    for csv_file in dataset_files:
        # Remove -RA.csv suffix to get dataset name
        dataset_name = csv_file.stem.replace("-RA", "")

        # Count pairs in dataset
        try:
            df = pd.read_csv(csv_file, comment='#')
            n_pairs = len(df)
            category = df['category'].unique()[0] if 'category' in df.columns else 'unknown'
            click.echo(f"  - {dataset_name} ({n_pairs} pairs, category: {category})")
        except Exception as e:
            click.echo(f"  - {dataset_name} (error reading file: {e})")

    click.echo()


@cli.command()
def list_models():
    """List active models (matching Streamlit app dropdown)"""

    from config import MODEL_INFO

    # Filter only active models
    active_models = {name: info for name, info in MODEL_INFO.items()
                     if info.get('is_active', True)}

    click.echo(f"\nActive models ({len(active_models)} total):")
    click.echo("Format: model-name (path/identifier)\n")

    for model_name in sorted(active_models.keys()):
        model_config = active_models[model_name]
        path = model_config.get('path', '')
        alias = model_config.get('alias', '')

        display_parts = []
        if path:
            display_parts.append(path)
        if alias and alias != model_name:
            display_parts.append(f"alias: {alias}")

        display = " | ".join(display_parts) if display_parts else "no path"

        click.echo(f"  {model_name}")
        click.echo(f"    → {display}")

    click.echo("\n  💡 Tips:")
    click.echo("     - Use the model name exactly as shown (left side)")
    click.echo("     - Use 'list-available-models' to see all models (including inactive)")
    click.echo("     - Models are shared with Semantic Affinity benchmark")
    click.echo()


@cli.command()
def list_available_models():
    """List all models in config (including inactive models)"""

    from config import MODEL_INFO

    click.echo(f"\nAll models in config ({len(MODEL_INFO)} total):")
    click.echo("Format: model-name (path/identifier) [STATUS]\n")

    # Sort by model name
    for model_name in sorted(MODEL_INFO.keys()):
        model_config = MODEL_INFO[model_name]
        path = model_config.get('path', '')
        alias = model_config.get('alias', '')
        is_active = model_config.get('is_active', True)

        # Build display string
        display_parts = []
        if path:
            display_parts.append(path)
        if alias and alias != model_name:
            display_parts.append(f"alias: {alias}")

        display = " | ".join(display_parts) if display_parts else "no path"

        status = "" if is_active else " [INACTIVE]"

        click.echo(f"  {model_name}{status}")
        click.echo(f"    → {display}")

    click.echo()


@cli.command()
def list_languages():
    """List all supported languages for RA computation"""

    click.echo("\nSupported languages for Relational Affinity analysis:")
    click.echo("Format: language-name (code) → CSV column suffix\n")

    # Sort by language name
    for lang_name in sorted(LANG_NAME_TO_CODE.keys()):
        code = LANG_NAME_TO_CODE[lang_name]
        csv_suffix = LANG_CODE_TO_CSV_SUFFIX.get(code, 'N/A')
        click.echo(f"  {lang_name.title():<12} ({code}) → word1_{csv_suffix}, word2_{csv_suffix}")

    click.echo("\n  💡 Usage:")
    click.echo("     - Use language names (e.g., --languages english,chinese,spanish)")
    click.echo("     - Default: english,chinese")
    click.echo("     - CSV datasets must have matching columns (e.g., word1_en, word2_en)")
    click.echo("     - Check dataset columns with: head -1 data/input/NeurIPS-01-family-relations-RA.csv")
    click.echo()


if __name__ == '__main__':
    cli()
