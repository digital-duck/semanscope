#!/usr/bin/env python3
"""
CLI Batch Benchmark Tool for Semantic Affinity Analysis

This tool mirrors the Streamlit SA workflow for batch/headless execution:
- Uses embedding cache (shared with Streamlit app)
- Auto-saves SA scores as JSON
- Auto-generates and saves PHATE visualizations (PNG + PDF)
- Compatible with model_manager.py architecture
- Results can be reviewed later in Streamlit app

Usage Examples:
conda activate semanscope
cd /home/papagame/projects/Proj-Geometry-of-Meaning/semanscope

# Single run
$ python cli_batch_benchmark.py run --models "LaBSE, Universal-Sentence-Encoder-Multilingual, Multilingual-E5-Large-Instruct-v2, Sentence-BERT Multilingual,  Qwen3-Embedding-0.6B, Qwen3-Embedding-4B (OpenRouter), Qwen3-Embedding-8B (OpenRouter), Gemini-Embedding-001 (OpenRouter),  OpenAI Text-Embedding-Ada-002 (OpenRouter), OpenAI Text-Embedding-3-Small (OpenRouter), OpenAI Text-Embedding-3-Large (OpenRouter),   mBERT, XLM-RoBERTa-v2" --datasets "ICML-1-peterg, ICML-2-cultural-nuance, ICML-3-challenge, ICML-4-zinets" --languages "chinese,english" --no-viz 2>&1 | tee cli_batch_benchmark-2025-12-24.txt 


# for NeurIPS dual-metric SA+RA paper: EN-ZH, all models, all datasets
##########################################################################
$ python cli_batch_benchmark_sa.py run --datasets "NeurIPS-01-family-relations,NeurIPS-02-royalty-hierarchy,NeurIPS-03-gendered-occupations,NeurIPS-04-comparative-superlative,NeurIPS-05-opposite-relations,NeurIPS-06-animal-gender,NeurIPS-07-math-numbers,NeurIPS-07-sequential-mathematical, 
NeurIPS-08-hierarchical-relations,NeurIPS-09-part-whole,NeurIPS-10-size-scale,NeurIPS-11-cause-effect"  --models "LaBSE, Universal-Sentence-Encoder-Multilingual, Multilingual-E5-Large-Instruct-v2, Sentence-BERT Multilingual,  Qwen3-Embedding-0.6B, Qwen3-Embedding-8B (OpenRouter), Gemini-Embedding-001 (OpenRouter),  OpenAI Text-Embedding-Ada-002 (OpenRouter), OpenAI Text-Embedding-3-Small (OpenRouter), OpenAI Text-Embedding-3-Large (OpenRouter),   mBERT, XLM-RoBERTa-v2, EmbeddingGemma-300M, DistilBERT Multilingual, BGE-M3 (Ollama), Snowflake-Arctic-Embed2 (Ollama), Qwen3-Embedding-4B (Ollama)"  --languages "english,chinese" --no-viz 2>&1 | tee ./batch_benchmark/cli_batch_benchmark_sa-2025-12-29-all-models-all-datasets-EN-ZH.txt

$ python cli_batch_benchmark_sa.py run --datasets "NeurIPS-01-family-relations,NeurIPS-02-royalty-hierarchy,NeurIPS-03-gendered-occupations,NeurIPS-04-comparative-superlative,NeurIPS-05-opposite-relations,NeurIPS-06-animal-gender,NeurIPS-07-math-numbers,NeurIPS-07-sequential-mathematical, 
NeurIPS-08-hierarchical-relations,NeurIPS-09-part-whole,NeurIPS-10-size-scale,NeurIPS-11-cause-effect"  --models "Voyage-3 (Voyage AI), Voyage-Multilingual-2 (Voyage AI), LaBSE, Universal-Sentence-Encoder-Multilingual, Multilingual-E5-Large-Instruct-v2, Sentence-BERT Multilingual,  Qwen3-Embedding-0.6B, Qwen3-Embedding-8B (OpenRouter), Gemini-Embedding-001 (OpenRouter),  OpenAI Text-Embedding-Ada-002 (OpenRouter), OpenAI Text-Embedding-3-Small (OpenRouter), OpenAI Text-Embedding-3-Large (OpenRouter),   mBERT, XLM-RoBERTa-v2, EmbeddingGemma-300M, DistilBERT Multilingual, BGE-M3 (Ollama), Snowflake-Arctic-Embed2 (Ollama), Qwen3-Embedding-4B (Ollama)"  --languages "english,chinese" --no-viz 2>&1 | tee ./batch_benchmark/cli_batch_benchmark_sa-2025-12-31-all-models-all-datasets-EN-ZH.txt


$ python cli_batch_benchmark_sa.py run --datasets "NeurIPS-01-family-relations,NeurIPS-02-royalty-hierarchy,NeurIPS-03-gendered-occupations,NeurIPS-04-comparative-superlative,NeurIPS-05-opposite-relations,NeurIPS-06-animal-gender,NeurIPS-07-math-numbers,NeurIPS-07-sequential-mathematical, 
NeurIPS-08-hierarchical-relations,NeurIPS-09-part-whole,NeurIPS-10-size-scale,NeurIPS-11-cause-effect"  --models "Voyage-3 (Voyage AI), Voyage-Multilingual-2 (Voyage AI), LaBSE, Universal-Sentence-Encoder-Multilingual, Multilingual-E5-Large-Instruct-v2, Sentence-BERT Multilingual,  Qwen3-Embedding-0.6B, Qwen3-Embedding-8B (OpenRouter), Gemini-Embedding-001 (OpenRouter),  OpenAI Text-Embedding-Ada-002 (OpenRouter), OpenAI Text-Embedding-3-Small (OpenRouter), OpenAI Text-Embedding-3-Large (OpenRouter),   mBERT, XLM-RoBERTa-v2, EmbeddingGemma-300M, DistilBERT Multilingual, BGE-M3 (Ollama), Snowflake-Arctic-Embed2 (Ollama), Qwen3-Embedding-4B (Ollama)"  --languages "english,spanish" --no-viz 2>&1 | tee ./batch_benchmark/cli_batch_benchmark_sa-2025-12-31-all-models-all-datasets-EN-ES.txt



cd batch_benchmark
python parse_batch_results_sa.py \
    --log-file cli_batch_benchmark_sa-2025-12-29-all-models-all-datasets-EN-ZH.txt \
    --output 2025-12-29-all-models-all-datasets-EN-ZH-sa.csv \
    --pivot \
    --pivot-cosine 2025-12-29-all-models-all-datasets-EN-ZH-sa-pivot-cos.csv \
    --pivot-euclidean 2025-12-29-all-models-all-datasets-EN-ZH-sa-pivot-euc.csv

    
# re-run with v2.5 datasets on 2026-01-02
$ python cli_batch_benchmark_sa.py run --datasets "NeurIPS-01-family-relations-v2.5,NeurIPS-02-royalty-hierarchy-v2.5, NeurIPS-03-gendered-occupations-v2.5, NeurIPS-04-comparative-superlative-v2.5,NeurIPS-05-opposite-relations-v2.5, NeurIPS-06-animal-gender-v2.5, NeurIPS-07-sequential-mathematical-v2.5, 
NeurIPS-08-hierarchical-relations-v2.5, NeurIPS-09-part-whole-v2.5, NeurIPS-10-size-scale-v2.5, NeurIPS-11-cause-effect-v2.5" --models "Voyage-3 (Voyage AI), Voyage-Multilingual-2 (Voyage AI), LaBSE, Universal-Sentence-Encoder-Multilingual, Multilingual-E5-Large-Instruct-v2, Sentence-BERT Multilingual,  Qwen3-Embedding-0.6B, Qwen3-Embedding-8B (OpenRouter), Gemini-Embedding-001 (OpenRouter),  OpenAI Text-Embedding-Ada-002 (OpenRouter), OpenAI Text-Embedding-3-Small (OpenRouter), OpenAI Text-Embedding-3-Large (OpenRouter),   mBERT, XLM-RoBERTa-v2, EmbeddingGemma-300M, DistilBERT Multilingual, BGE-M3 (Ollama), Snowflake-Arctic-Embed2 (Ollama), Qwen3-Embedding-4B (Ollama)"  --languages "english,chinese,spanish,german,turkish" --no-viz 2>&1 | tee ./batch_benchmark/cli_batch_benchmark_sa-2026-01-02-all-models-all-datasets-EN-ZH-ES-DE-TR.txt



cd batch_benchmark
python parse_batch_results_sa.py \
    --log-file cli_batch_benchmark_sa-2026-01-02-all-models-all-datasets-EN-ZH-ES-DE-TR.txt \
    --output 2026-01-02-all-models-all-datasets-EN-ZH-ES-DE-TR-sa.csv \
    --pivot \
    --pivot-cosine 2026-01-02-all-models-all-datasets-EN-ZH-ES-DE-TR-sa-pivot-cos.csv \
    --pivot-euclidean 2026-01-02-all-models-all-datasets-EN-ZH-ES-DE-TR-sa-pivot-euc.csv

Author: Claude Sonnet 4.5 (Anthropic)
Date: 2025-12-15
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
import hashlib
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_PATH, COLOR_MAP
from models.model_manager import get_model
from components.semantic_affinity import (
    SemanticAffinityAnalyzer,
    load_translation_dataset,
    expand_multi_meaning_translations
)
from utils.embedding_cache import get_embedding_cache
from components.dimension_reduction import DimensionReducer
from sklearn.preprocessing import StandardScaler


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


def round_floats(obj, decimals=3):
    """Recursively round all float values in a data structure to specified decimal places"""
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
    """
    Get embeddings for a list of words using specified model

    This function is used by the embedding cache as the callback
    when embeddings need to be computed.
    """
    if not words:
        return np.array([])

    # Load model
    model = get_model(model_name)

    # Get embeddings (model handles language internally)
    embeddings = model.get_embeddings(words, lang="en", debug_flag=False)

    if embeddings is None:
        raise ValueError(f"Model {model_name} returned None for {len(words)} words. Check model initialization errors above.")

    return np.array(embeddings)


def compute_sa_single(
    dataset_name: str,
    model_name: str,
    languages: List[str],
    metrics: List[str] = ['euclidean', 'cosine'],
    output_dir: Optional[Path] = None,
    force_recompute_embeddings: bool = False,
    save_visualizations: bool = True,
    verbose: bool = True,
    max_batch_size: int = 1000
) -> Dict[str, Any]:
    """
    Compute Semantic Affinity for a single configuration

    This function mirrors the Streamlit SA page workflow exactly:
    1. Load dataset with expansion
    2. Get embeddings (cache-aware)
    3. Compute SA for each metric
    4. Generate PHATE visualization
    5. Auto-save results + charts

    Args:
        dataset_name: Dataset name without '-SA' suffix
        model_name: Model identifier
        languages: List of language names (e.g., ['chinese', 'english'])
        metrics: Distance metrics to compute
        output_dir: Where to save results (default: data/batch_results)
        force_recompute_embeddings: Ignore cache and recompute
        save_visualizations: Generate and save PHATE charts
        verbose: Print progress messages
        max_batch_size: Maximum words per batch for memory efficiency (default: 1000)

    Returns:
        Dictionary with results, metadata, and file paths
    """

    if output_dir is None:
        output_dir = DATA_PATH / "batch_results"  # Use DATA_PATH from config (points to root/data)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        click.echo(f"\n{'='*80}")
        click.echo(f"Dataset: {dataset_name}")
        click.echo(f"Model: {model_name}")
        click.echo(f"Languages: {', '.join(languages)}")
        click.echo(f"Metrics: {', '.join(metrics)}")
        click.echo(f"{'='*80}\n")

    try:
        start_time = time.time()

        # Convert language names to codes
        language_codes = [LANG_NAME_TO_CODE[lang.lower()] for lang in languages]

        # 1. Load dataset
        if verbose:
            click.echo("📂 Loading dataset...")

        csv_path = DATA_PATH / "input" / f"{dataset_name}-SA.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {csv_path}")

        # Load original dataset
        word_translations_original, _ = load_translation_dataset(
            str(csv_path),
            languages
        )

        # Expand for cross-product
        word_translations_expanded, original_indices = expand_multi_meaning_translations(
            word_translations_original,
            language_codes
        )

        if verbose:
            click.echo(f"  Original: {len(word_translations_original)} words")
            click.echo(f"  Expanded: {len(word_translations_expanded)} word pairs\n")

        # 2. Get embeddings (cache-aware)
        if verbose:
            click.echo("🔮 Generating embeddings (cache-aware)...")

        embedding_cache = get_embedding_cache()

        embeddings_dict_expanded = {}
        embeddings_dict_original = {}
        cache_stats = {'total_cached': 0, 'total_computed': 0}

        for lang_code in language_codes:
            lang_name = CODE_TO_LANG_NAME[lang_code]

            # Get unique words for both datasets
            words_expanded = [t[lang_code] for t in word_translations_expanded if lang_code in t]

            # FIX: Split multi-meaning words in original dataset (e.g., "十|完整" → ["十", "完整"])
            # This matches the Streamlit app behavior in pages/6_📐_Semantic_Affinity.py lines 812-820
            words_original_split = []
            for trans in word_translations_original:
                if lang_code in trans:
                    word_str = str(trans[lang_code])
                    # Split by '|' to handle multi-meaning words
                    individual_words = [w.strip() for w in word_str.split('|') if w.strip()]
                    words_original_split.extend(individual_words)

            unique_words_expanded = sorted(set(words_expanded))
            unique_words_original = sorted(set(words_original_split))
            all_unique_words = sorted(set(unique_words_expanded) | set(unique_words_original))

            if verbose:
                click.echo(f"  {lang_name}: {len(all_unique_words)} unique words")

            # Get embeddings with cache
            all_embeddings, cached_count, computed_count = embedding_cache.get_embeddings(
                words=all_unique_words,
                model_name=model_name,
                lang_code=lang_code,
                embedding_func=lambda words: get_embeddings_for_words(words, model_name, lang_code),
                force_recompute=force_recompute_embeddings
            )

            cache_stats['total_cached'] += cached_count
            cache_stats['total_computed'] += computed_count

            if verbose:
                click.echo(f"    ⚡ Cached: {cached_count}, Computed: {computed_count}")

            # Build word-to-embedding index
            word_to_embedding = {word: all_embeddings[i] for i, word in enumerate(all_unique_words)}

            # Map to expanded dataset
            embeddings_expanded = np.array([word_to_embedding[word] for word in words_expanded])
            embeddings_dict_expanded[lang_code] = embeddings_expanded

            # Map to original dataset
            # FIX: Use FIRST word in multi-meaning entries (e.g., "十" from "十|完整")
            # This matches Streamlit app behavior in pages/6_📐_Semantic_Affinity.py lines 891-901
            embeddings_original_list = []
            for trans in word_translations_original:
                if lang_code in trans:
                    word_str = str(trans[lang_code])
                    # Use first word in multi-meaning (e.g., "十" from "十|完整")
                    word = word_str.split('|')[0].strip()
                    if word in word_to_embedding:
                        embeddings_original_list.append(word_to_embedding[word])
            embeddings_dict_original[lang_code] = np.array(embeddings_original_list)

        # Save cache
        embedding_cache.save_master_cache()

        if verbose:
            click.echo(f"\n  📊 Total: {cache_stats['total_cached']} cached, {cache_stats['total_computed']} computed\n")

        # 3. Compute SA for each metric
        if verbose:
            click.echo("📏 Computing Semantic Affinity...")

        analyzer = SemanticAffinityAnalyzer(max_batch_size=max_batch_size)
        results_dict = {}

        for metric in metrics:
            if verbose:
                click.echo(f"  {metric.title()}:")

            result = analyzer.compute_semantic_affinity(
                embeddings_dict=embeddings_dict_expanded,
                word_translations=word_translations_expanded,
                original_embeddings_dict=embeddings_dict_original,
                original_word_translations=word_translations_original,
                metric=metric
            )

            if verbose:
                click.echo(f"    SA Score: {result.score:.4f} ± {result.sem:.4f}")
                click.echo(f"    Inter-Spread: {result.vertical_spread:.4f} ± {result.vertical_spread_sem:.4f}")
                click.echo(f"    Intra-Spread: {result.horizontal_spread:.4f} ± {result.horizontal_spread_sem:.4f}")

            results_dict[metric] = {
                'sa_score': float(result.score),
                'sem': float(result.sem),
                'std': float(result.std),
                'inter_spread': float(result.vertical_spread),
                'inter_spread_sem': float(result.vertical_spread_sem),
                'intra_spread': float(result.horizontal_spread),
                'intra_spread_sem': float(result.horizontal_spread_sem),
                'ratio': float(result.semantic_ratio),
                'status': result.status,
                'n_words': result.n_words,
                'n_languages': result.n_languages,
            }

        # 4. Generate PHATE visualization
        viz_files = {}
        if save_visualizations:
            if verbose:
                click.echo(f"\n📊 Generating PHATE visualization...")

            try:
                # Combine all embeddings
                all_embeddings_list = []
                all_labels_list = []
                all_colors_list = []

                for lang_code in language_codes:
                    lang_name = CODE_TO_LANG_NAME[lang_code]
                    words = [t[lang_code] for t in word_translations_expanded if lang_code in t]
                    unique_words = sorted(set(words))

                    embeddings = embeddings_dict_expanded[lang_code]

                    # Get unique embeddings
                    word_to_idx = {word: i for i, word in enumerate([t[lang_code] for t in word_translations_expanded if lang_code in t])}
                    unique_embeddings = np.array([embeddings[word_to_idx[word]] for word in unique_words])

                    all_embeddings_list.append(unique_embeddings)
                    all_labels_list.extend(unique_words)

                    # Assign colors
                    color = COLOR_MAP.get(lang_code, '#999999')
                    all_colors_list.extend([color] * len(unique_words))

                combined_embeddings = np.vstack(all_embeddings_list)

                # Normalize and deduplicate
                scaler = StandardScaler()
                combined_embeddings = scaler.fit_transform(combined_embeddings)

                unique_embeddings, unique_indices = np.unique(combined_embeddings, axis=0, return_index=True)
                combined_embeddings = unique_embeddings
                all_labels_list = [all_labels_list[i] for i in sorted(unique_indices)]
                all_colors_list = [all_colors_list[i] for i in sorted(unique_indices)]

                # Run PHATE
                reducer = DimensionReducer()
                reduced_embeddings = reducer.reduce_dimensions(
                    combined_embeddings,
                    method="PHATE",
                    dimensions=2
                )

                if reduced_embeddings is not None:
                    # Save visualization using matplotlib matching Streamlit app style
                    import matplotlib.pyplot as plt
                    from matplotlib.font_manager import FontProperties
                    import matplotlib
                    import warnings

                    # Suppress font warnings for CJK characters (PDFs render correctly despite warnings)
                    warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from font.*')

                    # Configure Chinese font support to avoid glyph warnings
                    try:
                        # Try multiple CJK font options
                        matplotlib.rcParams['font.sans-serif'] = [
                            'Noto Sans CJK SC', 'Noto Sans CJK TC', 'SimHei', 'WenQuanYi Micro Hei',
                            'AR PL UMing TW MBE', 'DejaVu Sans'
                        ]
                        matplotlib.rcParams['axes.unicode_minus'] = False
                    except Exception:
                        pass  # Fallback to default fonts if CJK fonts not available

                    fig, ax = plt.subplots(figsize=(14, 10), dpi=150)

                    # Plot by language with word labels (matching Streamlit app)
                    for lang_code in language_codes:
                        lang_name = CODE_TO_LANG_NAME[lang_code]
                        color = COLOR_MAP.get(lang_code, '#999999')

                        # Get indices for this language
                        indices = [i for i, label in enumerate(all_labels_list)
                                  if all_colors_list[i] == color]

                        if indices:
                            lang_embeddings = reduced_embeddings[indices]
                            lang_labels = [all_labels_list[i] for i in indices]

                            # Scatter plot points
                            ax.scatter(lang_embeddings[:, 0], lang_embeddings[:, 1],
                                      c=color, label=lang_name, alpha=0.6, s=40, edgecolors='white', linewidths=0.5)

                            # Add word labels to each point (matching Streamlit app)
                            for j, (x, y, word) in enumerate(zip(lang_embeddings[:, 0], lang_embeddings[:, 1], lang_labels)):
                                ax.text(x, y, word, fontsize=8, alpha=0.7,
                                       ha='center', va='center',
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.15, edgecolor='none'))

                    # Build legend with SA scores (matching Streamlit app style)
                    legend_labels = []
                    for metric_name in ['euclidean', 'cosine']:
                        if metric_name in results_dict:
                            result_obj = results_dict[metric_name]
                            metric_label = "SA_eucl" if metric_name == 'euclidean' else "SA_cos"
                            # Access sa_score from dict (not object attribute)
                            sa_score = result_obj['sa_score'] if isinstance(result_obj, dict) else result_obj.score
                            sa_sem = result_obj['sem'] if isinstance(result_obj, dict) else result_obj.sem
                            legend_labels.append(f"{metric_label} = {sa_score:.4f} ± {sa_sem:.4f}")

                    # Add legend text to title area instead of overlaying on plot
                    title_text = f'PHATE: {model_name}\n{dataset_name} ({"+".join(languages)})'
                    if legend_labels:
                        title_text += '\n' + '   '.join(legend_labels)

                    ax.set_xlabel('PHATE 1', fontsize=12)
                    ax.set_ylabel('PHATE 2', fontsize=12)
                    ax.set_title(title_text, fontsize=11, pad=20)

                    # Add language legend (smaller, in corner)
                    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
                    ax.grid(True, alpha=0.3)

                    # Save PNG and PDF
                    lang_codes_str = '-'.join([LANG_NAME_TO_CODE[l.lower()] for l in languages])
                    safe_model_name = model_name.replace('/', '_').replace(' ', '_')
                    safe_dataset_name = dataset_name.replace('/', '_').replace(' ', '_')

                    base_filename = f"phate-{safe_dataset_name}-{safe_model_name}-{lang_codes_str}"

                    png_path = output_dir / f"{base_filename}.png"
                    pdf_path = output_dir / f"{base_filename}.pdf"

                    fig.savefig(png_path, dpi=150, bbox_inches='tight')
                    fig.savefig(pdf_path, bbox_inches='tight')
                    plt.close(fig)

                    viz_files['png'] = str(png_path.absolute())
                    viz_files['pdf'] = str(pdf_path.absolute())

                    if verbose:
                        click.echo(f"  ✅ PNG: {png_path}")
                        click.echo(f"  ✅ PDF: {pdf_path}")

            except Exception as e:
                if verbose:
                    click.echo(f"  ⚠️  Visualization failed: {str(e)}")
                viz_files['error'] = str(e)

        # 5. Save results as JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = model_name.replace('/', '_').replace(' ', '_')
        safe_dataset_name = dataset_name.replace('/', '_').replace(' ', '_')
        lang_codes_str = '-'.join([LANG_NAME_TO_CODE[l.lower()] for l in languages])

        json_filename = f"sa-{safe_dataset_name}-{safe_model_name}-{lang_codes_str}-{timestamp}.json"
        json_path = output_dir / json_filename

        output_data = {
            'metadata': {
                'dataset': dataset_name,
                'model': model_name,
                'languages': languages,
                'language_codes': language_codes,
                'metrics': metrics,
                'n_words_original': len(word_translations_original),
                'n_words_expanded': len(word_translations_expanded),
                'timestamp': timestamp,
                'datetime': datetime.now().isoformat(),
                'cache_stats': cache_stats,
                'execution_time_seconds': time.time() - start_time
            },
            'results': results_dict,
            'visualizations': viz_files
        }

        # Round all floats to 4 decimal places for cleaner output
        output_data = round_floats(output_data, decimals=4)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        if verbose:
            click.echo(f"\n💾 Results saved: {json_path}")
            click.echo(f"⏱️  Total time: {output_data['metadata']['execution_time_seconds']:.2f}s\n")

        return output_data

    except Exception as e:
        if verbose:
            click.echo(f"\n❌ Error: {str(e)}", err=True)
        import traceback
        traceback.print_exc()
        return None


@click.group()
def cli():
    """CLI Batch Benchmark Tool for Semantic Affinity Analysis"""
    pass


@cli.command()
@click.option('--datasets', required=True, help='Comma-separated dataset names (e.g., ICML-1-peterg,ICML-2-cultural)')
@click.option('--models', required=True, help='Comma-separated model names (e.g., LaBSE,Qwen3-Embedding-0.6B)')
@click.option('--languages', required=True, help='Comma-separated languages (e.g., chinese,english,spanish)')
@click.option('--metrics', multiple=True, default=['euclidean', 'cosine'], help='Distance metrics')
@click.option('--output-dir', type=click.Path(), default=None, help='Output directory (default: <project_root>/data/batch_results)')
@click.option('--force-recompute', is_flag=True, help='Ignore embedding cache')
@click.option('--no-viz', is_flag=True, help='Skip PHATE visualization')
@click.option('--continue-on-error', is_flag=True, help='Continue processing other models/datasets when an error occurs (default: stop on first error)')
def run(datasets, models, languages, metrics, output_dir, force_recompute, no_viz, continue_on_error):
    """Run SA computation with flexible model/dataset/language combinations

    This command runs a cross-product of models × datasets, with pairwise language combinations.

    Examples:
        # Single model, single dataset, single language pair
        --models LaBSE --datasets ICML-1-peterg --languages chinese,english

        # Multiple models on one dataset
        --models "LaBSE,Qwen3-Embedding-0.6B" --datasets ICML-1-peterg --languages chinese,english

        # One model on multiple datasets with multiple language pairs
        --models LaBSE --datasets "ICML-1-peterg,ICML-2-cultural" --languages "chinese,english,spanish"
        (runs: 1 model × 2 datasets × 3 language pairs = 6 computations)
    """
    from itertools import combinations

    # Parse comma-separated inputs (strip whitespace and filter empty strings)
    model_list = [m.strip() for m in models.strip().split(',') if m.strip()]
    dataset_list = [d.strip() for d in datasets.strip().split(',') if d.strip()]
    lang_list = [l.strip() for l in languages.strip().split(',') if l.strip()]

    # Generate pairwise language combinations
    if len(lang_list) < 2:
        click.echo("❌ Error: Need at least 2 languages for pairwise comparison", err=True)
        sys.exit(1)

    lang_pairs = list(combinations(lang_list, 2))

    # Calculate total runs
    total_runs = len(model_list) * len(dataset_list) * len(lang_pairs)

    click.echo(f"\n{'='*80}")
    click.echo(f"🚀 Batch Run Configuration")
    click.echo(f"{'='*80}")
    click.echo(f"Models ({len(model_list)}): {', '.join(model_list)}")
    click.echo(f"Datasets ({len(dataset_list)}): {', '.join(dataset_list)}")
    click.echo(f"Languages ({len(lang_list)}): {', '.join(lang_list)}")
    click.echo(f"Language pairs ({len(lang_pairs)}): {', '.join(['-'.join(pair) for pair in lang_pairs])}")
    click.echo(f"Total runs: {len(model_list)} models × {len(dataset_list)} datasets × {len(lang_pairs)} lang pairs = {total_runs}")
    click.echo(f"{'='*80}\n")

    results = []
    completed = 0

    # Outer loop: models × datasets
    for model_name in model_list:
        for dataset_name in dataset_list:
            # Inner loop: pairwise language combinations
            for lang_pair in lang_pairs:
                completed += 1

                click.echo(f"\n{'='*80}")
                click.echo(f"[{completed}/{total_runs}] Progress: {completed/total_runs*100:.1f}%")
                click.echo(f"Model: {model_name}")
                click.echo(f"Dataset: {dataset_name}")
                click.echo(f"Languages: {' + '.join(lang_pair)}")
                click.echo(f"{'='*80}")

                result = compute_sa_single(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    languages=list(lang_pair),
                    metrics=list(metrics),
                    output_dir=Path(output_dir) if output_dir else None,
                    force_recompute_embeddings=force_recompute,
                    save_visualizations=not no_viz,
                    verbose=True
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
    """List all available datasets"""

    input_dir = DATA_PATH / "input"

    if not input_dir.exists():
        click.echo(f"Error: Input directory not found: {input_dir}", err=True)
        sys.exit(1)

    # Find all *-SA.csv files
    dataset_files = sorted(input_dir.glob("*-SA.csv"))

    if not dataset_files:
        click.echo("No datasets found in data/input/", err=True)
        sys.exit(1)

    click.echo("\nAvailable datasets:")
    for csv_file in dataset_files:
        # Remove -SA.csv suffix to get dataset name
        dataset_name = csv_file.stem.replace("-SA", "")

        # Count words in dataset
        try:
            df = pd.read_csv(csv_file, comment='#')
            n_words = len(df)
            click.echo(f"  - {dataset_name} ({n_words} words)")
        except Exception as e:
            click.echo(f"  - {dataset_name} (error reading file: {e})")

    click.echo()


@cli.command()
def list_models():
    """List active models (matching Streamlit app dropdown)"""

    # Import MODEL_INFO from config
    from config import MODEL_INFO

    # Filter only active models (same as Streamlit app)
    active_models = {name: info for name, info in MODEL_INFO.items()
                     if info.get('is_active', True)}

    click.echo(f"\nActive models ({len(active_models)} total):")
    click.echo("Format: model-name (path/identifier)\n")

    # Sort by model name
    for model_name in sorted(active_models.keys()):
        model_config = active_models[model_name]
        path = model_config.get('path', '')
        alias = model_config.get('alias', '')

        # Build display string
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
    click.echo()


@cli.command()
def list_available_models():
    """List all models in config (including inactive models)"""

    # Import MODEL_INFO from config
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
    """List all supported languages"""

    click.echo("\nSupported languages:")
    click.echo("\nLanguage codes accepted by --languages parameter:")
    click.echo("  (Use comma-separated, e.g., --languages chinese,english,spanish)\n")

    # Display in a nice table format
    for lang_name, lang_code in sorted(LANG_NAME_TO_CODE.items()):
        click.echo(f"  {lang_name.ljust(12)} → {lang_code}")

    click.echo(f"\n  Total: {len(LANG_NAME_TO_CODE)} languages supported")

    # Check which languages are available in datasets
    input_dir = DATA_PATH / "input"
    if input_dir.exists():
        # Find unique language codes from dataset files
        dataset_langs = set()
        for csv_file in input_dir.glob("*-SA.csv"):
            try:
                df = pd.read_csv(csv_file, comment='#', nrows=0)  # Just read headers
                for col in df.columns:
                    if col.lower() in LANG_NAME_TO_CODE:
                        dataset_langs.add(col.lower())
            except Exception:
                pass

        if dataset_langs:
            click.echo(f"\n  Languages found in datasets: {len(dataset_langs)}")
            click.echo("  " + ", ".join(sorted(dataset_langs)))

    click.echo()


if __name__ == '__main__':
    cli()
