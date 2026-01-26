"""
Semantic Affinity Metric - Benchmark Embedding Models

This page implements the Semantic Affinity (SA) metric for benchmarking
multilingual embedding models using translation datasets.

Reference: docs/semantic-affinity.md

Key Concept:
Semantic Affinity (SA) = 1 / (1 + SR), where SR = vertical_spread / horizontal_spread
- SA ∈ (0, 1]: Bounded, normalized metric
- Higher SA = stronger cross-lingual alignment
- SA = 1.0: Perfect alignment (SR=0)
- SA = 0.5: Neutral (SR=1, translations at vocabulary baseline)
- SA > 0.5: Strong affinity (SR<1, translations closer than baseline)
- SA < 0.5: Weak affinity (SR>1, translations farther than baseline)
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import hashlib
from multiprocessing import cpu_count
import time
import traceback
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler


from semanscope.config import (
    ST_APP_NAME, ST_ICON, DATA_PATH, MODEL_INFO, COLOR_MAP,
    get_language_code_from_name, get_model_language_code,
    get_active_models_by_architecture, get_active_models_with_headers,
    DEFAULT_MODEL, DEFAULT_DATASET, DEFAULT_METHOD
)
from semanscope.components.semantic_affinity import (
    SemanticAffinityAnalyzer,
    load_translation_dataset,
    expand_multi_meaning_translations
)
from semanscope.components.embedding_viz import get_active_models
from semanscope.components.dimension_reduction import DimensionReducer
from semanscope.components.plotting import PlotManager
from semanscope.components.plotting_echarts import EChartsPlotManager
from semanscope.models.model_manager import get_model
from semanscope.utils.embedding_cache import get_embedding_cache
from semanscope.utils.global_settings import get_global_viz_settings


# ============================================================================
# Dataset Loading Helper Functions
# ============================================================================

def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize dataframe column names to lowercase for robust dataset handling.

    This makes the code case-insensitive - works with both:
    - "English,Chinese,German" (capitalized from old datasets)
    - "english,chinese,german" (lowercase from new merge_langs_to_sa.py)

    Args:
        df: DataFrame with potentially mixed-case column names

    Returns:
        DataFrame with all column names converted to lowercase
    """
    df.columns = [col.lower() for col in df.columns]
    return df


def load_dataset_robust(dataset_path: Path) -> Tuple[pd.DataFrame, bool, str]:
    """
    Robustly load dataset CSV with automatic column name normalization.

    Args:
        dataset_path: Path to CSV file

    Returns:
        Tuple of (dataframe, success, error_message)
    """
    try:
        # Load CSV with comment character support
        df = pd.read_csv(dataset_path, comment='#')

        # Normalize column names to lowercase
        df = normalize_dataframe_columns(df)

        return df, True, ""

    except FileNotFoundError:
        return None, False, f"Dataset file not found: {dataset_path}"
    except pd.errors.EmptyDataError:
        return None, False, "Dataset file is empty"
    except Exception as e:
        return None, False, f"Error loading dataset: {str(e)}"


# Page config
st.set_page_config(
    page_title="Semantic Affinity Benchmark",
    page_icon="📐",
    layout="wide"
)

# Language name to code mapping (for internal use)
LANG_NAME_TO_CODE = {
    'chinese': 'chn',
    'english': 'enu',
    'spanish': 'spa',
    'french': 'fra',
    'german': 'deu',
    'russian': 'rus',
    'korean': 'kor',
    'arabic': 'ara',
    'japanese': 'jpn',
    'vietnamese': 'vie',
    'thai': 'tha',
    'hebrew': 'heb',
    'hindi': 'hin',
    'greek': 'grk',
    'persian': 'fas',
    'turkish': 'tur',
    'georgian': 'kat',
    'armenian': 'hye'
}

# Reverse mapping for code to name
CODE_TO_LANG_NAME = {v: k.title() for k, v in LANG_NAME_TO_CODE.items()}


def split_icml_dataset_by_lang(dataset_name: str):
    """
    Split a multi-language ICML dataset into separate language-specific CSV files,
    to be used for PHATE visualization or other language-specific analyses.

    Args:
        dataset_name: Base name of the dataset (e.g., 'ICML-1-zinets-translated')

    Features:
    - Expands multi-meaning words separated by '|' into individual entries
      Example: "ten|complete" → two separate rows: "ten" and "complete"
    - Preserves metadata (category, difficulty) for each expanded word
    - Skips empty/NaN entries automatically

    Output:
    - Filename convention: '{dataset_name}-{lang_code}.txt'
    - Files saved to: data/input/{dataset_name}-{lang_code}.txt
    - File format: 'word,domain,type,note' 
    - domain = category (for color-coding in visualizations)
    - type = difficulty or cultural_sensitivity
    - leave 'domain,type,note' empty for now

    Returns:
        None (saves files to data/input/)
    """
    try:
        # Construct input file path
        input_path = DATA_PATH / "input" / f"{dataset_name}-SA.csv"

        if not input_path.exists():
            st.error(f"Dataset file not found: {input_path}")
            return

        # Read the CSV file with robust column name normalization
        df, success, error = load_dataset_robust(input_path)
        if not success:
            st.error(error)
            return

        # Get language columns (exclude 'category', 'difficulty', 'cultural_sensitivity', etc.)
        metadata_columns = ['category', 'difficulty', 'cultural_sensitivity', 'type', 'domain', 'note']
        language_columns = [col for col in df.columns if col.lower() not in metadata_columns]

        # Determine if we have metadata columns
        has_category = 'category' in df.columns
        has_difficulty = 'difficulty' in df.columns
        has_cultural = 'cultural_sensitivity' in df.columns

        # Split and save each language
        files_created = []
        expansion_stats = {}  # Track expansion for statistics

        for lang_name in language_columns:
            # Map language name to code
            lang_code = LANG_NAME_TO_CODE.get(lang_name.lower())
            if not lang_code:
                continue

            # Create output filename
            output_filename = f"{dataset_name}-{lang_code}.txt"
            output_path = DATA_PATH / "input" / output_filename

            # Prepare output data
            output_lines = []

            # Counters for statistics
            original_rows = 0
            expanded_rows = 0

            # Add header (always include all columns)
            output_lines.append("word,domain,type,note")

            # Add data rows
            for idx, row in df.iterrows():
                word_str = str(row[lang_name]).strip()

                # Skip empty words
                if not word_str or word_str == 'nan':
                    continue

                original_rows += 1

                # Expand words containing "|" into multiple entries
                individual_words = [w.strip() for w in word_str.split('|') if w.strip()]
                expanded_rows += len(individual_words)

                # Process each individual word
                for word in individual_words:
                    # Leave domain, type, note empty for now
                    output_lines.append(f"{word},,,")

            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(output_lines))

            files_created.append(output_filename)
            expansion_stats[lang_code] = {
                'original': original_rows,
                'expanded': expanded_rows,
                'filename': output_filename
            }

        # Show success message with expansion statistics
        if files_created:
            st.success(f"✅ Created {len(files_created)} language files:")

            # Show statistics in a more compact format
            total_original = sum(stats['original'] for stats in expansion_stats.values())
            total_expanded = sum(stats['expanded'] for stats in expansion_stats.values())

            if total_expanded > total_original:
                expansion_pct = ((total_expanded - total_original) / total_original) * 100
                st.info(f"📊 Expanded {total_original} rows → {total_expanded} rows (+{expansion_pct:.1f}% from '|' separation)")

            # Show file list
            for lang_code, stats in expansion_stats.items():
                if stats['expanded'] > stats['original']:
                    st.text(f"  • {stats['filename']} ({stats['original']} → {stats['expanded']} words)")
                else:
                    st.text(f"  • {stats['filename']} ({stats['expanded']} words)")
        else:
            st.warning("No language files were created. Check dataset format.")

    except Exception as e:
        st.error(f"Error splitting dataset: {str(e)}")
        
        if st.session_state.get('debug_mode', False):
            st.code(traceback.format_exc())


def discover_sa_datasets() -> List[str]:
    """Discover all *-SA.csv files in the input directory"""
    input_dir = DATA_PATH / "input"
    if not input_dir.exists():
        return []

    sa_files = list(input_dir.glob("*-SA.csv"))
    # Return base names without path, extension, and '-SA' suffix
    return [f.stem.replace('-SA', '') for f in sorted(sa_files)]


def load_and_validate_dataset(dataset_name: str) -> Tuple[pd.DataFrame, List[str], bool, str]:
    """
    Load dataset and validate it has at least 2 language columns

    Lines starting with '#' are treated as comments and skipped.

    Args:
        dataset_name: Name without '-SA' suffix (e.g., 'ICML-control')

    Returns:
        (dataframe, available_languages, is_valid, error_message)
    """
    # Add '-SA' suffix back for file loading
    dataset_path = DATA_PATH / "input" / f"{dataset_name}-SA.csv"

    if not dataset_path.exists():
        return None, [], False, f"Dataset file not found: {dataset_path}"

    try:
        # Load CSV with robust column name normalization
        df, success, error = load_dataset_robust(dataset_path)
        if not success:
            return None, [], False, error

        # Get column names (these should be language names)
        columns = list(df.columns)

        # Filter for recognized language columns
        available_languages = []
        for col in columns:
            col_lower = col.lower()
            if col_lower in LANG_NAME_TO_CODE:
                available_languages.append(col.title())

        # Validate at least 2 languages
        if len(available_languages) < 2:
            return df, available_languages, False, f"Dataset must have at least 2 language columns. Found: {available_languages}"

        return df, available_languages, True, ""

    except Exception as e:
        return None, [], False, f"Error loading dataset: {str(e)}"


def render_sidebar():
    """Render sidebar with controls"""
    with st.sidebar:
        st.subheader("⚙️ Settings")

        # ==========================================
        # Distance metric selection
        # st.markdown("### 📏 Distance Metric")
        c0,c1,c2=st.columns([2,3,3])
        with c0:
            st.markdown("Methods")
        with c1:
            cosine_enabled = st.checkbox(
                "Cosine",
                value=True,
                help="Cosine: 1 - cos(θ) - angle-based, magnitude-invariant (PRIMARY)"
            )
        with c2:
            euclidean_enabled = st.checkbox(
                "Euclidean",
                value=True,
                help="Euclidean: √(Σ(x-y)²) - scale-dependent (SUPPLEMENTARY)"
            )

        # Collect enabled metrics (COSINE FIRST)
        selected_metrics = []
        if cosine_enabled:
            selected_metrics.append('cosine')
        if euclidean_enabled:
            selected_metrics.append('euclidean')

        # ==========================================
        # Model selection (organized by architecture: BERT, Hybrid, LLM with visual dividers)
        # st.markdown("### 🤖 Embedding Model")
        model_names_with_headers, model_info_dict = get_active_models_with_headers()

        # Use default from global settings (config.py)
        default_index = model_names_with_headers.index(DEFAULT_MODEL) if DEFAULT_MODEL in model_names_with_headers else 1  # Skip first header

        model_name = st.selectbox(
            "Select embedding model",
            options=model_names_with_headers,
            index=default_index,
            help="Models organized by architecture: BERT-based → Hybrid → LLM-based"
        )

        # Check if user selected a header (disabled option)
        if model_name.startswith("━━━━━━"):
            st.warning("⚠️ Please select a model, not a group header")
            return None, [], None, [], None, None, None, False, False, None

        # Store current model in session state for cache management
        st.session_state['selected_model'] = model_name

        # Show model info
        if model_name in MODEL_INFO:
            with st.expander("ℹ️ Model Info"):
                st.info(MODEL_INFO[model_name]['help'])

                # Indicate if model is multilingual
                if "multilingual" in MODEL_INFO[model_name]['help'].lower():
                    st.success("✓ Multilingual model - no language parameter needed")


        # ==========================================
        # st.markdown("### 📂 Datasets")
        available_datasets = discover_sa_datasets()

        if not available_datasets:
            st.error("No *-SA.csv datasets found in data/input/")
            return None, [], None, None, None, None, False

        # Use default from global settings (config.py)
        dataset_name = st.selectbox(
            "Select dataset",
            options=available_datasets,
            index=available_datasets.index(DEFAULT_DATASET) if DEFAULT_DATASET in available_datasets else 0,
            help="Datasets with pattern *-SA.csv containing translation pairs"
        )

        # Load and validate dataset
        df, available_languages, is_valid, error_msg = load_and_validate_dataset(dataset_name)

        if not is_valid:
            st.error(error_msg)
            return None, [], None, None, None, False

        with st.expander("ℹ️ Dataset Info", expanded=False):
            # Dataset selection
            c_1, c_2 = st.columns([2, 2])
            with c_1:
                # Refresh button to reload datasets
                if st.button("🔄 Refresh"):
                    st.rerun()
            with c_2:
                # convert ICML dataset to multi-language dataset for PHATE visualization
                if st.button("✂️ Split by lang"):
                    split_icml_dataset_by_lang(dataset_name)
                    st.rerun()
            st.info(f"Available Datasets: {available_datasets}")
            st.success(f"✓ {len(available_languages)} languages: {', '.join(available_languages)}")

        # ==========================================
        # Language selection
        if len(available_languages) > 1:
            default_langs = available_languages[:2]
        # elif len(available_languages) == 2:
        #     default_langs = available_languages[:2]
        else:
            default_langs = available_languages
        # st.markdown("### 🌐 Languages")
        selected_languages = st.multiselect(
            "Select target languages",
            options=available_languages,
            default=default_langs,
            help="Choose 2 or more languages for cross-lingual analysis"
        )



        # ==========================================
        # Word Search feature
        with st.expander("🔍 Word Search", expanded=False):
            st.markdown("Search and highlight specific words in the PHATE visualization")

            enable_highlight = st.checkbox("Enable Word Search", value=False,
                                          help="Search and highlight specific words with custom colors and sizes")

            if enable_highlight:
                keywords_input = st.text_area(
                    "Search Words (one per line)",
                    placeholder="一\n二\n三",
                    height=100,
                    help="Enter words to search and highlight, one per line"
                )

                keywords = [kw.strip() for kw in keywords_input.strip().split('\n') if kw.strip()] if keywords_input.strip() else []

                col1, col2 = st.columns(2)
                with col1:
                    highlight_color = st.color_picker("Highlight Color", "#FFD700",
                                                      help="Color for highlighted search results")
                with col2:
                    highlight_size = st.slider("Point Size", 10, 50, 20,
                                              help="Point size for highlighted search results")

                highlight_font_size = st.slider("Label Font Size", 10, 30, 16,
                                               help="Font size for highlighted labels")

                highlight_config = {
                    'enabled': True,
                    'keywords': keywords,
                    'color': highlight_color,
                    'size': highlight_size,
                    'font_size': highlight_font_size
                } if keywords else None
            else:
                highlight_config = None


        # ==========================================
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            # Debug mode toggle
            st.markdown("#### 🐛 Debug Mode")
            debug_mode = st.checkbox(
                "Show debug messages",
                value=False,
                help="Show detailed validation and debugging information"
            )
            st.session_state['debug_mode'] = debug_mode

            collapse_epsilon = st.number_input(
                "Collapse detection threshold",
                min_value=1e-10,
                max_value=1e-3,
                value=1e-6,
                format="%.2e",
                help="Models with max distance below this are flagged as COLLAPSED"
            )

            # Parallel processing option
            n_cpus = cpu_count()

            use_parallel = st.checkbox(
                f"⚡ Enable parallel processing ({n_cpus} CPUs available)",
                value=True,
                help=f"Parallelize per-word spread calculations across {n_cpus} CPU cores for faster computation"
            )

            if use_parallel:
                n_jobs = st.slider(
                    "Number of parallel jobs",
                    min_value=-1,
                    max_value=n_cpus,
                    value=(n_cpus-1) if n_cpus > 2 else n_cpus,
                    help="Number of CPU cores to use (-1 = all available)"
                )
            else:
                n_jobs = 1
            # special case: -1 means use all available CPUs
            n_jobs = n_cpus if n_jobs == -1 else n_jobs

            # Batch size for memory efficiency
            max_batch_size = st.number_input(
                "Max batch size (words per batch)",
                min_value=200,
                max_value=10000,
                value=1000,
                step=200,
                help="Maximum number of words to process in one batch. Lower values use less memory but may be slower. Increase if you have more RAM available."
            )



            # API batch size setting
            st.markdown("#### 🌐 API Settings")
            api_batch_size = st.number_input(
                "OpenRouter batch size",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="Number of words per API batch. Larger = faster but may timeout. Recommended: 50-200"
            )
            st.session_state['api_batch_size'] = api_batch_size

            # Embedding cache options
            st.markdown("#### 💾 Embedding Cache")

            force_embedding = st.checkbox(
                "🔄 Force recompute embeddings",
                value=False,
                help="Ignore cached embeddings and recompute from scratch (useful for testing or if cache is corrupted)"
            )

            # Store in session state for use during computation
            st.session_state['force_embedding_recompute'] = force_embedding

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear current model",type="primary"):
                    # Get current model from main page
                    current_model = st.session_state.get('selected_model', model_name if 'model_name' in locals() else None)
                    if current_model:
                        cache = get_embedding_cache()
                        cache.clear_cache(model_name=current_model)
                        st.success(f"Cache cleared for: {current_model}")
                    else:
                        st.warning("No model selected")

            with col2:
                if st.button("🗑️ Clear ALL cache"):
                    cache = get_embedding_cache()
                    cache.clear_cache()
                    st.success("All embedding cache cleared!")




        # Validation
        if not selected_metrics:
            st.warning("⚠️ Please select at least one distance metric")
            return

        # ==========================================
        c_compute_btn, c_cache_control = st.columns([3,3])
        with c_compute_btn:
            # Compute button
            compute_button = st.button("Compute SA", type="primary")
        with c_cache_control:
            use_cache = st.checkbox(
                "Caching?",
                value=True,
                help="Cache results to avoid recomputation with same configuration"
            )

        # # Demo: Hello button
        # if st.button("👋 Hello", use_container_width=True):
        #     st.success("👋 Hello! Welcome to Semantic Affinity Benchmark!")
        #     st.balloons()

        st.markdown("---")

        # Instructions
        with st.expander("ℹ️ How to Use", expanded=False):
            st.markdown("""
            **How it works:**
            1. Select a dataset with translation pairs (`*-SA.csv` format)
            2. Choose 2+ target languages from the dataset
            3. Select an embedding model to benchmark
            4. Compute SA score: **higher = stronger cross-lingual affinity**

            **Interpretation (SA ∈ [0, 1]):**
            - **SA > 0.67**: Excellent alignment (SR < 0.5, translations much closer than baseline) ✅
            - **SA ≈ 0.55**: Good alignment (SR ≈ 0.8, translations reasonably close)
            - **SA = 0.50**: Neutral (SR = 1.0, translations at vocabulary baseline)
            - **SA < 0.40**: Poor alignment (SR > 1.5, translations farther than baseline) ❌
            - **SA → 0**: Collapsed model (SR → ∞, all embeddings identical)

            **Formula:** SA = 1 / (1 + SR), where SR = vertical_spread / horizontal_spread

            ---

            ### 🚀 CLI Batch Tool for Large-Scale Benchmarking

            For systematic benchmarking (multiple models × multiple datasets), use the CLI batch tool:

            ```bash
            # Navigate to project directory
            cd src

            # Single model-dataset pair
            python cli_batch_benchmark.py run \\
                --dataset ICML-zinets-translated \\
                --model LaBSE \\
                --languages "chinese english spanish"

            # Test one model across all datasets
            python cli_batch_benchmark.py batch-datasets \\
                --model LaBSE \\
                --languages "chinese english spanish french german russian korean arabic"

            # Full grid: all models × all datasets (overnight run)
            python cli_batch_benchmark.py grid \\
                --languages "chinese english spanish"
            ```

            **Features:**
            - ✅ Auto-saves SA scores as JSON + metadata
            - ✅ Auto-generates PHATE visualizations (PNG + PDF)
            - ✅ Shared embedding cache (faster re-runs)
            - ✅ Parallel processing across CPU cores
            - ✅ Memory-efficient batch processing

            **Output:** Results saved to `data/batch_results/` with filenames like:
            `SA-{dataset}-{model}-{langs}-{metric}-{timestamp}.json`

            See `BATCH_BENCHMARK_README.md` for full documentation.
            """)

        # Cache management
        with st.expander("💾 Cache Management", expanded=False):
            if 'sa_cache' in st.session_state and st.session_state.sa_cache:
                cache_count = len(st.session_state.sa_cache)
                st.info(f"📦 {cache_count} cached result(s)")

                # Show cache entries
                for idx, (key, data) in enumerate(st.session_state.sa_cache.items(), 1):
                    config = data.get('config', {})
                    st.markdown(f"**Cache {idx}:**")
                    st.text(f"Dataset: {config.get('dataset', 'N/A')}")
                    st.text(f"Languages: {', '.join(config.get('languages', []))}")
                    st.text(f"Model: {config.get('model', 'N/A')[:30]}...")
                    st.text(f"Metric: {config.get('metric', 'N/A')}")
                    st.markdown("---")

                if st.button("🗑️ Clear All Cache"):
                    st.session_state.sa_cache = {}
                    st.success("Cache cleared!")
                    st.rerun()
            else:
                st.info("No cached results yet")

    return dataset_name, selected_languages, model_name, selected_metrics, collapse_epsilon, n_jobs, max_batch_size, use_cache, compute_button, highlight_config


def get_embeddings_for_words(
    words: List[str],
    model_name: str,
    lang_code: str = None
) -> np.ndarray:
    """
    Get embeddings for a list of words using the specified model

    For multilingual models, the lang parameter is optional and often ignored.
    We pass a generic language code for compatibility, but most modern multilingual
    models auto-detect language or use a shared multilingual space.

    Args:
        words: List of words/phrases to embed
        model_name: Name of the embedding model
        lang_code: Optional language code (for models that need it)

    Returns:
        Array of embeddings with shape (n_words, embedding_dim)
    """
    try:
        model = get_model(model_name)

        # For multilingual models, we typically don't need to specify language
        # The model.get_embeddings() method accepts lang parameter but may ignore it
        # We pass "en" as a safe default, but modern models handle all languages
        embeddings = model.get_embeddings(words, lang="en", debug_flag=False)

        # Return embeddings (validation happens upstream after caching)
        return np.array(embeddings)
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        return None


def main():
    st.markdown("### 📐 Semantic Affinity Benchmark")

    # Brief introduction
    st.markdown("""
    Benchmark multilingual embedding models using the **Semantic Affinity (SA)** metric.
    Higher SA = stronger cross-lingual alignment (SA ∈ [0, 1]). See sidebar for usage instructions.
    """)

    # Render sidebar and get settings
    result = render_sidebar()
    if result is None or result[0] is None:
        st.warning("⚠️ Please select a valid dataset with at least 2 language columns")
        return

    dataset_name, selected_languages, model_name, selected_metrics, collapse_epsilon, n_jobs, max_batch_size, use_cache, compute_button, highlight_config = result

    # Validation
    if len(selected_languages) < 2:
        st.warning("⚠️ Please select at least 2 languages for cross-lingual analysis")
        return

    # Load dataset for display with robust column name normalization
    dataset_path = DATA_PATH / "input" / f"{dataset_name}-SA.csv"
    df, success, error = load_dataset_robust(dataset_path)
    if not success:
        st.error(error)
        return

    c1, c2 = st.columns([2, 2])
    # Main panel - show dataset info and configuration in expanders
    with c1:
        with st.expander("📊 Dataset Information", expanded=False):
            st.markdown(f"**Dataset**: `{dataset_name}-SA.csv` ({len(df)} words)")

            # Show selected language columns
            selected_cols = [lang.lower() for lang in selected_languages]
            display_df = df[selected_cols].head(10)
            st.dataframe(display_df, width='stretch', height=300)

            if st.checkbox("📋 View all rows", key="view_all_dataset"):
                st.dataframe(df[selected_cols], width='stretch', height=400)

    with c2:
        with st.expander("⚙️ Configuration", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Languages", len(selected_languages))
            with col2:
                st.metric("Total Words", len(df))
            with col3:
                st.metric("Model", model_name.split()[0] if ' ' in model_name else model_name[:15])
            with col4:
                metrics_str = " + ".join([m.title() for m in selected_metrics])
                st.metric("Metrics", metrics_str)

            st.markdown("**Selected Languages:**")
            st.write(", ".join(selected_languages))
            st.markdown("**Model:**")
            st.write(model_name)

    # Compute Semantic Affinity when button is clicked
    msg_dict = {}
    if compute_button:
        # Create cache key from configuration

        cache_key_data = {
            'dataset': dataset_name,
            'languages': sorted(selected_languages),  # Sorted for consistency
            'model': model_name,
            'metrics': sorted(selected_metrics),  # Include all selected metrics
            'collapse_epsilon': collapse_epsilon
        }
        cache_key = hashlib.md5(str(cache_key_data).encode()).hexdigest()

        # Initialize cache in session state
        if 'sa_cache' not in st.session_state:
            st.session_state.sa_cache = {}

        # Check cache (only if caching is enabled)
        cache_hit = False
        if use_cache and cache_key in st.session_state.sa_cache:
            st.success("⚡ Using cached results (same configuration)")
            cached_data = st.session_state.sa_cache[cache_key]
            results_dict = cached_data['results_dict']  # Changed to dict for multiple metrics
            embeddings_dict = cached_data['embeddings_dict']
            word_translations = cached_data['word_translations']
            language_codes = cached_data['language_codes']
            cache_hit = True

        if not cache_hit:
            with st.spinner("🔄 Computing Semantic Affinity..."):

                # Load translation dataset
                try:
                    word_translations_original, language_codes = load_translation_dataset(
                        str(dataset_path),
                        selected_languages
                    )

                    original_word_count = len(word_translations_original)

                    # Expand multi-meaning translations (cross-product approach)
                    # Example: {'chn': '十|完整', 'enu': 'ten|complete'} becomes 4 pairs:
                    #   (十, ten), (十, complete), (完整, ten), (完整, complete)
                    word_translations_expanded, original_indices = expand_multi_meaning_translations(
                        word_translations_original, language_codes
                    )

                    expansion_ratio = len(word_translations_expanded) / original_word_count if original_word_count > 0 else 1.0
                    if len(word_translations_expanded) > original_word_count:
                        st.info(f"🔄 Expanded multi-meaning words: {original_word_count} → {len(word_translations_expanded)} word pairs (×{expansion_ratio:.2f})")

                    # Keep both original and expanded for computation
                    word_translations = word_translations_expanded  # For expanded embeddings
                    word_translations_for_horizontal = word_translations_original  # For horizontal spread

                    # st.info(f"Loaded {len(word_translations)} words across {len(language_codes)} languages")
                    msg_dict["summary"] = f"Loaded {len(word_translations)} word pairs across {len(language_codes)} languages"

                except Exception as e:
                    st.error(f"Error loading dataset: {str(e)}")
                    return

            # Generate embeddings for BOTH expanded and original datasets
            embeddings_dict_expanded = {}
            embeddings_dict_original = {}


            total_embedding_start = time.time()

            # Get global embedding cache (shared across all Semanscope pages)
            embedding_cache = get_embedding_cache()

            # Check for force recompute flag in advanced settings
            force_recompute = st.session_state.get('force_embedding_recompute', False)

            progress_bar = st.progress(0)

            for lang_idx, lang_name in enumerate(selected_languages):
                lang_code = LANG_NAME_TO_CODE.get(lang_name.lower())
                if not lang_code:
                    st.warning(f"Unknown language: {lang_name}, skipping...")
                    continue

                # Extract UNIQUE words only (not expanded pairs!)
                # For expanded: get all unique words from translations
                unique_words_expanded = set()
                for trans in word_translations:
                    word = trans.get(lang_code, '')
                    if word:
                        unique_words_expanded.add(word)
                unique_words_expanded = sorted(list(unique_words_expanded))

                # For original: get all unique words (flatten multi-meaning)
                unique_words_original = set()
                for trans in word_translations_for_horizontal:
                    word_str = trans.get(lang_code, '')
                    if word_str:
                        # Split by '|' to handle multi-meaning words
                        for word in word_str.split('|'):
                            unique_words_original.add(word.strip())
                unique_words_original = sorted(list(unique_words_original))

                # Combine all unique words we need
                all_unique_words = sorted(set(unique_words_expanded) | set(unique_words_original))

                # Get embeddings for ALL unique words (cache-aware)
                with st.spinner(f"Processing {lang_name} ({len(all_unique_words)} unique words)..."):
                    all_embeddings, cached_count, computed_count = embedding_cache.get_embeddings(
                        words=all_unique_words,
                        model_name=model_name,
                        lang_code=lang_code,
                        embedding_func=lambda words: get_embeddings_for_words(words, model_name, lang_code),
                        force_recompute=force_recompute
                    )

                # Validate embeddings upstream - filter out NaN/Inf before any downstream processing
                debug_mode = st.session_state.get('debug_mode', False)

                if debug_mode:
                    st.info(f"🔍 {lang_name}: Checking {len(all_embeddings) if all_embeddings is not None else 0} embeddings for NaN/Inf...")

                if all_embeddings is None or len(all_embeddings) == 0:
                    st.error(f"❌ {lang_name}: No embeddings generated!")
                    continue

                # Check for NaN/Inf
                has_nan_rows = np.isnan(all_embeddings).any(axis=1)
                has_inf_rows = np.isinf(all_embeddings).any(axis=1)
                valid_mask = ~(has_nan_rows | has_inf_rows)
                n_invalid = (~valid_mask).sum()

                if debug_mode:
                    st.info(f"🔍 {lang_name}: NaN check - {np.isnan(all_embeddings).any()} ({has_nan_rows.sum()} rows), Inf check - {np.isinf(all_embeddings).any()} ({has_inf_rows.sum()} rows)")

                if n_invalid > 0:
                    invalid_words = [all_unique_words[i] for i in range(len(all_unique_words)) if not valid_mask[i]]
                    st.warning(f"⚠️ {lang_name}: Excluding {n_invalid}/{len(all_unique_words)} invalid embeddings (NaN/Inf): {', '.join(invalid_words[:10])}{'...' if len(invalid_words) > 10 else ''}")

                    # Filter to valid embeddings only
                    all_embeddings = all_embeddings[valid_mask]
                    all_unique_words = [word for i, word in enumerate(all_unique_words) if valid_mask[i]]

                    # If all embeddings are invalid, skip this language
                    if len(all_embeddings) == 0:
                        st.error(f"❌ {lang_name}: All embeddings are invalid! Skipping this language.")
                        continue
                elif debug_mode:
                    st.success(f"✅ {lang_name}: All {len(all_embeddings)} embeddings are valid (no NaN/Inf)")

                if computed_count > 0:
                    st.success(f"⚡ {lang_name}: {cached_count} cached, {computed_count} computed ({len(all_unique_words)} valid unique words)")
                else:
                    st.success(f"⚡ {lang_name}: All {cached_count} unique words from cache")

                # Build word-to-embedding index (only valid embeddings)
                word_to_embedding = {word: all_embeddings[i] for i, word in enumerate(all_unique_words)}

                # Map expanded translations to embeddings (in correct order)
                # Skip entries with missing embeddings
                embeddings_expanded_list = []
                valid_expanded_indices = []
                for idx, trans in enumerate(word_translations):
                    word = trans.get(lang_code, '')
                    if word and word in word_to_embedding:
                        embeddings_expanded_list.append(word_to_embedding[word])
                        valid_expanded_indices.append(idx)
                    # Silently skip missing embeddings (already warned upstream)

                embeddings_dict_expanded[lang_code] = np.array(embeddings_expanded_list)

                # Map original translations to embeddings (use first occurrence for multi-meaning)
                embeddings_original_list = []
                valid_original_indices = []
                for idx, trans in enumerate(word_translations_for_horizontal):
                    word_str = trans.get(lang_code, '')
                    if word_str:
                        # Use first word in multi-meaning (e.g., "十" from "十|完整")
                        word = word_str.split('|')[0].strip()
                        if word in word_to_embedding:
                            embeddings_original_list.append(word_to_embedding[word])
                            valid_original_indices.append(idx)
                    # Silently skip missing embeddings (already warned upstream)

                embeddings_dict_original[lang_code] = np.array(embeddings_original_list)
                msg_dict[lang_name] = f"✓ {lang_name}: {embeddings_dict_expanded[lang_code].shape}"

                progress_bar.progress((lang_idx + 1) / len(selected_languages))

            # Save cache if any embeddings were computed
            embedding_cache.save_master_cache()

            progress_bar.empty()

            total_embedding_time = time.time() - total_embedding_start
            st.info(f"⏱️ Total embedding generation: {total_embedding_time:.2f}s")


            if msg_dict:
                c_0,c_1,c_2,c_3,c_4,c_5,c_6,c_7,c_8 = st.columns([2,1,1,1,1,1,1,1,1])
                with c_0:
                    st.write(msg_dict["summary"])
                with c_1:
                    st.write(msg_dict.get("Chinese",""))
                with c_2:
                    st.write(msg_dict.get("English",""))
                with c_3:
                    st.write(msg_dict.get("Spanish",""))
                with c_4:
                    st.write(msg_dict.get("French",""))
                with c_5:
                    st.write(msg_dict.get("German",""))
                with c_6:
                    st.write(msg_dict.get("Russian",""))
                with c_7:   
                    st.write(msg_dict.get("Korean",""))
                with c_8:
                    st.write(msg_dict.get("Arabic",""))


            # Compute Semantic Affinity for each selected metric
            st.markdown("### 🧮 Computing Semantic Affinity...")

            # Show parallel processing info
            if n_jobs > 1:
                if len(word_translations) >= 500:
                    st.info(f"⚡ Using {n_jobs} CPU cores for parallel computation ({len(word_translations)} words ≥ 500 threshold)")
                else:
                    st.info(f"📊 Sequential mode ({len(word_translations)} words < 500 threshold, parallel overhead not beneficial)")

            analyzer = SemanticAffinityAnalyzer(
                collapse_epsilon=collapse_epsilon,
                n_jobs=n_jobs,
                max_batch_size=max_batch_size
            )

            # Store results for each metric
            results_dict = {}

            try:
                # Parallelize dual metric computation (Euclidean + Cosine)
                if len(selected_metrics) > 1 and n_jobs > 1:

                    def compute_metric(metric):
                        return metric, analyzer.compute_semantic_affinity(
                            embeddings_dict=embeddings_dict_expanded,
                            word_translations=word_translations,
                            original_embeddings_dict=embeddings_dict_original,
                            original_word_translations=word_translations_for_horizontal,
                            metric=metric
                        )

                    dual_metric_start = time.time()
                    with st.spinner(f"Computing SR with {len(selected_metrics)} metrics in parallel..."):
                        # Use 'loky' backend for true parallel execution
                        metric_results = Parallel(n_jobs=len(selected_metrics), backend='loky')(
                            delayed(compute_metric)(metric) for metric in selected_metrics
                        )

                        # Unpack results
                        for metric, result in metric_results:
                            results_dict[metric] = result

                    dual_metric_time = time.time() - dual_metric_start
                    st.info(f"⏱️ Dual metric computation (parallel): {dual_metric_time:.2f}s")
                else:
                    # Sequential execution for single metric or n_jobs=1
                    for metric in selected_metrics:
                        metric_start = time.time()
                        with st.spinner(f"Computing SR with {metric} metric..."):
                            result = analyzer.compute_semantic_affinity(
                                embeddings_dict=embeddings_dict_expanded,
                                word_translations=word_translations,
                                original_embeddings_dict=embeddings_dict_original,
                                original_word_translations=word_translations_for_horizontal,
                                metric=metric
                            )
                            results_dict[metric] = result
                        metric_time = time.time() - metric_start
                        st.info(f"⏱️ {metric.title()} computation: {metric_time:.2f}s")

                # Store in cache (only if caching is enabled)
                if use_cache:
                    st.session_state.sa_cache[cache_key] = {
                        'results_dict': results_dict,
                        'embeddings_dict': embeddings_dict_expanded,
                        'word_translations': word_translations,
                        'language_codes': language_codes,
                        'config': cache_key_data
                    }

                # Set embeddings_dict for visualization (use expanded version)
                embeddings_dict = embeddings_dict_expanded

            except Exception as e:
                st.error(f"Error computing Semantic Affinity: {str(e)}")
                
                st.code(traceback.format_exc())
                return

        # Display results (whether cached or freshly computed)
        st.markdown("---")
        st.markdown("### 📊 Results")

        # Display results side-by-side for each metric (COSINE FIRST)
        if len(selected_metrics) == 2:
            # Two columns: Cosine (primary) and Euclidean (supplementary)
            col_cos, col_euc = st.columns(2)

            for metric in ['cosine', 'euclidean']:
                if metric not in results_dict:
                    continue

                result = results_dict[metric]
                col = col_cos if metric == 'cosine' else col_euc

                with col:

                    # Metrics in sub-columns
                    m0, m1, m2, m3, m4 = st.columns(5)

                    with m0:
                        st.markdown(f"#### {metric.title()}")

                    with m1:
                        if result.status == 'OK':
                            st.metric("SA Score", f"{result.score:.4f}")
                        else:
                            st.metric("SA Score", result.status)

                    with m2:
                        st.metric("SEM (±)", f"{result.sem:.4f}")

                    with m3:
                        st.metric("Inter-Spread", f"{result.vertical_spread:.4f}")

                    with m4:
                        st.metric("Intra-Spread", f"{result.horizontal_spread:.4f}")
        else:
            # Single metric - use full width
            for metric in selected_metrics:
                result = results_dict[metric]

                st.markdown(f"### {metric.title()} Distance")

                col1_, col2_, col3_, col4_ = st.columns(4)

                with col1_:
                    if result.status == 'OK':
                        st.metric("SA Score", f"{result.score:.4f}")
                    else:
                        st.metric("SA Score", result.status)

                with col2_:
                    st.metric("SEM (±)", f"{result.sem:.4f}", help="Standard Error of Mean - propagated uncertainty")

                with col3_:
                    st.metric("Inter-Spread", f"{result.vertical_spread:.4f}", help=f"Cross-lingual ± {result.vertical_spread_sem:.4f}")

                with col4_:
                    st.metric("Intra-Spread", f"{result.horizontal_spread:.4f}", help=f"Baseline ± {result.horizontal_spread_sem:.4f}")


        # PHATE Visualization (independent of SR metric calculations)
        # st.markdown("---")
        st.markdown("### 📊 PHATE Manifold Visualization")
        # st.markdown("Visualize embedding manifold structure and cross-lingual clustering.")

        # Check if chart already exists to skip expensive PHATE computation
        # Create cache key for PHATE visualization
        phate_cache_data = {
            'dataset': dataset_name,
            'model': model_name,
            'languages': sorted(selected_languages),
            'n_words': len(word_translations)
        }
        phate_cache_str = str(sorted(phate_cache_data.items()))
        phate_cache_hash = hashlib.md5(phate_cache_str.encode()).hexdigest()[:8]

        # Expected chart filename
        lang_codes_str = '-'.join([LANG_NAME_TO_CODE.get(lang.lower(), lang[:3]) for lang in sorted(selected_languages)])
        expected_chart_file = f"echarts-{dataset_name}-phate-{model_name.split('/')[-1]}-{lang_codes_str}.png"
        chart_path = Path(f"../data/images/echarts/{expected_chart_file}")

        skip_phate = chart_path.exists()

        if skip_phate:
            st.success(f"⚡ Chart already exists: {expected_chart_file} - skipping PHATE computation")
            st.image(str(chart_path), caption=f"PHATE Visualization ({', '.join(selected_languages)})", width='stretch')
        else:
            with st.spinner("Computing PHATE dimensionality reduction..."):
                try:
                    # Combine all embeddings for visualization
                    all_embeddings_list = []
                    all_labels_list = []
                    all_colors_list = []

                    # Map language codes to color map keys
                    lang_code_to_color_key = {
                        'chn': 'chinese',
                        'enu': 'english',
                        'spa': 'spanish',
                        'fra': 'french',
                        'deu': 'german',
                        'rus': 'russian',
                        'kor': 'korean',
                        'ara': 'arabic',
                        'jpn': 'japanese',
                        'vie': 'vietnamese',
                        'tha': 'thai',
                        'heb': 'hebrew',
                        'hin': 'hindi',
                        'grk': 'greek',
                        'fas': 'persian',
                        'tur': 'turkish',
                        'kat': 'georgian',
                        'hye': 'armenian'
                    }

                    for lang_code, embeddings in embeddings_dict.items():
                        # Get UNIQUE words for this language (not expanded pairs)
                        lang_words = [trans.get(lang_code, '') for trans in word_translations]
                        lang_words_filtered = [w for w in lang_words if w]

                        # Get unique words and their indices
                        unique_words = []
                        unique_indices = []
                        seen = set()
                        for idx, word in enumerate(lang_words_filtered):
                            if word not in seen:
                                unique_words.append(word)
                                unique_indices.append(idx)
                                seen.add(word)

                        # Get embeddings for unique words only
                        unique_embeddings = embeddings[unique_indices]

                        # Debug: Check for NaN in this language's embeddings
                        has_nan = np.isnan(unique_embeddings).any()
                        has_inf = np.isinf(unique_embeddings).any()
                        if has_nan or has_inf:
                            st.warning(f"⚠️ PHATE prep: {lang_code} has NaN={has_nan}, Inf={has_inf}, shape={unique_embeddings.shape}")

                        all_embeddings_list.append(unique_embeddings)
                        all_labels_list.extend(unique_words)

                        # Get color from COLOR_MAP using language code mapping
                        color_key = lang_code_to_color_key.get(lang_code, 'english')
                        lang_color = COLOR_MAP.get(color_key, COLOR_MAP['english'])
                        all_colors_list.extend([lang_color] * len(unique_words))

                    # Stack all embeddings
                    debug_mode = st.session_state.get('debug_mode', False)

                    if debug_mode:
                        st.info(f"🔍 Debug: Stacking {len(all_embeddings_list)} language arrays with shapes: {[arr.shape for arr in all_embeddings_list]}")

                    combined_embeddings = np.vstack(all_embeddings_list)

                    if debug_mode:
                        st.info(f"🔍 Debug: Combined shape: {combined_embeddings.shape}, has NaN: {np.isnan(combined_embeddings).any()}, has Inf: {np.isinf(combined_embeddings).any()}")

                    # Double-check for NaN/Inf values before PHATE (safety check)
                    valid_mask = ~(np.isnan(combined_embeddings).any(axis=1) | np.isinf(combined_embeddings).any(axis=1))
                    n_invalid = (~valid_mask).sum()

                    if debug_mode:
                        st.info(f"🔍 Debug: Found {n_invalid} invalid rows out of {len(combined_embeddings)}")

                    if n_invalid > 0:
                        st.warning(f"⚠️ Filtering {n_invalid} invalid embeddings before PHATE")
                        combined_embeddings = combined_embeddings[valid_mask]
                        all_labels_list = [label for i, label in enumerate(all_labels_list) if valid_mask[i]]
                        all_colors_list = [color for i, color in enumerate(all_colors_list) if valid_mask[i]]

                        if len(combined_embeddings) == 0:
                            st.error("❌ All embeddings contain NaN/Inf values. Cannot compute PHATE.")
                            st.stop()

                        st.success(f"✅ Filtered to {len(combined_embeddings)} valid embeddings for PHATE")

                    # Apply PHATE with preprocessing
                    # Check for duplicate embeddings (can cause PHATE issues)
                    unique_embeddings, unique_indices = np.unique(combined_embeddings, axis=0, return_index=True)
                    n_duplicates = len(combined_embeddings) - len(unique_embeddings)

                    if n_duplicates > 0:
                        # st.warning(f"⚠️ Found {n_duplicates} duplicate embeddings. Using unique embeddings only for PHATE.")
                        combined_embeddings = unique_embeddings
                        all_labels_list = [all_labels_list[i] for i in sorted(unique_indices)]
                        all_colors_list = [all_colors_list[i] for i in sorted(unique_indices)]

                    # Normalize embeddings to prevent PHATE numerical instability
                    scaler = StandardScaler()
                    combined_embeddings_normalized = scaler.fit_transform(combined_embeddings)

                    # REMOVED: Random noise (line 1215-1217) - DimensionReducer already handles
                    # numerical stability with deterministic seeded noise (random_state=42)
                    # This makes PHATE results reproducible and consistent with batch benchmark

                    if debug_mode:
                        st.info(f"🔍 Debug: After normalization - shape: {combined_embeddings_normalized.shape}, has NaN: {np.isnan(combined_embeddings_normalized).any()}, has Inf: {np.isinf(combined_embeddings_normalized).any()}")

                    try:
                        reducer = DimensionReducer()
                        reduced_embeddings = reducer.reduce_dimensions(
                            combined_embeddings_normalized,
                            method="PHATE",
                            dimensions=2
                        )

                        if reduced_embeddings is None:
                            st.error("❌ PHATE returned None - likely internal error")
                            st.warning("⚠️ Skipping PHATE visualization. SA scores above are still valid.")
                            return

                    except Exception as e:
                        st.error(f"❌ PHATE failed with exception: {str(e)}")
                        st.warning("⚠️ Skipping PHATE visualization. SA scores above are still valid.")
                        
                        st.code(traceback.format_exc())
                        return

                    # Check if Publication Mode is enabled
                    viz_settings = get_global_viz_settings()
                    publication_mode = viz_settings.get('publication_mode', False)

                    # Create language codes list for title
                    lang_codes = list(embeddings_dict.keys())

                    # Prepare SA metrics text for legend (compact 2-line format, COSINE FIRST)
                    sa_metrics_text = None
                    if results_dict:
                        metrics_lines = []
                        for metric_name in ['cosine', 'euclidean']:
                            if metric_name in results_dict:
                                result = results_dict[metric_name]
                                metric_label = "SA_cos" if metric_name == 'cosine' else "SA_eucl"
                                metrics_lines.append(f"{metric_label} = {result.score:.4f} ± {result.sem:.4f}")
                        sa_metrics_text = "\n".join(metrics_lines) if metrics_lines else None

                    # Use Plotly (vector graphics) in Publication Mode, ECharts otherwise
                    if publication_mode:
                        # Publication Mode: Use Plotly for vector PDF export
                        st.info("📊 **Publication Mode**: Using Plotly for vector graphics (scalable PDFs)")

                        plotter = PlotManager()
                        plotter.plot_2d(
                            embeddings=reduced_embeddings,
                            labels=all_labels_list,
                            colors=all_colors_list,
                            title=f"PHATE Visualization",
                            clustering=False,
                            method_name="PHATE",
                            model_name=model_name,
                            dataset_name=dataset_name,
                            lang_codes=lang_codes,
                            word_search_config=highlight_config,  # Word search highlighting (Plotly parameter name)
                            sa_metrics_text=sa_metrics_text  # SA metrics for legend
                        )
                    else:
                        # Interactive Mode: Use ECharts for better interactivity
                        plotter = EChartsPlotManager()

                        # Generate unique chart key to disable caching (for debugging)

                        chart_key = f"sa_phate_viz_{int(time.time() * 1000)}"

                        plotter.plot_2d(
                            embeddings=reduced_embeddings,
                            labels=all_labels_list,
                            colors=all_colors_list,
                            title=f"PHATE Visualization",
                            clustering=False,
                            method_name="PHATE",
                            model_name=model_name,
                            dataset_name=dataset_name,
                            lang_codes=lang_codes,
                            chart_key=chart_key,  # Unique key per computation
                            highlight_config=highlight_config,  # Word search highlighting
                            sa_metrics_text=sa_metrics_text  # SA metrics for legend
                        )

                    st.info("""
                    **Manifold Structure**:
                    - Look for language-based clusters in the visualization
                    - Tighter clusters indicate better cross-lingual alignment
                    - Colors represent different languages
                    """)

                except Exception as e:
                    st.error(f"Error creating PHATE visualization: {str(e)}")
                    
                    st.code(traceback.format_exc())


        st.markdown("---")
        report_col0, report_col1, report_col2 = st.columns([1, 1, 1])

        # Per-word scores (use first metric)
        first_result = results_dict[selected_metrics[0]]
        if first_result.per_word_scores:

            with report_col1:
                st.markdown(f"#### 📊 Per-Word Analysis ({selected_metrics[0].title()})")
                # Create dataframe of per-word scores
                scores_df = pd.DataFrame([
                    {"Word": word, "Normalized Spread": score}
                    for word, score in first_result.per_word_scores.items()
                ]).sort_values("Normalized Spread", ascending=False)

                st.markdown(f"**All {len(scores_df)} words ranked by normalized spread:**")
                st.dataframe(scores_df, width='stretch', height=400)

            with report_col2:
                # Outliers
                if first_result.outliers:
                    st.markdown("**🔍 Top 10 Outliers** (hardest to align):")
                    outliers_df = pd.DataFrame([
                        {"Word": word, "Score": f"{score:.4f}"}
                        for word, score in first_result.outliers
                    ])
                    st.dataframe(outliers_df, width='stretch')

            with report_col0:
                # Summary for paper
                st.markdown("### 📄 Summary for Paper")

                # Build summary table with all metrics
                for metric in selected_metrics:
                    result = results_dict[metric]

                    summary_text = f"""
**{metric.title()} Distance:**

| Property | Value |
|----------|-------|
| Model | {model_name} |
| Dataset | {dataset_name} |
| Languages | {', '.join(selected_languages)} ({len(selected_languages)}) |
| Words | {result.n_words} |
| Intra-Spread | {result.horizontal_spread:.4f} ± {result.horizontal_spread_sem:.4f} |
| Inter-Spread | {result.vertical_spread:.4f} ± {result.vertical_spread_sem:.4f} |
| Status | {result.status} |
| **SA Score** | **{result.score:.4f} ± {result.sem:.4f}** |
| SR (reference) | {result.semantic_ratio:.4f} |
| Std Dev | {result.std:.4f} |
        """
                    st.markdown(summary_text)
                    if len(selected_metrics) > 1:
                        st.markdown("---")



                # Download results
                st.markdown("### 💾 Export Results")

                # Prepare CSV export with all metrics
                export_data = []
                for metric in selected_metrics:
                    result = results_dict[metric]
                    export_data.append({
                        "dataset": dataset_name,
                        "model": model_name,
                        "languages": "|".join(selected_languages),
                        "n_languages": len(selected_languages),
                        "n_words": result.n_words,
                        "metric": metric,
                        "sa_score": result.score,  # Primary metric (Semantic Affinity)
                        "sr_score": result.semantic_ratio,  # Reference (Semantic Ratio)
                        "sem": result.sem,
                        "std": result.std,
                        "horizontal_spread": result.horizontal_spread,
                        "horizontal_spread_sem": result.horizontal_spread_sem,
                        "vertical_spread": result.vertical_spread,
                        "vertical_spread_sem": result.vertical_spread_sem,
                        "status": result.status
                    })

                export_df = pd.DataFrame(export_data)

                csv_data = export_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv_data,
                    file_name=f"semantic_affinity_{model_name.replace(' ', '_')}_{dataset_name}.csv",
                    mime="text/csv"
                )




if __name__ == "__main__":
    main()
else:
    main()
