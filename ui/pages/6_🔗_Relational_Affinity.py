"""
Relational Affinity Metric v2.0 - Multi-Language Support

This page implements the Relational Affinity (RA) metric for benchmarking
how well multilingual embedding models preserve relational structure across languages.

VERSION 2.0 CHANGES:
- Flexible N-language support (1, 2, 3, or more languages)
- Dynamic language selection from UI
- Separate tracking of intra-lingual (within) and inter-lingual (between) RA
- Clean UI with consolidated debug output
- Language-agnostic naming conventions

Reference: docs/conference/NeurIPS/2026/relational-affinity/2-metric.md
Design: docs/design/RA-Multi-Language-Refactor.md

Key Concept:
Relational Vector (rel_vec) = emb(word2) - emb(word1)
- Captures semantic transformation from word1 → word2
- Example: rel_vec("husband", "wife") represents male→female relation

Relational Affinity (RA):
- Primary: RA_cosine = cosine_similarity(rel_vec_i, rel_vec_j) ∈ [-1, 1]
- Secondary: RA_euclidean = 2 * ||rel_vec_i - rel_vec_j|| / (||rel_vec_i|| + ||rel_vec_j||) ∈ [0, 2]

Higher RA_cosine = stronger relational alignment
Lower RA_euclidean = stronger relational alignment
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import re
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import phate
    PHATE_AVAILABLE = True
except ImportError:
    PHATE_AVAILABLE = False

from semanscope.config import DATA_PATH, MODEL_INFO, get_active_models_with_headers, DEFAULT_METHOD, DEFAULT_MODEL, DEFAULT_DATASET
from semanscope.models.model_manager import get_model
from semanscope.utils.embedding_cache import get_embedding_cache

# Page config
st.set_page_config(
    page_title="Relational Affinity v2 (Multi-Language)",
    page_icon="🔗",
    layout="wide"
)

# ============================================================================
# Constants
# ============================================================================

# Language code mapping for API calls
LANG_CODE_MAP = {
    'en': 'enu', 'enu': 'enu',
    'zh': 'chn', 'chn': 'chn',
    'es': 'esp', 'esp': 'esp',
    'fr': 'fra', 'fra': 'fra',
    'de': 'deu', 'deu': 'deu',
    'tr': 'tur', 'tur': 'tur',
    'ja': 'jpn', 'jpn': 'jpn',
    'ko': 'kor', 'kor': 'kor',
    'ar': 'ara', 'ara': 'ara',
    'ru': 'rus', 'rus': 'rus',
    'pt': 'por', 'por': 'por',
    'it': 'ita', 'ita': 'ita',
}

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class EdgeCaseStats:
    """Statistics for edge cases encountered during RA computation"""
    total_pairs: int
    valid_pairs: int
    excluded_zero_vec: int
    excluded_oov: int
    coverage: float
    negative_ra_count: int
    negative_ra_pairs: List[str]


@dataclass
class RAResult:
    """Results from RA computation"""
    metric: str  # 'cosine' or 'euclidean'
    language: str  # 'english', 'chinese', or 'cross-lingual'
    mean_ra: float
    std_ra: float
    min_ra: float
    max_ra: float
    negative_count: int  # Only for cosine
    pairwise_scores: np.ndarray  # NxN matrix
    edge_cases: EdgeCaseStats


# ============================================================================
# Core RA Metric Functions
# ============================================================================

def compute_rel_vec(
    emb_model,
    word1: str,
    word2: str,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Compute relational vector: emb(word2) - emb(word1)

    Returns:
        rel_vec: Relational vector (difference)

    Raises:
        ValueError: If ||rel_vec|| < epsilon (degenerate case)
        KeyError: If word not in vocabulary (OOV)
    """
    try:
        emb1 = emb_model.get_embeddings([word1], lang="en", debug_flag=False)[0]
        emb2 = emb_model.get_embeddings([word2], lang="en", debug_flag=False)[0]
    except Exception as e:
        raise KeyError(f"Word not in vocabulary: {e}")

    rel_vec = emb2 - emb1

    # Check for degenerate case (zero vector)
    norm = np.linalg.norm(rel_vec)
    if norm < epsilon:
        raise ValueError(
            f"Zero relational vector: ||emb({word2}) - emb({word1})|| = {norm:.2e}"
        )

    return rel_vec


def ra_cosine(rel_vec1: np.ndarray, rel_vec2: np.ndarray) -> float:
    """
    Compute Relational Affinity using cosine similarity

    RA_cos = (v1 · v2) / (||v1|| × ||v2||)

    Returns:
        float in [-1, 1]
            +1: Perfect alignment (same direction)
             0: Orthogonal (unrelated)
            -1: Opposite directions (inverted relation)

    Note: Negative values are meaningful - they indicate opposite relational directions.
    """
    dot_product = np.dot(rel_vec1, rel_vec2)
    norm1 = np.linalg.norm(rel_vec1)
    norm2 = np.linalg.norm(rel_vec2)

    cos_sim = dot_product / (norm1 * norm2)
    return float(cos_sim)


def ra_euclidean(rel_vec1: np.ndarray, rel_vec2: np.ndarray) -> float:
    """
    Compute Relational Affinity using normalized Euclidean distance

    RA_euc = 2 * ||v1 - v2|| / (||v1|| + ||v2||)

    Returns:
        float in [0, ∞), typically [0, 2]
            0: Identical vectors
            <0.5: Strong similarity
            >1.0: Weak similarity
    """
    diff = rel_vec1 - rel_vec2
    dist = np.linalg.norm(diff)

    norm1 = np.linalg.norm(rel_vec1)
    norm2 = np.linalg.norm(rel_vec2)

    normalized_dist = 2 * dist / (norm1 + norm2)
    return float(normalized_dist)


def pairwise_ra(
    rel_vecs: List[np.ndarray],
    metric: str = 'cosine',
    show_progress: bool = False
) -> np.ndarray:
    """
    Compute all pairwise RA scores (N×(N-1)/2 comparisons)

    Returns:
        NxN symmetric matrix
    """
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

    # Compute pairwise RA scores
    total_comparisons = N * (N - 1) // 2
    comparison_count = 0

    for i in range(N):
        for j in range(i, N):
            if i == j:
                ra_matrix[i, j] = diagonal_value
            else:
                score = ra_func(rel_vecs[i], rel_vecs[j])
                ra_matrix[i, j] = score
                ra_matrix[j, i] = score  # Symmetric
                comparison_count += 1

    return ra_matrix


def aggregate_ra_category(ra_matrix: np.ndarray, metric: str = 'cosine') -> Dict:
    """
    Aggregate pairwise RA scores for a category

    Returns:
        {mean, std, min, max, negative_count (if cosine)}
    """
    N = ra_matrix.shape[0]

    # Extract upper triangle (excluding diagonal)
    upper_triangle_indices = np.triu_indices(N, k=1)
    off_diagonal = ra_matrix[upper_triangle_indices]

    # Compute statistics
    result = {
        'mean': float(np.mean(off_diagonal)),
        'std': float(np.std(off_diagonal)),
        'min': float(np.min(off_diagonal)),
        'max': float(np.max(off_diagonal))
    }

    # Count negative RAs (only for cosine)
    if metric == 'cosine':
        result['negative_count'] = int(np.sum(off_diagonal < 0))

    return result


# ============================================================================
# PHATE Visualization Functions
# ============================================================================

def get_embeddings_batch(words: List[str], model_name: str, lang_code: str) -> np.ndarray:
    """Get embeddings for a list of words using cache"""
    embedding_cache = get_embedding_cache()
    model = get_model(model_name)

    def embedding_func(words_to_embed):
        return model.get_embeddings(words_to_embed, lang=lang_code, debug_flag=False)

    embeddings, cached, computed = embedding_cache.get_embeddings(
        words=words,
        model_name=model_name,
        lang_code=lang_code,
        embedding_func=embedding_func,
        force_recompute=False
    )

    # L2 normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    return embeddings


def run_phate(embeddings: np.ndarray, n_components: int = 2, knn: int = 5, decay: int = 15) -> np.ndarray:
    """Run PHATE dimensionality reduction"""
    if not PHATE_AVAILABLE:
        raise ImportError("phate-python not installed. Install with: pip install phate")

    phate_op = phate.PHATE(
        n_components=n_components,
        knn=knn,
        decay=decay,
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    reduced = phate_op.fit_transform(embeddings)
    return reduced


def compute_ra_cosine_phate(rel_vec1: np.ndarray, rel_vec2: np.ndarray) -> float:
    """Compute cosine similarity between relational vectors (for PHATE arrow coloring)"""
    dot_product = np.dot(rel_vec1, rel_vec2)
    norm1 = np.linalg.norm(rel_vec1)
    norm2 = np.linalg.norm(rel_vec2)
    return float(dot_product / (norm1 * norm2))


def create_phate_plot_with_arrows(
    phate_coords: np.ndarray,
    word_list: List[str],
    word_to_idx: Dict[str, int],
    df: pd.DataFrame,
    title: str,
    lang_prefix: str,  # 'en', 'zh', 'es', etc.
    show_labels: bool = True,
    color_by_ra: bool = False,
    ra_scores: List[float] = None,
    pair_filter: List[Tuple[str, str]] = None,  # List of (word1, word2) pairs to show
    label_font_size: int = 11,
    point_size: int = 7,
    ra_cosine: float = None,
    ra_euclidean: float = None
) -> go.Figure:
    """
    Create PHATE plot with relational vector arrows

    Args:
        phate_coords: Nx2 array of PHATE coordinates
        word_list: List of words (matches phate_coords order)
        word_to_idx: Dict mapping word → index in phate_coords
        df: DataFrame with relation pairs
        title: Plot title
        lang_prefix: Language code to select columns (e.g., 'en', 'zh', 'es')
        show_labels: Whether to show word labels
        color_by_ra: Color arrows by RA score
        ra_scores: List of RA scores per pair (if color_by_ra=True)
        pair_filter: Optional list of (word1, word2) pairs to display
    """
    fig = go.Figure()

    # Plot words as scatter points
    fig.add_trace(go.Scatter(
        x=phate_coords[:, 0],
        y=phate_coords[:, 1],
        mode='markers+text' if show_labels else 'markers',
        text=word_list if show_labels else None,
        textposition='top center',
        textfont=dict(size=label_font_size, color='#2c3e50'),
        marker=dict(
            size=point_size,
            color='#1f4788',  # Dark blue (publication quality)
            opacity=0.9,
            line=dict(width=1.2, color='#0d2a5c')
        ),
        name='Words',
        hovertemplate='%{text}<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
    ))

    # Draw arrows for relational vectors (only if pair_filter is specified)
    word1_col = f'word1_{lang_prefix}'
    word2_col = f'word2_{lang_prefix}'

    arrow_x = []
    arrow_y = []
    arrow_colors = []
    arrow_texts = []

    # Only draw arrows if user has specified word pairs
    if pair_filter is not None and len(pair_filter) > 0:
        for idx, row in df.iterrows():
            word1 = row[word1_col]
            word2 = row[word2_col]

            # Check if both words are in vocabulary
            if word1 not in word_to_idx or word2 not in word_to_idx:
                continue

            # Only show pairs specified in filter
            if (word1, word2) not in pair_filter:
                continue

            idx1 = word_to_idx[word1]
            idx2 = word_to_idx[word2]

            x1, y1 = phate_coords[idx1]
            x2, y2 = phate_coords[idx2]

            # Store arrow data
            arrow_x.extend([x1, x2, None])
            arrow_y.extend([y1, y2, None])

            if color_by_ra and ra_scores is not None:
                arrow_colors.append(ra_scores[idx])

            arrow_texts.append(f"{word1} → {word2}")

    # Determine arrow color scheme
    if color_by_ra and ra_scores is not None:
        # Color by RA score (gradient from red=bad to green=good)
        colorscale = [[0, 'red'], [0.5, 'yellow'], [1, 'green']]
        arrow_color = arrow_colors
        showscale = True
        colorbar_title = "RA Score"
    else:
        # Default color - vibrant red/pink
        arrow_color = '#e74c3c'  # Bright red
        showscale = False
        colorbar_title = None

    # Add arrows as lines
    for i in range(0, len(arrow_x), 3):
        if arrow_x[i] is None:
            continue

        # Determine line color for this arrow
        if color_by_ra and arrow_colors:
            line_color = f'rgba(255, {int(100 + 155*arrow_colors[i//3])}, 100, 0.8)'
        else:
            line_color = arrow_color

        fig.add_trace(go.Scatter(
            x=[arrow_x[i], arrow_x[i+1]],
            y=[arrow_y[i], arrow_y[i+1]],
            mode='lines',
            line=dict(
                color=line_color,
                width=2.5
            ),
            showlegend=False,
            hoverinfo='text',
            text=arrow_texts[i//3] if i//3 < len(arrow_texts) else ''
        ))

        # Add arrowhead
        dx = arrow_x[i+1] - arrow_x[i]
        dy = arrow_y[i+1] - arrow_y[i]
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            # Normalize and scale arrowhead
            dx_norm = dx / length
            dy_norm = dy / length
            arrow_size = 0.02 * max(phate_coords[:, 0].max() - phate_coords[:, 0].min(),
                                    phate_coords[:, 1].max() - phate_coords[:, 1].min())

            # Arrowhead as annotation
            fig.add_annotation(
                x=arrow_x[i+1],
                y=arrow_y[i+1],
                ax=arrow_x[i+1] - dx_norm * arrow_size,
                ay=arrow_y[i+1] - dy_norm * arrow_size,
                xref='x',
                yref='y',
                axref='x',
                ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2.5,
                arrowcolor=line_color
            )

    fig.update_layout(
        title=dict(
            text=title,
            x=0,
            xanchor='left',
            font=dict(size=14, color='#2c3e50')
        ),
        xaxis_title="X1",
        yaxis_title="X2",
        width=700,
        height=700,
        hovermode='closest',
        template='plotly_white',
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#2c3e50')
    )

    # Add RA metrics legend in upper-right corner (high position to avoid data overlay)
    if ra_cosine is not None or ra_euclidean is not None:
        legend_lines = []
        if ra_cosine is not None:
            legend_lines.append(f"RA (cos): {ra_cosine:.4f}")
        if ra_euclidean is not None:
            legend_lines.append(f"RA (euc): {ra_euclidean:.4f}")

        legend_text = "<br>".join(legend_lines)

        fig.add_annotation(
            text=legend_text,
            xref="paper", yref="paper",
            x=0.97, y=1.10,
            xanchor="right", yanchor="top",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#1f4788",
            borderwidth=1.5,
            borderpad=8,
            font=dict(size=11, color='#2c3e50', family='monospace')
        )

    return fig


# ============================================================================
# Dataset Loading
# ============================================================================

def discover_ra_datasets() -> List[str]:
    """Discover all NeurIPS-*-RA.csv files"""
    input_dir = DATA_PATH / "input"
    if not input_dir.exists():
        return []

    ra_files = list(input_dir.glob("NeurIPS-*-RA.csv"))
    # Return clean names without -RA.csv suffix
    return [f.stem.replace('-RA', '') for f in sorted(ra_files)]


def load_ra_dataset(dataset_name: str) -> Tuple[pd.DataFrame, bool, str]:
    """
    Load RA dataset

    Expected columns:
    - relation_id, category, word1_en, word2_en, word1_zh, word2_zh,
      relation_type, verified, notes

    Args:
        dataset_name: Name without -RA.csv suffix (e.g., 'NeurIPS-1-family-relations-enu-chn')

    Returns:
        (dataframe, is_valid, error_message)
    """
    # Add -RA.csv suffix back for file loading
    dataset_path = DATA_PATH / "input" / f"{dataset_name}-RA.csv"

    if not dataset_path.exists():
        return None, False, f"Dataset not found: {dataset_path}"

    try:
        df = pd.read_csv(dataset_path, comment='#')

        # Validate required columns
        required_cols = ['word1_en', 'word2_en', 'word1_zh', 'word2_zh']
        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            return df, False, f"Missing columns: {missing}"

        return df, True, ""

    except Exception as e:
        return None, False, f"Error loading dataset: {str(e)}"


def detect_languages(df: pd.DataFrame) -> Dict[str, str]:
    """
    Detect available languages from dataframe columns

    Returns:
        Dict mapping lang_code -> display_name
        e.g., {'en': 'English', 'zh': 'Chinese', 'es': 'Spanish'}
    """
    # Language code to display name mapping
    LANG_NAMES = {
        'en': 'English',
        'enu': 'English',
        'zh': 'Chinese',
        'chn': 'Chinese',
        'es': 'Spanish',
        'esp': 'Spanish',
        'fr': 'French',
        'fra': 'French',
        'de': 'German',
        'deu': 'German',
        'tr': 'Turkish',
        'tur': 'Turkish',
        'ja': 'Japanese',
        'jpn': 'Japanese',
        'ko': 'Korean',
        'kor': 'Korean',
        'ar': 'Arabic',
        'ara': 'Arabic',
        'ru': 'Russian',
        'rus': 'Russian',
        'pt': 'Portuguese',
        'por': 'Portuguese',
        'it': 'Italian',
        'ita': 'Italian',
    }

    detected = {}

    # Look for word1_XX and word2_XX columns
    for col in df.columns:
        if col.startswith('word1_') or col.startswith('word2_'):
            lang_code = col.split('_')[1]  # Extract XX from word1_XX
            if lang_code in LANG_NAMES and lang_code not in detected:
                detected[lang_code] = LANG_NAMES[lang_code]

    return detected


def get_language_pairs(df: pd.DataFrame) -> List[Tuple[str, str, str]]:
    """
    Get all valid language pairs from dataframe

    Returns:
        List of (lang_code, display_name, column_suffix) tuples
        e.g., [('en', 'English', 'en'), ('zh', 'Chinese', 'zh'), ...]
    """
    lang_map = detect_languages(df)

    # Return list of tuples
    return [(code, name, code) for code, name in sorted(lang_map.items())]


# ============================================================================
# Sidebar
# ============================================================================

def render_sidebar():
    """Render sidebar with controls"""
    with st.sidebar:
        st.subheader("⚙️ Settings")

        # ================================================================
        # (A) MODEL SELECTION
        # ================================================================
        st.markdown("### 🤖 Model")

        model_names_with_headers, model_info_dict = get_active_models_with_headers()

        # Use default from global settings (config.py)
        default_index = model_names_with_headers.index(DEFAULT_MODEL) if DEFAULT_MODEL in model_names_with_headers else 1

        model_name = st.selectbox(
            "Embedding Model",
            options=model_names_with_headers,
            index=default_index,
            help="Models organized by architecture",
            label_visibility="collapsed"
        )

        # Check if header selected
        if model_name.startswith("━━━━━━"):
            st.warning("⚠️ Please select a model, not a group header")
            return None

        # Store in session state
        st.session_state['selected_model'] = model_name

        # Show model info
        if model_name in MODEL_INFO:
            with st.expander("ℹ️ Model Info", expanded=False):
                st.info(MODEL_INFO[model_name]['help'])


        # ================================================================
        # (B) DATASET SELECTION & DATA VIEW
        # ================================================================
        col_title, col_refresh = st.columns([5, 1])
        with col_title:
            st.markdown("### 📂 Dataset")
        with col_refresh:
            if st.button("🔄", help="Refresh dataset list", width='stretch'):
                st.rerun()

        available_datasets = discover_ra_datasets()

        if not available_datasets:
            st.error("No NeurIPS-*-RA.csv datasets found")
            return None

        # Use default from global settings (config.py)
        dataset_name = st.selectbox(
            "Relational Dataset",
            options=available_datasets,
            index=available_datasets.index(DEFAULT_DATASET) if DEFAULT_DATASET in available_datasets else 0,
            help="NeurIPS relational benchmark datasets",
            label_visibility="collapsed"
        )

        # Load dataset
        df, is_valid, error_msg = load_ra_dataset(dataset_name)

        if not is_valid:
            st.error(error_msg)
            return None

        # Detect available languages
        languages = get_language_pairs(df)

        c1,c2 = st.columns([3,3])
        # Dataset info
        with c1:
            if 'category' in df.columns:
                categories = df['category'].unique()
                st.caption(f"📂 Categories: {', '.join(categories)}")
        with c2:
            st.caption(f"✅ **{len(df)} relations**")
        lang_names = [name for _, name, _ in languages]
        # st.caption(f"🌐 Languages: {', '.join(lang_names)}")

        # ================================================================
        # DATA VIEW - Show word pairs in text_areas (DYNAMIC)
        # ================================================================
        with st.expander("📋 View Data", expanded=False):
            # st.markdown("**Select languages:**")
            # st.caption("Selection applies to data view and PHATE visualization")

            # Create checkboxes for each language
            enabled_languages = {}
            lang_check_cols = st.columns(len(languages))
            for idx, (lang_code, lang_name, col_suffix) in enumerate(languages):
                with lang_check_cols[idx]:
                    # Default: Enable EN and ZH if present
                    default_enabled = (idx == 0 or idx == 2) # idx < 3
                    enabled = st.checkbox(
                        lang_name,
                        value=default_enabled,
                        key=f"lang_select_{lang_code}",
                        help=f"Show {lang_name} data"
                    )
                    enabled_languages[col_suffix] = enabled

            # Count enabled languages
            num_enabled = sum(enabled_languages.values())
            if num_enabled == 0:
                st.warning("⚠️ Please select at least one language")
            # else:
            #     st.caption(f"✅ {num_enabled} language(s) selected")

            # st.caption("**Word pairs in this dataset:**")

            # Filter languages based on selection
            selected_languages = [
                (lang_code, lang_name, col_suffix)
                for lang_code, lang_name, col_suffix in languages
                if enabled_languages.get(col_suffix, False)
            ]

            # Create columns dynamically based on selected languages
            if selected_languages:
                cols = st.columns(len(selected_languages))

                for idx, (lang_code, lang_name, col_suffix) in enumerate(selected_languages):
                    with cols[idx]:
                        # st.caption(f"**{lang_name} ({lang_code.upper()})**")
                        # Format: word1, word2
                        pairs_text = "\n".join([
                            f"{row[f'word1_{col_suffix}']}, {row[f'word2_{col_suffix}']}"
                            for _, row in df.iterrows()
                        ])
                        st.text_area(
                            f"{lang_name} pairs",
                            value=pairs_text,
                            height=200,
                            help="Copy pairs to use in word-pair filter below",
                            label_visibility="collapsed",
                            key=f"ra_data_view_{dataset_name}_{lang_code}"
                        )

        # ================================================================
        # PHATE ARROWS (optional)
        # ================================================================
        with st.expander("Overlay Relational Vector Arrows on PHATE", expanded=False):
            # st.markdown("**Enter word pairs to draw arrows in PHATE visualization:**")
            # st.caption("_Leave empty for scatter plot only. RA metrics always use all pairs in dataset._")

            pair_filters = {}  # lang_code -> list of pairs

            # Use selected languages (from View Data checkboxes)
            if not selected_languages:
                st.warning("⚠️ No languages selected. Please select languages in 'View Data' section above.")
            else:
                # Create columns dynamically based on selected languages
                filter_cols = st.columns(len(selected_languages))

                for idx, (lang_code, lang_name, col_suffix) in enumerate(selected_languages):
                    with filter_cols[idx]:
                        st.caption(f"**{lang_name}**")

                        # Auto-populate first 4 word pairs from dataset
                        default_pairs = '\n'.join([
                            f"{row[f'word1_{col_suffix}']}, {row[f'word2_{col_suffix}']}"
                            for _, row in df.head(4).iterrows()
                        ])

                        pairs_input = st.text_area(
                            f"{lang_name} (word1, word2)",
                            value=default_pairs,
                            height=120,
                            help="Format: word1, word2 (one pair per line). Auto-populated with first 3 pairs from dataset.",
                            label_visibility="collapsed",
                            key=f"ra_filter_{dataset_name}_{lang_code}"
                        )

                        # Parse word pairs
                        if pairs_input.strip():
                            pair_list = []
                            for line in pairs_input.strip().split('\n'):
                                if ',' in line:
                                    parts = [p.strip() for p in line.split(',')]
                                    if len(parts) >= 2:
                                        pair_list.append((parts[0], parts[1]))

                            if pair_list:
                                pair_filters[col_suffix] = pair_list
                                st.caption(f"✅ {len(pair_list)} arrow(s) specified")

        # Show feedback if arrows are specified
        if pair_filters:
            # Count how many pairs match the specified arrows
            matched_count = 0
            for idx, row in df.iterrows():
                for lang_suffix, filter_pairs in pair_filters.items():
                    if (row[f'word1_{lang_suffix}'], row[f'word2_{lang_suffix}']) in filter_pairs:
                        matched_count += 1
                        break

            if matched_count == 0:
                st.warning("⚠️ Specified pairs not found in dataset. PHATE will show scatter plot only.")

        # ================================================================
        # METRIC SELECTION & ADVANCED SETTINGS
        # ================================================================


        # Advanced settings
        with st.expander("🔧 Advanced Settings", expanded=False):
            # ================================================================
            # Metrics SETTINGS
            # ================================================================

            col0, col1, col2 = st.columns([3, 5, 5])
            with col0:
                st.markdown("### Metrics")
            with col1:
                cosine_enabled = st.checkbox("Cosine", value=True, help="Primary metric (direction-focused)")
            with col2:
                euclidean_enabled = st.checkbox("Euclidean", value=True, help="Secondary metric (magnitude-aware)")

            selected_metrics = []
            if cosine_enabled:
                selected_metrics.append('cosine')
            if euclidean_enabled:
                selected_metrics.append('euclidean')

            if not selected_metrics:
                st.warning("⚠️ Select at least one metric")
                return None


            debug_mode = st.checkbox("Debug Mode", value=True, help="Show all intermediate calculations")
            st.session_state['ra_debug_mode'] = debug_mode


            st.markdown("### **Cross-lingual RA Calculation Mode**")
            cross_lingual_mode = st.radio(
                "Mode",
                options=['unified', 'cross-only'],
                index=0,
                help="""
                **Unified** (default): Pool all EN+ZH vectors, compute all pairwise RA
                - Total comparisons: (N_en + N_zh) × (N_en + N_zh - 1) / 2
                - Includes: EN-EN, ZH-ZH, EN-ZH
                - Use for: General model quality benchmarking

                **Cross-only**: Compute only EN↔ZH cross-lingual pairs
                - Total comparisons: N_en × N_zh
                - Includes: EN-ZH only (excludes EN-EN, ZH-ZH)
                - Use for: Translation quality & cross-lingual alignment analysis
                """,
                label_visibility="collapsed"
            )

            force_embedding = st.checkbox(
                "Force recompute embeddings",
                value=False,
                help="Ignore cache and recompute"
            )
            st.session_state['force_embedding_recompute'] = force_embedding

            # ================================================================
            # PHATE VISUALIZATION SETTINGS
            # ================================================================
            st.markdown("### 📊 PHATE Visualization")
            if not PHATE_AVAILABLE:
                st.warning("⚠️ PHATE not available. Install with: `pip install phate`")
                enable_phate = False
            else:
                enable_phate = st.checkbox("Enable PHATE visualization", value=True,
                                          help="Visualize relational vectors as arrows in 2D space")

            if enable_phate and PHATE_AVAILABLE:
                st.caption("💡 Language selection is configured in 'View Data' section above")

                st.markdown("**PHATE Parameters:**")

                col__1, col__2 = st.columns(2)
                with col__1:
                    knn = st.slider("k-NN", min_value=3, max_value=20, value=5,
                                help="Number of nearest neighbors")
                with col__2:
                    decay = st.slider("Decay", min_value=5, max_value=50, value=15,
                                    help="Decay rate for kernel")

                st.markdown("**Visualization Options:**")
                show_labels = st.checkbox("Show word labels", value=True)

                col1, col2 = st.columns(2)
                with col1:
                    label_font_size = st.slider("Label font size", min_value=6, max_value=14, value=11,
                                               help="Font size for word labels")
                with col2:
                    point_size = st.slider("Point size", min_value=4, max_value=12, value=7,
                                          help="Size of scatter points")

                color_by_ra = st.checkbox("Color arrows by RA score", value=False,
                                        help="Requires RA computation")

            else:
                knn, decay, show_labels, color_by_ra = 5, 15, True, False
                label_font_size, point_size = 11, 7
                enabled_languages = {}


            c1_ , c2_ = st.columns(2)
            with c1_:
                if debug_mode:
                    display_limit = st.number_input(
                        "Debug display limit (rows)",
                        min_value=0,
                        max_value=1000,
                        value=0,
                        help="Number of rows to show in debug tables (0 = show all)"
                    )
                    st.session_state['ra_display_limit'] = display_limit

            with c2_:
                epsilon = st.number_input(
                    "Zero vector threshold",
                    min_value=1e-10,
                    max_value=1e-3,
                    value=1e-8,
                    format="%.2e",
                    help="Threshold for detecting degenerate relational vectors"
                )


        # ================================================================
        # (C) COMPUTE BUTTON
        # ================================================================
        compute_button = st.button("🚀 Compute & Visualize", type="primary", width='stretch')

        st.markdown("---")

        # Instructions
        with st.expander("ℹ️ How to Use", expanded=False):
            st.markdown("""
            #### Overview
            Measure and visualize how well multilingual embeddings preserve **relational structure** across languages.

            **Metrics**: Compute RA scores (cosine/euclidean) for monolingual and cross-lingual relations.

            **Visualization**: View relational vectors as arrows in 2D PHATE space to see geometric alignment.

            Higher RA = stronger relational alignment. Configure settings in the sidebar and click "Compute & Visualize".

                        
            **Relational Affinity measures:**
            - How consistently embedding models preserve semantic relations
            - Cross-lingual relational structure (EN ↔ ZH)

            **Metrics:**
            - **RA_cosine**: Direction alignment (higher = better, range [-1, 1])
            - **RA_euclidean**: Magnitude difference (lower = better, range [0, 2])

            **Interpretation (Cosine):**
            - RA > 0.6: Strong relational preservation ✅
            - RA ≈ 0.4: Moderate preservation
            - RA < 0.2: Weak preservation ❌
            - RA < 0: Inverse relations (unexpected)

            **Examples:**
            - husband→wife vs king→queen (expect RA ≈ 0.7)
            - hot→cold vs big→small (expect RA ≈ 0.5)

            **Debug Mode Outputs:**
            When enabled, shows detailed intermediate calculations for each pair
            """)

            st.info("""
            ⚠️ **PHATE preserves distances, not angles!**
            - Arrow **length** is meaningful
            - Arrow **direction** in 2D is NOT reliable
            - Focus on **clustering** and **length consistency**
            """)


    return {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'df': df,
        'selected_metrics': selected_metrics,
        'epsilon': epsilon,
        'compute_button': compute_button,
        'enable_phate': enable_phate if PHATE_AVAILABLE else False,
        'knn': knn,
        'decay': decay,
        'show_labels': show_labels,
        'ra_debug_mode': st.session_state.get('ra_debug_mode', False),
        'label_font_size': label_font_size,
        'point_size': point_size,
        'color_by_ra': color_by_ra,
        'pair_filters': pair_filters,
        'enabled_languages': enabled_languages,
        'selected_languages': selected_languages,  # v2: for multi-language RA
        'languages': languages  # v2: all available languages in dataset
    }


# ============================================================================
# Main Computation
# ============================================================================

def compute_relational_affinity(
    model_name: str,
    df: pd.DataFrame,
    selected_languages: List[Tuple[str, str, str]],  # [(lang_code, lang_name, col_suffix), ...]
    metric: str,
    epsilon: float = 1e-8
) -> Dict:
    """
    Compute RA for a dataset with flexible multi-language support

    Args:
        model_name: Name of embedding model
        df: DataFrame with word pairs
        selected_languages: List of (lang_code, lang_name, col_suffix) tuples for selected languages
        metric: 'cosine' or 'euclidean'
        epsilon: Threshold for zero vectors

    Returns:
        {
            'rel_vecs_by_lang': {col_suffix: [rel_vec, ...], ...},  # For PHATE charting
            'pair_labels_by_lang': {col_suffix: ['word1→word2', ...], ...},  # For PHATE charting
            'ra_monolingual': {col_suffix: RAResult, ...},  # Intra-lingual RA
            'ra_cross_lingual': {(col_suffix1, col_suffix2): RAResult, ...},  # Inter-lingual RA
            'all_comparisons': [(type, lang1, lang2, idx1, idx2, score), ...],  # All pairwise
            'selected_languages': selected_languages  # Pass through
        }
    """
    debug_mode = st.session_state.get('ra_debug_mode', False)
    # Load model
    model = get_model(model_name)

    # Get embedding cache
    embedding_cache = get_embedding_cache()
    force_recompute = st.session_state.get('force_embedding_recompute', False)

    # Step 1: Collect all unique words for each selected language
    all_words_by_lang = {}  # {col_suffix: [word1, word2, ...], ...}

    for lang_code, lang_name, col_suffix in selected_languages:
        words_set = set()
        for _, row in df.iterrows():
            words_set.add(row[f'word1_{col_suffix}'])
            words_set.add(row[f'word2_{col_suffix}'])
        all_words_by_lang[col_suffix] = sorted(list(words_set))

    # Step 2: Get embeddings for each selected language (cached)
    embeddings_by_lang = {}  # {col_suffix: np.ndarray, ...}
    word_to_emb_by_lang = {}  # {col_suffix: {word: emb, ...}, ...}

    for lang_code, lang_name, col_suffix in selected_languages:
        api_code = LANG_CODE_MAP.get(col_suffix, col_suffix)
        words = all_words_by_lang[col_suffix]

        with st.spinner(f"Loading {lang_name} embeddings ({len(words)} words)..."):
            embeddings, cached, computed = embedding_cache.get_embeddings(
                words=words,
                model_name=model_name,
                lang_code=api_code,
                embedding_func=lambda words, lang=lang_code: np.array([
                    model.get_embeddings([w], lang=lang, debug_flag=False)[0]
                    for w in words
                ]),
                force_recompute=force_recompute
            )

        # L2 normalize all embeddings to unit length (critical for RA!)
        embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_by_lang[col_suffix] = embeddings_normalized

        # Build word-to-embedding map
        word_to_emb_by_lang[col_suffix] = {
            word: embeddings_normalized[i] for i, word in enumerate(words)
        }

    # Save cache
    embedding_cache.save_master_cache()

    # Step 3: Compute relational vectors for each selected language
    rel_vecs_by_lang = {}  # {col_suffix: [rel_vec1, rel_vec2, ...], ...}
    pair_labels_by_lang = {}  # {col_suffix: ['word1→word2', ...], ...}
    pair_metrics_by_lang = {}  # {col_suffix: [metrics_dict, ...], ...} - for debug
    excluded_by_lang = {}  # {col_suffix: {'zero_vec': count, 'oov': count}, ...}

    total_pairs = len(df)

    for lang_code, lang_name, col_suffix in selected_languages:
        rel_vecs = []
        pair_labels = []
        pair_metrics = []  # Store intermediate calculations for debug
        excluded_zero_vec = 0
        excluded_oov = 0

        word_to_emb = word_to_emb_by_lang[col_suffix]

        for idx, row in df.iterrows():
            w1 = row[f'word1_{col_suffix}']
            w2 = row[f'word2_{col_suffix}']

            try:
                # Get embeddings
                emb1 = word_to_emb[w1]
                emb2 = word_to_emb[w2]

                # Compute relational vector
                rel_vec = emb2 - emb1

                # Check for zero vector
                norm_rel_vec = np.linalg.norm(rel_vec)
                if norm_rel_vec < epsilon:
                    excluded_zero_vec += 1
                    if debug_mode:
                        st.warning(f"⚠️ Zero vector in {lang_name}: {w1}→{w2}, ||rel_vec||={norm_rel_vec:.2e}")
                    continue

                # Normalize relational vector to unit length
                rel_vec = rel_vec / norm_rel_vec

                # Compute intermediate metrics (for debug)
                if debug_mode:
                    norm_emb1 = np.linalg.norm(emb1)
                    norm_emb2 = np.linalg.norm(emb2)
                    cos_sim = np.dot(emb1, emb2) / (norm_emb1 * norm_emb2)
                    # Note: Since embeddings are L2-normalized, euc_raw = euc_norm = rel_norm
                    # Keep only rel_norm as it's the most conceptually meaningful

                    pair_metrics.append({
                        'cos_sim': float(cos_sim),
                        'rel_norm': float(norm_rel_vec)
                    })

                # Valid pair
                rel_vecs.append(rel_vec)
                pair_labels.append(f"{w1}→{w2}")

            except KeyError as e:
                excluded_oov += 1
                if debug_mode:
                    st.warning(f"⚠️ OOV in {lang_name}: {w1}, {w2}")

        # Store results
        rel_vecs_by_lang[col_suffix] = rel_vecs
        pair_labels_by_lang[col_suffix] = pair_labels
        pair_metrics_by_lang[col_suffix] = pair_metrics
        excluded_by_lang[col_suffix] = {
            'zero_vec': excluded_zero_vec,
            'oov': excluded_oov
        }

    # Calculate overall coverage (use first language as reference)
    first_lang_suffix = selected_languages[0][2]
    valid_pairs = len(rel_vecs_by_lang[first_lang_suffix])
    coverage = valid_pairs / total_pairs if total_pairs > 0 else 0.0

    # Step 4: Compute intra-lingual RA (monolingual) for each language
    ra_monolingual = {}  # {col_suffix: RAResult, ...}

    for lang_code, lang_name, col_suffix in selected_languages:
        rel_vecs = rel_vecs_by_lang[col_suffix]

        if len(rel_vecs) == 0:
            continue

        with st.spinner(f"Computing intra-lingual RA ({lang_name}, {metric})..."):
            ra_matrix = pairwise_ra(rel_vecs, metric=metric)

        # Aggregate stats
        stats = aggregate_ra_category(ra_matrix, metric=metric)

        # Create edge case stats
        edge_cases = EdgeCaseStats(
            total_pairs=total_pairs,
            valid_pairs=valid_pairs,
            excluded_zero_vec=excluded_by_lang[col_suffix]['zero_vec'],
            excluded_oov=excluded_by_lang[col_suffix]['oov'],
            coverage=coverage,
            negative_ra_count=stats.get('negative_count', 0),
            negative_ra_pairs=[]
        )

        # Create RAResult
        ra_result = RAResult(
            metric=metric,
            language=lang_name,
            mean_ra=stats['mean'],
            std_ra=stats['std'],
            min_ra=stats['min'],
            max_ra=stats['max'],
            negative_count=stats.get('negative_count', 0),
            pairwise_scores=ra_matrix,
            edge_cases=edge_cases
        )

        ra_monolingual[col_suffix] = ra_result

    # Step 5: Compute inter-lingual RA (cross-lingual) for all language pairs
    ra_cross_lingual = {}  # {(col_suffix1, col_suffix2): RAResult, ...}
    all_comparisons = []  # [(type, lang1, lang2, idx1, idx2, score), ...]

    # First, add monolingual comparisons to all_comparisons (for debug table)
    for lang_code, lang_name, col_suffix in selected_languages:
        rel_vecs = rel_vecs_by_lang[col_suffix]
        N = len(rel_vecs)

        for i in range(N):
            for j in range(i + 1, N):
                if metric == 'cosine':
                    score = ra_cosine(rel_vecs[i], rel_vecs[j])
                else:
                    score = ra_euclidean(rel_vecs[i], rel_vecs[j])

                all_comparisons.append((
                    f'{col_suffix}-{col_suffix}',  # Type
                    col_suffix,  # lang1
                    col_suffix,  # lang2
                    i,  # idx1
                    j,  # idx2
                    score
                ))

    # Then, compute cross-lingual comparisons
    num_langs = len(selected_languages)
    for i in range(num_langs):
        for j in range(i + 1, num_langs):
            lang1_code, lang1_name, lang1_suffix = selected_languages[i]
            lang2_code, lang2_name, lang2_suffix = selected_languages[j]

            rel_vecs1 = rel_vecs_by_lang[lang1_suffix]
            rel_vecs2 = rel_vecs_by_lang[lang2_suffix]

            N1 = len(rel_vecs1)
            N2 = len(rel_vecs2)

            if N1 == 0 or N2 == 0:
                continue

            with st.spinner(f"Computing inter-lingual RA ({lang1_name}↔{lang2_name}, {metric})..."):
                # Compute all cross-lingual comparisons
                cross_scores = []
                negative_pairs = []

                for idx1 in range(N1):
                    for idx2 in range(N2):
                        if metric == 'cosine':
                            score = ra_cosine(rel_vecs1[idx1], rel_vecs2[idx2])
                        else:
                            score = ra_euclidean(rel_vecs1[idx1], rel_vecs2[idx2])

                        cross_scores.append(score)

                        # Track for debug
                        all_comparisons.append((
                            f'{lang1_suffix}-{lang2_suffix}',  # Type
                            lang1_suffix,  # lang1
                            lang2_suffix,  # lang2
                            idx1,  # idx1
                            idx2,  # idx2
                            score
                        ))

                        if metric == 'cosine' and score < 0:
                            pair_label1 = pair_labels_by_lang[lang1_suffix][idx1]
                            pair_label2 = pair_labels_by_lang[lang2_suffix][idx2]
                            negative_pairs.append(f"{pair_label1} vs {pair_label2}")

                # Compute stats
                stats = {
                    'mean': float(np.mean(cross_scores)),
                    'std': float(np.std(cross_scores)),
                    'min': float(np.min(cross_scores)),
                    'max': float(np.max(cross_scores))
                }

                if metric == 'cosine':
                    stats['negative_count'] = len(negative_pairs)

                # Create edge case stats
                edge_cases = EdgeCaseStats(
                    total_pairs=total_pairs,
                    valid_pairs=valid_pairs,
                    excluded_zero_vec=0,
                    excluded_oov=0,
                    coverage=coverage,
                    negative_ra_count=len(negative_pairs) if metric == 'cosine' else 0,
                    negative_ra_pairs=negative_pairs
                )

                # Create RAResult
                ra_result = RAResult(
                    metric=metric,
                    language=f'{lang1_name}-{lang2_name}',
                    mean_ra=stats['mean'],
                    std_ra=stats['std'],
                    min_ra=stats['min'],
                    max_ra=stats['max'],
                    negative_count=stats.get('negative_count', 0),
                    pairwise_scores=np.array(cross_scores).reshape(N1, N2),
                    edge_cases=edge_cases
                )

                ra_cross_lingual[(lang1_suffix, lang2_suffix)] = ra_result

    # Store debug info in session state (if debug mode)
    if debug_mode:
        st.session_state['ra_debug_valid_pairs'] = valid_pairs
        st.session_state['ra_debug_total_pairs'] = total_pairs
        st.session_state['ra_debug_coverage'] = coverage
        st.session_state['ra_debug_excluded_by_lang'] = excluded_by_lang

    return {
        'rel_vecs_by_lang': rel_vecs_by_lang,
        'pair_labels_by_lang': pair_labels_by_lang,
        'pair_metrics_by_lang': pair_metrics_by_lang,  # For debug table
        'ra_monolingual': ra_monolingual,
        'ra_cross_lingual': ra_cross_lingual,
        'all_comparisons': all_comparisons,
        'selected_languages': selected_languages
    }




# ============================================================================
# Main App
# ============================================================================

def main():
    st.markdown("### 🔗 Relational Affinity Analysis v2.0")

    # Render sidebar
    config = render_sidebar()
    if config is None:
        st.warning("⚠️ Please configure settings in sidebar")
        return

    # Extract config
    model_name = config['model_name']
    dataset_name = config['dataset_name']
    df = config['df']
    selected_metrics = config['selected_metrics']
    epsilon = config['epsilon']
    compute_button = config['compute_button']
    selected_languages = config.get('selected_languages', [])

    # Check if any languages are selected
    if not selected_languages:
        st.warning("⚠️ Please select at least one language in the 'View Data' section of the sidebar")
        return

    # Compute RA
    if compute_button:
        # Compute for each metric
        results_dict = {}

        for metric in selected_metrics:
            results = compute_relational_affinity(
                model_name=model_name,
                df=df,
                selected_languages=selected_languages,
                metric=metric,
                epsilon=epsilon
            )

            results_dict[metric] = results

        # Display debug info (consolidated, only once for all metrics)
        if config.get('ra_debug_mode', False):
            valid_pairs = st.session_state.get('ra_debug_valid_pairs', 0)
            total_pairs = st.session_state.get('ra_debug_total_pairs', 0)
            coverage = st.session_state.get('ra_debug_coverage', 0.0)
            excluded_by_lang = st.session_state.get('ra_debug_excluded_by_lang', {})

            if len(selected_metrics) > 0:
                with st.expander("🔍 Debug: Pairwise RA Comparisons", expanded=False):
                    # Coverage info
                    st.info(f"✓ Valid pairs: {valid_pairs}/{total_pairs} ({coverage*100:.1f}%)")
                    
                    # Excluded breakdown per language
                    excluded_lines = []
                    for col_suffix, counts in excluded_by_lang.items():
                        lang_name = next((name for code, name, suffix in selected_languages if suffix == col_suffix), col_suffix)
                        excluded_lines.append(f"{lang_name}: {counts['zero_vec']} zero-vec, {counts['oov']} OOV")
                    if excluded_lines:
                        st.info(f"✗ Excluded: " + " | ".join(excluded_lines))

                    # Column definitions
                    st.markdown("### 📋 Column Definitions")
                    st.markdown("""
                    - **Type**: Comparison type (e.g., en-en, zh-zh, en-zh)
                    - **Pair 1**, **Pair 2**: The two relational pairs being compared
                    - **Status**: Quality indicator based on `RA_cos`
                      - ✅ Strong: RA_cos > 0.5
                      - ⚠️ Moderate: 0.3 ≤ RA_cos ≤ 0.5
                      - ❌ Weak: 0 ≤ RA_cos < 0.3
                      - 🔴 Negative: RA_cos < 0 (opposite directions!)
                    - **RA_cos**: Relational Affinity (cosine) — directional alignment [-1, 1]
                    - **RA_euc**: Relational Affinity (euclidean) — normalized distance [0, 2]
                    - **pair1_cos_sim**, **pair2_cos_sim**: Cosine similarity between word embeddings in each pair
                    - **pair1_rel_norm**, **pair2_rel_norm**: Magnitude of relational vector (||word2 - word1||)
                    """)

                    # Get all_comparisons from first metric
                    first_metric = selected_metrics[0]
                    all_comparisons = results_dict[first_metric].get('all_comparisons', [])
                    pair_labels_by_lang = results_dict[first_metric].get('pair_labels_by_lang', {})
                    pair_metrics_by_lang = results_dict[first_metric].get('pair_metrics_by_lang', {})

                    st.markdown(f"### 📊 All Pairwise Comparisons ({len(all_comparisons)} total)")

                    # Build unified table with both RA_cos and RA_euc + intermediate calculations
                    comparison_rows = []

                    # Create a lookup for scores by pair and metric
                    scores_by_pair_metric = {}  # (idx1, idx2, lang1, lang2) -> {metric: score}

                    for metric in selected_metrics:
                        metric_comparisons = results_dict[metric].get('all_comparisons', [])
                        for comp_type, lang1, lang2, idx1, idx2, score in metric_comparisons:
                            key = (idx1, idx2, lang1, lang2)
                            if key not in scores_by_pair_metric:
                                scores_by_pair_metric[key] = {'type': comp_type}
                            scores_by_pair_metric[key][metric] = score

                    # Build rows
                    for (idx1, idx2, lang1, lang2), scores in scores_by_pair_metric.items():
                        # Get pair labels
                        label1 = pair_labels_by_lang.get(lang1, [])[idx1] if idx1 < len(pair_labels_by_lang.get(lang1, [])) else f"pair{idx1}"
                        label2 = pair_labels_by_lang.get(lang2, [])[idx2] if idx2 < len(pair_labels_by_lang.get(lang2, [])) else f"pair{idx2}"

                        # Determine status based on cosine score (if available)
                        status = "N/A"
                        if 'cosine' in scores:
                            cos_score = scores['cosine']
                            if cos_score > 0.5:
                                status = "✅ Strong"
                            elif cos_score >= 0.3:
                                status = "⚠️ Moderate"
                            elif cos_score >= 0:
                                status = "❌ Weak"
                            else:
                                status = "🔴 Negative"

                        row = {
                            'Type': scores['type'],
                            'Pair 1': label1,
                            'Pair 2': label2,
                            'Status': status
                        }

                        # Add metric columns
                        if 'cosine' in scores:
                            row['RA_cos'] = f"{scores['cosine']:.4f}"
                        if 'euclidean' in scores:
                            row['RA_euc'] = f"{scores['euclidean']:.4f}"

                        # Add intermediate calculations in desired order:
                        # pair1_cos_sim, pair2_cos_sim, pair1_rel_norm, pair2_rel_norm

                        # First add cos_sim for both pairs
                        if lang1 in pair_metrics_by_lang and idx1 < len(pair_metrics_by_lang[lang1]):
                            metrics1 = pair_metrics_by_lang[lang1][idx1]
                            row['pair1_cos_sim'] = f"{metrics1['cos_sim']:.4f}"

                        if lang2 in pair_metrics_by_lang and idx2 < len(pair_metrics_by_lang[lang2]):
                            metrics2 = pair_metrics_by_lang[lang2][idx2]
                            row['pair2_cos_sim'] = f"{metrics2['cos_sim']:.4f}"

                        # Then add rel_norm for both pairs
                        if lang1 in pair_metrics_by_lang and idx1 < len(pair_metrics_by_lang[lang1]):
                            metrics1 = pair_metrics_by_lang[lang1][idx1]
                            row['pair1_rel_norm'] = f"{metrics1['rel_norm']:.4f}"

                        if lang2 in pair_metrics_by_lang and idx2 < len(pair_metrics_by_lang[lang2]):
                            metrics2 = pair_metrics_by_lang[lang2][idx2]
                            row['pair2_rel_norm'] = f"{metrics2['rel_norm']:.4f}"

                        comparison_rows.append(row)

                    if comparison_rows:
                        df_debug = pd.DataFrame(comparison_rows)

                        # Sort by RA_cos if available
                        if 'RA_cos' in df_debug.columns:
                            df_debug['_sort_key'] = df_debug['RA_cos'].astype(float)
                            df_debug = df_debug.sort_values('_sort_key', ascending=True).reset_index(drop=True)
                            df_debug = df_debug.drop(columns=['_sort_key'])

                            # Count negative pairs
                            negative_count = sum(df_debug['Status'] == '🔴 Negative')
                            if negative_count > 0:
                                st.error(f"⚠️ Found {negative_count}/{len(comparison_rows)} pair(s) with NEGATIVE RA (opposite directions!)")
                            else:
                                st.success(f"✓ All {len(comparison_rows)} pairs have positive RA")

                        st.dataframe(df_debug, width='stretch', height=400)

        # Display results
        st.markdown(f"#### 📊 Results")

        # Display side-by-side for dual metrics
        if len(selected_metrics) == 2:
            col_cos, col_euc = st.columns(2)

            for metric in ['cosine', 'euclidean']:
                if metric not in results_dict:
                    continue

                results = results_dict[metric]
                ra_monolingual = results.get('ra_monolingual', {})
                ra_cross_lingual = results.get('ra_cross_lingual', {})

                col = col_cos if metric == 'cosine' else col_euc

                with col:
                    # Detailed stats expander
                    with st.expander("📈 Detailed Stats"):
                        all_results = list(ra_monolingual.values()) + list(ra_cross_lingual.values())
                        stats_df = pd.DataFrame({
                            'Language/Pair': [r.language for r in all_results],
                            'Mean': [r.mean_ra for r in all_results],
                            'Std': [r.std_ra for r in all_results],
                            'Min': [r.min_ra for r in all_results],
                            'Max': [r.max_ra for r in all_results]
                        })
                        st.dataframe(stats_df, width='stretch')

                    # Horizontal metrics display
                    # Calculate number of columns needed: 1 (title) + intra + inter
                    num_cols = 1 + len(ra_monolingual) + len(ra_cross_lingual)
                    metric_cols = st.columns(num_cols)

                    # Column 0: Metric title
                    with metric_cols[0]:
                        st.markdown(f"##### {metric.title()}")

                    # Columns for intra-lingual (monolingual) results
                    col_idx = 1
                    for col_suffix, ra_result in ra_monolingual.items():
                        with metric_cols[col_idx]:
                            st.metric(
                                label=ra_result.language,
                                value=f"{ra_result.mean_ra:.4f}",
                                delta=f"σ={ra_result.std_ra:.4f}"
                            )
                        col_idx += 1

                    # Columns for inter-lingual (cross-lingual) results
                    for (lang1, lang2), ra_result in ra_cross_lingual.items():
                        with metric_cols[col_idx]:
                            st.metric(
                                label=ra_result.language,
                                value=f"{ra_result.mean_ra:.4f}",
                                delta=f"σ={ra_result.std_ra:.4f}"
                            )
                        col_idx += 1

        elif len(selected_metrics) == 1:
            # Single metric display - horizontal layout
            metric = selected_metrics[0]
            results = results_dict[metric]
            ra_monolingual = results.get('ra_monolingual', {})
            ra_cross_lingual = results.get('ra_cross_lingual', {})

            st.markdown(f"### {metric.title()}")

            # Calculate number of columns needed
            num_cols = len(ra_monolingual) + len(ra_cross_lingual)
            if num_cols > 0:
                metric_cols = st.columns(num_cols)

                col_idx = 0
                # Display intra-lingual first
                for col_suffix, ra_result in ra_monolingual.items():
                    with metric_cols[col_idx]:
                        st.metric(
                            label=ra_result.language,
                            value=f"{ra_result.mean_ra:.4f}",
                            delta=f"σ={ra_result.std_ra:.4f}"
                        )
                    col_idx += 1

                # Then inter-lingual
                for (lang1, lang2), ra_result in ra_cross_lingual.items():
                    with metric_cols[col_idx]:
                        st.metric(
                            label=ra_result.language,
                            value=f"{ra_result.mean_ra:.4f}",
                            delta=f"σ={ra_result.std_ra:.4f}"
                        )
                    col_idx += 1

        # PHATE Visualization (per language)
        if config.get('enable_phate', False) and PHATE_AVAILABLE:
            st.markdown("---")
            st.markdown("### 📊 PHATE Visualization")

            # Create PDF output directory if it doesn't exist
            pdf_output_dir = DATA_PATH / "images"/"PDF"/"RA"
            pdf_output_dir.mkdir(parents=True, exist_ok=True)

            # Clean up dataset name (remove NeurIPS- prefix and language suffix)
            dataset_clean = dataset_name.replace('NeurIPS-', '').split('-enu-')[0].split('-chn-')[0]
            method_name_upper = DEFAULT_METHOD.upper()  # For chart display
            method_name_lower = DEFAULT_METHOD.lower()  # For filename

            # Get first metric results for rel_vecs
            first_metric = selected_metrics[0]
            first_results = results_dict[first_metric]
            rel_vecs_by_lang = first_results.get('rel_vecs_by_lang', {})
            pair_labels_by_lang = first_results.get('pair_labels_by_lang', {})

            # Create PHATE plots for each selected language
            phate_cols = st.columns(min(len(selected_languages), 3))

            for idx, (lang_code, lang_name, col_suffix) in enumerate(selected_languages):
                if col_suffix not in rel_vecs_by_lang:
                    continue

                col = phate_cols[idx % len(phate_cols)]

                with col:
                    # Get word embeddings for this language
                    words = df[f'word1_{col_suffix}'].tolist() + df[f'word2_{col_suffix}'].tolist()
                    words = sorted(list(set(words)))

                    # Get embeddings using cache
                    embeddings = get_embeddings_batch(words, model_name, lang_code)

                    # Run PHATE
                    phate_coords = run_phate(embeddings, n_components=2, knn=config['knn'], decay=config['decay'])

                    # Create word-to-index map
                    word_to_idx = {word: i for i, word in enumerate(words)}

                    # Get RA scores for this language (from both metrics if available)
                    ra_cosine_val = None
                    if 'cosine' in results_dict:
                        ra_mono_cos = results_dict['cosine'].get('ra_monolingual', {}).get(col_suffix)
                        ra_cosine_val = ra_mono_cos.mean_ra if ra_mono_cos else None

                    ra_euclidean_val = None
                    if 'euclidean' in results_dict:
                        ra_mono_euc = results_dict['euclidean'].get('ra_monolingual', {}).get(col_suffix)
                        ra_euclidean_val = ra_mono_euc.mean_ra if ra_mono_euc else None

                    # Get pair filter for this language
                    pair_filter = config.get('pair_filters', {}).get(col_suffix, [])

                    # Create chart title: <method>-<model>-<dataset>-<language>
                    chart_title = f"{method_name_upper}-{model_name}-{dataset_clean}-{lang_code.upper()}"

                    # Create plot
                    fig = create_phate_plot_with_arrows(
                        phate_coords=phate_coords,
                        word_list=words,
                        word_to_idx=word_to_idx,
                        df=df,
                        title=chart_title,
                        lang_prefix=col_suffix,
                        show_labels=config.get('show_labels', True),
                        color_by_ra=config.get('color_by_ra', False),
                        ra_scores=None,
                        pair_filter=pair_filter if pair_filter else None,
                        label_font_size=config.get('label_font_size', 11),
                        point_size=config.get('point_size', 7),
                        ra_cosine=ra_cosine_val,
                        ra_euclidean=ra_euclidean_val
                    )

                    st.plotly_chart(fig, width='stretch')

                    # Auto-save as PDF
                    try:
                        # Create filename: method-model-dataset-language.pdf (e.g., phate-labse-1-family-relations-en.pdf)
                        # Sanitize model_name: replace spaces, parentheses with hyphens
                        model_name_clean = model_name.replace(' ', '-').replace('(', '-').replace(')', '-')
                        # Remove multiple consecutive hyphens
                        model_name_clean = re.sub(r'-+', '-', model_name_clean).strip('-')
                        # Lowercase entire filename for Unix compatibility
                        pdf_filename = f"{method_name_lower}-{model_name_clean}-{dataset_clean}-{lang_code}.pdf".lower()
                        pdf_path = pdf_output_dir / pdf_filename

                        # Save figure as PDF (using modern Plotly API without deprecated keyword args)
                        fig.write_image(str(pdf_path), format='pdf')

                        # Show success message with full path
                        st.success(f"💾 Saved: `{pdf_path}`")
                    except Exception as e:
                        # Silently fail if kaleido not installed
                        if "kaleido" in str(e).lower():
                            st.warning("⚠️ Install `kaleido` package to enable PDF export: `pip install kaleido`")
                        else:
                            st.error(f"⚠️ PDF save failed: {str(e)}")


if __name__ == "__main__":
    main()
else:
    main()
