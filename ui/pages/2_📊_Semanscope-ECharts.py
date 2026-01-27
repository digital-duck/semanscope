import streamlit as st
import numpy as np
import os
from semanscope.components.embedding_viz import EmbeddingVisualizer
from semanscope.components.dimension_reduction import DimensionReducer
from semanscope.components.plotting_echarts import EChartsPlotManager
from semanscope.components.geometric_analysis import GeometricAnalyzer
from semanscope.components.shared.enter_text_data import EnterTextDataWidget

from semanscope.config import (
    check_login,
    DEFAULT_N_CLUSTERS,
    COLOR_MAP,
    LANGUAGE_CODE_MAP,
    DEFAULT_LANG,
    DEFAULT_INSTRUCTION_MSG_1,
    get_language_code_from_name,
    get_model_language_code
)
from semanscope.utils.title_filename_helper import create_title_and_filename
from semanscope.utils.global_settings import (
    get_global_viz_settings,
    get_global_publication_settings,
    get_global_geometric_analysis,
    is_global_geometric_analysis_enabled,
    GlobalSettingsManager
)

# Page config
st.set_page_config(
    page_title="Semanscope - ECharts",
    page_icon="📊",
    layout="wide"
)

class EChartsEmbeddingVisualizer(EmbeddingVisualizer):
    """Enhanced embedding visualizer using Apache ECharts for plotting"""

    def __init__(self):
        super().__init__()
        self.echarts_plot_manager = EChartsPlotManager()

    def create_plot(self, reduced_embeddings, labels, colors, model_name, method_name,
                   dimensions="2D", do_clustering=False, n_clusters=DEFAULT_N_CLUSTERS, dataset_name="", highlight_config=None, lang_codes=None):
        """Create visualization using ECharts and also create a Plotly figure for PDF export"""

        # Create enhanced plot title with language codes
        title = self.echarts_plot_manager.create_title(method_name, model_name, dataset_name, lang_codes)

        # Apply rotation if set
        if hasattr(st.session_state, 'plot_rotation') and st.session_state.plot_rotation != 0:
            if dimensions == "2D":
                # Apply 2D rotation
                angle = np.radians(st.session_state.plot_rotation)
                rotation_matrix = np.array([
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]
                ])
                reduced_embeddings = reduced_embeddings @ rotation_matrix.T

        # Create appropriate plot based on dimensions (display_chart=True to show chart immediately)
        if dimensions == "3D":
            plot_option = self.echarts_plot_manager.plot_3d(
                reduced_embeddings, labels, colors, title,
                clustering=do_clustering, n_clusters=n_clusters,
                method_name=method_name, model_name=model_name, dataset_name=dataset_name,
                highlight_config=highlight_config, display_chart=True, lang_codes=lang_codes
            )
        else:
            plot_option = self.echarts_plot_manager.plot_2d(
                reduced_embeddings, labels, colors, title,
                clustering=do_clustering, n_clusters=n_clusters,
                method_name=method_name, model_name=model_name, dataset_name=dataset_name,
                highlight_config=highlight_config, display_chart=True, lang_codes=lang_codes
            )

        # Store the chart configuration for potential export
        st.session_state.current_echarts_config = plot_option

        # Create a Plotly figure for PDF export (without displaying it)
        plotly_fig = self._create_hidden_plotly_figure(
            reduced_embeddings, labels, colors, method_name, model_name,
            dataset_name, dimensions, do_clustering, n_clusters
        )

        # Store the Plotly figure for PDF export (without displaying it)
        st.session_state.current_figure = plotly_fig

        return plot_option

    def _create_hidden_plotly_figure(self, reduced_embeddings, labels, colors, method_name, model_name, dataset_name, dimensions, do_clustering, n_clusters):
        """Create a Plotly figure for PDF export without displaying it"""
        import plotly.graph_objects as go
        import pandas as pd
        from sklearn.cluster import KMeans

        # Create a simplified title for Plotly
        title = f"{method_name} | {model_name}"
        if dataset_name:
            title += f" | {dataset_name}"

        # Create figure
        fig = go.Figure()

        if dimensions == "3D":
            # Create 3D scatter plot
            if do_clustering:
                # Perform clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(reduced_embeddings)

                fig.add_trace(go.Scatter3d(
                    x=reduced_embeddings[:, 0],
                    y=reduced_embeddings[:, 1],
                    z=reduced_embeddings[:, 2],
                    mode='markers+text',
                    text=labels,
                    textposition='top center',
                    marker=dict(
                        size=8,
                        color=colors,
                        opacity=0.8
                    )
                ))
            else:
                fig.add_trace(go.Scatter3d(
                    x=reduced_embeddings[:, 0],
                    y=reduced_embeddings[:, 1],
                    z=reduced_embeddings[:, 2],
                    mode='markers+text',
                    text=labels,
                    textposition='top center',
                    marker=dict(
                        size=8,
                        color=colors,
                        opacity=0.8
                    )
                ))

            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title="Component 1",
                    yaxis_title="Component 2",
                    zaxis_title="Component 3"
                ),
                showlegend=False,
                width=1400,
                height=1100
            )
        else:
            # Create 2D scatter plot
            if do_clustering:
                # Perform clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(reduced_embeddings)

            fig.add_trace(go.Scatter(
                x=reduced_embeddings[:, 0],
                y=reduced_embeddings[:, 1],
                mode='markers+text',
                text=labels,
                textposition='top center',
                marker=dict(
                    size=8,
                    color=colors,
                    opacity=0.8
                )
            ))

            fig.update_layout(
                title=title,
                xaxis_title="Component 1",
                yaxis_title="Component 2",
                showlegend=False,
                width=1400,
                height=1100,
                xaxis_scaleanchor="y",
                xaxis_scaleratio=1
            )

        return fig

    def save_plot_image(self, input_name: str, model_name: str, method_name: str, chinese_selected: bool, english_selected: bool, dimensions: str = "2D"):
        """Save the current plot as image with ECharts identifier in filename"""
        if st.session_state.current_figure is None:
            st.warning("No plot to save. Please generate a visualization first.")
            return ""

        # Get publication settings to determine export format and directory
        pub_settings = st.session_state.get('global_settings', {}).get('publication', {})
        publication_mode = pub_settings.get('publication_mode', False)
        export_format = pub_settings.get('export_format', 'PNG').upper()
        export_dpi = pub_settings.get('export_dpi', 300)
        plot_width = pub_settings.get('plot_width', 1400)
        plot_height = pub_settings.get('plot_height', 1100)

        # Determine output directory based on format
        if export_format == 'PDF' and publication_mode:
            output_dir = self.images_dir / "PDF"
        else:
            output_dir = self.images_dir

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create sanitized filename components
        safe_input = self.sanitize_filename(input_name)
        safe_model = self.sanitize_filename(model_name)
        safe_method = self.sanitize_filename(method_name)

        # Collect active languages from session state using centralized approach
        lang_tags = []
        lang_code_map = LANGUAGE_CODE_MAP

        # Check for language selections from multiple possible key prefixes
        key_prefixes = ['echarts_cfg_', 'main_', 'dual_', '']

        for prefix in key_prefixes:
            # Get language selections for this prefix
            lang1 = st.session_state.get(f'{prefix}lang1', 'Chinese')
            lang2 = st.session_state.get(f'{prefix}lang2', 'English')
            lang3 = st.session_state.get(f'{prefix}lang3', 'German')

            lang1_code = lang_code_map.get(lang1, "chn")
            lang2_code = lang_code_map.get(lang2, "enu")
            lang3_code = lang_code_map.get(lang3, "deu")

            # Check which languages are actually selected via checkboxes
            if st.session_state.get(f'{lang1_code}_include_checkbox', False):
                lang_tags.append(lang1_code.upper())
            if st.session_state.get(f'{lang2_code}_include_checkbox', False) and lang2_code.upper() not in lang_tags:
                lang_tags.append(lang2_code.upper())
            if st.session_state.get(f'{lang3_code}_include_checkbox', False) and lang3_code.upper() not in lang_tags:
                lang_tags.append(lang3_code.upper())

            # If we found languages with this prefix, break
            if lang_tags:
                break

        # Fallback to boolean parameters if session state detection fails
        if not lang_tags:
            if chinese_selected:
                lang_tags.append("CHN")
            if english_selected:
                lang_tags.append("ENU")

        # Use centralized helper to generate consistent filename
        file_extension = export_format.lower() if publication_mode else "png"
        _, base_filename = create_title_and_filename(
            [method_name],
            [model_name],
            input_name,
            lang_tags,
            file_extension
        )

        # Add ECharts prefix and 3D suffix
        name_part, ext_part = base_filename.rsplit('.', 1)
        dim_suffix = "-3d" if dimensions == "3D" else ""
        filename = f"echarts-{name_part}{dim_suffix}.{ext_part}"

        # Store the generated filename in session state for reuse
        st.session_state['echarts_generated_filename'] = filename
        file_path = output_dir / filename

        try:
            # Save the figure in the specified format
            if publication_mode and export_format == 'PDF':
                st.session_state.current_figure.write_image(
                    str(file_path),
                    format="pdf",
                    width=plot_width,
                    height=plot_height
                )
            elif publication_mode and export_format == 'SVG':
                st.session_state.current_figure.write_image(
                    str(file_path),
                    format="svg",
                    width=plot_width,
                    height=plot_height
                )
            else:
                # Default PNG format
                scale_factor = export_dpi / 96 if publication_mode else 2
                st.session_state.current_figure.write_image(
                    str(file_path),
                    format="png",
                    width=plot_width if publication_mode else 1200,
                    height=plot_height if publication_mode else 800,
                    scale=scale_factor
                )

            return filename
        except Exception as e:
            st.error(f"Error saving image: {e}")
            return ""

@st.fragment
def save_plot_image(visualizer, current_input, model_name, method_name, chinese_selected, english_selected):
    return visualizer.save_plot_image(current_input, model_name, method_name, chinese_selected, english_selected)

@st.fragment
def generate_visualization_echarts(visualizer, reducer, source_words, target_words_dict, colors, model_name, method_name, dimensions, do_clustering, n_clusters, source_selected=True, target_selected_dict=None, highlight_config=None, selected_languages=None, dataset_name="User Input"):
    """Generate embeddings and ECharts visualization with full multilingual support

    Args:
        selected_languages: (NEW) Dictionary with lang1/lang2/lang3 structure (preferred)
        source_words, target_words_dict: (LEGACY) Old source/target paradigm (backward compatibility)
    """
    if target_selected_dict is None:
        target_selected_dict = {}

    # Use centralized model language mapping from config

    # NEW PARADIGM: Use selected_languages if provided
    if selected_languages is not None:
        # DEDUPLICATION: Build unique datasets from selected_languages
        unique_datasets = {}
        for lang_key in ['lang1', 'lang2', 'lang3']:
            lang_data = selected_languages[lang_key]
            if lang_data['selected'] and lang_data['words']:
                lang_code = lang_data['code']
                if lang_code not in unique_datasets:
                    # First occurrence of this language - keep it
                    unique_datasets[lang_code] = {
                        'name': lang_data['name'],
                        'code': lang_code,
                        'words': lang_data['words'],
                        'positions': [lang_key]
                    }
                else:
                    # Duplicate language - just track the position
                    unique_datasets[lang_code]['positions'].append(lang_key)

        # Check if any unique datasets found
        if len(unique_datasets) == 0:
            st.warning("Please select at least one language and enter words/phrases.")
            return False

        # Show deduplication info if duplicates were found
        duplicate_count = sum(len(info['positions']) - 1 for info in unique_datasets.values())
        if duplicate_count > 0:
            duplicated_langs = [f"{info['name']} (positions: {', '.join(info['positions'])})"
                              for info in unique_datasets.values() if len(info['positions']) > 1]
            st.info(f"🔄 **Deduplication Applied**: Processed {len(unique_datasets)} unique datasets (skipped {duplicate_count} duplicates). Duplicated: {', '.join(duplicated_langs)}")

        # Process embeddings for all unique language datasets
        all_embeddings = []
        labels = []
        deduplicated_colors = []  # Build colors for deduplicated data only

        # Process each unique language dataset once
        for lang_code, dataset_info in unique_datasets.items():
            if dataset_info['words']:
                model_lang = get_model_language_code(lang_code)
                embeddings = visualizer.get_embeddings(dataset_info['words'], model_name, model_lang)
                if embeddings is not None and len(embeddings) == len(dataset_info['words']):
                    all_embeddings.append(embeddings)
                    labels.extend(dataset_info['words'])

                    # Build colors for this unique dataset - check for semantic colors first
                    semantic_colors_key = f"{lang_code}_semantic_colors"
                    word_color_map = st.session_state.get(semantic_colors_key, {})
                    
                    if word_color_map:
                        # Use semantic/domain colors if available
                        dataset_colors = [word_color_map.get(word, '#FF00FF') for word in dataset_info['words']]
                    else:
                        # Fallback to language colors
                        from semanscope.config import COLOR_MAP
                        lang_color_map = {"enu": "english", "chn": "chinese", "fra": "french", "spa": "spanish",
                                         "deu": "german", "ara": "arabic", "heb": "hebrew", "hin": "hindi",
                                         "jpn": "japanese", "kor": "korean", "rus": "russian", "tha": "thai",
                                         "vie": "vietnamese"}
                        color_key = lang_color_map.get(lang_code, "english")
                        actual_color = COLOR_MAP.get(color_key, COLOR_MAP.get("english", "#1f77b4"))
                        dataset_colors = [actual_color] * len(dataset_info['words'])
                    
                    deduplicated_colors.extend(dataset_colors)

        if not all_embeddings:
            st.error("Failed to generate embeddings for any language.")
            return False

        combined_embeddings = np.vstack(all_embeddings)
        # COMMENTED OUT: This was overwriting domain colors with language colors
        # Use deduplicated colors instead of the original colors parameter
        # colors = deduplicated_colors

    else:
        # LEGACY PARADIGM: Use old source/target approach (backward compatibility)
        # Check if any language is selected and has words
        has_source = source_words and source_selected
        has_any_target = any(words and target_selected_dict.get(lang_code, False)
                            for lang_code, words in target_words_dict.items())

        if not (has_source or has_any_target):
            st.warning("Please select at least one language and enter words/phrases.")
            return False

        # Process embeddings for all selected languages
        all_embeddings = []
        labels = []

        # Process source language with dynamic language detection
        if has_source:
            # Get the first selected language from session state
            lang1 = st.session_state.get('lang1', 'Chinese')
            lang1_code = get_language_code_from_name(lang1)
            source_model_lang = get_model_language_code(lang1_code)

            source_embeddings = visualizer.get_embeddings(source_words, model_name, source_model_lang)
            if source_embeddings is not None:
                all_embeddings.append(source_embeddings)
                labels.extend(source_words)

        # Process each target language
        for lang_code, words in target_words_dict.items():
            if words and target_selected_dict.get(lang_code, False):
                # Skip target language processing if it's the same as source language (avoid duplication)
                if has_source and lang_code == lang1_code:
                    continue

                model_lang = get_model_language_code(lang_code)

                embeddings = visualizer.get_embeddings(words, model_name, model_lang)
                if embeddings is not None:
                    all_embeddings.append(embeddings)
                    labels.extend(words)

        if not all_embeddings:
            st.error("Failed to generate embeddings for any language.")
            return False

        combined_embeddings = np.vstack(all_embeddings)

    # Reduce dimensions
    dims = 3 if dimensions == "3D" else 2
    reduced_embeddings = reducer.reduce_dimensions(
        combined_embeddings,
        method=method_name,
        dimensions=dims
    )

    if reduced_embeddings is None:
        return False

    # Clear previous visualization data to prevent memory buildup
    if 'current_figure' in st.session_state:
        st.session_state.current_figure = None

    # Dataset name is now passed as parameter

    # Collect active language codes for chart title
    active_lang_codes = []
    if selected_languages is not None:
        # NEW PARADIGM: Collect from deduplicated unique_datasets
        if 'unique_datasets' in locals():
            # Use deduplicated data if available
            active_lang_codes = list(unique_datasets.keys())
        else:
            # Fallback to original logic (shouldn't happen with deduplication)
            for lang_key in ['lang1', 'lang2', 'lang3']:
                lang_data = selected_languages[lang_key]
                if lang_data['selected'] and lang_data['words']:
                    if lang_data['code'] not in active_lang_codes:
                        active_lang_codes.append(lang_data['code'])
    else:
        # LEGACY PARADIGM: Collect from source/target
        if source_words and source_selected:
            lang1 = st.session_state.get('lang1', 'Chinese')
            lang1_code = get_language_code_from_name(lang1)
            active_lang_codes.append(lang1_code)

        for lang_code, words in target_words_dict.items():
            if words and target_selected_dict.get(lang_code, False):
                if lang_code not in active_lang_codes:  # Avoid duplicates
                    active_lang_codes.append(lang_code)

    # Use the colors passed from the component (contains proper semantic colors)
    # Store data in session state for rotation and analysis
    st.session_state.visualization_data = {
        'reduced_embeddings': reduced_embeddings,
        'labels': labels,
        'colors': colors,  # Use colors passed from component, not recalculated ones
        'model_name': model_name,
        'method_name': method_name,
        'dimensions': dimensions,
        'do_clustering': do_clustering,
        'n_clusters': n_clusters,
        'dataset_name': dataset_name,
        'lang_codes': active_lang_codes
    }

    # Store the current visualization parameters for auto-save use
    st.session_state.current_viz_params = {
        'method_name': method_name,
        'model_name': model_name,
        'dataset_name': dataset_name,
        'active_languages': active_lang_codes
    }

    # Create ECharts visualization - use the colors passed from component
    visualizer.create_plot(
        reduced_embeddings,
        labels,
        colors,  # Use colors passed from component
        model_name,
        method_name,
        dimensions,
        do_clustering,
        n_clusters,
        dataset_name,
        highlight_config,
        active_lang_codes
    )

    # Note: Auto-save is now handled universally by the plot_2d() function in plotting_echarts.py
    # No duplicate auto-save logic needed here

    return True

def perform_geometric_analysis(analyzer, params):
    """Perform comprehensive geometric analysis on visualization data"""
    if 'visualization_data' not in st.session_state:
        st.error("No visualization data available for geometric analysis")
        return

    viz_data = st.session_state.visualization_data
    embeddings = viz_data['reduced_embeddings']
    labels = viz_data['labels']
    colors = viz_data.get('colors', [])

    # Store analysis results
    analysis_results = {}

    # Clustering Analysis
    if params.get('enable_clustering', False):
        clustering_results = analyzer.analyze_clustering(
            embeddings,
            params['n_clusters'],
            params['density_radius'],
            labels
        )
        analysis_results['clustering'] = clustering_results

    # Branching Analysis
    if params.get('enable_branching', False):
        branching_results = analyzer.analyze_branching(
            embeddings,
            labels,
            params['connectivity_threshold']
        )
        analysis_results['branching'] = branching_results

    # Void Analysis
    if params.get('enable_void', False):
        void_results = analyzer.analyze_voids(
            embeddings,
            params['void_confidence']
        )
        analysis_results['void'] = void_results

    # Store results in session state
    st.session_state.geometric_analysis_results = analysis_results

    # Save metrics and display results
    try:
        input_name = st.session_state.get('cfg_input_text_entered', 'untitled')
        if not input_name or input_name == 'untitled':
            input_name = st.session_state.get('cfg_input_text_selected', 'MISSING_DATASET_3')

        languages = []
        unique_colors = set(colors) if colors else set()
        if 'chinese' in unique_colors:
            languages.append('chinese')
        if 'english' in unique_colors:
            languages.append('english')

        model_name = viz_data.get('model_name', 'unknown-model')
        method_name = viz_data.get('method_name', 'unknown-method')

        save_json = params.get('save_json_files', False)
        saved_files = analyzer.save_metrics_to_files(
            analysis_results, input_name, model_name, method_name, languages, save_json
        )

        analyzer.display_metrics_save_status(saved_files)

    except Exception as e:
        st.warning(f"Could not save metrics automatically: {str(e)}")

    # Display results
    display_geometric_analysis_results(analyzer, analysis_results, embeddings, labels)

def display_geometric_analysis_results(analyzer, results, embeddings, labels):
    """Display geometric analysis results in the UI"""
    with st.expander("🔬 Geometric Analysis Results", expanded=False):

        # Create tabs for different analysis types
        tabs = []
        tab_names = []

        if 'clustering' in results:
            tab_names.append("🔍 Clustering")
        if 'branching' in results:
            tab_names.append("🌿 Branching")
        if 'void' in results:
            tab_names.append("=s Voids")
        if len(results) > 1:
            tab_names.append("📊 Summary")

        if tab_names:
            tabs = st.tabs(tab_names)

            tab_idx = 0

            # Clustering tab
            if 'clustering' in results:
                with tabs[tab_idx]:
                    analyzer.display_clustering_metrics(results['clustering'])
                tab_idx += 1

            # Branching tab
            if 'branching' in results:
                with tabs[tab_idx]:
                    analyzer.display_branching_metrics(results['branching'])
                tab_idx += 1

            # Void tab
            if 'void' in results:
                with tabs[tab_idx]:
                    analyzer.display_void_metrics(results['void'])
                tab_idx += 1

            # Summary tab
            if len(results) > 1 and tab_idx < len(tabs):
                with tabs[tab_idx]:
                    st.subheader("📈 Key Metrics Summary")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if 'clustering' in results and 'basic_metrics' in results['clustering']:
                            silhouette = results['clustering']['basic_metrics'].get('silhouette_score', 0)
                            st.metric("Clustering Quality", f"{silhouette:.3f}",
                                    help="Silhouette score: measures how well-separated clusters are.")

                    with col2:
                        if 'branching' in results and 'linearity_scores' in results['branching']:
                            linearity = results['branching']['linearity_scores'].get('overall_linearity', 0)
                            st.metric("Overall Linearity", f"{linearity:.3f}",
                                    help="Principal component variance ratio: measures linearity of data layout.")

                    with col3:
                        if 'void' in results and 'void_regions' in results['void']:
                            void_count = results['void']['void_regions'].get('num_voids', 0)
                            st.metric("Void Regions Found", void_count,
                                    help="Number of empty regions detected in the embedding space.")

def main(debug_flag=False):
    # Check login status
    check_login()

    st.subheader("📊 Semanscope - ECharts")

    st.info("👈 Configure text input and visualization settings in the sidebar and click **Visualize** to start 2D / 3D charting")

    # Initialize components
    visualizer = EChartsEmbeddingVisualizer()
    reducer = DimensionReducer()
    geometric_analyzer = GeometricAnalyzer()

    # Use global settings for most controls, but handle dimensions locally
    viz_settings = get_global_viz_settings()
    model_name = viz_settings['model_name']
    method_name = viz_settings['method_name']
    do_clustering = viz_settings['do_clustering']
    n_clusters = viz_settings['n_clusters']

    # Get ECharts settings from global settings
    echarts_settings = GlobalSettingsManager.get_echarts_settings()

    # Get input words first (text input at top of sidebar)
    # Get input words using unified component
    with st.sidebar:
        enter_text_widget = EnterTextDataWidget(key_prefix="echarts_")
        text_data = enter_text_widget.render(visualizer)

        
        # Convert to legacy format for compatibility
        languages_data = text_data['languages']
        selected_languages = {}
        source_words = []
        target_words_dict = {}
        colors = []
        source_selected = False
        target_selected_dict = {}
        
        # Build selected_languages structure and legacy compatibility
        for i, (lang_name, lang_code, text_content, is_selected) in enumerate(languages_data):
            lang_key = f"lang{i+1}"
            words = [word.strip() for word in text_content.split('\n') if word.strip()] if text_content else []
            
            selected_languages[lang_key] = {
                'name': lang_name,
                'code': lang_code,
                'words': words,
                'selected': is_selected
            }
            
            # Legacy compatibility with domain-aware color mapping (EXACT COPY from main Semanscope page)
            if i == 0 and is_selected:  # First language as source
                source_words = words
                source_selected = is_selected
                # Check for semantic colors stored in session state (from Load Text button)
                semantic_colors_key = f"{lang_code}_semantic_colors"
                word_color_map = st.session_state.get(semantic_colors_key, {})
                if not word_color_map:
                    # Try to load semantic colors directly if not in session state
                    input_name = text_data.get('input_name_selected', '')
                    # Fallback: try to get from session state directly if text_data doesn't have it
                    if not input_name:
                        input_name = st.session_state.get('input_name_selected', '')
                    # Additional fallback: if we're seeing ACL-2-word-v2 pattern, use that
                    if not input_name and any('ACL-2-word-v2' in str(w) for w in words[:5]):
                        input_name = 'ACL-2-word-v2'

                    if input_name and hasattr(visualizer, 'load_semantic_data_from_file'):
                        try:
                            semantic_words, word_color_map = visualizer.load_semantic_data_from_file(input_name, lang_code)
                            if word_color_map:
                                # Store in session state for future use
                                st.session_state[semantic_colors_key] = word_color_map
                        except Exception as e:
                            st.error(f"Error loading semantic colors: {e}")
                            word_color_map = {}

                if word_color_map:  # Use domain colors if available
                    # Better fallback color for missing words - use a bright purple to make missing mappings obvious
                    word_colors = [word_color_map.get(word, '#FF00FF') for word in words]
                    colors.extend(word_colors)
                    # Debug: Show successful color mapping
                    if debug_flag: st.success(f"🎨 Using domain colors for {lang_code}: {len(word_color_map)} mappings found")
                    # Debug: Log color assignment (silent)
                    color_stats = {}
                    for color in word_colors:
                        color_stats[color] = color_stats.get(color, 0) + 1
                else:
                    st.warning(f"⚠️ No domain colors found for {lang_code}. Using language fallback colors.")
                    # Fallback to language colors
                    colors.extend([COLOR_MAP.get(lang_name.lower(), COLOR_MAP.get(lang_code, '#666666'))] * len(words))
            elif is_selected:  # Other languages as targets
                target_words_dict[lang_code] = words
                target_selected_dict[lang_code] = is_selected
                # Check for semantic colors stored in session state (from Load Text button)
                semantic_colors_key = f"{lang_code}_semantic_colors"
                word_color_map = st.session_state.get(semantic_colors_key, {})
                if not word_color_map:
                    # Try to load semantic colors directly if not in session state
                    input_name = text_data.get('input_name_selected', '')
                    # Fallback: try to get from session state directly if text_data doesn't have it
                    if not input_name:
                        input_name = st.session_state.get('input_name_selected', '')
                    # Additional fallback: if we're seeing ACL-2-word-v2 pattern, use that
                    if not input_name and any('ACL-2-word-v2' in str(w) for w in words[:5]):
                        input_name = 'ACL-2-word-v2'

                    if input_name and hasattr(visualizer, 'load_semantic_data_from_file'):
                        try:
                            semantic_words, word_color_map = visualizer.load_semantic_data_from_file(input_name, lang_code)
                            if word_color_map:
                                # Store in session state for future use
                                st.session_state[semantic_colors_key] = word_color_map
                        except Exception as e:
                            st.error(f"Error loading semantic colors: {e}")
                            word_color_map = {}
                
                if word_color_map:  # Use domain colors if available
                    # Better fallback color for missing words - use a bright purple to make missing mappings obvious
                    word_colors = [word_color_map.get(word, '#FF00FF') for word in words]
                    colors.extend(word_colors)
                    # Debug: Log color assignment (silent)
                    color_stats = {}
                    for color in word_colors:
                        color_stats[color] = color_stats.get(color, 0) + 1
                else:
                    # Fallback to language colors
                    colors.extend([COLOR_MAP.get(lang_name.lower(), COLOR_MAP.get(lang_code, '#666666'))] * len(words))
        
        # Action buttons
        # st.markdown("---")
        btn_visualize = st.button(
            "Visualize",
            type="primary",
            width='stretch',
            help="Generate ECharts visualization"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            btn_rotate_90 = st.button(
                "🔄 Rotate 90°",
                width='stretch',
                help="Rotate the plot by 90 degrees"
            )
        with col2:
            btn_save_png = st.button(
                "💾 Save PNG",
                width='stretch',
                help="Save current visualization as PNG"
            )

    # Organize sidebar in logical sections
    with st.sidebar:

        # Dimensions control (only available on ECharts page)
        # with st.expander("📐 Visualization Dimensions", expanded=False):
        dimensions = st.radio(
            "Choose Dimensions",
            options=["2D", "3D"],
            horizontal=True,
            index=0,  # Default to 3D since this is the ECharts 3D page
            help="Select 2D or 3D visualization (only available on ECharts page)",
            key="echarts_dimensions"
        )

        # Word Search Settings
        highlight_config = visualizer.echarts_plot_manager.get_highlight_settings()


        # Show global settings link
        GlobalSettingsManager.render_current_settings_summary()

        with st.expander("⚙️ PNG Export Settings", expanded=False):

            # Save Config button
            # COMMENTED OUT: JSON config not needed
            # if st.button("💾 Save Config JSON", help="Save ECharts configuration as JSON"):
            #     if 'current_echarts_config' in st.session_state:
            #         current_input = st.session_state.get('cfg_input_text_selected', 'MISSING_DATASET_4')
            #
            #         # Determine active languages for filename using new trilingual structure
            #         active_languages = []
            #
            #         # Use deduplicated data if available from visualization
            #         if 'visualization_data' in st.session_state and 'lang_codes' in st.session_state.visualization_data:
            #             # Use language codes from the last successful visualization (already deduplicated)
            #             active_languages = st.session_state.visualization_data['lang_codes']
            #         else:
            #             # Fallback: Check all three language slots (without deduplication)
            #             for lang_key in ['lang1', 'lang2', 'lang3']:
            #                 if lang_key in selected_languages:
            #                     lang_data = selected_languages[lang_key]
            #                     if lang_data['selected'] and lang_data['words']:
            #                         if lang_data['code'] not in active_languages:
            #                             active_languages.append(lang_data['code'])
            #
            #         # Get visualization parameters from session state (same as chart title)
            #         viz_params = st.session_state.get('current_viz_params', {})
            #         viz_method_name = viz_params.get('method_name', method_name)
            #         viz_model_name = viz_params.get('model_name', model_name)
            #         viz_dataset_name = viz_params.get('dataset_name', 'MISSING_DATASET_4')
            #         viz_active_languages = viz_params.get('active_languages', active_languages)
            #
            #         # Generate standardized filename using centralized helper with SAME parameters as chart title
            #         _, standardized_filename = create_title_and_filename(
            #             [viz_method_name],
            #             [viz_model_name],
            #             viz_dataset_name,
            #             viz_active_languages,
            #             "json"
            #         )
            #
            #         # Create echarts filename with prefix
            #         echarts_json_filename = f"echarts-{standardized_filename}"
            #
            #         saved_file = visualizer.echarts_plot_manager.save_echarts_as_png(
            #             st.session_state.current_echarts_config,
            #             [],  # Empty filename_parts since we're using external_filename
            #             dimensions,
            #             external_filename=echarts_json_filename
            #         )
            #         if saved_file:
            #             pass  # Success message handled by auto-save notification
            #     else:
            #         st.warning("⚠️ No chart configuration available. Generate a visualization first.")

            # st.markdown("---")  # Separator

            # Check selenium availability and use global ECharts settings
            auto_save_status = visualizer.echarts_plot_manager.get_auto_save_status()

            # Use global ECharts settings for auto-save
            enable_auto_png = echarts_settings['auto_save_enabled'] and auto_save_status['available']
            if not auto_save_status['available']:
                enable_auto_png = False
                st.warning("⚠️ Selenium not available. Install with: `pip install selenium webdriver-manager`")
            elif echarts_settings['auto_save_enabled']:
                st.caption("✅ Auto-PNG export enabled (from global settings)")
            else:
                st.caption("ℹ️ Auto-PNG export disabled (from global settings)")

            # Use global settings for PNG dimensions
            png_width = echarts_settings['png_width']
            png_height = echarts_settings['png_height']

            st.info(f"📏 Using global PNG dimensions: {png_width}x{png_height}px\n\nTo modify these settings, go to the **Settings** page.")

            st.markdown("---")  # Separator

            # Info about ECharts features
            st.markdown("**ℹ️ About ECharts Features**")
            st.markdown("""
            **Apache ECharts** brings enhanced interactivity and beautiful visualizations to semantic exploration:

            - **🎯 Enhanced Interactivity**: Smooth zoom, pan, and hover interactions
            - **🎨 Rich Visual Effects**: Beautiful animations and transitions
            - **📊 Advanced Clustering**: Dynamic cluster visualization with customizable boundaries
            - **📱 Responsive Design**: Optimized for different screen sizes
            - **⚡ Performance**: Optimized rendering for large datasets

            **Note**: 3D visualizations use ECharts GL for enhanced 3D rendering capabilities.
            """)

            st.markdown("---")  # Separator

            # ECharts usage tips
            st.markdown("**💡 ECharts Interaction Tips**")
            st.markdown("""
            **Mouse Controls:**
            - **Left Click + Drag**: Pan around the visualization
            - **Mouse Wheel**: Zoom in/out
            - **Hover**: See detailed information for each point
            - **Click Legend**: Toggle cluster visibility (in clustering mode)

            **3D Controls (3D mode only):**
            - **Left Click + Drag**: Rotate the 3D scene
            - **Right Click + Drag**: Pan the 3D scene
            - **Mouse Wheel**: Zoom in/out

            **Settings:**
            - Adjust text size, point size, and other visual properties in the sidebar
            - Toggle animations for smoother or faster rendering
            - Enable/disable grid lines for cleaner appearance
            """)

        # Store settings in session state
        st.session_state.echarts_auto_save = {
            'enabled': enable_auto_png,
            'width': png_width,
            'height': png_height,
            'available': auto_save_status['available']
        }



    # Use global geometric analysis settings (no UI display needed)
    enable_geometric_analysis = is_global_geometric_analysis_enabled()
    analysis_params = get_global_geometric_analysis()

    # Handle rotate button
    if btn_rotate_90:
        st.session_state.plot_rotation = (st.session_state.plot_rotation + 90) % 360
        if 'visualization_data' in st.session_state:
            viz_data = st.session_state.visualization_data
            plot_option = visualizer.create_plot(
                viz_data['reduced_embeddings'],
                viz_data['labels'],
                viz_data['colors'],
                viz_data['model_name'],
                viz_data['method_name'],
                viz_data['dimensions'],
                viz_data['do_clustering'],
                viz_data['n_clusters'],
                viz_data.get('dataset_name', 'User Input'),
                highlight_config,
                viz_data.get('lang_codes', [])
            )

            # Display the ECharts visualization
            from streamlit_echarts import st_echarts
            chart_height = f"{1100}px"  # Use publication height
            st_echarts(
                options=plot_option,
                height=chart_height,
                key=f"echarts_main_{viz_data['dimensions']}"
            )
        else:
            st.warning("Please generate a visualization first by clicking 'Visualize'")

    # Handle save image button - now saves ECharts config
    if btn_save_png:
        current_input = st.session_state.get('cfg_input_text_selected', 'MISSING_DATASET_5')

        # Save ECharts configuration if available
        if 'current_echarts_config' in st.session_state:
            # Determine active languages for filename using new trilingual structure
            active_languages = []

            # Use deduplicated data if available from visualization
            if 'visualization_data' in st.session_state and 'lang_codes' in st.session_state.visualization_data:
                # Use language codes from the last successful visualization (already deduplicated)
                active_languages = st.session_state.visualization_data['lang_codes']
            else:
                # Fallback: Check all three language slots (without deduplication)
                for lang_key in ['lang1', 'lang2', 'lang3']:
                    if lang_key in selected_languages:
                        lang_data = selected_languages[lang_key]
                        if lang_data['selected'] and lang_data['words']:
                            if lang_data['code'] not in active_languages:
                                active_languages.append(lang_data['code'])

            # Get visualization parameters from session state (same as chart title)
            viz_params = st.session_state.get('current_viz_params', {})
            viz_method_name = viz_params.get('method_name', method_name)
            viz_model_name = viz_params.get('model_name', model_name)
            viz_dataset_name = viz_params.get('dataset_name', 'MISSING_DATASET_5')
            viz_active_languages = viz_params.get('active_languages', active_languages)

            # Generate standardized filename using centralized helper with SAME parameters as chart title
            _, standardized_filename = create_title_and_filename(
                [viz_method_name],
                [viz_model_name],
                viz_dataset_name,
                viz_active_languages,
                "json"
            )

            # Create echarts filename with prefix
            echarts_json_filename = f"echarts-{standardized_filename}"

            saved_file = visualizer.echarts_plot_manager.save_echarts_as_png(
                st.session_state.current_echarts_config,
                [],  # Empty filename_parts since we're using external_filename
                dimensions,
                external_filename=echarts_json_filename
            )
            if saved_file:
                pass  # Success message handled by auto-save notification
        else:
            # Fallback to regular PNG save if no ECharts config
            # Determine which languages are selected for compatibility with save function
            manual_chinese_selected = source_selected and source_words  # Source words treated as Chinese for compatibility
            manual_english_selected = target_selected_dict.get('enu', False)  # Check if English is selected in targets

            file_png = save_plot_image(visualizer, current_input, model_name, method_name, manual_chinese_selected, manual_english_selected)
            if file_png:
                st.sidebar.success(f"Image saved as: {file_png}")
                st.image(f"data/images/{file_png}", caption=f"{file_png}", width='stretch')
            else:
                st.error("Failed to save image.")

    # Handle visualize button
    # Debug info removed - color-coding working correctly

    if btn_visualize:
        # Use new paradigm with selected_languages
        # Get dataset name from text_data with multiple fallbacks
        dataset_name = 'User Input'
        if text_data:
            dataset_name = text_data.get('input_name_selected', '')
        if not dataset_name:
            dataset_name = st.session_state.get('input_name_selected', '')
        if not dataset_name:
            # Check if we can detect ACL-2-word-v2 from the actual word patterns
            if source_words and any('ACL' in str(w) for w in source_words[:10]):
                dataset_name = 'ACL-2-word-v2'
            else:
                dataset_name = 'User Input'

        # Debug: Show what dataset name was detected (FORCE CACHE INVALIDATION)
        if debug_flag: st.info(f"🏷️ Dataset name detected: '{dataset_name}' [DEBUG v2.0]")
        success = generate_visualization_echarts(visualizer, reducer, source_words, target_words_dict, colors, model_name, method_name, dimensions, do_clustering, n_clusters, source_selected, target_selected_dict, highlight_config, selected_languages, dataset_name)

        # Perform geometric analysis if enabled and visualization was successful (only for 2D)
        if success and enable_geometric_analysis and analysis_params and 'visualization_data' in st.session_state:
            if dimensions == "2D":
                with st.spinner("🔬 Performing geometric analysis..."):
                    perform_geometric_analysis(geometric_analyzer, analysis_params)
            else:
                st.info("9 Geometric analysis is only available for 2D visualizations.")
    else:
        # Clean main panel - show simple placeholder
        # st.info(DEFAULT_INSTRUCTION_MSG_1)
        show_overview()


def show_overview():
    with st.expander("Overview", expanded=False):
        st.markdown("Advanced 3D/2D semantic visualization with interactive charts and word search capabilities")
        st.markdown("""
        ### 🎯 Advanced Chart Features:

        **🔍 Word Search**: Locate and highlight specific words in your visualizations
        - Smart text matching that handles punctuation and variations
        - Customizable colors and sizes for search results
        - Support for multiple search terms with different colors

        **🌐 3D & 2D Visualization**: Interactive Apache ECharts powered charts
        - Smooth 3D rotation and zoom capabilities
        - Toggle between 2D and 3D perspectives
        - High-performance rendering for large datasets

        **💾 Export & Sharing**: Save your visualizations and configurations
        - PNG export with customizable resolution and DPI
        - JSON configuration export for reproducibility
        - Rotation controls for optimal viewing angles

        **🎨 Advanced Styling**: Professional chart customization
        - Dark/light theme support
        - Customizable point sizes and colors
        - Grid and axis controls for publication-ready charts
        """)


if __name__ == "__main__":
    main()
