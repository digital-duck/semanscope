import streamlit as st
import numpy as np
from semanscope.components.embedding_viz import EmbeddingVisualizer
from semanscope.components.dimension_reduction import DimensionReducer
from semanscope.components.geometric_analysis import GeometricAnalyzer
from semanscope.components.shared.enter_text_data import EnterTextDataWidget
from semanscope.components.word_search import WordSearchManager

from semanscope.config import (
    check_login,
    COLOR_MAP,
    LANGUAGE_CODE_MAP,
    DEFAULT_LANG,
    DEFAULT_INSTRUCTION_MSG_1,
    DATA_PATH,
    get_language_code_from_name,
    get_model_language_code
)
from semanscope.utils.download_helpers import handle_download_button
from semanscope.utils.global_settings import (
    GlobalSettingsManager,
    get_global_viz_settings,
    get_global_geometric_analysis,
    is_global_geometric_analysis_enabled
)


# Page config
st.set_page_config(
    page_title="Semanscope",
    page_icon="🧭",
    layout="wide"
)

@st.fragment 
def save_plot_image(visualizer, current_input, model_name, method_name, chinese_selected, english_selected):
    return visualizer.save_plot_image(current_input, model_name, method_name, chinese_selected, english_selected)

@st.fragment
def generate_visualization(visualizer, reducer, source_words, target_words_dict, colors, model_name, method_name, dimensions, do_clustering, n_clusters, source_selected=True, target_selected_dict=None, selected_languages=None, debug_flag=False, word_search_config=None):
    """Generate embeddings and visualization with full multilingual support

    Args:
        selected_languages: (NEW) Dictionary with lang1/lang2/lang3 structure (preferred)
        source_words, target_words_dict: (LEGACY) Old source/target paradigm (backward compatibility)
    """
    if target_selected_dict is None:
        target_selected_dict = {}

    # Debug flag test
    if debug_flag:
        with st.expander("🔍 DEBUG OUTPUT", expanded=True):
            st.success(f"✅ DEBUG MODE ENABLED - Using model: {model_name}")
            st.info("Debug messages will appear here as embeddings are processed...")

            # Debug: Show what languages and words are actually being processed
            if selected_languages is not None:
                st.write("**Languages being processed:**")
                for lang_key in ['lang1', 'lang2', 'lang3']:
                    lang_data = selected_languages[lang_key]
                    if lang_data['selected'] and lang_data['words']:
                        st.write(f"• {lang_data['name']} ({lang_data['code']}): {len(lang_data['words'])} words")
                        # Show first few words
                        preview_words = lang_data['words'][:10]
                        st.write(f"  📝 Sample words: {', '.join(preview_words)}")
                        if len(lang_data['words']) > 10:
                            st.write(f"  ... and {len(lang_data['words']) - 10} more")

    # Use centralized model language mapping from config

    # NEW PARADIGM: Use selected_languages if provided
    if selected_languages is not None:
        # Modern trilingual approach - treat all languages equally
        has_any_language = any(
            lang_data['selected'] and lang_data['words']
            for lang_data in selected_languages.values()
        )

        if not has_any_language:
            st.warning("Please select at least one language and enter words/phrases.")
            return False

        # DEDUPLICATION: Remove duplicate datasets before processing
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

        # Process embeddings for unique languages only
        all_embeddings = []
        labels = []
        deduplicated_colors = []  # Build colors for deduplicated data only

        # Process each unique language dataset once
        for lang_code, dataset_info in unique_datasets.items():
            model_lang = get_model_language_code(lang_code)
            embeddings = visualizer.get_embeddings(dataset_info['words'], model_name, model_lang, debug_flag)
            if embeddings is not None:
                # Ensure embeddings and words have same length
                if len(embeddings) == len(dataset_info['words']):
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
                        dataset_colors = [COLOR_MAP[color_key]] * len(dataset_info['words'])
                    
                    deduplicated_colors.extend(dataset_colors)
                else:
                    st.warning(f"⚠️ Embedding count mismatch for {dataset_info['name']}: {len(embeddings)} embeddings vs {len(dataset_info['words'])} words. Skipping this dataset.")

        # Show deduplication info if duplicates were found
        duplicate_count = sum(len(info['positions']) - 1 for info in unique_datasets.values())
        if duplicate_count > 0:
            duplicated_langs = [f"{info['name']} (positions: {', '.join(info['positions'])})"
                              for info in unique_datasets.values() if len(info['positions']) > 1]
            st.info(f"🔄 **Deduplication Applied**: Processed {len(unique_datasets)} unique datasets (skipped {duplicate_count} duplicates). Duplicated: {', '.join(duplicated_langs)}")

        if not all_embeddings:
            st.error("Failed to generate embeddings for any language.")
            return False

        combined_embeddings = np.vstack(all_embeddings)
        # Use deduplicated colors instead of the original colors parameter
        colors = deduplicated_colors

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

            source_embeddings = visualizer.get_embeddings(source_words, model_name, source_model_lang, debug_flag)
            if source_embeddings is not None:
                # Ensure embeddings and words have same length
                if len(source_embeddings) == len(source_words):
                    all_embeddings.append(source_embeddings)
                    labels.extend(source_words)
                else:
                    st.warning(f"⚠️ Embedding count mismatch for source: {len(source_embeddings)} embeddings vs {len(source_words)} words.")

        # Process each target language
        for lang_code, words in target_words_dict.items():
            if words and target_selected_dict.get(lang_code, False):
                # Skip target language processing if it's the same as source language (avoid duplication)
                if has_source and lang_code == lang1_code:
                    continue

                model_lang = get_model_language_code(lang_code)

                embeddings = visualizer.get_embeddings(words, model_name, model_lang, debug_flag)
                if embeddings is not None:
                    # Ensure embeddings and words have same length
                    if len(embeddings) == len(words):
                        all_embeddings.append(embeddings)
                        labels.extend(words)
                    else:
                        st.warning(f"⚠️ Embedding count mismatch for {lang_code}: {len(embeddings)} embeddings vs {len(words)} words.")

        if not all_embeddings:
            st.error("Failed to generate embeddings for any language.")
            return False

        combined_embeddings = np.vstack(all_embeddings)

    # Reduce dimensions with research-oriented error handling and cross-page caching
    dims = 3 if dimensions == "3D" else 2
    try:
        # Use cached dimension reduction when possible
        # Dataset = all text labels, combined language info for cache key
        cache_lang = "multi" if len(set([get_language_code_from_name(lang) for lang in [
            st.session_state.get('lang1', 'Chinese'),
            st.session_state.get('lang2', 'English')
        ]])) > 1 else get_language_code_from_name(st.session_state.get('lang1', 'Chinese'))

        reduced_embeddings = reducer.reduce_dimensions_with_cache(
            combined_embeddings,
            method=method_name,
            dimensions=dims,
            dataset=labels,  # Use text labels as dataset identifier
            lang=cache_lang,  # Combined language code
            model=model_name  # Model name for cache key
        )

        if reduced_embeddings is None:
            st.error(f"🔬 **Research Error**: {method_name} returned None result")
            st.info("💡 **Research Note**: This indicates the method failed to process your data. Consider trying a different dimensionality reduction method or examining your input data for issues.")
            return False

    except Exception as e:
        # Method-specific failure - let the user know this is a research finding
        st.error(f"🔬 **Research Finding**: {method_name} is incompatible with this dataset")

        with st.expander("🔍 Method Failure Analysis", expanded=True):
            st.markdown("**Research Insight**: This failure provides valuable information about the limitations of different dimensionality reduction methods with your specific data type.")
            st.code(str(e))

        st.info("💡 **Next Steps**: Try a different dimensionality reduction method from the sidebar to compare which methods work best with your data.")
        return False

    # Clear previous visualization data to prevent memory buildup
    if 'current_figure' in st.session_state:
        st.session_state.current_figure = None

    # Get dataset name from selected input (use same key as selectbox widget)
    dataset_name = st.session_state.get('main_cfg_input_text_selected', 'User Input')
    # Additional fallbacks for compatibility
    if not dataset_name or dataset_name == 'User Input':
        dataset_name = st.session_state.get('input_name_selected', 'User Input')
    if not dataset_name or dataset_name == 'User Input':
        dataset_name = st.session_state.get('cfg_input_text_selected', 'User Input')

    # Collect active language codes for chart title
    active_lang_codes = []
    if selected_languages is not None:
        # NEW PARADIGM: Collect from selected_languages
        for lang_key in ['lang1', 'lang2', 'lang3']:
            lang_data = selected_languages[lang_key]
            if lang_data['selected'] and lang_data['words']:
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
    # Store data in session state for rotation
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

    # Create visualization - use the colors passed from component
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
        active_lang_codes,
        word_search_config
    )

    # Display collected debug logs if debug mode was enabled
    if debug_flag and 'debug_logs' in st.session_state and st.session_state.debug_logs:
        with st.expander("🔍 DETAILED DEBUG LOGS", expanded=True):
            st.write("**Complete Debug Information:**")
            for log_msg in st.session_state.debug_logs:
                st.write(f"• {log_msg}")
            # Clear logs for next run
            st.session_state.debug_logs = []
    
    # Auto-save the visualization using standardized helper
    try:
        # Get current input name for filename (consistent with title logic)
        current_input = st.session_state.get('main_cfg_input_text_selected', 'untitled')
        if not current_input or current_input == 'untitled':
            current_input = st.session_state.get('cfg_input_text_entered', 'untitled')
        if not current_input or current_input == 'untitled':
            current_input = st.session_state.get('input_name_selected', 'sample_1')
            # Fallback to dual-view page key
            if not current_input:
                current_input = st.session_state.get('cfg_input_text_selected', 'sample_1')

        # Determine currently selected languages from the UI (not from legacy data)
        current_selected_languages = []

        # Use the selected_languages from the new paradigm to get actual current selection
        if selected_languages:
            for lang_key in ['lang1', 'lang2', 'lang3']:
                lang_data = selected_languages[lang_key]
                if lang_data['selected'] and lang_data['words']:
                    current_selected_languages.append(lang_data['code'])

        # Clear any previous session state language flags
        for lang_code in ['chn', 'enu', 'deu', 'fra', 'spa', 'ara', 'heb', 'hin', 'jpn', 'kor', 'rus', 'tha', 'vie']:
            if f'{lang_code}_include_checkbox' in st.session_state:
                del st.session_state[f'{lang_code}_include_checkbox']

        # Set session state keys only for currently selected languages
        for lang_code in current_selected_languages:
            st.session_state[f'{lang_code}_include_checkbox'] = True

        # Auto-save now handled automatically by plot_2d() in plotting_echarts.py
        # No manual auto-save needed here anymore

    except Exception as error:
        st.warning(f"An error occurred: {str(error)}")
    
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
    
    # Save metrics to files automatically
    try:
        # Get input name from session state (consistent with title logic)
        input_name = st.session_state.get('main_cfg_input_text_selected', 'untitled')
        if not input_name or input_name == 'untitled':
            input_name = st.session_state.get('cfg_input_text_entered', 'untitled')
        if not input_name or input_name == 'untitled':
            input_name = st.session_state.get('cfg_input_text_selected', 'sample_1')
        
        # Determine languages from visualization data
        languages = []
        
        # Check for unique language types in colors array
        unique_colors = set(colors) if colors else set()
        if 'chinese' in unique_colors:
            languages.append('chinese')
        if 'english' in unique_colors:
            languages.append('english')
        
        # Get model and method info
        model_name = viz_data.get('model_name', 'unknown-model')
        method_name = viz_data.get('method_name', 'unknown-method')
        
        # Save metrics
        save_json = params.get('save_json_files', False)
        saved_files = analyzer.save_metrics_to_files(
            analysis_results, input_name, model_name, method_name, languages, save_json
        )
        
        # Display save status
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
            tab_names.append("🕳️ Voids")
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
            
            # Summary tab (if multiple analyses were performed)
            if len(results) > 1 and tab_idx < len(tabs):
                with tabs[tab_idx]:
                    # Summary statistics first
                    st.subheader("📈 Key Metrics Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'clustering' in results and 'basic_metrics' in results['clustering']:
                            silhouette = results['clustering']['basic_metrics'].get('silhouette_score', 0)
                            st.metric("Clustering Quality", f"{silhouette:.3f}", 
                                    help="Silhouette score: measures how well-separated clusters are. Range [-1,1], >0.5 is good, >0.7 is excellent")
                    
                    with col2:
                        if 'branching' in results and 'linearity_scores' in results['branching']:
                            linearity = results['branching']['linearity_scores'].get('overall_linearity', 0)
                            st.metric("Overall Linearity", f"{linearity:.3f}",
                                    help="Principal component variance ratio: measures how linear/straight the data layout is. Higher values indicate more linear arrangement")
                    
                    with col3:
                        if 'void' in results and 'void_regions' in results['void']:
                            void_count = results['void']['void_regions'].get('num_voids', 0)
                            st.metric("Void Regions Found", void_count,
                                    help="Number of empty regions detected in the embedding space where no data points exist")
                    
                    st.markdown("---")
                    
                    # Clustering visualization only
                    st.subheader("🎯 Clustering Analysis Visualization")
                    
                    try:
                        # Create simplified clustering plot
                        viz_data = st.session_state.visualization_data
                        model_name = viz_data.get('model_name', 'unknown-model')
                        method_name = viz_data.get('method_name', 'unknown-method')
                        dataset_name = viz_data.get('dataset_name', 'User Input')
                        
                        clustering_fig = analyzer.create_comprehensive_analysis_plot(
                            embeddings, labels, 
                            results.get('clustering', {}),
                            results.get('branching', {}),
                            results.get('void', {}),
                            model_name, method_name, dataset_name
                        )
                        st.plotly_chart(clustering_fig, width='stretch')
                        
                        # Add download button for clustering chart
                        handle_download_button(clustering_fig, model_name, method_name, dataset_name, "clustering", "main")
                        
                        # Save plot as PNG
                        try:
                            # Get input and model info for PNG filename (consistent with title logic)
                            input_name = st.session_state.get('main_cfg_input_text_selected', 'untitled')
                            if not input_name or input_name == 'untitled':
                                input_name = st.session_state.get('cfg_input_text_entered', 'untitled')
                            if not input_name or input_name == 'untitled':
                                input_name = st.session_state.get('cfg_input_text_selected', 'sample_1')
                            
                            viz_data = st.session_state.visualization_data
                            model_name = viz_data.get('model_name', 'unknown-model')
                            method_name = viz_data.get('method_name', 'unknown-method')
                            
                            png_filename = analyzer.save_summary_plot_as_png(clustering_fig, input_name, model_name, method_name)
                            if png_filename:
                                st.success(f"📸 Clustering visualization saved as: {DATA_PATH / 'metrics' / png_filename}")
                            
                        except Exception as png_error:
                            st.warning(f"Could not save PNG: {str(png_error)}")
                        
                    except Exception as e:
                        st.error(f"Error creating clustering visualization: {str(e)}")

def main():
    # Check login status
    check_login()

    # Get global settings
    viz_settings = get_global_viz_settings()

    st.subheader(f"🧭 Semanscope")

    st.info("👈 Configure text input and visualization settings in the sidebar and click **Visualize** to start semantic analysis")

    # Initialize components
    visualizer = EmbeddingVisualizer()
    reducer = DimensionReducer()
    geometric_analyzer = GeometricAnalyzer()
    word_search_manager = WordSearchManager()

    # Use global settings instead of sidebar controls
    model_name = viz_settings['model_name']
    method_name = viz_settings['method_name']
    dimensions = viz_settings['dimensions']
    do_clustering = viz_settings['do_clustering']
    n_clusters = viz_settings['n_clusters']

    # Get geometric analysis settings from global configuration
    enable_geometric_analysis = is_global_geometric_analysis_enabled()
    analysis_params = get_global_geometric_analysis()

    # Get input words using unified component
    with st.sidebar:
        enter_text_widget = EnterTextDataWidget(key_prefix="main_")
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
            
            # Legacy compatibility with domain-aware color mapping
            if i == 0 and is_selected:  # First language as source
                source_words = words
                source_selected = is_selected
                # Check for semantic colors stored in session state (from Load Text button)
                semantic_colors_key = f"{lang_code}_semantic_colors"
                word_color_map = st.session_state.get(semantic_colors_key, {})
                if not word_color_map:
                    # Try to load semantic colors directly if not in session state
                    input_name = text_data.get('input_name_selected', '')
                    if input_name and hasattr(visualizer, 'load_semantic_data_from_file'):
                        try:
                            semantic_words, word_color_map = visualizer.load_semantic_data_from_file(input_name, lang_code)
                            if word_color_map:
                                # Store in session state for future use
                                st.session_state[semantic_colors_key] = word_color_map
                                # Debug: Show some color mapping info
                                if len(word_color_map) > 0:
                                    sample_mappings = list(word_color_map.items())[:5]
                                    st.info(f"🎨 Loaded {len(word_color_map)} domain colors for {lang_code}. Sample: {sample_mappings}")
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
            elif is_selected:  # Other languages as targets
                target_words_dict[lang_code] = words
                target_selected_dict[lang_code] = is_selected
                # Check for semantic colors stored in session state (from Load Text button)
                semantic_colors_key = f"{lang_code}_semantic_colors"
                word_color_map = st.session_state.get(semantic_colors_key, {})
                if not word_color_map:
                    # Try to load semantic colors directly if not in session state
                    input_name = text_data.get('input_name_selected', '')
                    if input_name and hasattr(visualizer, 'load_semantic_data_from_file'):
                        try:
                            semantic_words, word_color_map = visualizer.load_semantic_data_from_file(input_name, lang_code)
                            if word_color_map:
                                # Store in session state for future use
                                st.session_state[semantic_colors_key] = word_color_map
                                # Debug: Show some color mapping info
                                if len(word_color_map) > 0:
                                    sample_mappings = list(word_color_map.items())[:5]
                                    st.info(f"🎨 Loaded {len(word_color_map)} domain colors for {lang_code}. Sample: {sample_mappings}")
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
            help="Generate semantic visualization"
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

    # Add sidebar configuration
    with st.sidebar:
        # Display link to global settings
        GlobalSettingsManager.render_current_settings_summary()

        # Word Search settings
        word_search_config = word_search_manager.get_word_search_settings(key_prefix="main")

        # Debug settings
        with st.expander("🔍 Debug Settings", expanded=False):
            debug_flag = st.checkbox(
                "Debug Character Encoding",
                value=False,
                help="Show detailed debugging information for character encoding failures",
                key="debug_character_encoding_main"
            )


    # Handle rotate button - reuse existing visualization data
    if btn_rotate_90:
        st.session_state.plot_rotation = (st.session_state.plot_rotation + 90) % 360
        # If we have existing visualization data, redraw with rotation
        if 'visualization_data' in st.session_state:
            viz_data = st.session_state.visualization_data
            visualizer.create_plot(
                viz_data['reduced_embeddings'],
                viz_data['labels'],
                viz_data['colors'],
                viz_data['model_name'],
                viz_data['method_name'],
                viz_data['dimensions'],
                viz_data['do_clustering'],
                viz_data['n_clusters'],
                viz_data.get('dataset_name', 'User Input'),
                viz_data.get('lang_codes', []),
                word_search_config
            )
        else:
            st.warning("Please generate a visualization first by clicking 'Visualize'")
    
    # Handle save image button
    if btn_save_png:
        # Get current input name (consistent with title and auto-save logic)
        current_input = st.session_state.get('main_cfg_input_text_selected', 'untitled')
        if not current_input or current_input == 'untitled':
            current_input = st.session_state.get('cfg_input_text_selected', 'untitled')

        # Determine active languages for filename based on trilingual selection
        active_languages = []

        # Check all three language slots
        lang1 = st.session_state.get('lang1', 'Chinese')
        lang2 = st.session_state.get('lang2', 'English')
        lang3 = st.session_state.get('lang3', 'German')

        # Add source language if selected
        if source_selected and source_words:
            lang1_code = get_language_code_from_name(lang1)
            active_languages.append(lang1_code)

        # Add target languages if selected
        for lang_code, words in target_words_dict.items():
            if target_selected_dict.get(lang_code, False) and words:
                if lang_code not in active_languages:
                    active_languages.append(lang_code)

        # For backward compatibility with save function, determine Chinese and English flags
        chinese_selected = 'chn' in active_languages
        english_selected = 'enu' in active_languages

        file_png = save_plot_image(visualizer, current_input, model_name, method_name, chinese_selected, english_selected)
        if file_png:
            st.sidebar.success(f"Image saved as: {file_png}")
            st.image(str(DATA_PATH / "images" / file_png), caption=f"{file_png}", width='stretch')
        else:
            st.error("Failed to save image.")

    # Handle visualize button
    if btn_visualize:
        # Check if 3D is selected and show warning
        if dimensions == "3D":
            st.warning("🚧 **3D visualization is disabled on this page**")
            st.info("💡 **Please use the 'Semanscope-ECharts' page for interactive 3D visualizations with better performance and controls.**")
            st.stop()

        # Use new paradigm with selected_languages (only for 2D)
        success = generate_visualization(visualizer, reducer, source_words, target_words_dict, colors, model_name, method_name, dimensions, do_clustering, n_clusters, source_selected, target_selected_dict, selected_languages, debug_flag, word_search_config)
        
        # Perform geometric analysis if enabled and visualization was successful (only for 2D)
        if success and enable_geometric_analysis and analysis_params and 'visualization_data' in st.session_state:
            if dimensions == "2D":
                with st.spinner("🔬 Performing geometric analysis..."):
                    perform_geometric_analysis(geometric_analyzer, analysis_params)
            else:
                st.info("ℹ️ Geometric analysis is only available for 2D visualizations.")
    else:
        # Clean main panel - show simple placeholder
        # st.info(DEFAULT_INSTRUCTION_MSG_1)
        show_overview()

def show_overview():
    with st.expander("Overview", expanded=False):
        st.markdown("Core semantic visualization with embedding analysis and geometric insights")
        st.markdown("""
        ### 🎯 Core Features:

        **📊 Semantic Visualization**: Transform text into interactive 2D scatter plots
        - Multiple embedding models (Sentence-BERT, Qwen, BGE-M3, etc.)
        - Various dimensionality reduction methods (PHATE, t-SNE, UMAP, PCA)

        **🌐 Multilingual Support**: Analyze text across different languages
        - Optimized for Chinese-English semantic alignment
        - Support for 15+ languages including Arabic, Hebrew, Japanese, Korean

        **🔬 Geometric Analysis**: Mathematical insights into semantic relationships
        - Convex hull analysis for semantic boundaries
        - Centroid calculations and cluster metrics
        - Distance and angle measurements between concepts

        **🎨 Clustering & Customization**: Organize and style your visualizations
        - K-means clustering with adjustable cluster counts
        - Customizable colors and visualization parameters
        """)



if __name__ == "__main__":
    main()