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
    DEFAULT_MODEL,DEFAULT_METHOD,DEFAULT_DIMENSION,
    DEFAULT_INSTRUCTION_MSG_1,
    DEFAULT_TOP3_MODELS,
    get_language_code_from_name,
    get_model_language_code,
    MODEL_INFO,
    METHOD_INFO,
    get_active_models
)
from semanscope.utils.global_settings import (
    get_global_viz_settings,
    get_global_publication_settings,
    get_global_geometric_analysis,
    is_global_geometric_analysis_enabled,
    GlobalSettingsManager
)
from semanscope.utils.title_filename_helper import create_title_and_filename
from semanscope.components.embedding_viz import get_active_methods

# Page config
st.set_page_config(
    page_title="Semanscope - Compare",
    page_icon="⚖️",
    layout="wide"
)

def filter_nan_embeddings(embeddings, words, model_name, debug_flag=False):
    """Filter out NaN values from embeddings and corresponding words"""
    if embeddings is None:
        st.warning(f"🔍 **{model_name}**: Embeddings is None, returning empty")
        return None, []

    # Debug: Check embeddings shape and type
    if debug_flag: st.info(f"🔍 **{model_name}**: Checking embeddings shape: {embeddings.shape}, type: {type(embeddings)}")

    # Check for NaN values
    nan_mask = np.isnan(embeddings).any(axis=1)
    inf_mask = np.isinf(embeddings).any(axis=1)
    bad_mask = nan_mask | inf_mask

    if bad_mask.any():
        # Count how many embeddings contain NaN/Inf
        nan_count = nan_mask.sum()
        inf_count = inf_mask.sum()
        total_bad = bad_mask.sum()
        total_count = len(embeddings)

        st.warning(f"⚠️ **{model_name}**: Found {total_bad}/{total_count} bad embeddings (NaN: {nan_count}, Inf: {inf_count}). Filtering them out.")

        # Filter out rows with NaN/Inf values
        valid_mask = ~bad_mask
        clean_embeddings = embeddings[valid_mask]
        clean_words = [words[i] for i in range(len(words)) if valid_mask[i]]

        if len(clean_embeddings) == 0:
            st.error(f"❌ **{model_name}**: All embeddings contain NaN/Inf values. Cannot proceed.")
            return None, []

        if debug_flag: st.info(f"✅ **{model_name}**: Using {len(clean_embeddings)}/{total_count} valid embeddings after filtering.")

        # Final sanity check
        if np.isnan(clean_embeddings).any() or np.isinf(clean_embeddings).any():
            st.error(f"❌ **{model_name}**: Clean embeddings still contain NaN/Inf values!")
            return None, []

        return clean_embeddings, clean_words

    if debug_flag: st.info(f"✅ **{model_name}**: No NaN/Inf values detected in {len(embeddings)} embeddings")
    return embeddings, words

class ComparisonVisualizer(EmbeddingVisualizer):
    """Enhanced embedding visualizer for cross-comparisons using Apache ECharts"""

    def __init__(self):
        super().__init__()
        self.echarts_plot_manager = EChartsPlotManager()
        # Comparison colors: Red, Blue, Green
        self.comparison_colors = ['#DC143C', '#0000FF', '#008000']  # Crimson, Blue, Green

    def get_semantic_colors_for_labels(self, labels, language_code, dataset_name):
        """Get semantic colors for labels if available, fallback to None"""
        try:
            # Check session state for semantic colors
            semantic_colors_key = f"{language_code}_semantic_colors"
            word_color_map = st.session_state.get(semantic_colors_key, {})

            # If not in session, try to load from file
            if not word_color_map:
                if hasattr(self, 'load_semantic_data_from_file') and dataset_name:
                    try:
                        _, word_color_map = self.load_semantic_data_from_file(dataset_name, language_code)
                        if word_color_map:
                            # Store in session state for future use
                            st.session_state[semantic_colors_key] = word_color_map
                    except Exception:
                        word_color_map = {}

            # Map labels to colors if mapping exists
            if word_color_map:
                semantic_colors = []
                for label in labels:
                    if label in word_color_map:
                        semantic_colors.append(word_color_map[label])
                    else:
                        semantic_colors.append(None)  # Will use fallback
                return semantic_colors

        except Exception as e:
            # Silently handle errors and fall back to comparison colors
            pass

        return None

    def get_short_aliases(self):
        """Get short aliases for models and methods"""
        model_aliases = {
            'Qwen3-Embedding-0.6B': 'Qwen3-0.6B',
            'sentence-bert-multilingual': 'BERT-Multi',
            'EmbeddingGemma-300M': 'Gemma-300M',
            'Snowflake-Arctic-Embed2 (Ollama)': 'Arctic-E2',
            'Snowflake-Arctic-Embed (Ollama)': 'Arctic-E1',
            'BGE-M3 (Ollama)': 'BGE-M3',
            'Paraphrase-Multilingual (Ollama)': 'Para-Multi'
        }
        
        method_aliases = {
            'PHATE': 'PHATE',
            't-SNE': 'tSNE', 
            'UMAP': 'UMAP',
            'Isomap': 'Isomap',
            'PCA': 'PCA',
            'MDS': 'MDS',
            'SpectralEmbedding': 'Spectral',
            'LocallyLinearEmbedding': 'LLE'
        }
        
        return model_aliases, method_aliases

    def generate_comparison_title(self, comparison_type, config, dataset_name):
        """Generate dynamic chart title based on comparison configuration"""
        model_aliases, method_aliases = self.get_short_aliases()
        
        if comparison_type == "By Lang":
            # Format: [Method] X, [Model] Y, [Dataset] Z, [Languages] A+B+C
            method_alias = method_aliases.get(config['method'], config['method'])
            model_alias = model_aliases.get(config['model'], config['model'])
            lang_codes = "+".join(config['languages'])
            return f"[Method] {method_alias}, [Model] {model_alias}, [Dataset] {dataset_name}, [Languages] {lang_codes}"
            
        elif comparison_type == "By Model":
            # Format: [Method] X, [Models] A+B+C, [Dataset] Z, [Languages] Y1+Y2
            method_alias = method_aliases.get(config['method'], config['method'])
            model_aliases_list = [model_aliases.get(m, m) for m in config['models']]
            models_str = "+".join(model_aliases_list)
            langs_str = "+".join(config.get('languages', [config.get('language', 'unknown')]))
            return f"[Method] {method_alias}, [Models] {models_str}, [Dataset] {dataset_name}, [Languages] {langs_str}"

        elif comparison_type == "By Method":
            # Format: [Methods] A+B+C, [Model] Y, [Dataset] Z, [Languages] X1+X2
            method_aliases_list = [method_aliases.get(m, m) for m in config['methods']]
            methods_str = "+".join(method_aliases_list)
            model_alias = model_aliases.get(config['model'], config['model'])
            langs_str = "+".join(config.get('languages', [config.get('language', 'unknown')]))
            return f"[Methods] {methods_str}, [Model] {model_alias}, [Dataset] {dataset_name}, [Languages] {langs_str}"
        
        return "Comparison Analysis"

    def create_comparison_plot(self, embeddings_list, labels_list, comparison_type, config, dataset_name, dimensions="2D", enable_debug=False, highlight_config=None):
        """Create comparison visualization with Red/Blue/Green colors"""
        
        # Color indicators for chart title
        color_indicators = ['(R)', '(B)', '(G)']  # Red, Blue, Green
        
        # Prepare parameters for ECharts title generation
        if comparison_type == "By Lang":
            method_name = config['method']
            model_name = config['model']
            # Add color indicators to language codes
            lang_codes_with_colors = []
            for i, lang_code in enumerate(config['languages']):
                color_indicator = color_indicators[i % len(color_indicators)]
                lang_codes_with_colors.append(f"{lang_code}{color_indicator}")
            lang_codes = lang_codes_with_colors
        elif comparison_type == "By Model":
            method_name = config['method']
            # Add color indicators to model names
            models_with_colors = []
            for i, model in enumerate(config['models']):
                color_indicator = color_indicators[i % len(color_indicators)]
                models_with_colors.append(f"{model}{color_indicator}")
            model_name = "+".join(models_with_colors)
            lang_codes = config.get('languages', [config.get('language', 'unknown')])
        elif comparison_type == "By Method":
            # Add color indicators to method names
            methods_with_colors = []
            for i, method in enumerate(config['methods']):
                color_indicator = color_indicators[i % len(color_indicators)]
                methods_with_colors.append(f"{method}{color_indicator}")
            method_name = "+".join(methods_with_colors)
            model_name = config['model']
            lang_codes = config.get('languages', [config.get('language', 'unknown')])
        
        # Apply short aliases for method and model names
        model_aliases, method_aliases = self.get_short_aliases()
        
        # Convert method name to alias (handling color indicators)
        if "+" in method_name:
            # Multiple methods with color indicators
            methods = method_name.split("+")
            aliased_methods = []
            for method in methods:
                # Extract color indicator if present
                if method.endswith(('(R)', '(B)', '(G)')):
                    base_method = method[:-3]  # Remove color indicator
                    color_indicator = method[-3:]  # Get color indicator
                    aliased_method = method_aliases.get(base_method, base_method)
                    aliased_methods.append(f"{aliased_method}{color_indicator}")
                else:
                    aliased_methods.append(method_aliases.get(method, method))
            method_name = "+".join(aliased_methods)
        else:
            # Single method
            if method_name.endswith(('(R)', '(B)', '(G)')):
                base_method = method_name[:-3]
                color_indicator = method_name[-3:]
                aliased_method = method_aliases.get(base_method, base_method)
                method_name = f"{aliased_method}{color_indicator}"
            else:
                method_name = method_aliases.get(method_name, method_name)
            
        # Convert model name to alias (handling color indicators)
        if "+" in model_name:
            # Multiple models with color indicators
            models = model_name.split("+")
            aliased_models = []
            for model in models:
                # Extract color indicator if present
                if model.endswith(('(R)', '(B)', '(G)')):
                    base_model = model[:-3]  # Remove color indicator
                    color_indicator = model[-3:]  # Get color indicator
                    aliased_model = model_aliases.get(base_model, base_model)
                    aliased_models.append(f"{aliased_model}{color_indicator}")
                else:
                    aliased_models.append(model_aliases.get(model, model))
            model_name = "+".join(aliased_models)
        else:
            # Single model
            if model_name.endswith(('(R)', '(B)', '(G)')):
                base_model = model_name[:-3]
                color_indicator = model_name[-3:]
                aliased_model = model_aliases.get(base_model, base_model)
                model_name = f"{aliased_model}{color_indicator}"
            else:
                model_name = model_aliases.get(model_name, model_name)
        
        # Auto-scale embeddings to balance different method/model scales
        scaled_embeddings_list = []
        all_labels = []
        all_colors = []

        # Try to get semantic colors if available
        semantic_colors_available = False
        language_code = None

        # Determine the primary language code for semantic color lookup
        if comparison_type == "By Lang" and 'languages' in config:
            # For language comparison, use the first language for semantic colors
            language_code = config['languages'][0] if config['languages'] else None
        elif 'language' in config:
            # For other comparison types, use the single language
            language_code = config['language']

        # Check if semantic colors are available for this dataset and language
        semantic_colors_for_dataset = None
        if language_code and dataset_name:
            semantic_colors_for_dataset = self.get_semantic_colors_for_labels(
                [label for labels in labels_list for label in labels],
                language_code,
                dataset_name
            )
            if semantic_colors_for_dataset:
                semantic_colors_available = True

        for i, (embeddings, labels) in enumerate(zip(embeddings_list, labels_list)):
            # Apply auto-scaling: normalize each variation to [-1, 1] range
            if len(embeddings_list) > 1:  # Only scale when comparing multiple variations
                # Get min/max for each dimension
                mins = np.min(embeddings, axis=0)
                maxs = np.max(embeddings, axis=0)

                # Avoid division by zero
                ranges = maxs - mins
                ranges[ranges == 0] = 1.0

                # Scale to [-1, 1] range
                scaled_embeddings = 2 * (embeddings - mins) / ranges - 1
                scaled_embeddings_list.append(scaled_embeddings)
            else:
                # No scaling needed for single variation
                scaled_embeddings_list.append(embeddings)

            all_labels.extend(labels)

            # Assign colors: Use semantic colors if available, otherwise comparison colors
            if semantic_colors_available and language_code and dataset_name:
                # Get semantic colors for this specific set of labels
                semantic_colors = self.get_semantic_colors_for_labels(labels, language_code, dataset_name)
                if semantic_colors:
                    # Use semantic colors, falling back to comparison color for unmapped words
                    comparison_color = self.comparison_colors[i % len(self.comparison_colors)]
                    colors_for_labels = []
                    for semantic_color in semantic_colors:
                        if semantic_color is not None:
                            colors_for_labels.append(semantic_color)
                        else:
                            colors_for_labels.append(comparison_color)
                    all_colors.extend(colors_for_labels)
                else:
                    # Fallback to comparison colors
                    comparison_color = self.comparison_colors[i % len(self.comparison_colors)]
                    all_colors.extend([comparison_color] * len(labels))
            else:
                # Fallback to comparison colors: Red, Blue, Green
                comparison_color = self.comparison_colors[i % len(self.comparison_colors)]
                all_colors.extend([comparison_color] * len(labels))
        
        # Concatenate scaled embeddings
        combined_embeddings = np.vstack(scaled_embeddings_list)
        
        # Debug: Check lang_codes value (only if debug enabled)
        if enable_debug:
            st.info(f"🔍 **Debug**: lang_codes = {lang_codes}, comparison_type = {comparison_type}")

        # Create ECharts plot with proper parameters for title generation
        if dimensions == "3D":
            plot_option = self.echarts_plot_manager.plot_3d(
                combined_embeddings, all_labels, all_colors, "",  # Empty title, let ECharts generate it
                clustering=False, n_clusters=DEFAULT_N_CLUSTERS,
                method_name=method_name, model_name=model_name, dataset_name=dataset_name,
                highlight_config=highlight_config, display_chart=True, lang_codes=lang_codes
            )
        else:
            plot_option = self.echarts_plot_manager.plot_2d(
                combined_embeddings, all_labels, all_colors, "",  # Empty title, let ECharts generate it
                clustering=False, n_clusters=DEFAULT_N_CLUSTERS,
                method_name=method_name, model_name=model_name, dataset_name=dataset_name,
                highlight_config=highlight_config, display_chart=True, lang_codes=lang_codes
            )
        
        # Store chart configuration for export
        st.session_state.current_echarts_config = plot_option
        
        return plot_option

    def _generate_standardized_filename(self, comparison_type, config, dataset_name, img_format="png", enable_debug=False):
        """Generate standardized filename using centralized helper"""

        # Extract lists of methods, models, and languages from config
        if comparison_type == "By Lang":
            method_names = [config['method']]
            model_names = [config['model']]
            lang_codes = config['languages']

        elif comparison_type == "By Model":
            method_names = [config['method']]
            model_names = config['models']
            lang_codes = config.get('languages', [config.get('language', 'unknown')])

        elif comparison_type == "By Method":
            method_names = config['methods']
            model_names = [config['model']]
            lang_codes = config.get('languages', [config.get('language', 'unknown')])

        else:
            # Fallback
            method_names = [config.get('method', 'unknown')]
            model_names = [config.get('model', 'unknown')]
            lang_codes = [config.get('language', 'unknown')]

        # Use centralized helper to generate title and filename
        _, filename = create_title_and_filename(method_names, model_names, dataset_name, lang_codes, img_format)

        # Remove extension since we'll add it back later
        filename_without_ext = filename.rsplit('.', 1)[0]

        if enable_debug:
            st.info(f"🔍 **Filename Debug**: Generated '{filename_without_ext}' from config: {config}")

        return filename_without_ext

    def create_stacked_plots(self, embeddings_list, labels_list, comparison_type, config, dataset_name, dimensions="2D", enable_debug=False, highlight_config=None):
        """Create stacked comparison plots (top to bottom)"""
        
        # Determine variation names for individual titles
        if comparison_type == "By Lang":
            variation_names = config['languages']
            base_method = config['method']
            base_model = config['model']
        elif comparison_type == "By Model":
            variation_names = config['models']
            base_method = config['method']
            base_model = None  # Will be set individually
        elif comparison_type == "By Method":
            variation_names = config['methods']
            base_method = None  # Will be set individually
            base_model = config['model']
        
        # Apply aliases
        model_aliases, method_aliases = self.get_short_aliases()
        
        # Create individual plots stacked vertically
        for i, (embeddings, labels, variation_name) in enumerate(zip(embeddings_list, labels_list, variation_names)):
            # Generate individual title
            if comparison_type == "By Lang":
                method_alias = method_aliases.get(base_method, base_method)
                model_alias = model_aliases.get(base_model, base_model)
                title = f"[Method] {method_alias}, [Model] {model_alias}, [Dataset] {dataset_name}, [Language] {variation_name}"
            elif comparison_type == "By Model":
                method_alias = method_aliases.get(base_method, base_method)
                model_alias = model_aliases.get(variation_name, variation_name)
                langs_str = "+".join(config.get('languages', [config.get('language', 'unknown')]))
                title = f"[Method] {method_alias}, [Model] {model_alias}, [Dataset] {dataset_name}, [Languages] {langs_str}"
            elif comparison_type == "By Method":
                method_alias = method_aliases.get(variation_name, variation_name)
                model_alias = model_aliases.get(base_model, base_model)
                langs_str = "+".join(config.get('languages', [config.get('language', 'unknown')]))
                title = f"[Method] {method_alias}, [Model] {model_alias}, [Dataset] {dataset_name}, [Languages] {langs_str}"
            
            # Add section divider
            st.markdown("---")
            st.markdown(f"##### Visualizing: {variation_name}")
            
            # Use semantic colors if available, otherwise single comparison color
            language_code = None
            if comparison_type == "By Lang":
                language_code = variation_name  # variation_name is the language code
            elif 'language' in config:
                language_code = config['language']

            # Try to get semantic colors for this variation
            semantic_colors = None
            if language_code and dataset_name:
                semantic_colors = self.get_semantic_colors_for_labels(labels, language_code, dataset_name)

            if semantic_colors:
                # Use semantic colors, falling back to comparison color for unmapped words
                single_color = self.comparison_colors[0]  # Red fallback
                colors = []
                for semantic_color in semantic_colors:
                    if semantic_color is not None:
                        colors.append(semantic_color)
                    else:
                        colors.append(single_color)
            else:
                # Use single color for individual plots (first color - red)
                single_color = self.comparison_colors[0]
                colors = [single_color] * len(labels)
            
            # Create individual ECharts plot with unique key
            unique_key = f"compare_{comparison_type.lower().replace(' ', '_')}_{i}_{variation_name}"
            
            # Determine correct method_name and model_name based on comparison type
            if comparison_type == "By Lang":
                actual_method_name = base_method
                actual_model_name = base_model
            elif comparison_type == "By Model":
                actual_method_name = base_method
                actual_model_name = variation_name  # variation_name is the model being compared
            elif comparison_type == "By Method":
                actual_method_name = variation_name  # variation_name is the method being compared
                actual_model_name = base_model
            else:
                actual_method_name = "Unknown"
                actual_model_name = "Unknown"
            
            # Prepare lang_codes for ECharts title generation
            if comparison_type in ["By Model", "By Method"]:
                lang_codes_for_chart = config.get('languages', [config.get('language', 'unknown')])
            else:  # "By Lang"
                lang_codes_for_chart = config.get('languages', [])

            if dimensions == "3D":
                plot_option = self.echarts_plot_manager.plot_3d(
                    embeddings, labels, colors, title,
                    clustering=False, n_clusters=DEFAULT_N_CLUSTERS,
                    method_name=actual_method_name, model_name=actual_model_name, dataset_name=dataset_name,
                    highlight_config=highlight_config, display_chart=True, lang_codes=lang_codes_for_chart, chart_key=unique_key
                )
            else:
                plot_option = self.echarts_plot_manager.plot_2d(
                    embeddings, labels, colors, title,
                    clustering=False, n_clusters=DEFAULT_N_CLUSTERS,
                    method_name=actual_method_name, model_name=actual_model_name, dataset_name=dataset_name,
                    highlight_config=highlight_config, display_chart=True, lang_codes=lang_codes_for_chart, chart_key=unique_key
                )

def main(debug_flag=False):
    check_login()
    
    st.subheader("⚖️ Semanscope - Compare")
    # Show instructions when no visualization is active
    st.info("👆 Configure comparison settings in the sidebar and click 'Visualize' to start to compare by Lang/Model/Method")


    # Initialize visualizer
    visualizer = ComparisonVisualizer()
    reducer = DimensionReducer()
    
    # Get global settings
    viz_settings = get_global_viz_settings()
    
    # Sidebar
    with st.sidebar:
        # st.subheader("🎛️ Comparison Controls")

        c1, c2 = st.columns([3, 3])

        with c1:
            # 1. Compare Option Selection
            comparison_type = st.radio(
                "Comparison Type",
                options=["By Lang", "By Model", "By Method"],
                index=0,
                help="Choose what variable to compare across",
                key="comparison_type"
            )
        
        with c2:
            show_in_same_chart = st.checkbox(
                "Show in same chart",
                value=False,
                help="Checked: Overlay all variations with different colors. For 'By Lang' comparison, embeddings will be combined before dimension reduction for better semantic clustering. Unchecked: Show separate charts stacked vertically",
                key="show_in_same_chart"
            )

            # Debug Control
            enable_debug = st.checkbox(
                "🔧 Enable Debug",
                value=False,
                help="Show debug information for troubleshooting",
                key="compare_debug_enabled"
            )

        # st.markdown("---")
        
        # 2. Dynamic Controls based on comparison type
        if comparison_type == "By Lang":
            # Use Enter Text Data component for language handling
            text_widget = EnterTextDataWidget(key_prefix="compare_lang_", max_languages=3)
            text_data = text_widget.render(visualizer)
            
            # Fixed controls for Model and Method
            # st.subheader("Fixed Variables")
            
            # Model selection
            active_models = get_active_models()
            model_names = sorted(list(active_models.keys()))
            if DEFAULT_MODEL in model_names:
                default_model_index = model_names.index(DEFAULT_MODEL)
            else:
                default_model_index = 0
            selected_model = st.selectbox(
                "Model",
                options=model_names,
                index=default_model_index,
                help="Fixed model for all language comparisons",
                key="fixed_model"
            )
            
            # Method selection  
            active_methods = get_active_methods()
            method_names = sorted(list(active_methods.keys()))
            if DEFAULT_METHOD in method_names:
                default_method_index = method_names.index(DEFAULT_METHOD)
            else:
                default_method_index = 0
            selected_method = st.selectbox(
                "Method", 
                options=method_names,
                index=default_method_index,
                help="Fixed method for all language comparisons",
                key="fixed_method"
            )
            
        elif comparison_type == "By Model":
            # Multiselect for models
            st.warning("⚠️ Maximum 3 models supported for comparison")
            active_models = get_active_models()
            model_names = sorted(list(active_models.keys()))
            selected_models = st.multiselect(
                "Models to Compare",
                options=model_names,
                default=DEFAULT_TOP3_MODELS,
                max_selections=3,
                help="Select 2-3 models to compare",
                key="compare_models"
            )
            
            # Enter Text Data (will use selected language only)
            text_widget = EnterTextDataWidget(key_prefix="compare_model_", max_languages=3)
            text_data = text_widget.render(visualizer)
            # st.info("💡 Using the selected language (with checkbox ☑️) for model comparison")
            
            # Fixed Method
            # st.subheader("Fixed Variables")
            active_methods = get_active_methods()
            method_names = sorted(list(active_methods.keys()))
            if DEFAULT_METHOD in method_names:
                default_method_index = method_names.index(DEFAULT_METHOD)
            else:
                default_method_index = 0
            selected_method = st.selectbox(
                "Method",
                options=method_names, 
                index=default_method_index,
                help="Fixed method for all model comparisons",
                key="fixed_method_model"
            )
            
        elif comparison_type == "By Method":
            # Multiselect for methods
            st.warning("⚠️ Maximum 3 methods supported for comparison")
            active_methods = get_active_methods()
            method_names = sorted(list(active_methods.keys()))
            selected_methods = st.multiselect(
                "Methods to Compare",
                options=method_names,
                default=method_names[:2] if len(method_names) >= 2 else method_names,
                max_selections=3,
                help="Select 2-3 dimensionality reduction methods to compare",
                key="compare_methods"
            )
            
            # Enter Text Data (will use selected language only)
            text_widget = EnterTextDataWidget(key_prefix="compare_method_", max_languages=3)
            text_data = text_widget.render(visualizer)
            st.info("💡 Using the selected language (with checkbox ☑️) for method comparison")
            
            # Fixed Model
            # st.subheader("Fixed Variables")
            active_models = get_active_models()
            model_names = sorted(list(active_models.keys()))
            if DEFAULT_MODEL in model_names:
                default_model_index = model_names.index(DEFAULT_MODEL)
            else:
                default_model_index = 0
            selected_model = st.selectbox(
                "Model",
                options=model_names,
                index=default_model_index,
                help="Fixed model for all method comparisons", 
                key="fixed_model_method"
            )
        
        # st.markdown("---")
        
        # 3. Visualization Settings
        # st.subheader("⚙️ Visualization Settings")
        
        dimension_options = ["2D", "3D"]
        if DEFAULT_DIMENSION in dimension_options:
            default_dimension_index = dimension_options.index(DEFAULT_DIMENSION)
        else:
            default_dimension_index = 0

        dimensions = st.radio(
            "Dimensions",
            options=dimension_options,
            index=default_dimension_index,
            help="Select visualization dimensions",
            horizontal=True,
            key="viz_dimensions"
        )

        # 4. Visualize Button
        visualize_button = st.button(
            "🎨 Visualize",
            type="primary",
            width='stretch',
            help="Generate comparison visualization"
        )

        # Word Search settings
        highlight_config = visualizer.echarts_plot_manager.get_highlight_settings()

        # Display link to global settings
        GlobalSettingsManager.render_current_settings_summary()

        # st.markdown("---")
        # 5. Dataset Preview
        if 'text_data' in locals() and text_data:
            with st.expander("Settings:", expanded=False):
                dataset_name = text_data.get('input_name_selected', 'User Input')
                st.write(f"**Dataset:** {dataset_name}")
                
                # Show selected languages/models/methods
                if comparison_type == "By Lang":
                    selected_langs = [lang[1] for lang in text_data['languages'] if lang[3]]  # lang_code for selected
                    if selected_langs:
                        st.write(f"**Languages:** {', '.join(selected_langs)}")
                    st.write(f"**Model:** {selected_model}")
                    st.write(f"**Method:** {selected_method}")
                    
                elif comparison_type == "By Model":
                    if selected_models:
                        st.write(f"**Models:** {', '.join(selected_models)}")
                    # Get ALL SELECTED languages
                    selected_langs = [lang[1] for lang in text_data['languages'] if lang[3]]  # lang_code for selected
                    if selected_langs:
                        st.write(f"**Languages:** {', '.join(selected_langs)}")
                    st.write(f"**Method:** {selected_method}")
                    
                elif comparison_type == "By Method":
                    if selected_methods:
                        st.write(f"**Methods:** {', '.join(selected_methods)}")
                    # Get ALL SELECTED languages
                    selected_langs = [lang[1] for lang in text_data['languages'] if lang[3]]  # lang_code for selected
                    if selected_langs:
                        st.write(f"**Languages:** {', '.join(selected_langs)}")
                    st.write(f"**Model:** {selected_model}")
        
        
    
    # Main content area
    if visualize_button:
        if 'text_data' not in locals() or not text_data:
            st.error("❌ Please configure text data first")
            return
            
        dataset_name = text_data.get('input_name_selected', 'User Input')
        
        with st.spinner("🔄 Generating comparison visualization..."):
            try:
                if comparison_type == "By Lang":
                    # Compare by languages
                    selected_langs = [(lang[0], lang[1], lang[2]) for lang in text_data['languages'] if lang[3]]  # (name, code, text)

                    # DEBUG: Show what languages are selected
                    if enable_debug:
                        st.info(f"🔍 **Debug - Language Selection**:")
                        st.write(f"  Total languages from widget: {len(text_data['languages'])}")
                        for i, lang in enumerate(text_data['languages']):
                            st.write(f"  Lang {i+1}: {lang[0]} ({lang[1]}) - Selected: {lang[3]}, Text length: {len(lang[2].strip()) if lang[2] else 0}")
                        st.write(f"  Filtered selected languages: {len(selected_langs)}")
                        for lang in selected_langs:
                            word_count = len([w for w in lang[2].strip().split('\n') if w.strip()]) if lang[2] else 0
                            st.write(f"  - {lang[0]} ({lang[1]}): {word_count} words")

                    if len(selected_langs) < 1:
                        st.error("❌ Please select at least 1 language for comparison")
                        return
                    
                    embeddings_list = []
                    labels_list = []
                    
                    # Check if we should combine embeddings before dimension reduction
                    if show_in_same_chart and len(selected_langs) > 1:
                        # Combine embeddings from different languages before dimension reduction
                        combined_embeddings = []
                        combined_words = []
                        lang_boundaries = []  # Track which words belong to which language
                        start_idx = 0
                        
                        st.info(f"🔄 **Enhanced semantic clustering**: Combining embeddings from {len(selected_langs)} languages before dimension reduction to allow natural clustering of semantically similar words across languages")
                        if enable_debug:
                            st.info(f"🔍 **Debug - Combining embeddings**: Processing {len(selected_langs)} languages before dimension reduction")
                        
                        for lang_name, lang_code, text_content in selected_langs:
                            if text_content.strip():
                                words = text_content.strip().split('\n')
                                try:
                                    embeddings = visualizer.get_embeddings(words, selected_model, lang_code)

                                    # Filter NaN values
                                    embeddings, words = filter_nan_embeddings(embeddings, words, selected_model, enable_debug)
                                    if embeddings is None:
                                        continue

                                    combined_embeddings.append(embeddings)
                                    combined_words.extend(words)
                                    lang_boundaries.append((start_idx, start_idx + len(words), lang_code, lang_name))
                                    start_idx += len(words)
                                    
                                    if enable_debug:
                                        st.info(f"🔍 **{lang_name}**: Added {len(words)} words, embeddings shape: {embeddings.shape}")
                                        
                                except Exception as e:
                                    st.error(f"❌ Error processing {lang_name}: {str(e)}")
                                    continue
                        
                        if combined_embeddings:
                            # Concatenate all embeddings
                            all_embeddings = np.vstack(combined_embeddings)
                            
                            if enable_debug:
                                st.info(f"🔍 **Combined**: Total embeddings shape: {all_embeddings.shape}, Total words: {len(combined_words)}")
                            
                            # Apply dimension reduction to the combined embeddings
                            dims = 3 if dimensions == "3D" else 2
                            
                            # Create a combined dataset identifier for caching
                            lang_codes = [boundary[2] for boundary in lang_boundaries]
                            combined_dataset_id = f"combined_{'+'.join(lang_codes)}"
                            
                            reduced_embeddings = reducer.reduce_dimensions_with_cache(
                                all_embeddings, selected_method, dims,
                                dataset=combined_dataset_id, lang='+'.join(lang_codes), model=selected_model
                            )
                            
                            # Validate shape
                            if reduced_embeddings.shape[1] != dims:
                                st.error(f"❌ Combined dimension reduction failed: expected {dims}D, got {reduced_embeddings.shape[1]}D")
                                return
                            
                            # Split the reduced embeddings back into language-specific groups
                            for start_idx, end_idx, lang_code, lang_name in lang_boundaries:
                                lang_reduced_embeddings = reduced_embeddings[start_idx:end_idx]
                                lang_words = combined_words[start_idx:end_idx]
                                
                                embeddings_list.append(lang_reduced_embeddings)
                                labels_list.append(lang_words)
                                
                                if enable_debug:
                                    st.info(f"🔍 **{lang_name} final**: {len(lang_words)} words, reduced shape: {lang_reduced_embeddings.shape}")
                        
                    else:
                        # Original logic: process each language separately
                        for lang_name, lang_code, text_content in selected_langs:
                            if enable_debug:
                                st.info(f"🔍 **Processing language**: {lang_name} ({lang_code})")
                                word_count = len([w for w in text_content.strip().split('\n') if w.strip()]) if text_content else 0
                                st.write(f"  Text content length: {len(text_content) if text_content else 0}, Words: {word_count}")

                            if text_content.strip():
                                words = text_content.strip().split('\n')
                                if enable_debug:
                                    st.write(f"  Processing {len(words)} words...")

                                try:
                                    embeddings = visualizer.get_embeddings(words, selected_model, lang_code)

                                    if enable_debug:
                                        st.write(f"  Got embeddings shape: {embeddings.shape if embeddings is not None else 'None'}")

                                    # Filter NaN values
                                    embeddings, words = filter_nan_embeddings(embeddings, words, selected_model, enable_debug)
                                    if embeddings is None:
                                        if enable_debug:
                                            st.warning(f"  ⚠️ Embeddings is None after filtering, skipping {lang_name}")
                                        continue

                                    dims = 3 if dimensions == "3D" else 2
                                    reduced_embeddings = reducer.reduce_dimensions_with_cache(
                                        embeddings, selected_method, dims,
                                        dataset=words, lang=lang_code, model=selected_model
                                    )

                                    if enable_debug:
                                        st.write(f"  Reduced embeddings shape: {reduced_embeddings.shape}")

                                    # Validate shape
                                    if reduced_embeddings.shape[1] != dims:
                                        st.error(f"❌ Dimension reduction failed for {lang_name}: expected {dims}D, got {reduced_embeddings.shape[1]}D")
                                        continue

                                    embeddings_list.append(reduced_embeddings)
                                    labels_list.append(words)

                                    if enable_debug:
                                        st.success(f"  ✅ Successfully processed {lang_name}: {len(words)} words added to visualization")
                                except Exception as e:
                                    st.error(f"❌ Error processing {lang_name}: {str(e)}")
                                    if enable_debug:
                                        import traceback
                                        st.code(traceback.format_exc())
                                    continue
                            else:
                                if enable_debug:
                                    st.warning(f"  ⚠️ Skipping {lang_name} - empty text content")
                    
                    # Prepare config for title
                    config = {
                        'method': selected_method,
                        'model': selected_model,
                        'languages': [lang[1] for lang in selected_langs]  # lang codes
                    }
                    
                elif comparison_type == "By Model":
                    # Compare by models
                    if len(selected_models) < 1:
                        st.error("❌ Please select at least 1 model for comparison")
                        return

                    # Get ALL SELECTED languages
                    selected_langs = [(lang[0], lang[1], lang[2]) for lang in text_data['languages'] if lang[3]]  # (name, code, text)

                    if not selected_langs:
                        st.error("❌ Please select at least one language")
                        return

                    # Combine words from all selected languages
                    all_words = []
                    lang_boundaries = []  # Track which words belong to which language
                    start_idx = 0

                    for lang_name, lang_code, text_content in selected_langs:
                        if text_content.strip():
                            words = text_content.strip().split('\n')
                            all_words.extend(words)
                            lang_boundaries.append((start_idx, start_idx + len(words), lang_code, lang_name))
                            start_idx += len(words)

                    if not all_words:
                        st.error("❌ No text data provided for selected languages")
                        return

                    if enable_debug:
                        st.info(f"🔍 **By Model - Processing {len(selected_langs)} languages**: {', '.join([l[1] for l in selected_langs])}")
                        st.write(f"  Total words across all languages: {len(all_words)}")
                        for start, end, code, name in lang_boundaries:
                            st.write(f"  - {name} ({code}): {end-start} words")

                    embeddings_list = []
                    labels_list = []

                    # Create combined language identifier for caching
                    combined_lang = '+'.join([lang[1] for lang in selected_langs])

                    for model in selected_models:
                        try:
                            if enable_debug:
                                st.info(f"🔍 **Processing model**: {model} with {len(all_words)} words from {len(selected_langs)} language(s)")

                            # Get embeddings for all words (model handles multilingual automatically)
                            embeddings = visualizer.get_embeddings(all_words, model, combined_lang)

                            if enable_debug:
                                st.write(f"  Got embeddings shape: {embeddings.shape if embeddings is not None else 'None'}")

                            # Filter NaN values
                            clean_embeddings, clean_words = filter_nan_embeddings(embeddings, all_words, model, enable_debug)
                            if clean_embeddings is None:
                                if enable_debug:
                                    st.warning(f"  ⚠️ Embeddings is None after filtering for {model}")
                                continue

                            dims = 3 if dimensions == "3D" else 2

                            # Additional NaN check right before dimension reduction
                            if np.isnan(clean_embeddings).any() or np.isinf(clean_embeddings).any():
                                st.error(f"❌ **{model}**: Clean embeddings still have NaN/Inf before dimension reduction!")
                                continue

                            if enable_debug:
                                st.info(f"🔍 **{model}**: Calling dimension reduction with {len(clean_embeddings)} clean embeddings")

                            reduced_embeddings = reducer.reduce_dimensions_with_cache(
                                clean_embeddings, selected_method, dims,
                                dataset=clean_words, lang=combined_lang, model=model
                            )

                            # Check if dimension reduction failed
                            if reduced_embeddings is None:
                                st.error(f"❌ **{model}**: Dimension reduction returned None")
                                continue

                            # Validate shape
                            if reduced_embeddings.shape[1] != dims:
                                st.error(f"❌ Dimension reduction failed for {model}: expected {dims}D, got {reduced_embeddings.shape[1]}D")
                                continue

                            embeddings_list.append(reduced_embeddings)
                            labels_list.append(clean_words)

                            if enable_debug:
                                st.success(f"  ✅ Successfully processed {model}: {len(clean_words)} words")

                        except Exception as e:
                            st.error(f"❌ Error processing {model}: {str(e)}")
                            if enable_debug:
                                import traceback
                                st.code(traceback.format_exc())
                            continue

                    # Prepare config for title
                    config = {
                        'method': selected_method,
                        'models': selected_models,
                        'languages': [lang[1] for lang in selected_langs]  # lang codes
                    }
                    
                elif comparison_type == "By Method":
                    # Compare by methods
                    if len(selected_methods) < 1:
                        st.error("❌ Please select at least 1 method for comparison")
                        return

                    # Get ALL SELECTED languages
                    selected_langs = [(lang[0], lang[1], lang[2]) for lang in text_data['languages'] if lang[3]]  # (name, code, text)

                    if not selected_langs:
                        st.error("❌ Please select at least one language")
                        return

                    # Combine words from all selected languages
                    all_words = []
                    lang_boundaries = []  # Track which words belong to which language
                    start_idx = 0

                    for lang_name, lang_code, text_content in selected_langs:
                        if text_content.strip():
                            words = text_content.strip().split('\n')
                            all_words.extend(words)
                            lang_boundaries.append((start_idx, start_idx + len(words), lang_code, lang_name))
                            start_idx += len(words)

                    if not all_words:
                        st.error("❌ No text data provided for selected languages")
                        return

                    # Create combined language identifier for caching
                    combined_lang = '+'.join([lang[1] for lang in selected_langs])

                    if enable_debug:
                        st.info(f"🔍 **By Method - Processing {len(selected_langs)} languages**: {', '.join([l[1] for l in selected_langs])}")
                        st.write(f"  Total words across all languages: {len(all_words)}")
                        for start, end, code, name in lang_boundaries:
                            st.write(f"  - {name} ({code}): {end-start} words")

                    # Get embeddings once for all methods
                    embeddings = visualizer.get_embeddings(all_words, selected_model, combined_lang)

                    # Filter NaN values
                    embeddings, all_words = filter_nan_embeddings(embeddings, all_words, selected_model, enable_debug)
                    if embeddings is None:
                        st.error(f"❌ Cannot proceed with {selected_model}: all embeddings contain NaN values")
                        return

                    embeddings_list = []
                    labels_list = []

                    for method in selected_methods:
                        try:
                            if enable_debug:
                                st.info(f"🔍 **Processing method**: {method} with {len(all_words)} words")

                            dims = 3 if dimensions == "3D" else 2
                            reduced_embeddings = reducer.reduce_dimensions_with_cache(
                                embeddings, method, dims,
                                dataset=all_words, lang=combined_lang, model=selected_model
                            )

                            if enable_debug:
                                st.write(f"  Reduced embeddings shape: {reduced_embeddings.shape if reduced_embeddings is not None else 'None'}")

                            # Validate shape
                            if reduced_embeddings.shape[1] != dims:
                                st.error(f"❌ Dimension reduction failed for {method}: expected {dims}D, got {reduced_embeddings.shape[1]}D")
                                continue

                            embeddings_list.append(reduced_embeddings)
                            labels_list.append(all_words)

                            if enable_debug:
                                st.success(f"  ✅ Successfully processed {method}: {len(all_words)} words")

                        except Exception as e:
                            st.error(f"❌ Error processing {method}: {str(e)}")
                            if enable_debug:
                                import traceback
                                st.code(traceback.format_exc())
                            continue

                    # Prepare config for title
                    config = {
                        'methods': selected_methods,
                        'model': selected_model,
                        'languages': [lang[1] for lang in selected_langs]  # lang codes
                    }
                
                # Create comparison visualization
                if embeddings_list and labels_list and len(embeddings_list) >= 1:
                    # Show visualization for 1 or more valid embeddings
                    if show_in_same_chart:
                        # Combined chart with color coding
                        if enable_debug:
                            st.info(f"🔍 **Config Debug**: config = {config}")

                        plot_option = visualizer.create_comparison_plot(
                            embeddings_list, labels_list, comparison_type, config, dataset_name, dimensions, enable_debug, highlight_config
                        )

                        # Auto-save now handled automatically by plot_2d() in plotting_echarts.py
                        if enable_debug:
                            st.info("🔍 **Auto-save**: Now handled automatically by core plot_2d() function")
                    else:
                        # Stacked charts (top to bottom)
                        visualizer.create_stacked_plots(
                            embeddings_list, labels_list, comparison_type, config, dataset_name, dimensions, enable_debug, highlight_config
                        )

                        # Auto-save now handled automatically by plot_2d() in plotting_echarts.py
                        if enable_debug:
                            st.info("🔍 **Auto-save (Stacked)**: Now handled automatically by core plot_2d() function")

                    if len(embeddings_list) == 1:
                        st.success(f"✅ {comparison_type} visualization completed successfully! (Single selection)")
                    else:
                        st.success(f"✅ {comparison_type} comparison completed successfully!")
                    
                    # Additional controls
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔄 Rotate 90°"):
                            # Implement rotation if needed
                            st.info("Rotation feature coming soon!")
                    
                    with col2:
                        if st.button("💾 Save PNG"):
                            # Implement PNG save if needed
                            st.info("PNG save feature coming soon!")
                    
                elif len(embeddings_list) < 1:
                    st.error("❌ Need at least 1 successful embedding for visualization. Please check your data or try a different method.")
                else:
                    st.error("❌ Failed to generate embeddings for visualization")
                    
            except Exception as e:
                st.error(f"❌ Error during comparison: {str(e)}")
                st.exception(e)
    
    else:
        show_overview()

def show_overview():
    with st.expander("Overview", expanded=False):
        st.markdown("Cross-compare up to 3 variations by Language, Model, or Method")   
        st.markdown("""
        ### 🎯 Comparison Options:
        
        **🌐 By Lang**: Compare how the same dataset appears across different languages
        - Uses the same model and method for all languages
        - Colors: Red, Blue, Green for different languages
        
        **🤖 By Model**: Compare how different embedding models represent the same data
        - Uses the same method and selected language for all models
        - Colors: Red, Blue, Green for different models

        **🔬 By Method**: Compare how different dimensionality reduction methods visualize the same embeddings
        - Uses the same model and selected language for all methods
        - Colors: Red, Blue, Green for different methods
        """)


if __name__ == "__main__":
    main()