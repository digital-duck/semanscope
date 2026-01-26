import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from semanscope.components.embedding_viz import EmbeddingVisualizer, get_active_methods
from semanscope.components.shared.enter_text_data import EnterTextDataWidget
from semanscope.components.dimension_reduction import DimensionReducer
from semanscope.components.geometric_analysis import GeometricAnalyzer
from semanscope.utils.title_filename_helper import create_title_and_filename, create_chart_title
from semanscope.config import (
    check_login,
    PLOT_WIDTH,
    PLOT_HEIGHT,
    COLOR_MAP,
    MODEL_INFO,
    METHOD_INFO,
    DEFAULT_MODEL,
    DEFAULT_METHOD,
    DEFAULT_INSTRUCTION_MSG_1,
    LANGUAGE_CODE_MAP,
    DEFAULT_LANG_SET,
    sample_chn_input_data,
    sample_enu_input_data,
    SRC_DIR, DATA_PATH,
    get_sorted_language_names,
    get_language_codes_with_prefix,
    get_language_code_from_name,
    get_language_name_from_code
)
from pathlib import Path
from semanscope.utils.download_helpers import handle_download_button
from semanscope.components.shared.publication_settings import PublicationSettingsWidget
from semanscope.utils.global_settings import (
    get_global_viz_settings,
    get_global_publication_settings,
    get_global_geometric_analysis,
    is_global_geometric_analysis_enabled,
    GlobalSettingsManager
)

# Page config
st.set_page_config(
    page_title="Zoom",
    page_icon="🔍",
    layout="wide"
)

DEFAULT_STEP_SIZE = 0.005

class EnhancedDualViewManager:
    """Enhanced dual-view with center/size based zoom controls"""
    
    def __init__(self):
        self.overview_config = {
            'marker_size': 6,
            'opacity': 0.8,
            'show_text': False
        }
        self.detail_config = {
            'marker_size': 16,
            'opacity': 1.0,
            'show_text': True,
            'textfont_size': 16
        }

    def center_size_to_bounds(self, center_x, center_y, width, height):
        """Convert center/size to min/max bounds"""
        return {
            'x_min': center_x - width/2,
            'x_max': center_x + width/2,
            'y_min': center_y - height/2,
            'y_max': center_y + height/2
        }

    def create_enhanced_dual_view(self, embeddings, labels, colors, title, zoom_params, model_name=None, method_name=None, dataset_name=None, lang_codes=None):
        """Create separate overview and detail figures for the enhanced dual view"""
        
        # Convert center/size to bounds
        viewport_coords = self.center_size_to_bounds(
            zoom_params['center_x'], zoom_params['center_y'],
            zoom_params['width'], zoom_params['height']
        )
        
        # Support for semantic color coding - check if colors are hex codes
        has_semantic_colors = any(isinstance(c, str) and c.startswith('#') for c in colors)

        if has_semantic_colors:
            # Use semantic colors directly instead of language-based separation
            color_values = []
            for c in colors:
                if isinstance(c, str) and c.startswith('#'):
                    color_values.append(c)
                elif c == 'chinese':
                    color_values.append('#FF0000')  # Red fallback
                elif c == 'english':
                    color_values.append('#0000FF')  # Blue fallback
                else:
                    color_values.append('#808080')  # Gray fallback
        else:
            # Use traditional language-based colors for backward compatibility
            color_values = []
            for c in colors:
                if c == 'chinese':
                    color_values.append('#FF0000')  # Red
                elif c == 'english':
                    color_values.append('#0000FF')  # Blue
                else:
                    color_values.append('#808080')  # Gray fallback

        # Create masks for traditional language separation (when not using semantic colors)
        if not has_semantic_colors:
            chinese_mask = np.array([color == 'chinese' for color in colors])
            english_mask = np.array([color == 'english' for color in colors])
        else:
            # For semantic colors, we don't separate by language
            chinese_mask = np.array([False] * len(colors))
            english_mask = np.array([False] * len(colors))
        
        # Create overview figure
        overview_fig = go.Figure()

        if has_semantic_colors:
            # OVERVIEW - All points with semantic colors
            overview_fig.add_trace(go.Scatter(
                x=embeddings[:, 0],
                y=embeddings[:, 1],
                mode='markers',
                marker=dict(size=6, color=color_values, opacity=0.8, line=dict(width=1, color='white')),
                text=labels,
                hovertemplate='<b>%{text}</b><br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>',
                name="Semantic Colors",
                showlegend=False
            ))
        else:
            # OVERVIEW - Traditional language-based separation
            if np.any(chinese_mask):
                overview_fig.add_trace(go.Scatter(
                    x=embeddings[chinese_mask, 0],
                    y=embeddings[chinese_mask, 1],
                    mode='markers',
                    marker=dict(size=6, color='red', opacity=0.8, line=dict(width=1, color='white')),
                    text=[labels[i] for i in range(len(labels)) if chinese_mask[i]],
                    hovertemplate='<b>%{text}</b><br>中文<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>',
                    name="中文",
                    showlegend=False
                ))

            if np.any(english_mask):
                overview_fig.add_trace(go.Scatter(
                    x=embeddings[english_mask, 0],
                    y=embeddings[english_mask, 1],
                    mode='markers',
                    marker=dict(size=6, color='blue', opacity=0.8, line=dict(width=1, color='white')),
                    text=[labels[i] for i in range(len(labels)) if english_mask[i]],
                    hovertemplate='<b>%{text}</b><br>English<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>',
                    name="English",
                    showlegend=False
                ))
        
        # Add zoom box to overview
        overview_fig.add_shape(
            type="rect",
            x0=viewport_coords['x_min'], y0=viewport_coords['y_min'],
            x1=viewport_coords['x_max'], y1=viewport_coords['y_max'],
            line=dict(color="orange", width=3),
            fillcolor="rgba(255, 165, 0, 0.2)"
        )
        
        # Set overview axis ranges
        data_x_min, data_x_max = embeddings[:, 0].min(), embeddings[:, 0].max()
        data_y_min, data_y_max = embeddings[:, 1].min(), embeddings[:, 1].max()
        x_padding = (data_x_max - data_x_min) * 0.1
        y_padding = (data_y_max - data_y_min) * 0.1
        
        overview_fig.update_xaxes(
            range=[data_x_min - x_padding, data_x_max + x_padding],
            title_text="x",
            showgrid=True, 
            gridwidth=1,
            gridcolor='#D0D0D0',
            griddash='dot'
        )
        overview_fig.update_yaxes(
            range=[data_y_min - y_padding, data_y_max + y_padding],
            title_text="y",
            showgrid=True, 
            gridwidth=1,
            gridcolor='#D0D0D0',
            griddash='dot',
            scaleanchor="x", scaleratio=1
        )
        
        # Create overview title using centralized helper
        if method_name and model_name and dataset_name:
            overview_title = create_chart_title(
                [method_name] if method_name else [],
                [model_name] if model_name else [],
                dataset_name,
                lang_codes or []
            )
        else:
            overview_title = "Overview"
        
        overview_fig.update_layout(
            title=dict(
                text=overview_title,
                font=dict(size=18, family='Arial, sans-serif'),
                x=0.5,  # Center align title
                xanchor='center'
            ),
            dragmode='pan',
            hovermode='closest',
            showlegend=False,
            height=700,  # More square aspect ratio
            width=800,   # Controlled width to reduce white space
            plot_bgcolor='white',
            font=dict(family='Arial, sans-serif'),
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        # Create detail figure
        detail_fig = go.Figure()
        
        # DETAIL - Only points within viewport
        viewport_mask = (
            (embeddings[:, 0] >= viewport_coords['x_min']) &
            (embeddings[:, 0] <= viewport_coords['x_max']) &
            (embeddings[:, 1] >= viewport_coords['y_min']) &
            (embeddings[:, 1] <= viewport_coords['y_max'])
        )
        
        if has_semantic_colors:
            # DETAIL - All points in viewport with semantic colors
            if np.any(viewport_mask):
                viewport_colors = [color_values[i] for i in range(len(color_values)) if viewport_mask[i]]
                detail_fig.add_trace(go.Scatter(
                    x=embeddings[viewport_mask, 0],
                    y=embeddings[viewport_mask, 1],
                    mode='markers+text',
                    marker=dict(size=16, color=viewport_colors, opacity=1.0, line=dict(width=2, color='white')),
                    text=[labels[i] for i in range(len(labels)) if viewport_mask[i]],
                    textposition="top center",
                    textfont=dict(size=16, color='black', family='Arial Black'),
                    hovertemplate='<b>%{text}</b><br>Detail View<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>',
                    name="Semantic Colors",
                    showlegend=False
                ))
        else:
            # DETAIL - Traditional language-based separation
            chinese_viewport = chinese_mask & viewport_mask
            english_viewport = english_mask & viewport_mask

            if np.any(chinese_viewport):
                detail_fig.add_trace(go.Scatter(
                    x=embeddings[chinese_viewport, 0],
                    y=embeddings[chinese_viewport, 1],
                    mode='markers+text',
                    marker=dict(size=16, color='red', opacity=1.0, line=dict(width=2, color='white')),
                    text=[labels[i] for i in range(len(labels)) if chinese_viewport[i]],
                    textposition="top center",
                    textfont=dict(size=16, color='red', family='Arial Black'),
                    hovertemplate='<b>%{text}</b><br>中文 (Detail)<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>',
                    name="中文",
                    showlegend=False
                ))

            if np.any(english_viewport):
                detail_fig.add_trace(go.Scatter(
                    x=embeddings[english_viewport, 0],
                    y=embeddings[english_viewport, 1],
                    mode='markers+text',
                    marker=dict(size=16, color='blue', opacity=1.0, line=dict(width=2, color='white')),
                    text=[labels[i] for i in range(len(labels)) if english_viewport[i]],
                    textposition="top center",
                    textfont=dict(size=16, color='blue', family='Arial Black'),
                    hovertemplate='<b>%{text}</b><br>English (Detail)<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>',
                    name="English",
                    showlegend=False
                ))
        
        # Detail: Viewport range with 10% margin
        x_range = viewport_coords['x_max'] - viewport_coords['x_min']
        y_range = viewport_coords['y_max'] - viewport_coords['y_min']
        x_margin = x_range * 0.1
        y_margin = y_range * 0.1
        
        detail_fig.update_xaxes(
            range=[viewport_coords['x_min'] - x_margin, viewport_coords['x_max'] + x_margin],
            title_text="x",
            showgrid=True, 
            gridwidth=1,
            gridcolor='#D0D0D0',
            griddash='dot'
        )
        detail_fig.update_yaxes(
            range=[viewport_coords['y_min'] - y_margin, viewport_coords['y_max'] + y_margin],
            title_text="y",
            showgrid=True, 
            gridwidth=1,
            gridcolor='#D0D0D0',
            griddash='dot',
            scaleanchor="x", scaleratio=1
        )
        
        # Create detail view title using centralized helper
        if method_name and model_name and dataset_name:
            detail_title = create_chart_title(
                [method_name] if method_name else [],
                [model_name] if model_name else [],
                dataset_name,
                lang_codes or []
            )
        else:
            detail_title = "Detail View"
        
        detail_fig.update_layout(
            title=dict(
                text=detail_title,
                font=dict(size=18, family='Arial, sans-serif'),
                x=0.5,  # Center align title
                xanchor='center'
            ),
            dragmode='pan',
            hovermode='closest',
            showlegend=False,
            height=900,
            plot_bgcolor='white',
            font=dict(family='Arial, sans-serif'),
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        # Count points in viewport
        points_in_viewport = viewport_mask.sum()
        
        return overview_fig, detail_fig, points_in_viewport, viewport_mask

def perform_dual_view_geometric_analysis(analyzer, params, embeddings, labels, model_name=None, method_name=None):
    """Perform comprehensive geometric analysis for dual view"""
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
    st.session_state.dual_view_geometric_analysis = analysis_results
    
    # Save metrics to files automatically
    try:
        # Get input name from session state
        input_name = st.session_state.get('cfg_input_text_entered', 'untitled')
        if not input_name or input_name == 'untitled':
            input_name = st.session_state.get('cfg_input_text_selected', 'dual_view')
        
        # Determine languages from enhanced data
        if 'enhanced_data' in st.session_state:
            enhanced_data = st.session_state.enhanced_data
            colors = enhanced_data.get('colors', [])
            languages = []
            if 'chinese' in colors:
                languages.append('chinese')
            if 'english' in colors:
                languages.append('english')
        else:
            languages = ['unknown']
        
        # Use provided model and method names, or fallback to defaults
        if model_name is None:
            model_name = 'dual-view-model'
        if method_name is None:
            method_name = 'dual-view-method'
        
        # Save metrics
        save_json = params.get('save_json_files', False)
        saved_files = analyzer.save_metrics_to_files(
            analysis_results, input_name, model_name, method_name, languages, save_json
        )
        
        # Display save status
        analyzer.display_metrics_save_status(saved_files)
        
    except Exception as e:
        st.warning(f"Could not save dual view metrics automatically: {str(e)}")

def display_dual_view_geometric_analysis(model_name=None, method_name=None):
    """Display geometric analysis results for dual view"""
    if 'dual_view_geometric_analysis' not in st.session_state:
        return
    
    results = st.session_state.dual_view_geometric_analysis
    
    if not results:
        return
    
    with st.expander("🔬 Geometric Analysis Results - Dual View", expanded=False):
        
        # Display analysis results without nested expanders
        from semanscope.components.geometric_analysis import GeometricAnalyzer
        analyzer = GeometricAnalyzer()
        
        if 'clustering' in results:
            st.subheader("🔍 Clustering Analysis")
            analyzer.display_clustering_metrics(results['clustering'])
        
        if 'branching' in results:
            st.subheader("🌿 Branching Analysis")
            analyzer.display_branching_metrics(results['branching'])
        
        if 'void' in results:
            st.subheader("🕳️ Void Analysis")
            analyzer.display_void_metrics(results['void'])
        
        # Summary visualization if multiple analyses exist
        if len(results) > 1 and 'enhanced_data' in st.session_state:
            st.subheader("📊 Comprehensive Analysis Visualization")
            try:
                from semanscope.components.geometric_analysis import GeometricAnalyzer
                analyzer = GeometricAnalyzer()
                
                enhanced_data = st.session_state.enhanced_data
                embeddings = enhanced_data['embeddings']
                labels = enhanced_data['labels']
                
                # Get dataset information for consistent title
                dataset_name = st.session_state.get('dual_input_text_selected', 'User Input')
                
                comprehensive_fig = analyzer.create_comprehensive_analysis_plot(
                    embeddings, labels,
                    results.get('clustering', {}),
                    results.get('branching', {}),
                    results.get('void', {}),
                    model_name, method_name, dataset_name
                )
                st.plotly_chart(comprehensive_fig, width='stretch')
                
                # Auto-save the clustering chart using standardized helper (Plotly version)
                try:
                    from semanscope.utils.auto_save_helper import get_auto_save_settings, display_auto_save_success
                    from semanscope.utils.title_filename_helper import create_title_and_filename

                    # Check if auto-save is enabled globally
                    auto_save_settings = get_auto_save_settings()
                    if not auto_save_settings.get('enabled', False):
                        # Skip auto-save if disabled
                        pass
                    else:
                        # Get current input name for filename
                        current_input = st.session_state.get('cfg_input_text_entered', 'untitled')
                        if not current_input or current_input == 'untitled':
                            current_input = st.session_state.get('cfg_input_text_selected', 'sample_1')

                        # Get publication settings for proper formatting
                        pub_settings = st.session_state.get('global_settings', {}).get('publication', {})
                        publication_mode = pub_settings.get('publication_mode', False)
                        export_format = pub_settings.get('export_format', 'PNG').upper() if publication_mode else 'PNG'
                        export_dpi = pub_settings.get('export_dpi', 300)

                        # Determine active languages (fallback to empty if not available)
                        active_languages = []

                        # Generate standardized filename
                        _, standardized_filename = create_title_and_filename(
                            [f"{method_name}-clustering"],
                            [model_name],
                            current_input,
                            active_languages if active_languages else ['multi'],
                            export_format.lower()
                        )

                        # Add zoom prefix
                        zoom_filename = f"zoom-{standardized_filename}"

                        # Auto-save the clustering chart
                        images_dir = DATA_PATH / "images"
                        images_dir.mkdir(parents=True, exist_ok=True)

                        # Use auto-save settings for dimensions
                        width = auto_save_settings.get('width', 800)
                        height = auto_save_settings.get('height', 700)

                        # Save with high quality
                        img_bytes = comprehensive_fig.to_image(
                            format=export_format.lower(),
                            width=width,
                            height=height,
                            scale=export_dpi/96
                        )

                        # Write to file
                        clustering_path = images_dir / zoom_filename
                        clustering_path.write_bytes(img_bytes)

                        display_auto_save_success([zoom_filename], "🔍")

                except Exception as auto_save_error:
                    st.warning(f"Could not auto-save clustering chart: {str(auto_save_error)}")
                
                # Add download button for clustering chart
                handle_download_button(comprehensive_fig, model_name, method_name, dataset_name, "clustering", "dual_view")
                    
            except Exception as e:
                st.error(f"Error creating comprehensive analysis plot: {str(e)}")

def setup_sidebar_controls():
    """Setup sidebar controls and return settings from global configuration"""
    # Use global settings instead of local sidebar controls
    viz_settings = get_global_viz_settings()
    publication_settings = get_global_publication_settings()

    # Store publication settings in session state for backward compatibility
    st.session_state.dual_view_publication_settings = publication_settings

    return {
        'model_name': viz_settings['model_name'],
        'method_name': viz_settings['method_name'],
        'publication_settings': publication_settings
    }


def handle_text_input():
    """Handle text input UI and return processed text data"""
    # Initialize visualizer and shared component
    visualizer = EmbeddingVisualizer()
    enter_text_widget = EnterTextDataWidget(key_prefix="dual_", max_languages=3)

    # Use the shared Enter Text Data widget (called in sidebar by main())
    text_data = enter_text_widget.render(visualizer)

    # Store the raw text_data in session state for use by get_selected_language_codes()
    st.session_state.dual_input_text_data = text_data

    # Convert the shared component's output to the format expected by Dual View
    languages_data = text_data['languages']  # List of (lang_name, lang_code, text_content, is_selected) tuples

    # Process text into word lists and create language_data structure
    language_data = {}

    for lang_name, lang_code, text, selected in languages_data:
        words = visualizer.process_text(text) if (selected and text) else []
        language_data[lang_code] = {
            'name': lang_name,
            'text': text,
            'selected': selected,
            'words': words
        }

    return language_data


def setup_geometric_analysis_controls():
    """Get geometric analysis settings from global configuration"""
    # Use global geometric analysis settings
    enable_geometric_analysis = is_global_geometric_analysis_enabled()
    analysis_params = get_global_geometric_analysis()

    return enable_geometric_analysis, analysis_params


def setup_zoom_controls():
    """Setup zoom controls and return zoom parameters"""  
    with st.sidebar:
        # Zoom controls
        with st.expander("🔭 Zoom Controls", expanded=False):
            # Global box size parameters
            global_width = 0.8
            global_height = 0.3
            
            # Initialize default zoom parameters
            if 'zoom_params' not in st.session_state:
                st.session_state.zoom_params = {
                    'center_x': 0.0, 'center_y': 0.0,
                    'width': 0.05, 'height': 0.05,  # Default zoom box size
                    'delta_x': 0.005, 'delta_y': 0.005  # Fixed panning step
                }
            
            zoom_params = st.session_state.zoom_params

            st.write("**Box Center:**")
            col1, col2 = st.columns(2)
            with col1:
                center_x = st.number_input("Center X", value=zoom_params['center_x'], step=DEFAULT_STEP_SIZE, format="%.3f")
            with col2:
                center_y = st.number_input("Center Y", value=zoom_params['center_y'], step=DEFAULT_STEP_SIZE, format="%.3f")
            
            st.write("**Box Size:**")
            col3, col4 = st.columns(2)
            with col3:
                width = st.number_input("Width", value=zoom_params['width'], step=DEFAULT_STEP_SIZE, format="%.3f")
            with col4:
                height = st.number_input("Height", value=zoom_params['height'], step=DEFAULT_STEP_SIZE, format="%.3f")
            
            st.write("**Panning Step:**")
            col5, col6 = st.columns(2)
            with col5:
                delta_x = st.number_input("Delta X", value=zoom_params['delta_x'], step=DEFAULT_STEP_SIZE, format="%.3f")
            with col6:
                delta_y = st.number_input("Delta Y", value=zoom_params['delta_y'], step=DEFAULT_STEP_SIZE, format="%.3f")
            
            col_update, col_reset = st.columns(2)
            with col_update:
                if st.button("🔄 Update Zoom", type="primary"):
                    # Apply pan movement to center
                    new_center_x = center_x + delta_x
                    new_center_y = center_y + delta_y
                    
                    st.session_state.zoom_params = {
                        'center_x': new_center_x, 'center_y': new_center_y,
                        'width': width, 'height': height,
                        'delta_x': delta_x, 'delta_y': delta_y
                    }
                    st.rerun()
            
            with col_reset:
                if st.button("🎯 Reset Zoom"):
                    st.session_state.zoom_params = {
                        'center_x': 0.0, 'center_y': 0.0,
                        'width': 0.05, 'height': 0.05,  # Default zoom box size
                        'delta_x': 0.005, 'delta_y': 0.005  # Fixed panning step
                    }
                    st.rerun()
        


def setup_action_buttons():
    """Setup action buttons and return button states"""
    with st.sidebar:
        # Generate button
        btn_visualize = st.button(
            "Visualize",
            type="primary",
            width='stretch',
            help="Generate semantic visualization"
        )

        btn_pan_col, btn_save_img_col = st.columns(2)
        
        with btn_pan_col:
            btn_pan = st.button(
                "Panning", 
                width='stretch',
                type="secondary", 
                help="Apply pan movement from zoom controls", 
                key="btn_panning"
            )
        with btn_save_img_col:
            btn_save_detail_img = st.button(
                "Save Image", 
                width='stretch',
                type="secondary", 
                help="Save Detail View to Image File", 
                key="btn_save_detail_img"
            )
            
        return btn_visualize, btn_pan, btn_save_detail_img


def get_selected_language_codes():
    """Get selected language codes from the dual view text data - only for checked languages"""
    lang_codes = []

    # Try to get from current text data (most reliable approach)
    # Check if we can get the language data from the EnterTextDataWidget
    if 'dual_input_text_data' in st.session_state:
        languages_data = st.session_state.dual_input_text_data.get('languages', [])
        for lang_name, lang_code, text_content, is_selected in languages_data:
            if is_selected:  # Only include if checkbox is checked
                lang_codes.append(lang_code.upper())
        return lang_codes

    # Fallback: try to get from enhanced data language information (but filter by selection)
    if 'enhanced_data' in st.session_state:
        enhanced_data = st.session_state.enhanced_data
        if 'language_info' in enhanced_data:
            # Get language codes from enhanced data, but we need to check selection status
            # This approach is less reliable since enhanced_data doesn't store selection flags
            language_info = enhanced_data['language_info']
            # For now, return all if we can't determine selection status
            # This is a fallback that may include unchecked languages
            return [lang_code.upper() for lang_code in language_info.keys()]

    # Final fallback: check session state for language selections
    for i in range(1, 4):  # Check lang1, lang2, lang3
        lang_key = f'dual_lang{i}'
        if lang_key in st.session_state:
            # Get language name and convert to code
            lang_name = st.session_state[lang_key]
            # Map common language names to codes
            lang_code_map = {
                'Chinese': 'CHN',
                'English': 'ENU',
                'Korean': 'KOR',
                'Japanese': 'JPN',
                'German': 'DEU',
                'French': 'FRA',
                'Spanish': 'SPA',
                'Arabic': 'ARA'
            }
            if lang_name in lang_code_map:
                lang_codes.append(lang_code_map[lang_name])

    return lang_codes

def show_overview():
    with st.expander("Overview", expanded=False):
        st.markdown("Detailed exploration with dual-view analysis and comprehensive geometric insights")
        st.markdown("""
        ### 🎯 Detailed Analysis Features:

        **🔬 Dual-View Visualization**: Side-by-side comparative analysis
        - Simultaneous visualization of different dimensional reductions
        - Compare different embedding models on the same data
        - Interactive synchronization between views

        **📊 Comprehensive Geometric Analysis**: Deep mathematical insights
        - Convex hull analysis with area and perimeter calculations
        - Centroid positioning and cluster dispersion metrics
        - Branching analysis for semantic tree structures
        - Void detection in semantic space

        **🎯 Enhanced Clustering**: Advanced cluster analysis tools
        - Silhouette scoring for cluster quality assessment
        - Inter-cluster distance measurements
        - Cluster density and compactness analysis
        - Outlier detection and boundary analysis

        **📈 Performance Metrics**: Quantitative evaluation tools
        - Dimensionality reduction stress analysis
        - Embedding quality metrics and validation
        - Statistical summaries and distributions
        - Export-ready analysis reports
        """)


def main():
    check_login()

    st.subheader("🔍 Semanscope - Zoom")
    st.info("👈 Configure text input and analysis settings in the sidebar and click **Visualize** to start charting with zooming/panning features")


    # Initialize components
    visualizer = EmbeddingVisualizer()
    reducer = DimensionReducer()
    dual_manager = EnhancedDualViewManager()
    geometric_analyzer = GeometricAnalyzer()

    # Setup sidebar components
    with st.sidebar:
        # Get text input using shared component
        language_data = handle_text_input()

    # Setup other sidebar components - now using global settings
    settings = setup_sidebar_controls()
    model_name = settings['model_name']
    method_name = settings['method_name']

    enable_geometric_analysis, analysis_params = setup_geometric_analysis_controls()
    setup_zoom_controls()
    btn_vis, btn_pan, btn_save_detail_img = setup_action_buttons()

    # Add sidebar configuration at the end
    with st.sidebar:
        # Display link to global settings
        GlobalSettingsManager.render_current_settings_summary()
    
    # Handle button actions
    if btn_vis:
        # DEDUPLICATION VALIDATION: Check if at least one unique dataset is selected
        lang_configs = [
            ("lang1", st.session_state.get('dual_lang1', 'English')),
            ("lang2", st.session_state.get('dual_lang2', 'English')),
            ("lang3", st.session_state.get('dual_lang3', 'English'))
        ]

        # Build unique datasets for validation (same logic as main generation)
        unique_validation_datasets = {}
        for i, (position, lang_name) in enumerate(lang_configs):
            lang_code = get_language_code_from_name(lang_name)

            # Get position-specific checkbox and text area state
            position_suffix = f"pos{i+1}"
            checkbox_key = f'dual_{lang_code}_include_checkbox_{position_suffix}'
            text_area_key = f'dual_{lang_code}_text_area_{position_suffix}'

            is_selected = st.session_state.get(checkbox_key, False)
            text_content = st.session_state.get(text_area_key, '')
            words = visualizer.process_text(text_content) if text_content else []

            if is_selected and words:
                if lang_code not in unique_validation_datasets:
                    # First occurrence of this language - keep it
                    unique_validation_datasets[lang_code] = {
                        'name': lang_name,
                        'code': lang_code,
                        'words': words,
                        'positions': [position]
                    }

        # Check if any unique datasets found
        if len(unique_validation_datasets) > 0:
            st.session_state.generate_requested = True
            st.rerun()
        else:
            st.warning("Please select at least one language and enter words/phrases to visualize")

    else:
        show_overview()




    # Handle save detail image
    if btn_save_detail_img:
        if 'enhanced_data' in st.session_state:
            # Get current input name from session state or use a default
            current_input_name = st.session_state.get('cfg_input_text_entered', 'dual-view')
            if not current_input_name or current_input_name == 'untitled':
                current_input_name = 'dual-view'
            
            # Recreate the detail figure for saving
            data = st.session_state.enhanced_data
            # Get dataset name
            dataset_name = st.session_state.get('dual_input_text_selected', 'User Input')
            
            # Get selected language codes
            lang_codes = get_selected_language_codes()

            overview_fig, detail_fig, points_count, viewport_mask = dual_manager.create_enhanced_dual_view(
                data['embeddings'],
                data['labels'],
                data['colors'],
                data['title'],
                st.session_state.zoom_params,
                model_name,
                method_name,
                dataset_name,
                lang_codes
            )
            
            # Determine selected languages for save function
            chinese_selected = language_data.get('chn', {}).get('selected', False)
            english_selected = language_data.get('enu', {}).get('selected', False)

            # Save the detail view image
            filename = visualizer.save_detail_view_image(
                detail_fig,
                current_input_name,
                model_name,
                method_name,
                chinese_selected,
                english_selected
            )
            
            if filename:
                st.success(f"Detail view saved as: {filename}")
            else:
                st.error("Failed to save detail view image")
        else:
            st.warning("No visualization to save. Please generate a visualization first.")

    # Display existing visualization if available
    if 'enhanced_data' in st.session_state:
        data = st.session_state.enhanced_data
        
        # Create visualization
        # Get dataset name
        dataset_name = st.session_state.get('dual_input_text_selected', 'User Input')
        
        # Get selected language codes
        lang_codes = get_selected_language_codes()

        overview_fig, detail_fig, points_count, viewport_mask = dual_manager.create_enhanced_dual_view(
            data['embeddings'],
            data['labels'],
            data['colors'],
            data['title'],
            st.session_state.zoom_params,
            model_name,
            method_name,
            dataset_name,
            lang_codes
        )
        
        # Detail view below (no container) with pan button
        st.plotly_chart(detail_fig, width='stretch', key="detail_view")
        
        # Add download button for detail view (always available)
        handle_download_button(detail_fig, model_name, method_name, dataset_name, "detail", "dual_view")
        
        if st.session_state.get('btn_panning', False):
            # Get current zoom params from sidebar
            current_params = st.session_state.zoom_params
            # Apply pan movement
            new_center_x = current_params['center_x'] + current_params['delta_x']
            new_center_y = current_params['center_y'] + current_params['delta_y']
            
            st.session_state.zoom_params = {
                'center_x': new_center_x, 'center_y': new_center_y,
                'width': current_params['width'], 'height': current_params['height'],
                'delta_x': current_params['delta_x'], 'delta_y': current_params['delta_y']
            }
            st.rerun()

        # Overview in expandable container
        with st.expander("📊 Overview", expanded=True):
            st.plotly_chart(overview_fig, width='stretch', config={'displayModeBar': False})
      
        # Stats in collapsible expander
        with st.expander("📊 Statistics", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Points", len(data['labels']))
            with col2:
                st.metric("Points in Zoom", points_count)
            with col3:
                coverage = (points_count / len(data['labels']) * 100) if len(data['labels']) > 0 else 0
                st.metric("Zoom Coverage", f"{coverage:.1f}%")
            
            # List words in zoom area
            if points_count > 0:
                st.write("**Words in Zoom Area:**")
                zoom_labels = [data['labels'][i] for i in range(len(data['labels'])) if viewport_mask[i]]
                zoom_colors = [data['colors'][i] for i in range(len(data['colors'])) if viewport_mask[i]]

                # Check if we have semantic colors (hex codes) or traditional language colors
                has_semantic_colors_in_data = any(isinstance(c, str) and c.startswith('#') for c in zoom_colors)

                if has_semantic_colors_in_data:
                    # For semantic colors, group by actual color values
                    color_groups = {}
                    for i, (label, color) in enumerate(zip(zoom_labels, zoom_colors)):
                        if color not in color_groups:
                            color_groups[color] = []
                        color_groups[color].append(label)

                    # Display groups of words by color
                    cols = st.columns(min(len(color_groups), 4))  # Max 4 columns
                    for idx, (color, words) in enumerate(color_groups.items()):
                        col_idx = idx % len(cols)
                        with cols[col_idx]:
                            if isinstance(color, str) and color.startswith('#'):
                                st.markdown(f"**Color {color}:**")
                            else:
                                st.markdown(f"**{color.capitalize()}:**")
                            st.write(", ".join(words))
                else:
                    # Traditional language-based separation
                    chinese_in_zoom = [label for i, label in enumerate(zoom_labels) if zoom_colors[i] == 'chinese']
                    english_in_zoom = [label for i, label in enumerate(zoom_labels) if zoom_colors[i] == 'english']

                    col_chn, col_eng = st.columns(2)
                    with col_chn:
                        if chinese_in_zoom:
                            st.write("**Chinese:**")
                            st.write(", ".join(chinese_in_zoom))

                    with col_eng:
                        if english_in_zoom:
                            st.write("**English:**")
                            st.write(", ".join(english_in_zoom))
        
        # Display geometric analysis results if available
        display_dual_view_geometric_analysis(model_name, method_name)
    
    # Handle generation request
    if st.session_state.get('generate_requested', False):
        st.session_state.generate_requested = False

        # Build language data like the main Semanscope page does
        lang_configs = [
            ("lang1", st.session_state.get('dual_lang1', 'English')),
            ("lang2", st.session_state.get('dual_lang2', 'English')),
            ("lang3", st.session_state.get('dual_lang3', 'English'))
        ]

        # Create selected_languages structure matching main page
        selected_languages = {}
        colors = []

        for i, (position, lang_name) in enumerate(lang_configs):
            lang_code = get_language_code_from_name(lang_name)
            lang_key = f"lang{i+1}"

            # Get position-specific checkbox state directly from session state
            position_suffix = f"pos{i+1}"
            checkbox_key = f'dual_{lang_code}_include_checkbox_{position_suffix}'
            text_area_key = f'dual_{lang_code}_text_area_{position_suffix}'

            is_selected = st.session_state.get(checkbox_key, False)
            text_content = st.session_state.get(text_area_key, '')

            # Process text to get words
            words = visualizer.process_text(text_content) if text_content else []

            selected_languages[lang_key] = {
                'name': lang_name,
                'code': lang_code,
                'words': words,
                'selected': is_selected
            }

            # Build colors for this language (DO NOT load here - trust session state from Load Text button)
            if is_selected and words:
                # Check for semantic colors stored in session state (from Load Text button)
                semantic_colors_key = f"{lang_code}_semantic_colors"
                word_color_map = st.session_state.get(semantic_colors_key, {})

                # Debug: Show what's in session state
                st.write(f"🔍 DEBUG: Session state keys containing 'color': {[k for k in st.session_state.keys() if 'color' in k.lower()]}")
                st.write(f"🔍 DEBUG: Looking for key: {semantic_colors_key}")

                if word_color_map:
                    st.success(f"🎨 Zoom: Found {len(word_color_map)} semantic colors for {lang_code} in session state")
                    sample_mappings = list(word_color_map.items())[:5]
                    st.info(f"Sample mappings: {sample_mappings}")
                else:
                    st.warning(f"🎨 Zoom: No semantic colors found for {lang_code} in session state")

                if word_color_map:  # Use domain colors if available
                    # Better fallback color for missing words - use a bright purple to make missing mappings obvious
                    word_colors = [word_color_map.get(word, '#FF00FF') for word in words]
                    colors.extend(word_colors)
                    # Debug: Log color assignment
                    color_stats = {}
                    for color in word_colors:
                        color_stats[color] = color_stats.get(color, 0) + 1
                    st.info(f"🎨 Color stats for {lang_code}: {color_stats}")
                else:
                    # Fallback to language colors from COLOR_MAP
                    from semanscope.config import COLOR_MAP
                    lang_color_map = {"enu": "english", "chn": "chinese", "fra": "french", "spa": "spanish",
                                     "deu": "german", "ara": "arabic", "heb": "hebrew", "hin": "hindi",
                                     "jpn": "japanese", "kor": "korean", "rus": "russian", "tha": "thai",
                                     "vie": "vietnamese"}
                    color_key = lang_color_map.get(lang_code, "english")
                    word_colors = [COLOR_MAP[color_key]] * len(words)
                    colors.extend(word_colors)
                    st.info(f"🎨 Using fallback language color {COLOR_MAP[color_key]} for {lang_code}")

        # Apply deduplication logic using the selected_languages structure
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

        # Check if any unique language has words
        has_words = len(unique_datasets) > 0

        if has_words:
            # Show deduplication info if duplicates were found
            duplicate_count = sum(len(info['positions']) - 1 for info in unique_datasets.values())
            if duplicate_count > 0:
                duplicated_langs = [f"{info['name']} (positions: {', '.join(info['positions'])})"
                                  for info in unique_datasets.values() if len(info['positions']) > 1]
                st.info(f"🔄 **Deduplication Applied**: Processed {len(unique_datasets)} unique datasets (skipped {duplicate_count} duplicates). Duplicated: {', '.join(duplicated_langs)}")

            with st.spinner("🔄 Processing embeddings..."):
                # Generate embeddings
                all_embeddings = []
                all_labels = []
                all_colors = []

                # Language code to model language mapping
                lang_to_model_code = {
                    "chn": "zh",
                    "enu": "en",
                    "fra": "fr",
                    "spa": "es",
                    "deu": "de",
                    "ara": "ar"
                }

                # Language code to color mapping (fallback)
                lang_color_map = {
                    "chn": "#FF0000",  # Red
                    "enu": "#0000FF",  # Blue
                    "fra": "#008000",  # Green
                    "spa": "#FFA500",  # Orange
                    "deu": "#800080",  # Purple
                    "ara": "#A52A2A",  # Brown
                }

                # Process each unique language dataset once (use same logic as main page)
                all_embeddings = []
                all_labels = []
                deduplicated_colors = []

                for lang_code, dataset_info in unique_datasets.items():
                    model_lang = lang_to_model_code.get(lang_code, "en")
                    embeddings = visualizer.get_embeddings(dataset_info['words'], model_name, model_lang)
                    if embeddings is not None:
                        # Ensure embeddings and words have same length
                        if len(embeddings) == len(dataset_info['words']):
                            all_embeddings.append(embeddings)
                            all_labels.extend(dataset_info['words'])

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

                all_colors = deduplicated_colors

                if all_embeddings:
                    combined_embeddings = np.vstack(all_embeddings)
                    
                    # Reduce dimensions
                    reduced_embeddings = reducer.reduce_dimensions(
                        combined_embeddings, 
                        method=method_name, 
                        dimensions=2
                    )
                    
                    if reduced_embeddings is not None:
                        # Update zoom parameters based on data range if not already set
                        if 'zoom_params' not in st.session_state:
                            x_center = reduced_embeddings[:, 0].mean()
                            y_center = reduced_embeddings[:, 1].mean()
                            
                            st.session_state.zoom_params = {
                                'center_x': x_center, 'center_y': y_center,
                                'width': 0.05, 'height': 0.05,  # Default zoom box size
                                'delta_x': 0.005, 'delta_y': 0.005  # Fixed panning step
                            }
                        
                        # Store data
                        st.session_state.enhanced_data = {
                            'embeddings': reduced_embeddings,
                            'labels': all_labels,
                            'colors': all_colors,
                            'title': f"{model_name} + {method_name}"
                        }
                        
                        # Perform geometric analysis if enabled
                        if enable_geometric_analysis and analysis_params:
                            with st.spinner("🔬 Performing geometric analysis..."):
                                perform_dual_view_geometric_analysis(
                                    geometric_analyzer, analysis_params, 
                                    reduced_embeddings, all_labels,
                                    model_name, method_name
                                )
                        
                        st.rerun()
                    else:
                        st.error("Failed to reduce dimensions")
                else:
                    st.error("Failed to generate embeddings")
        else:
            st.warning("Please select at least one language and enter words/phrases to visualize")



    with st.sidebar:
        # st.markdown("---")
        with st.expander("Usage Tips"):
            st.markdown("""
            **How to use Enhanced Dual Viewer:**
            1. **Zoom Center**: Set the center point of your zoom region
            2. **Zoom Size**: Define width/height of the zoom box
            3. **Pan Movement**: Use Delta X/Y to move the viewport
            4. **Update View**: Click 'Update Zoom' to apply center + pan changes
            5. **Detail View**: Shows large labels with 10% margin around zoom area
            """)

if __name__ == "__main__":
    main()
