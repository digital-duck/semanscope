import streamlit as st
import numpy as np
import os
from semanscope.components.embedding_viz import EmbeddingVisualizer
from semanscope.components.dimension_reduction import DimensionReducer
from semanscope.components.plotting_echarts import EChartsPlotManager
from semanscope.components.geometric_analysis import GeometricAnalyzer
from semanscope.components.shared.enter_text_data_multilingual import MultilingualEnterTextDataWidget

from semanscope.config import (
    check_login,
    DEFAULT_N_CLUSTERS,
    DEFAULT_MODEL,
    DEFAULT_METHOD,
    COLOR_MAP,
    LANGUAGE_CODE_MAP,
    DEFAULT_LANG,
    DEFAULT_INSTRUCTION_MSG_1,
    get_language_code_from_name,
    get_model_language_code,
    get_active_models,
    get_active_methods_config
)
from semanscope.utils.global_settings import (
    get_global_viz_settings,
    get_global_publication_settings,
    get_global_geometric_analysis,
    is_global_geometric_analysis_enabled,
    GlobalSettingsManager
)

# Page config
st.set_page_config(
    page_title="Multilingual Cross-Script Explorer",
    page_icon="🌐",
    layout="wide"
)

class MultilingualEChartsVisualizer(EmbeddingVisualizer):
    """Enhanced embedding visualizer for 9-language cross-script analysis using Apache ECharts"""

    def __init__(self):
        super().__init__()
        self.echarts_plot_manager = EChartsPlotManager()

    def create_plot(self, reduced_embeddings, labels, colors, model_name, method_name,
                   dimensions="2D", do_clustering=False, n_clusters=DEFAULT_N_CLUSTERS, 
                   dataset_name="", highlight_config=None, lang_codes=None):
        """Create multilingual visualization using ECharts optimized for cross-script analysis"""

        # Create enhanced plot title with all language codes
        title = self.echarts_plot_manager.create_title(method_name, model_name, dataset_name, lang_codes)

        # Apply rotation if set
        if hasattr(st.session_state, 'plot_rotation') and st.session_state.plot_rotation != 0:
            if dimensions == "2D":
                # Apply 2D rotation
                angle = np.radians(st.session_state.plot_rotation)
                cos_angle, sin_angle = np.cos(angle), np.sin(angle)
                rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
                reduced_embeddings = reduced_embeddings @ rotation_matrix.T

        # Create ECharts visualization optimized for multilingual analysis
        # Use display_chart=True to show the chart immediately like ECharts-3D page
        if dimensions == "2D":
            self.echarts_plot_manager.plot_2d(
                reduced_embeddings, labels, colors, title, 
                clustering=do_clustering, n_clusters=n_clusters,
                method_name=method_name, model_name=model_name, dataset_name=dataset_name,
                highlight_config=highlight_config, lang_codes=lang_codes,
                display_chart=True  # Display chart immediately via ECharts
            )
        else:  # 3D
            self.echarts_plot_manager.plot_3d(
                reduced_embeddings, labels, colors, title,
                clustering=do_clustering, n_clusters=n_clusters,
                method_name=method_name, model_name=model_name, dataset_name=dataset_name,
                highlight_config=highlight_config, lang_codes=lang_codes,
                display_chart=True  # Display chart immediately via ECharts
            )

        return True  # Return success instead of chart config

@st.fragment
def generate_multilingual_visualization(visualizer, reducer, multilingual_data, model_name, method_name,
                                       dimensions, do_clustering, n_clusters, color_map=COLOR_MAP, debug_flag=False, text_data=None, highlight_config=None):
    """Generate embeddings and visualization for up to 9 languages"""
    
    if debug_flag:
        with st.expander("🔍 MULTILINGUAL DEBUG OUTPUT", expanded=True):
            st.success(f"✅ DEBUG MODE ENABLED - Using model: {model_name}")
            st.info("Processing multilingual cross-script analysis...")
            
            # Debug: Show what languages are being processed
            active_languages = [lang_data for lang_data in multilingual_data if lang_data[3]]  # is_selected = True
            st.write(f"**Processing {len(active_languages)} active languages:**")
            for lang_name, lang_code, text_content, is_selected in active_languages:
                words = text_content.split('\n') if text_content else []
                st.write(f"• {lang_name} ({lang_code}): {len(words)} words")
                if words and len(words) > 0:
                    preview_words = words[:5]
                    st.write(f"  📝 Sample: {', '.join(preview_words)}")

    # Filter to only selected languages
    active_languages = [(name, code, content, selected) for name, code, content, selected in multilingual_data if selected and content.strip()]
    
    if not active_languages:
        st.warning("Please select at least one language and enter words/phrases.")
        return False

    # Process embeddings for each active language
    all_embeddings = []
    all_labels = []
    all_colors = []
    lang_codes_processed = []

    for lang_name, lang_code, text_content, _ in active_languages:
        words = [word.strip() for word in text_content.split('\n') if word.strip()]
        if not words:
            continue

        # Generate embeddings for this language
        model_lang = get_model_language_code(lang_code)
        embeddings = visualizer.get_embeddings(words, model_name, lang=model_lang)

        if embeddings is not None and len(embeddings) > 0:
            all_embeddings.extend(embeddings)
            all_labels.extend(words)

            # Simple and direct: use COLOR_MAP for language-based domain colors
            # No need for complex fallback logic - just use the proper domain colors directly
            lang_color_map = {
                "enu": "english", "chn": "chinese", "fra": "french", "spa": "spanish",
                "deu": "german", "ara": "arabic", "heb": "hebrew", "hin": "hindi",
                "jpn": "japanese", "kor": "korean", "rus": "russian", "tha": "thai",
                "vie": "vietnamese"
            }

            color_key = lang_color_map.get(lang_code, "english")
            actual_color = color_map[color_key]
            word_colors = [actual_color] * len(words)



            all_colors.extend(word_colors)
            lang_codes_processed.append(lang_code)


            if debug_flag:
                st.success(f"✅ Generated {len(embeddings)} embeddings for {lang_name}")
        else:
            if debug_flag:
                st.error(f"❌ Failed to generate embeddings for {lang_name}")

    if not all_embeddings:
        st.error("Failed to generate embeddings for any language.")
        return False

    # Reduce dimensions - convert dimensions string to integer
    try:
        # Convert dimensions from string to integer for reducer
        if dimensions == "2D":
            n_components = 2
        elif dimensions == "3D":
            n_components = 3
        else:
            n_components = 2  # Default fallback
            
        reduced_embeddings = reducer.reduce_dimensions(
            np.array(all_embeddings), method_name, n_components
        )
        
        if debug_flag:
            st.success(f"✅ Dimension reduction completed: {reduced_embeddings.shape}")
            
    except Exception as e:
        st.error(f"Dimension reduction failed: {str(e)}")
        return False

    # Get actual dataset name (not hardcoded)
    dataset_name = st.session_state.get('multilingual_cfg_input_text_selected', 'Cross-Script-Analysis')
    if not dataset_name:
        dataset_name = 'Cross-Script-Analysis'

    # Generate the plot (ECharts visualization displayed immediately)
    success = visualizer.create_plot(
        reduced_embeddings, all_labels, all_colors, model_name, method_name,
        dimensions, do_clustering, n_clusters, dataset_name, highlight_config=highlight_config, lang_codes=lang_codes_processed
    )

    if success:
        
        # Auto-save now handled automatically by plot_2d() in plotting_echarts.py
        # No manual auto-save needed here anymore
        st.info("💡 **For PDF format**: Use the ECharts download menu (📥) in the visualization to export as PDF")
        
        # Display language summary
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📊 Cross-Script Analysis Summary")
            for i, (lang_name, lang_code, _, _) in enumerate(active_languages):
                lang_color = color_map.get(lang_name.lower(), color_map.get(lang_code, '#666666'))
                word_count = len([w for w in all_labels if all_colors[all_labels.index(w)] == lang_color])
                st.markdown(f"• **{lang_name}** ({lang_code}): {word_count} words")
        
        with col2:
            st.markdown("### 🎛️ Visualization Controls")
            st.info(f"**Method**: {method_name}")
            st.info(f"**Model**: {model_name}")
            st.info(f"**Dimensions**: {dimensions}")
            if do_clustering:
                st.info(f"**Clusters**: {n_clusters}")

        return True
    
    return False

def main():
    """Main function for the Multilingual Cross-Script Explorer"""
    
    # Check login
    check_login()
    
    # Initialize visualizer and reducer
    visualizer = MultilingualEChartsVisualizer()
    reducer = DimensionReducer()
    
    # Page header
    st.subheader("🌐 Semanscope: Multilingual Analysis")
    st.info("👈 Configure multilingual text input and model settings in the sidebar and click **Visualize** to explore cross-language semantics")

    # Get available models and methods directly from config
    active_models = get_active_models()
    active_methods = get_active_methods_config()
    debug_flag = False

    # Sidebar configuration
    with st.sidebar:
        
        # Use the enhanced multilingual text input widget
        enter_text_widget = MultilingualEnterTextDataWidget(key_prefix="multilingual_")
        text_data = enter_text_widget.render(visualizer)
        
        multilingual_data = text_data['languages']
        expanded_view = text_data.get('expanded_view', False)
        
        # Dimensions control (2D/3D toggle like ECharts-3D page)
        # with st.expander("📐 Visualization Dimensions", expanded=False):
        dimensions = st.radio(
            "Choose Dimensions",
            options=["2D", "3D"],
            horizontal=True,
            index=0,  # Default to 2D for multilingual analysis
            help="Select 2D or 3D visualization using ECharts",
            key="multilingual_echarts_dimensions"
        )

        # Use global settings (controlled by Settings page)
        viz_settings = get_global_viz_settings()
        model_name = viz_settings['model_name']
        method_name = viz_settings['method_name']
        do_clustering = viz_settings['do_clustering']
        n_clusters = viz_settings['n_clusters']

        # Action buttons
        # st.markdown("---")
        btn_visualize = st.button(
            "Visualize",
            type="primary",
            width='stretch',
            help="Create multilingual embedding visualization"
        )

        # Word Search settings
        highlight_config = visualizer.echarts_plot_manager.get_highlight_settings()

        # Display link to global settings
        from semanscope.utils.global_settings import GlobalSettingsManager
        GlobalSettingsManager.render_current_settings_summary()

        # Move informational content to bottom of sidebar
        st.markdown("---")

        # Add description to sidebar
        st.markdown("""
        **Advanced cross-script semantic analysis supporting up to 9 languages simultaneously.**
        Perfect for comparing writing systems: *Latin, Chinese, Arabic, Greek, Korean, Japanese, and more.*

        📋 **Research Applications:**
        - Sub-character level analysis across writing systems
        - Cross-script geometric pattern discovery
        - Multilingual embedding model comparison
        - Writing system semantic structure analysis
        """)
        
        # st.markdown("---")

        st.markdown("### 📖 How to Use Cross-Script Analysis")
        with st.expander("Usage Instructions", expanded=False):
            st.markdown("""
            **Step 1: Select Dataset**
            - Choose from available multilingual datasets
            - Or use custom text input for each language

            **Step 2: Configure Languages**
            - Start with 3 languages in primary view
            - Click "Load Text" to expand to 9 languages
            - Mix different script families for best results

            **Step 3: Generate Visualization**
            - Click the Visualize button
            - Explore cross-script geometric patterns
            - Compare semantic structures across writing systems
            """)

        st.markdown("### 🎯 Recommended Configurations")
        with st.expander("Example Setups", expanded=False):
            st.markdown("""
            **📝 Alphabet Comparison**
            - English (Latin), Greek (Greek), Russian (Cyrillic)
            - Arabic (Arabic), Hebrew (Hebrew), Korean (Hangul)
            
            **🏗️ Character Structure**
            - Chinese (Logographic), Japanese (Mixed), Korean (Featural)
            - English (Alphabetic), Arabic (Abjad), Hindi (Abugida)
            
            **🌍 Language Families**
            - English (Germanic), French (Romance), German (Germanic)  
            - Chinese (Sino-Tibetan), Arabic (Semitic), Greek (Hellenic)
            """)

        with st.expander("🔬 Research Insights", expanded=False):
            st.markdown("""
            ### Cross-Script Analysis Applications:
            
            **1. Writing System Classification:**
            - Compare how different scripts cluster
            - Identify universal vs. script-specific patterns
            
            **2. Sub-Character Structure Analysis:**
            - Study radicals, graphemes, and basic writing units
            - Discover geometric patterns in character composition
            
            **3. Semantic Universals:**
            - Find concepts that cluster across all languages
            - Identify language-specific semantic organization
            """)

        # Debug settings (moved to bottom)
        if st.checkbox("🔍 Debug Mode", value=debug_flag):
            debug_flag = True

    # Main visualization area
    if btn_visualize:
        with st.spinner("Generating multilingual cross-script analysis..."):
            success = generate_multilingual_visualization(
                visualizer, reducer, multilingual_data, model_name, method_name,
                dimensions, do_clustering, n_clusters, COLOR_MAP, debug_flag, text_data, highlight_config
            )
            
            if success:
                st.success("✅ Cross-script analysis completed!")
                
                # Additional insights for research
                st.markdown("---")
                with st.expander("🔬 Research Insights", expanded=False):
                    st.markdown("""
                    ### Cross-Script Analysis Applications:
                    
                    **1. Writing System Classification:**
                    - Compare how different scripts (Latin, Chinese, Arabic) cluster
                    - Identify universal vs. script-specific patterns
                    
                    **2. Sub-Character Structure Analysis:**
                    - Study radicals, graphemes, and basic writing units
                    - Discover geometric patterns in character composition
                    
                    **3. Semantic Universals:**
                    - Find concepts that cluster across all languages
                    - Identify language-specific semantic organization
                    
                    **4. Model Evaluation:**
                    - Test how well models handle different writing systems
                    - Compare embedding quality across script families
                    """)
            else:
                st.error("❌ Visualization generation failed. Please check your inputs and try again.")
    else:
        # Clean main panel - show simple placeholder
        # st.info(DEFAULT_INSTRUCTION_MSG_1)
        show_overview()
            
def show_overview():
    with st.expander("Overview", expanded=False):
        st.markdown("Cross-linguistic semantic analysis with specialized multilingual embedding models")
        st.markdown("""
        ### 🎯 Multilingual Features:

        **🌍 Cross-Language Analysis**: Compare semantic relationships across different languages
        - Optimized models for multilingual embeddings (LASER, mT5, XLM-R)
        - Support for 15+ languages including CJK (Chinese, Japanese, Korean)
        - Direct cross-script comparison capabilities

        **🔗 Semantic Alignment**: Discover how concepts align across languages
        - Identify translation equivalents in semantic space
        - Measure cross-linguistic semantic distances
        - Visualize language-specific clustering patterns

        **📚 Script Diversity**: Handle diverse writing systems and character encodings
        - Latin, Cyrillic, Arabic, Hebrew, Devanagari support
        - CJK character handling with proper tokenization
        - Mixed-script text analysis capabilities

        **🎨 Language-Aware Visualization**: Color-coded by language for clarity
        - Distinct colors for each language/script
        - Language-specific legends and annotations
        - Comparative side-by-side visualizations
        """)
        

if __name__ == "__main__":
    main()