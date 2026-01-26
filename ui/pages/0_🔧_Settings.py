import streamlit as st
from semanscope.components.embedding_viz import EmbeddingVisualizer, get_active_methods
from pathlib import Path
from semanscope.components.geometric_analysis import GeometricAnalyzer
from semanscope.components.shared.publication_settings import PublicationSettingsWidget
from semanscope.config import (
    check_login,
    MODEL_INFO,
    METHOD_INFO,
    DEFAULT_MODEL,
    DEFAULT_METHOD,
    DEFAULT_N_CLUSTERS,
    DEFAULT_LANG_SET,
    DEFAULT_TTL,
    DEFAULT_DATASET,
    DEFAULT_DIMENSION,
    DEFAULT_CLUSTERING,
    DEFAULT_SPLIT_LINES,
    DATA_PATH,
    get_active_models,
    get_language_codes_with_prefix,
    get_all_language_names,
    get_all_language_codes,
    get_language_name_from_code,
    get_language_code_from_name,
    LANGUAGE_CODE_MAP
)
from semanscope.utils.cache_manager import get_cache_manager, get_cache_stats, cleanup_cache, clear_all_cache

def get_available_datasets():
    """Get list of available dataset names from data/input directory (like Semanscope page)"""
    input_dir = DATA_PATH / "input"
    if not input_dir.exists():
        return ["sample_1"]

    input_names = set()
    # Get all language codes with prefix using helper function
    lang_codes = get_language_codes_with_prefix("-")

    # Check both .txt and .csv files
    for extension in ["*.txt", "*.csv"]:
        for file_path in input_dir.glob(extension):
            # Skip color-code files
            if "color-code" in file_path.name:
                continue

            name_part = file_path.stem
            # Strip all possible language suffixes (only the last one found)
            for lang_code in lang_codes:
                if name_part.endswith(lang_code):
                    name_part = name_part[:-len(lang_code)]
                    break  # Only remove one suffix to avoid over-stripping
            input_names.add(name_part)

    return sorted(list(input_names)) if input_names else ["sample_1"]

# Page config
st.set_page_config(
    page_title="Settings",
    page_icon="🔧",
    layout="wide"
)

def initialize_global_settings():
    """Initialize global settings with defaults if not already set"""
    if 'global_settings' not in st.session_state:
        st.session_state.global_settings = {
            # Publication Settings
            'publication': {
                'textfont_size': 16,
                'point_size': 12,
                'plot_height': 800,
                'export_dpi': 300,
                'export_format': 'PNG'
            },

            # Geometric Analysis Settings
            'geometric_analysis': {
                'enabled': DEFAULT_CLUSTERING,
                'enable_clustering': DEFAULT_CLUSTERING,
                'enable_branching': DEFAULT_CLUSTERING,
                'enable_void': DEFAULT_CLUSTERING,
                'n_clusters': DEFAULT_N_CLUSTERS,
                'density_radius': 0.1,
                'connectivity_threshold': 0.8,
                'void_confidence': 0.95,
                'save_json_files': False
            },

            # ECharts specific settings
            'echarts': {
                'auto_save_enabled': True,
                'png_width': 800,
                'png_height': 800
            },

            # Color Coding Settings for semantic domains
            'color_coding': {
                'domain_colors': {
                    # Function words (Red family)
                    'articles_determiners': '#FF4444',
                    'prepositions': '#FF6666',
                    'conjunctions': '#FF8888',
                    'pronouns': '#FFAAAA',

                    # Abstract sequential (Green/Orange)
                    'numbers': '#44FF44',
                    'colors': '#FF8800',

                    # Content words (Blue family)
                    'family_kinship': '#4488FF',
                    'body_parts': '#8B4513',
                    'animals': '#FF8C00',
                    'food': '#FF6347',
                    'actions_verbs': '#4444FF',
                    'emotions': '#6666FF',
                    'nature_elements': '#228B22',
                    'time_temporal': '#4444FF',
                    'spatial_directional': '#888888',
                    'abstract_qualities': '#DDA0DD',

                    # Morphological families (Purple family)
                    'morphological_work': '#800080',
                    'morphological_light': '#9400D3',
                    'morphological_book': '#8A2BE2',
                    'morphological_子': '#800080',

                    # Fallback colors
                    'unknown': '#CCCCCC',
                    'default': '#666666'
                },
                'custom_domains': {},
                'color_scheme_name': 'default'
            },

            # Language Settings
            'languages': {
                'default_languages': DEFAULT_LANG_SET.copy(),
                'available_languages': get_all_language_codes(),
                'language_priority': DEFAULT_LANG_SET.copy()
            },

            # Cache Settings
            'cache_ttl_hours': DEFAULT_TTL,  # Default cache TTL from config

            # Default Global Variables
            'defaults': {
                'dataset': DEFAULT_DATASET,
                'method': DEFAULT_METHOD,
                'model': DEFAULT_MODEL,
                'dimension': DEFAULT_DIMENSION,
                'split_lines': DEFAULT_SPLIT_LINES
            }
        }

def save_global_settings():
    """Save current settings to session state"""
    # This function will be called when settings are updated
    st.success("✅ Settings saved and will be applied to all Semanscope pages")
    st.rerun()

def save_defaults_to_config_file(defaults_dict):
    """Save default settings to config.py file by updating the DEFAULT_* variables"""
    try:
        import os
        from pathlib import Path

        # Get the config.py file path
        config_file = Path(__file__).parent.parent / "config.py"

        if not config_file.exists():
            st.error(f"❌ Config file not found: {config_file}")
            return False

        # Read current config.py content
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Update each default value
        lines = content.split('\n')
        updated_lines = []

        for line in lines:
            if line.strip().startswith('DEFAULT_DATASET ='):
                updated_lines.append(f'DEFAULT_DATASET = "{defaults_dict["dataset"]}"  # "ACL-2-word-v2" # "ACL-6-Emoji" #   "ACL-1-Alphabets"')
            elif line.strip().startswith('DEFAULT_METHOD ='):
                updated_lines.append(f'DEFAULT_METHOD = "{defaults_dict["method"]}"')
            elif line.strip().startswith('DEFAULT_MODEL ='):
                updated_lines.append(f'DEFAULT_MODEL = "{defaults_dict["model"]}"  # MTEB #5, proven baseline for geometric studies')
            elif line.strip().startswith('DEFAULT_DIMENSION ='):
                updated_lines.append(f'DEFAULT_DIMENSION = "{defaults_dict["dimension"]}"')
            elif line.strip().startswith('DEFAULT_SPLIT_LINES ='):
                updated_lines.append(f'DEFAULT_SPLIT_LINES = {str(defaults_dict["split_lines"])}  # Enable line splitting/preprocessing by default')
            else:
                updated_lines.append(line)

        # Write the updated content back
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(updated_lines))

        return True

    except Exception as e:
        st.error(f"❌ Error updating config.py: {str(e)}")
        return False

def update_config_file(new_languages):
    """Update the DEFAULT_LANG_SET in config.py file (optional feature)"""
    import os

    # Use relative path to config.py in the semanscope package
    config_path = Path(__file__).parent.parent / "config.py"
    config_path = str(config_path.absolute())
    
    try:
        # Read current config file
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find and replace the DEFAULT_LANG_SET line
        import re
        pattern = r'DEFAULT_LANG_SET\s*=\s*\[.*?\]'
        new_line = f'DEFAULT_LANG_SET = {new_languages}'
        
        updated_content = re.sub(pattern, new_line, content)
        
        # Write back to file
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        return True
    except Exception as e:
        st.error(f"❌ Failed to update config.py: {str(e)}")
        return False

def get_user_language_settings():
    """Get user's language settings, falling back to config defaults"""
    if 'global_settings' in st.session_state and 'languages' in st.session_state.global_settings:
        return st.session_state.global_settings['languages']['default_languages']
    else:
        return DEFAULT_LANG_SET

def render_charting_settings():
    """Render consolidated charting settings (ECharts + Publication)"""
    st.subheader("📊 Charting Settings")
    st.markdown("Configure visualization and export settings for all chart types.")

    # Create 2-column layout: Settings (left, wider) + Current Values (right, narrower)
    settings_col, current_col = st.columns([3, 1])

    with settings_col:
        current_echarts = st.session_state.global_settings['echarts']

        # Auto-save settings section
        st.subheader("🎯 Auto-save Settings")

        col1, col2 = st.columns(2)

        with col1:
            # Auto-save settings
            current_echarts['auto_save_enabled'] = st.checkbox(
                "Auto-save 2D chart",
                value=current_echarts['auto_save_enabled'],
                help="Automatically save 2D visualizations. Note: ECharts pages will use PNG format even if PDF is selected below.",
                key="global_echarts_auto_save"
            )

        with col2:
            if current_echarts['auto_save_enabled']:
                # Get current publication format
                pub_format = st.session_state.get('global_settings', {}).get('publication', {}).get('export_format', 'PNG')
                st.caption(f"✅ Auto-save enabled ({pub_format} format)")
            else:
                st.caption("ℹ️ Auto-save disabled")

        # Auto-save Image Size section (only show if auto-save is enabled)
        if current_echarts['auto_save_enabled']:
            st.markdown("---")
            st.subheader("📐 Auto-save Image Size")

            # Image dimensions
            col3, col4 = st.columns(2)

            with col3:
                current_echarts['png_width'] = st.number_input(
                    "Image Width (px)",
                    min_value=600,
                    max_value=2400,
                    value=current_echarts['png_width'],
                    step=100,
                    help="Width of auto-saved images (PNG/PDF)",
                    key="global_png_width"
                )

            with col4:
                current_echarts['png_height'] = st.number_input(
                    "Image Height (px)",
                    min_value=400,
                    max_value=1600,
                    value=current_echarts['png_height'],
                    step=100,
                    help="Height of auto-saved images (PNG/PDF)",
                    key="global_png_height"
                )

            # Preset buttons
            st.write("**Quick Presets:**")
            col5, col6, col7 = st.columns(3)

            with col5:
                if st.button("📱 Small (600x600)", key="preset_small"):
                    st.session_state.global_settings['echarts']['png_width'] = 600
                    st.session_state.global_settings['echarts']['png_height'] = 600
                    st.rerun()

            with col6:
                if st.button("💻 Medium (800x800)", key="preset_medium"):
                    st.session_state.global_settings['echarts']['png_width'] = 800
                    st.session_state.global_settings['echarts']['png_height'] = 800
                    st.rerun()

            with col7:
                if st.button("📄 Large (1200x1200)", key="preset_large"):
                    st.session_state.global_settings['echarts']['png_width'] = 1200
                    st.session_state.global_settings['echarts']['png_height'] = 1200
                    st.rerun()

        # Publication Format section (moved to last)
        st.markdown("---")
        st.subheader("📄 Publication Format")

        # Use the shared publication settings widget
        publication_settings = PublicationSettingsWidget.render_publication_settings("global_settings")

        # Update global settings
        st.session_state.global_settings['publication'] = publication_settings

        # Save button
        st.markdown("---")
        if st.button("💾 Save Settings", type="primary", width='stretch', key="save_charting"):
            save_global_settings()

    with current_col:
        st.subheader("Current Values")

        # Current auto-save settings
        st.markdown("**Auto-save:**")
        auto_save_status = "Enabled" if current_echarts['auto_save_enabled'] else "Disabled"
        st.code(auto_save_status)

        # Current export settings
        current_pub = st.session_state.global_settings['publication']

        st.markdown("**Export Format:**")
        st.code(current_pub['export_format'])

        st.markdown("**Export DPI:**")
        st.code(str(current_pub['export_dpi']))

        # Current dimensions
        st.markdown("**Image Width:**")
        st.code(f"{current_echarts['png_width']}px")

        st.markdown("**Image Height:**")
        st.code(f"{current_echarts['png_height']}px")

        st.markdown("**Aspect Ratio:**")
        ratio = current_echarts['png_width'] / current_echarts['png_height']
        if ratio == 1.0:
            st.code("Square (1:1)")
        else:
            st.code(f"{ratio:.2f}:1")

        # Publication settings summary
        st.markdown("**Plot Width:**")
        st.code(f"{current_pub['plot_width']}px")

        st.markdown("**Plot Height:**")
        st.code(f"{current_pub['plot_height']}px")

def render_publication_settings():
    """Render publication settings section"""
    st.subheader("📄 Publication Settings")

    # Create 2-column layout: Settings (left, wider) + Current Values (right, narrower)
    settings_col, current_col = st.columns([3, 1])

    with settings_col:
        st.subheader("Controls")

        # Use the shared publication settings widget
        publication_settings = PublicationSettingsWidget.render_publication_settings("global_settings")

        # Update global settings
        st.session_state.global_settings['publication'] = publication_settings

        # Save button
        if st.button("💾 Save Settings", type="primary", width='stretch', key="save_pub"):
            save_global_settings()

    with current_col:
        st.subheader("Current Values")

        # Current publication settings display
        current_pub = st.session_state.global_settings['publication']

        st.markdown("**Text Font Size:**")
        st.code(f"{current_pub['textfont_size']}px")

        st.markdown("**Point Size:**")
        st.code(f"{current_pub['point_size']}px")

        st.markdown("**Width:**")
        st.code(f"{current_pub['plot_width']}px")

        st.markdown("**Height:**")
        st.code(f"{current_pub['plot_height']}px")

        st.markdown("**Export DPI:**")
        st.code(str(current_pub['export_dpi']))

        st.markdown("**Export Format:**")
        st.code(current_pub['export_format'])

def render_geometric_analysis_settings():
    """Render geometric analysis settings section"""
    st.subheader("🔬 Geometric Analysis Settings")

    # Create 2-column layout: Settings (left, wider) + Current Values (right, narrower)
    settings_col, current_col = st.columns([3, 1])

    with settings_col:

        current_settings = st.session_state.global_settings['geometric_analysis']

        # Main enable/disable toggle (includes clustering)
        enabled = st.checkbox(
            "Enable Geometric Analysis (includes Clustering)",
            value=current_settings['enabled'],
            help="Enable geometric analysis with automatic clustering, plus optional branching and void analysis",
            key="global_geometric_enabled"
        )

        current_settings['enabled'] = enabled

        if enabled:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Additional Analysis Types")

                # Set clustering to always be enabled when geometric analysis is enabled
                current_settings['enable_clustering'] = True
                st.info("🔗 **Clustering Analysis**: Always enabled with geometric analysis")

                current_settings['enable_branching'] = st.checkbox(
                    "Branching Analysis",
                    value=current_settings['enable_branching'],
                    help="Analyze branching patterns and linearity",
                    key="global_branching_analysis"
                )

                current_settings['enable_void'] = st.checkbox(
                    "Void Analysis",
                    value=current_settings['enable_void'],
                    help="Detect empty regions in embedding space",
                    key="global_void_analysis"
                )

            with col2:
                st.subheader("Parameters")

                # Clustering parameters
                if current_settings['enable_clustering']:
                    current_settings['n_clusters'] = st.slider(
                        "Analysis Clusters",
                        min_value=2,
                        max_value=15,
                        value=current_settings['n_clusters'],
                        help="Number of clusters for geometric analysis",
                        key="global_analysis_clusters"
                    )

                    current_settings['density_radius'] = st.slider(
                        "Density Radius",
                        min_value=0.01,
                        max_value=1.0,
                        value=current_settings['density_radius'],
                        step=0.01,
                        format="%.2f",
                        help="Radius for density-based analysis",
                        key="global_density_radius"
                    )

                # Branching parameters
                if current_settings['enable_branching']:
                    current_settings['connectivity_threshold'] = st.slider(
                        "Connectivity Threshold",
                        min_value=0.1,
                        max_value=1.0,
                        value=current_settings['connectivity_threshold'],
                        step=0.05,
                        format="%.2f",
                        help="Threshold for connectivity analysis",
                        key="global_connectivity_threshold"
                    )

                # Void parameters
                if current_settings['enable_void']:
                    current_settings['void_confidence'] = st.slider(
                        "Void Confidence",
                        min_value=0.5,
                        max_value=0.99,
                        value=current_settings['void_confidence'],
                        step=0.01,
                        format="%.2f",
                        help="Confidence level for void detection",
                        key="global_void_confidence"
                    )

            # Save options
            current_settings['save_json_files'] = st.checkbox(
                "Save JSON Files",
                value=current_settings['save_json_files'],
                help="Save detailed analysis results as JSON files",
                key="global_save_json"
            )
        else:
            # When geometric analysis is disabled, disable clustering too
            current_settings['enable_clustering'] = False
            st.info("ℹ️ Geometric analysis disabled - clustering will not be available")

        # Save button
        if st.button("💾 Save Settings", type="primary", width='stretch', key="save_geo"):
            save_global_settings()

    with current_col:
        st.subheader("Current Values")

        # Current geometric analysis settings display
        current_geo = st.session_state.global_settings['geometric_analysis']

        st.markdown("**Status:**")
        st.code("Enabled" if current_geo['enabled'] else "Disabled")

        if current_geo['enabled']:
            st.markdown("**Analysis Types:**")
            analysis_types = []
            if current_geo['enable_clustering']:
                analysis_types.append("Clustering")
            if current_geo['enable_branching']:
                analysis_types.append("Branching")
            if current_geo['enable_void']:
                analysis_types.append("Void")

            if analysis_types:
                for analysis_type in analysis_types:
                    st.code(f"✓ {analysis_type}")
            else:
                st.code("None selected")

            if current_geo['enable_clustering']:
                st.markdown("**Clusters:**")
                st.code(str(current_geo['n_clusters']))

                st.markdown("**Density Radius:**")
                st.code(f"{current_geo['density_radius']:.2f}")

            if current_geo['enable_branching']:
                st.markdown("**Connectivity:**")
                st.code(f"{current_geo['connectivity_threshold']:.2f}")

            if current_geo['enable_void']:
                st.markdown("**Void Confidence:**")
                st.code(f"{current_geo['void_confidence']:.2f}")

            st.markdown("**Save JSON:**")
            st.code("Yes" if current_geo['save_json_files'] else "No")

def render_echarts_settings():
    """Render ECharts-specific settings section"""
    st.subheader("📊 ECharts Settings")

    # Create 2-column layout: Settings (left, wider) + Current Values (right, narrower)
    settings_col, current_col = st.columns([3, 1])

    with settings_col:
        st.subheader("Controls")

        current_settings = st.session_state.global_settings['echarts']

        col1, col2 = st.columns(2)

        with col1:
            # Auto-save settings
            current_settings['auto_save_enabled'] = st.checkbox(
                "Auto-save Visualizations",
                value=current_settings['auto_save_enabled'],
                help="Automatically save visualizations using the format selected in Publication tab",
                key="global_echarts_auto_save"
            )

        with col2:
            if current_settings['auto_save_enabled']:
                # Get current publication format
                pub_format = st.session_state.get('global_settings', {}).get('publication', {}).get('export_format', 'PNG')
                st.caption(f"✅ Auto-save enabled ({pub_format} format from Publication tab)")
            else:
                st.caption("ℹ️ Auto-save disabled")

        # Image dimensions
        col3, col4 = st.columns(2)

        with col3:
            current_settings['png_width'] = st.number_input(
                "Image Width (px)",
                min_value=600,
                max_value=2400,
                value=current_settings['png_width'],
                step=100,
                help="Width of auto-saved images (PNG/PDF)",
                key="global_png_width"
            )

        with col4:
            current_settings['png_height'] = st.number_input(
                "Image Height (px)",
                min_value=400,
                max_value=1600,
                value=current_settings['png_height'],
                step=100,
                help="Height of auto-saved images (PNG/PDF)",
                key="global_png_height"
            )

        # Preset buttons
        st.write("**Quick Presets:**")
        col5, col6, col7 = st.columns(3)

        with col5:
            if st.button("📱 Small (600x600)", key="preset_small"):
                st.session_state.global_settings['echarts']['png_width'] = 600
                st.session_state.global_settings['echarts']['png_height'] = 600
                st.rerun()

        with col6:
            if st.button("💻 Medium (800x800)", key="preset_medium"):
                st.session_state.global_settings['echarts']['png_width'] = 800
                st.session_state.global_settings['echarts']['png_height'] = 800
                st.rerun()

        with col7:
            if st.button("📄 Large (1200x1200)", key="preset_large"):
                st.session_state.global_settings['echarts']['png_width'] = 1200
                st.session_state.global_settings['echarts']['png_height'] = 1200
                st.rerun()

        # Save button
        if st.button("💾 Save Settings", type="primary", width='stretch', key="save_echarts"):
            save_global_settings()

    with current_col:
        st.subheader("Current Values")

        # Current ECharts settings display
        current_echarts = st.session_state.global_settings['echarts']

        st.markdown("**Auto-save:**")
        st.code("Enabled" if current_echarts['auto_save_enabled'] else "Disabled")

        st.markdown("**PNG Width:**")
        st.code(f"{current_echarts['png_width']}px")

        st.markdown("**PNG Height:**")
        st.code(f"{current_echarts['png_height']}px")

        st.markdown("**Aspect Ratio:**")
        ratio = current_echarts['png_width'] / current_echarts['png_height']
        if ratio == 1.0:
            st.code("Square (1:1)")
        else:
            st.code(f"{ratio:.2f}:1")

def render_language_settings():
    """Render language configuration settings section"""
    st.subheader("🌐 Language Settings")
    st.markdown("Configure default languages and language priorities for Semanscope")

    # Create 2-column layout: Settings (left, wider) + Current Values (right, narrower)
    settings_col, current_col = st.columns([3, 1])

    with settings_col:

        # Ensure languages key exists in global_settings
        if 'languages' not in st.session_state.global_settings:
            st.session_state.global_settings['languages'] = {
                'default_languages': DEFAULT_LANG_SET.copy(),
                'available_languages': get_all_language_codes(),
                'language_priority': DEFAULT_LANG_SET.copy()
            }

        current_settings = st.session_state.global_settings['languages']

        # Info about current config
        st.info("📋 **Language Configuration**: Set which languages are available by default and their priority order for Semanscope.")

        # Language selection section
        st.subheader("🎯 Default Language Set")
        
        # Get all available languages
        all_language_names = get_all_language_names()
        current_default_codes = current_settings['default_languages']
        current_default_names = [get_language_name_from_code(code) for code in current_default_codes]

        # Multi-select for default languages
        selected_language_names = st.multiselect(
            "Select Default Languages",
            options=all_language_names,
            default=current_default_names,
            help="Choose which languages are available by default in Semanscope",
            key="default_languages_select"
        )

        # Convert back to language codes
        selected_language_codes = [get_language_code_from_name(name) for name in selected_language_names]
        
        # Update settings if changed
        if selected_language_codes != current_default_codes:
            current_settings['default_languages'] = selected_language_codes
            current_settings['language_priority'] = selected_language_codes

        # Language priority section
        st.subheader("📊 Language Priority Order")
        st.markdown("Drag and drop to reorder language priority (first = highest priority)")

        if selected_language_names:
            # Create a simple reordering interface using selectbox
            st.markdown("**Reorder Languages by Priority:**")
            
            # Get current priority order
            priority_codes = current_settings.get('language_priority', selected_language_codes)
            priority_names = [get_language_name_from_code(code) for code in priority_codes if code in selected_language_codes]
            
            # Add any newly selected languages that aren't in priority list
            for name in selected_language_names:
                if name not in priority_names:
                    priority_names.append(name)
            
            # Remove any languages that are no longer selected
            priority_names = [name for name in priority_names if name in selected_language_names]
            
            # Simple reordering interface
            if len(priority_names) > 1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Current Priority Order:**")
                    for i, name in enumerate(priority_names):
                        code = get_language_code_from_name(name)
                        st.write(f"{i+1}. {name} ({code})")
                
                with col2:
                    st.markdown("**Move Language:**")
                    if len(priority_names) > 0:
                        # Select language to move
                        lang_to_move = st.selectbox(
                            "Language to move",
                            options=priority_names,
                            key="lang_to_move"
                        )
                        
                        # Select new position
                        new_position = st.selectbox(
                            "New position",
                            options=list(range(1, len(priority_names) + 1)),
                            key="new_position"
                        )
                        
                        # Move button
                        if st.button("🔄 Move Language", key="move_lang"):
                            # Remove language from current position
                            priority_names.remove(lang_to_move)
                            # Insert at new position (convert from 1-based to 0-based)
                            priority_names.insert(new_position - 1, lang_to_move)
                            
                            # Update settings
                            priority_codes = [get_language_code_from_name(name) for name in priority_names]
                            current_settings['language_priority'] = priority_codes
                            st.success(f"✅ Moved {lang_to_move} to position {new_position}")
                            st.rerun()
            else:
                st.info("Select multiple languages to enable priority ordering")
        else:
            st.warning("⚠️ Please select at least one default language")

        # Quick presets section
        st.subheader("⚡ Quick Presets")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🇨🇳 East Asian Focus", key="preset_east_asian"):
                preset_codes = ["chn", "jpn", "kor", "enu"]
                current_settings['default_languages'] = preset_codes
                current_settings['language_priority'] = preset_codes
                st.success("✅ Applied East Asian preset")
                st.rerun()
        
        with col2:
            if st.button("🇪🇺 European Focus", key="preset_european"):
                preset_codes = ["enu", "deu", "fra", "spa"]
                current_settings['default_languages'] = preset_codes
                current_settings['language_priority'] = preset_codes
                st.success("✅ Applied European preset")
                st.rerun()
        
        with col3:
            if st.button("🌍 Global Diverse", key="preset_global"):
                preset_codes = ["enu", "chn", "ara", "hin", "spa", "fra"]
                current_settings['default_languages'] = preset_codes
                current_settings['language_priority'] = preset_codes
                st.success("✅ Applied Global Diverse preset")
                st.rerun()

        # Reset section
        st.subheader("🔄 Reset Options")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("↩️ Reset to Config Default", key="reset_to_config"):
                current_settings['default_languages'] = DEFAULT_LANG_SET.copy()
                current_settings['language_priority'] = DEFAULT_LANG_SET.copy()
                st.success("✅ Reset to config.py defaults")
                st.rerun()
        
        with col2:
            if st.button("🌐 Enable All Languages", key="enable_all"):
                all_codes = get_all_language_codes()
                current_settings['default_languages'] = all_codes
                current_settings['language_priority'] = all_codes
                st.success("✅ Enabled all available languages")
                st.rerun()

        # Advanced options
        st.subheader("⚙️ Advanced Options")
        
        # Option to update config.py file directly
        update_config_file_option = st.checkbox(
            "📝 Also update config.py file",
            value=False,
            help="Update the DEFAULT_LANG_SET variable in config.py to make changes persistent across app restarts",
            key="update_config_file"
        )
        
        if update_config_file_option:
            st.warning("⚠️ **Caution**: This will modify the config.py file directly. Changes will persist even after app restart.")

        # Save button
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Language Settings", type="primary", width='stretch', key="save_languages"):
                save_global_settings()
        
        with col2:
            if update_config_file_option:
                if st.button("💾 Save + Update config.py", type="secondary", width='stretch', key="save_and_update_config"):
                    # Save to session state
                    save_global_settings()
                    
                    # Also update config.py
                    current_langs = current_settings['default_languages']
                    if update_config_file(current_langs):
                        st.success("✅ Successfully updated config.py file")
                    st.rerun()

    with current_col:
        st.subheader("Current Configuration")

        # Current language settings display
        current_lang_settings = st.session_state.global_settings['languages']

        st.markdown("**Default Languages:**")
        current_codes = current_lang_settings['default_languages']
        st.code(f"{len(current_codes)} languages")
        
        for i, code in enumerate(current_codes):
            name = get_language_name_from_code(code)
            st.write(f"{i+1}. {name} ({code})")

        st.markdown("**From config.py:**")
        config_codes = DEFAULT_LANG_SET
        st.code(f"{len(config_codes)} languages")
        
        # Show if current differs from config
        if set(current_codes) != set(config_codes):
            st.warning("⚠️ Different from config.py")
        else:
            st.success("✅ Matches config.py")

        st.markdown("**Total Available:**")
        total_available = len(get_all_language_codes())
        st.code(f"{total_available} languages")

        # Show language codes mapping preview
        st.markdown("**Code Mapping Preview:**")
        preview_codes = current_codes[:5]  # Show first 5
        for code in preview_codes:
            name = get_language_name_from_code(code)
            st.code(f"{code} → {name}")
        
        if len(current_codes) > 5:
            st.code(f"...and {len(current_codes) - 5} more")

def render_defaults_settings():
    """Render defaults settings section for overriding global variables"""
    st.subheader("⚙️ Global Default Values")
    st.markdown("Override the default global variables used across Semanscope. These settings will be used as defaults throughout the application.")

    # Get current defaults settings
    current_settings = st.session_state.global_settings['defaults']

    # Create 2-column layout: Settings (left, wider) + Current Values (right, narrower)
    settings_col_1, settings_col_2, current_col, config_col = st.columns([3, 3, 1, 1])

    with settings_col_1:

        # Method selection
        st.markdown("#### 🔬 Default Method")
        active_methods = get_active_methods()
        method_options = sorted(list(active_methods.keys()))

        current_method = current_settings['method']
        method_index = method_options.index(current_method) if current_method in method_options else 0

        new_method = st.selectbox(
            "Default Dimensionality Reduction Method",
            options=method_options,
            index=method_index,
            help="Select the default method for dimensionality reduction",
            key="global_default_method"
        )

        if new_method != current_method:
            current_settings['method'] = new_method

        # Show method info
        if new_method in active_methods:
            st.info(f"**{new_method}**: {active_methods[new_method]['help']}")

        # Dataset selection
        st.markdown("#### 📊 Default Dataset")
        dataset_options = get_available_datasets()

        current_dataset = current_settings['dataset']
        dataset_index = dataset_options.index(current_dataset) if current_dataset in dataset_options else 0

        new_dataset = st.selectbox(
            "Default Dataset",
            options=dataset_options,
            index=dataset_index,
            help="Select the default dataset to use when no specific dataset is chosen",
            key="global_default_dataset"
        )

        if new_dataset != current_dataset:
            current_settings['dataset'] = new_dataset

        # Show dataset info
        # st.info(f"📂 Found {len(dataset_options)} available datasets from `data/input/` directory")

        # Dimension selection
        st.markdown("#### 📐 Default Dimension")
        dimension_options = ["2D", "3D"]

        current_dimension = current_settings['dimension']
        dimension_index = dimension_options.index(current_dimension) if current_dimension in dimension_options else 0

        new_dimension = st.radio(
            "Default Visualization Dimension",
            options=dimension_options,
            index=dimension_index,
            help="Select the default visualization dimension (2D or 3D)",
            horizontal=True,
            key="global_default_dimension"
        )

        if new_dimension != current_dimension:
            current_settings['dimension'] = new_dimension


    with settings_col_2:


        # Model selection
        st.markdown("#### 🤖 Default Model")
        # Get available models from config
        active_models = get_active_models()
        model_options = list(active_models.keys())

        current_model = current_settings['model']
        model_index = model_options.index(current_model) if current_model in model_options else 0

        new_model = st.selectbox(
            "Default Embedding Model",
            options=model_options,
            index=model_index,
            help="Select the default embedding model to use",
            key="global_default_model"
        )

        if new_model != current_model:
            current_settings['model'] = new_model

        # Show model info
        if new_model in MODEL_INFO:
            st.info(f"**{new_model}**: {MODEL_INFO[new_model]['help']}")





        # Split Lines checkbox
        st.markdown("#### 🔤 Text Processing")
        current_split_lines = current_settings.get('split_lines', DEFAULT_SPLIT_LINES)

        new_split_lines = st.checkbox(
            "Split line in dataset",
            value=current_split_lines,
            help="Enable preprocessing to split multi-word lines into individual words. Disable for sentence visualization.",
            key="global_default_split_lines"
        )

        if new_split_lines != current_split_lines:
            current_settings['split_lines'] = new_split_lines

        if new_split_lines:
            st.info("✅ Lines with multiple words will be split and deduplicated")
        else:
            st.info("📝 Lines will be treated as single units (e.g., for sentences)")


    # Save buttons
    col_save_session, col_save_config, col_reset, _ = st.columns([1, 1, 1, 1])

    with col_save_session:
        if st.button("💾 Save to Session", type="primary", width='stretch', key="save_defaults_session"):
            # This saves to st.session_state (current behavior)
            save_global_settings()
            st.success("✅ Saved to session state")

    with col_save_config:
        if st.button("📝 Save to Config.py", type="secondary", width='stretch', key="save_defaults_config"):
            # Save to both config.py file AND session state
            success = save_defaults_to_config_file(current_settings)
            if success:
                save_global_settings()  # Also save to session state
                st.success("✅ Saved to both config.py and session state!")
            else:
                st.error("❌ Failed to save to config.py, but saved to session state")
                save_global_settings()  # Still save to session as fallback

    with col_reset:
        if st.button("↩️ Reset to config.py Defaults", key="reset_defaults_to_config"):
            current_settings['dataset'] = DEFAULT_DATASET
            current_settings['method'] = DEFAULT_METHOD
            current_settings['model'] = DEFAULT_MODEL
            current_settings['dimension'] = DEFAULT_DIMENSION
            current_settings['split_lines'] = DEFAULT_SPLIT_LINES
            st.success("✅ Reset to config.py defaults")
            st.rerun()


    with current_col:
        st.markdown("#### Current")

        # Current defaults display
        current_defaults = st.session_state.global_settings['defaults']

        st.markdown("**Dataset:**")
        st.code(current_defaults['dataset'])

        st.markdown("**Method:**")
        st.code(current_defaults['method'])

        st.markdown("**Model:**")
        st.code(current_defaults['model'])

        st.markdown("**Dimension:**")
        st.code(current_defaults['dimension'])


    with config_col:
        # Show config.py defaults for comparison
        st.markdown("#### config.py")

        st.markdown("**Dataset:**")
        st.code(DEFAULT_DATASET)

        st.markdown("**Method:**")
        st.code(DEFAULT_METHOD)

        st.markdown("**Model:**")
        st.code(DEFAULT_MODEL)

        st.markdown("**Dimension:**")
        st.code(DEFAULT_DIMENSION)

        # Show if current differs from config
        config_defaults = {
            'dataset': DEFAULT_DATASET,
            'method': DEFAULT_METHOD,
            'model': DEFAULT_MODEL,
            'dimension': DEFAULT_DIMENSION
        }

        # if current_defaults != config_defaults:
        #     st.warning("⚠️ Different from config.py")
        # else:
        #     st.success("✅ Matches config.py")

def main():
    check_login()

    st.subheader("🔧 Global Settings - shared across all Semanscope pages")

    # Initialize settings
    initialize_global_settings()

    # Create tabs for different setting categories
    defaults_tab, charting_tab, color_coding_tab, geometric_analysis_tab, languages_tab, cache_tab = st.tabs([
        "⚙️ Defaults",
        "📊 Charting",
        "🎨 Color Coding",
        "🔬 Geometric Analysis",
        "🌐 Languages",
        "💾 Cache",
    ])

    with defaults_tab:
        render_defaults_settings()

    with languages_tab:
        render_language_settings()

    with charting_tab:
        render_charting_settings()

    with color_coding_tab:
        render_color_coding_settings()

    with geometric_analysis_tab:
        render_geometric_analysis_settings()

    with cache_tab:
        render_cache_settings()

def render_color_coding_settings():
    """Render color coding settings section for semantic domains"""
    st.subheader("🎨 Color Coding Settings")
    st.markdown("Configure colors for semantic domains used in visualizations. This supports ACL paper Figure 2 requirements for morphological network signatures.")

    # Create 2-column layout: Settings (left, wider) + Current Values (right, narrower)
    settings_col, current_col = st.columns([3, 1])

    with settings_col:
        st.subheader("Domain Color Configuration")

        current_settings = st.session_state.global_settings['color_coding']

        # Color scheme info
        st.info("🔬 **ACL Paper Support**: Color coding now stored in `config.py` for persistence across sessions. Case-insensitive domain matching!")
        st.info("💡 **Direct Editing**: For custom domains, edit `CUSTOM_SEMANTIC_DOMAINS` in `src/config.py`")

        # Predefined domains section
        st.subheader("📋 All Available Domains from Config")

        # Import all available domains from config
        from semanscope.config import SEMANTIC_DOMAIN_COLORS, CUSTOM_SEMANTIC_DOMAINS, get_all_domain_colors

        # Get all domains from config.py and organize by category for better UX
        all_config_domains = set(SEMANTIC_DOMAIN_COLORS.keys())

        domain_categories = {
            "Language Identification": [
                d for d in ['chinese', 'english', 'french', 'spanish', 'german', 'arabic', 'hebrew', 'hindi', 'japanese', 'korean', 'russian', 'thai', 'persian', 'turkish', 'georgian', 'armenian', 'vietnamese'] if d in all_config_domains
            ],
            "Latin Script Variants": [
                d for d in ['latin_uppercase', 'latin_lowercase', 'french_uppercase', 'french_lowercase', 'spanish_uppercase', 'spanish_lowercase', 'german_uppercase', 'german_lowercase', 'turkish_uppercase', 'turkish_lowercase', 'turkish_special', 'vietnamese_letters'] if d in all_config_domains
            ],
            "Arabic Script Family": [
                d for d in ['arabic_letters', 'arabic_trilateral_roots', 'persian_letters'] if d in all_config_domains
            ],
            "Asian Scripts": [
                d for d in ['elemental_chars', 'radicals_1d', 'radicals_2d', 'korean_letters', 'korean_consonants', 'korean_vowels', 'hiragana_letters', 'katakana_letters', 'thai_letters', 'hindi_letters'] if d in all_config_domains
            ],
            "European Scripts": [
                d for d in ['hebrew_letters', 'hebrew_roots', 'greek_uppercase', 'greek_lowercase', 'russian_letters', 'georgian_letters', 'armenian_uppercase', 'armenian_lowercase'] if d in all_config_domains
            ],
            "Core Semantic Categories": [
                d for d in ['people', 'objects', 'places', 'actions', 'attributes', 'concepts'] if d in all_config_domains
            ],
            "Function Words (Red Family)": [
                d for d in ['articles_determiners', 'prepositions', 'conjunctions', 'pronouns'] if d in all_config_domains
            ],
            "Abstract Sequential": [
                d for d in ['numbers', 'colors', 'mathematics', 'physics', 'time_temporal'] if d in all_config_domains
            ],
            "Content Words (Blue Family)": [
                d for d in ['family_kinship', 'body_parts', 'animals', 'food', 'actions_verbs', 'emotions', 'nature_elements', 'spatial_directional', 'abstract_qualities'] if d in all_config_domains
            ],
            "Morphological Families": [
                d for d in ['morphological_work', 'morphological_light', 'morphological_book', 'morphological_子'] if d in all_config_domains
            ],
            "Special Characters & Symbols": [
                d for d in ['numeric_letters', 'emoji_letters', 'punctuation', 'symbols'] if d in all_config_domains
            ],
            "Fallback Colors": [
                d for d in ['unknown', 'default'] if d in all_config_domains
            ]
        }

        # Remove empty categories
        domain_categories = {k: v for k, v in domain_categories.items() if v}

        # Create expandable sections for each category
        for category, domains in domain_categories.items():
            with st.expander(f"🎯 {category}", expanded=(category == "Language Identification")):
                # Use 4 columns for better space utilization
                cols = st.columns(4)
                for i, domain in enumerate(domains):
                    with cols[i % 4]:
                        # Get color from config.py directly since these might not be in session state
                        config_color = SEMANTIC_DOMAIN_COLORS.get(domain, '#666666')

                        # Use config color as default, session state takes precedence if exists
                        current_color = current_settings['domain_colors'].get(domain, config_color)

                        # Color picker for domain
                        new_color = st.color_picker(
                            f"{domain.replace('_', ' ').title()}",
                            value=current_color,
                            key=f"color_{domain}"
                        )

                        # Update session state if color changed
                        if new_color != current_color:
                            if 'domain_colors' not in current_settings:
                                current_settings['domain_colors'] = {}
                            current_settings['domain_colors'][domain] = new_color

        # Custom domains section
        st.subheader("➕ Custom Domains")
        st.markdown("Add your own semantic domains for specialized analysis (e.g., 子-network sub-domains)")

        # Add new custom domain
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            new_domain_name = st.text_input(
                "New Domain Name",
                placeholder="e.g., morphological_子_family, morphological_子_nature",
                help="Enter a unique domain name (use underscores instead of spaces)",
                key="new_domain_name"
            )
        with col2:
            new_domain_color = st.color_picker(
                "Color",
                value="#800080",
                key="new_domain_color"
            )
        with col3:
            if st.button("➕ Add Domain", type="secondary"):
                if new_domain_name and new_domain_name not in current_settings['domain_colors'] and new_domain_name not in current_settings['custom_domains']:
                    # Add to session state and save immediately
                    st.session_state.global_settings['color_coding']['custom_domains'][new_domain_name] = new_domain_color
                    st.success(f"✅ Added domain: {new_domain_name}")
                    save_global_settings()
                elif new_domain_name in current_settings['domain_colors'] or new_domain_name in current_settings['custom_domains']:
                    st.error("❌ Domain name already exists")
                else:
                    st.error("❌ Please enter a domain name")

        # Display and edit custom domains
        if current_settings['custom_domains']:
            st.markdown("**Current Custom Domains:**")
            custom_domains_to_remove = []

            for domain_name, domain_color in current_settings['custom_domains'].items():
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"🎯 {domain_name.replace('_', ' ').title()}")
                with col2:
                    # Color picker for custom domain
                    new_color = st.color_picker(
                        "Color",
                        value=domain_color,
                        key=f"custom_color_{domain_name}",
                        label_visibility="collapsed"
                    )
                    # Update session state immediately when color changes
                    if new_color != domain_color:
                        st.session_state.global_settings['color_coding']['custom_domains'][domain_name] = new_color
                with col3:
                    if st.button("🗑️", key=f"delete_{domain_name}", help="Delete this custom domain"):
                        custom_domains_to_remove.append(domain_name)

            # Remove domains marked for deletion
            for domain_name in custom_domains_to_remove:
                del st.session_state.global_settings['color_coding']['custom_domains'][domain_name]
                st.success(f"🗑️ Removed domain: {domain_name}")
                save_global_settings()
        else:
            st.info("ℹ️ No custom domains defined. Add domains above for specialized analysis.")

        # Color scheme management
        st.subheader("💾 Color Scheme Management")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Reset to Defaults", type="secondary"):
                # Reset to default colors
                st.session_state.global_settings['color_coding']['custom_domains'] = {}
                st.session_state.global_settings['color_coding']['color_scheme_name'] = 'default'
                st.success("🔄 Reset to default color scheme")
                st.rerun()

        with col2:
            current_settings['color_scheme_name'] = st.text_input(
                "Scheme Name",
                value=current_settings['color_scheme_name'],
                help="Name for this color configuration",
                key="color_scheme_name"
            )

        # Save button
        if st.button("💾 Save Color Settings", type="primary", width='stretch', key="save_colors"):
            save_global_settings()

    with current_col:
        st.subheader("Current Scheme")

        # Current color coding settings display
        current_colors = st.session_state.global_settings['color_coding']

        st.markdown("**Scheme Name:**")
        st.code(current_colors['color_scheme_name'])

        # Import config-based counts
        from semanscope.config import SEMANTIC_DOMAIN_COLORS, CUSTOM_SEMANTIC_DOMAINS, get_all_domain_colors

        st.markdown("**Config.py Domains:**")
        st.code(f"{len(SEMANTIC_DOMAIN_COLORS)} domains")

        st.markdown("**Custom Domains:**")
        st.code(f"{len(CUSTOM_SEMANTIC_DOMAINS)} domains")

        # Show category breakdown
        st.markdown("**Categories:**")
        for category, domains in domain_categories.items():
            if domains:  # Only show non-empty categories
                st.code(f"{category}: {len(domains)}")

        # Show total domains from config
        total_domains = len(get_all_domain_colors())
        st.markdown("**Total Available:**")
        st.code(f"{total_domains} domains")

        # Color preview section
        st.markdown("**Color Preview (from config.py):**")

        # Import and show colors from config
        from semanscope.config import get_domain_color, get_all_domain_colors

        # Show key ACL paper domains
        preview_domains = ['mathematics', 'physics', 'numbers', 'morphological_子', 'morphological_work', 'chinese', 'english']

        for domain in preview_domains:
            color = get_domain_color(domain)
            st.markdown(f'<div style="background-color: {color}; padding: 8px; margin: 2px; border-radius: 4px; color: white; text-align: center; font-size: 12px;">{domain.replace("_", " ").title()}</div>', unsafe_allow_html=True)

        # Show custom domain colors
        if current_colors['custom_domains']:
            st.markdown("**Custom Colors:**")
            for domain, color in list(current_colors['custom_domains'].items()):
                st.markdown(f'<div style="background-color: {color}; padding: 8px; margin: 2px; border-radius: 4px; color: white; text-align: center; font-size: 12px;">{domain.replace("_", " ").title()}</div>', unsafe_allow_html=True)


def render_cache_settings():
    """Render cache settings section for Semanscope calculations"""
    st.subheader("💾 Semanscope Cache Settings")
    st.markdown("🚀 **Performance Cache**: Optimize Semanscope performance by caching expensive embedding and dimensionality reduction calculations across pages.")

    # Cache TTL Configuration
    st.markdown("### ⏰ Cache Time-To-Live (TTL)")

    # Get current cache TTL setting
    current_ttl = st.session_state.global_settings.get('cache_ttl_hours', DEFAULT_TTL)

    col1, col2 = st.columns([1, 2])

    with col1:
        ttl_options = {
            0: "Disabled (No caching)",
            1: "1 hour",
            2: "2 hours",
            4: "4 hours",
            8: "8 hours (Default)",
            16: "16 hours",
            32: "32 hours",
            64: "64 hours",
            128: "128 hours"
        }

        new_ttl = st.selectbox(
            "Cache Duration",
            options=list(ttl_options.keys()),
            index=list(ttl_options.keys()).index(DEFAULT_TTL),
            format_func=lambda x: ttl_options[x],
            help="How long to keep cached results before recalculation",
            key="cache_ttl_setting"
        )

        if new_ttl != current_ttl:
            st.session_state.global_settings['cache_ttl_hours'] = new_ttl
            st.success(f"✅ Cache TTL updated to: {ttl_options[new_ttl]}")
            st.rerun()

    with col2:
        st.info(f"**Current Setting**: {ttl_options[current_ttl]}")

        if current_ttl == 0:
            st.warning("⚠️ Caching disabled - calculations will be repeated for each visualization")
        else:
            st.success(f"🚀 Cache enabled - results saved for {current_ttl} hour{'s' if current_ttl > 1 else ''}")

    st.markdown("---")

    # Cache Statistics and Management
    st.markdown("### 📊 Cache Statistics")

    try:
        cache_stats = get_cache_stats()

        # Display cache statistics in columns
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

        with stat_col1:
            st.metric("Total Files", cache_stats['total_files'])

        with stat_col2:
            st.metric("Valid Files", cache_stats['valid_files'])

        with stat_col3:
            st.metric("Expired Files", cache_stats['expired_files'])

        with stat_col4:
            st.metric("Total Size", f"{cache_stats['total_size_mb']} MB")

        # Prominent Clear Cache button
        st.markdown("---")
        clear_col1, clear_col2, clear_col3 = st.columns([1, 2, 1])
        with clear_col2:
            if st.button("🧹 Clear Cache", help="Clear all Semanscope cache files", type="primary", width='stretch'):
                removed = clear_all_cache()
                if removed > 0:
                    st.success(f"✅ Successfully cleared {removed} cache files")
                    st.rerun()
                else:
                    st.info("ℹ️ No cache files to clear")

        # Cache management buttons
        st.markdown("### 🛠️ Cache Management")

        mgmt_col1, mgmt_col2, mgmt_col3 = st.columns(3)

        with mgmt_col1:
            if st.button("🧹 Cleanup Expired", help="Remove expired cache files only"):
                removed = cleanup_cache()
                if removed > 0:
                    st.success(f"✅ Cleaned up {removed} expired cache files")
                    st.rerun()
                else:
                    st.info("ℹ️ No expired cache files to remove")

        with mgmt_col2:
            if st.button("🗑️ Clear All Cache", help="Remove ALL cache files", type="secondary"):
                removed = clear_all_cache()
                if removed > 0:
                    st.success(f"✅ Cleared {removed} cache files")
                    st.rerun()
                else:
                    st.info("ℹ️ No cache files to clear")

        with mgmt_col3:
            if st.button("🔄 Refresh Stats", help="Update cache statistics"):
                st.rerun()

        # Cache explanation
        st.markdown("---")
        st.markdown("### 🔍 How Semanscope Caching Works")

        with st.expander("🎯 Cache Key Format"):
            st.code("Format: <dataset_hash>-<lang>-<model>-<method>-<params_hash>")
            st.markdown("""
            **Components:**
            - **Dataset Hash**: MD5 hash of input texts (first 8 chars)
            - **Language**: Language code (e.g., 'en', 'zh', 'en-zh')
            - **Model**: Embedding model name
            - **Method**: Dimensionality reduction method (e.g., 'phate', 'tsne')
            - **Params Hash**: MD5 hash of method parameters (first 8 chars)
            """)

        with st.expander("🚀 Performance Benefits"):
            st.markdown("""
            **Cached Operations:**
            - ✅ **Embedding Generation**: Expensive model inference (seconds to minutes)
            - ✅ **PHATE Calculations**: Manifold learning with pairwise distances (O(n²) complexity)
            - ✅ **t-SNE/UMAP**: Iterative dimensionality reduction algorithms
            - ✅ **Cross-Page Sharing**: 2D → 3D transitions without recalculation

            **Cache Hit Scenarios:**
            - 🔄 Switching between 2D and 3D visualizations
            - 🔄 Changing visualization parameters (colors, point sizes)
            - 🔄 Moving between Semanscope pages with same dataset-model-method
            """)

        with st.expander("⚙️ Cache Architecture"):
            st.markdown("""
            **Technical Details:**
            - **Storage**: Local file system (`cache/` directory)
            - **Format**: Python pickle (.pkl) files for fast serialization
            - **TTL**: File modification time + configured hours
            - **Cleanup**: Automatic on TTL expiry + manual management
            - **Thread Safety**: File-based locking for concurrent access
            """)

    except Exception as e:
        st.error(f"❌ Cache system error: {str(e)}")
        st.info("💡 Try refreshing the page or check cache directory permissions")


if __name__ == "__main__":
    main()