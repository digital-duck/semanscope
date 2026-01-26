"""
Global Settings Manager for Semanscope Pages

This module provides utilities to access and manage global settings
that are shared across all Semanscope pages.
"""

import streamlit as st
from config import DEFAULT_MODEL, DEFAULT_METHOD, DEFAULT_N_CLUSTERS, DEFAULT_DATASET, DEFAULT_DIMENSION, DEFAULT_CLUSTERING, DEFAULT_SPLIT_LINES, get_all_domain_colors, get_domain_color


class GlobalSettingsManager:
    """Manager class for global settings across Semanscope pages"""

    @staticmethod
    def get_settings():
        """Get global settings, initializing with defaults if not present"""
        if 'global_settings' not in st.session_state:
            GlobalSettingsManager.initialize_defaults()
        else:
            # Ensure new fields exist in existing session state
            if 'defaults' not in st.session_state.global_settings:
                st.session_state.global_settings['defaults'] = {
                    'dataset': DEFAULT_DATASET,
                    'method': DEFAULT_METHOD,
                    'model': DEFAULT_MODEL,
                    'dimension': DEFAULT_DIMENSION,
                    'split_lines': DEFAULT_SPLIT_LINES
                }
            elif 'split_lines' not in st.session_state.global_settings['defaults']:
                st.session_state.global_settings['defaults']['split_lines'] = DEFAULT_SPLIT_LINES
        return st.session_state.global_settings

    @staticmethod
    def initialize_defaults():
        """Initialize global settings with default values"""
        st.session_state.global_settings = {
            # Visualization Settings
            'model_name': DEFAULT_MODEL,
            'method_name': DEFAULT_METHOD,
            'dimensions': '2D',
            'do_clustering': False,
            'n_clusters': DEFAULT_N_CLUSTERS,
            'debug_character_encoding': False,

            # Publication Settings
            'publication': {
                'publication_mode': False,
                'textfont_size': 16,
                'point_size': 12,
                'plot_width': 1400,
                'plot_height': 1100,
                'export_dpi': 300,
                'export_format': 'PDF'
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

            # Default Global Variables Override
            'defaults': {
                'dataset': DEFAULT_DATASET,
                'method': DEFAULT_METHOD,
                'model': DEFAULT_MODEL,
                'dimension': DEFAULT_DIMENSION,
                'split_lines': DEFAULT_SPLIT_LINES
            }
        }

    @staticmethod
    def get_visualization_settings():
        """Get visualization settings (model, method, dimensions, clustering)"""
        settings = GlobalSettingsManager.get_settings()
        # Get defaults override if available, otherwise use legacy settings
        defaults = settings.get('defaults', {})
        geometric_analysis = settings.get('geometric_analysis', {})

        # Clustering is now controlled by geometric analysis
        clustering_enabled = geometric_analysis.get('enabled', False) and geometric_analysis.get('enable_clustering', False)

        return {
            'model_name': defaults.get('model', settings.get('model_name', DEFAULT_MODEL)),
            'method_name': defaults.get('method', settings.get('method_name', DEFAULT_METHOD)),
            'dimensions': defaults.get('dimension', settings.get('dimensions', '2D')),
            'do_clustering': clustering_enabled,
            'n_clusters': geometric_analysis.get('n_clusters', settings.get('n_clusters', DEFAULT_N_CLUSTERS))
        }

    @staticmethod
    def get_publication_settings():
        """Get publication settings (fonts, DPI, format)"""
        settings = GlobalSettingsManager.get_settings()
        return settings['publication']

    @staticmethod
    def get_geometric_analysis_settings():
        """Get geometric analysis settings"""
        settings = GlobalSettingsManager.get_settings()
        return settings['geometric_analysis']

    @staticmethod
    def get_echarts_settings():
        """Get ECharts-specific settings"""
        settings = GlobalSettingsManager.get_settings()
        return settings['echarts']

    @staticmethod
    def get_color_coding_settings():
        """Get color coding settings for semantic domains"""
        settings = GlobalSettingsManager.get_settings()
        return settings['color_coding']

    @staticmethod
    def get_domain_color(domain_name):
        """Get color for a specific semantic domain"""
        color_settings = GlobalSettingsManager.get_color_coding_settings()

        # Check custom domains first
        if domain_name in color_settings['custom_domains']:
            return color_settings['custom_domains'][domain_name]

        # Then check predefined domains
        if domain_name in color_settings['domain_colors']:
            return color_settings['domain_colors'][domain_name]

        # Return default color if not found
        return color_settings['domain_colors'].get('default', '#666666')

    @staticmethod
    def get_all_domain_colors():
        """Get all domain colors (predefined + custom) as a single dictionary"""
        color_settings = GlobalSettingsManager.get_color_coding_settings()
        all_colors = {}

        # Add predefined domains
        all_colors.update(color_settings['domain_colors'])

        # Add custom domains (these override predefined if there are name conflicts)
        all_colors.update(color_settings['custom_domains'])

        return all_colors

    @staticmethod
    def get_defaults():
        """Get default global variables (with overrides from settings)"""
        settings = GlobalSettingsManager.get_settings()
        return settings['defaults']

    @staticmethod
    def get_default_dataset():
        """Get the default dataset (with override from settings)"""
        defaults = GlobalSettingsManager.get_defaults()
        return defaults['dataset']

    @staticmethod
    def get_default_method():
        """Get the default method (with override from settings)"""
        defaults = GlobalSettingsManager.get_defaults()
        return defaults['method']

    @staticmethod
    def get_default_model():
        """Get the default model (with override from settings)"""
        defaults = GlobalSettingsManager.get_defaults()
        return defaults['model']

    @staticmethod
    def get_default_dimension():
        """Get the default dimension (with override from settings)"""
        defaults = GlobalSettingsManager.get_defaults()
        return defaults['dimension']

    @staticmethod
    def get_default_clustering():
        """Get the default clustering setting (now controlled by geometric analysis)"""
        settings = GlobalSettingsManager.get_settings()
        geometric_analysis = settings.get('geometric_analysis', {})
        return geometric_analysis.get('enabled', False) and geometric_analysis.get('enable_clustering', False)

    @staticmethod
    def get_default_split_lines():
        """Get the default split lines setting (with override from settings)"""
        defaults = GlobalSettingsManager.get_defaults()
        return defaults.get('split_lines', DEFAULT_SPLIT_LINES)

    @staticmethod
    def update_setting(category, key, value):
        """Update a specific setting"""
        settings = GlobalSettingsManager.get_settings()
        if category in settings:
            settings[category][key] = value
        else:
            settings[key] = value

    @staticmethod
    def is_geometric_analysis_enabled():
        """Check if geometric analysis is enabled"""
        settings = GlobalSettingsManager.get_geometric_analysis_settings()
        return settings.get('enabled', True)

    @staticmethod
    def is_debug_character_encoding_enabled():
        """Check if debug character encoding is enabled"""
        settings = GlobalSettingsManager.get_settings()
        return settings.get('debug_character_encoding', False)

    @staticmethod
    def get_geometric_analysis_params():
        """Get geometric analysis parameters in the format expected by GeometricAnalyzer"""
        settings = GlobalSettingsManager.get_geometric_analysis_settings()
        if not settings.get('enabled', True):
            return None

        return {
            'enable_clustering': settings.get('enable_clustering', True),
            'enable_branching': settings.get('enable_branching', False),
            'enable_void': settings.get('enable_void', False),
            'n_clusters': settings.get('n_clusters', 5),
            'density_radius': settings.get('density_radius', 0.1),
            'connectivity_threshold': settings.get('connectivity_threshold', 0.8),
            'void_confidence': settings.get('void_confidence', 0.95),
            'save_json_files': settings.get('save_json_files', False)
        }

    @staticmethod
    def render_settings_link():
        """Render a link to the Settings page in sidebar"""
        st.sidebar.markdown("### ⚙️ Configuration")
        st.sidebar.info("💡 **Tip**: Configure global settings in the **Settings** page to apply across all Semanscope pages.")

    @staticmethod
    def render_current_settings_summary():
        """Render current settings summary at the end of sidebar"""
        # st.sidebar.markdown("---")
        # Get current settings summary
        viz_settings = GlobalSettingsManager.get_visualization_settings()
        pub_settings = GlobalSettingsManager.get_publication_settings()
        echarts_settings = GlobalSettingsManager.get_echarts_settings()
        color_settings = GlobalSettingsManager.get_color_coding_settings()

        with st.sidebar.expander("📋 Current Global Settings", expanded=False):
            # Visualization Settings
            st.markdown("**🎨 Visualization:**")
            st.write(f"• Model: {viz_settings['model_name']}")
            st.write(f"• Method: {viz_settings['method_name']}")
            st.write(f"• Dimensions: {viz_settings['dimensions']}")
            st.write(f"• Clustering: {'Enabled (via Geometric Analysis)' if viz_settings['do_clustering'] else 'Disabled'}")

            # Publication Settings
            st.markdown("**📄 Publication:**")
            st.write(f"• Text Size: {pub_settings['textfont_size']}px")
            st.write(f"• Point Size: {pub_settings['point_size']}px")
            st.write(f"• Plot Height: {pub_settings['plot_height']}px")
            st.write(f"• Export DPI: {pub_settings['export_dpi']}")

            # Geometric Analysis
            geo_enabled = GlobalSettingsManager.is_geometric_analysis_enabled()
            st.markdown("**🔬 Geometric Analysis:**")
            st.write(f"• Status: {'Enabled' if geo_enabled else 'Disabled'}")

            # ECharts Settings
            st.markdown("**📊 ECharts:**")
            st.write(f"• Auto-save: {'Enabled' if echarts_settings['auto_save_enabled'] else 'Disabled'}")
            st.write(f"• PNG Size: {echarts_settings['png_width']}x{echarts_settings['png_height']}")

            # Color Coding Settings
            st.markdown("**🎨 Color Coding:**")
            total_domains = len(color_settings['domain_colors']) + len(color_settings['custom_domains'])
            st.write(f"• Scheme: {color_settings['color_scheme_name']}")
            st.write(f"• Total Domains: {total_domains}")
            st.write(f"• Custom Domains: {len(color_settings['custom_domains'])}")


# Convenience functions for easy access
def get_global_viz_settings():
    """Convenience function to get visualization settings"""
    return GlobalSettingsManager.get_visualization_settings()

def get_global_publication_settings():
    """Convenience function to get publication settings"""
    return GlobalSettingsManager.get_publication_settings()

def get_global_geometric_analysis():
    """Convenience function to get geometric analysis settings"""
    return GlobalSettingsManager.get_geometric_analysis_params()

def is_global_geometric_analysis_enabled():
    """Convenience function to check if geometric analysis is enabled"""
    return GlobalSettingsManager.is_geometric_analysis_enabled()

def get_global_color_coding_settings():
    """Convenience function to get color coding settings"""
    return GlobalSettingsManager.get_color_coding_settings()

def get_global_domain_color(domain_name):
    """Convenience function to get color for a specific domain"""
    return GlobalSettingsManager.get_domain_color(domain_name)

def get_global_all_domain_colors():
    """Convenience function to get all domain colors"""
    return GlobalSettingsManager.get_all_domain_colors()

def get_global_defaults():
    """Convenience function to get default global variables"""
    return GlobalSettingsManager.get_defaults()

def get_global_default_dataset():
    """Convenience function to get the default dataset"""
    return GlobalSettingsManager.get_default_dataset()

def get_global_default_method():
    """Convenience function to get the default method"""
    return GlobalSettingsManager.get_default_method()

def get_global_default_model():
    """Convenience function to get the default model"""
    return GlobalSettingsManager.get_default_model()

def get_global_default_dimension():
    """Convenience function to get the default dimension"""
    return GlobalSettingsManager.get_default_dimension()

def get_global_default_clustering():
    """Convenience function to get the default clustering setting"""
    return GlobalSettingsManager.get_default_clustering()

def get_global_default_split_lines():
    """Convenience function to get the default split lines setting"""
    try:
        return GlobalSettingsManager.get_default_split_lines()
    except Exception:
        # Fallback to config default if there's any issue
        return DEFAULT_SPLIT_LINES