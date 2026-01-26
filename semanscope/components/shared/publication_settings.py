"""
Shared publication settings widget for eliminating code duplication
"""
import streamlit as st
from typing import Dict, Any
from config import (
    get_publication_settings,
    get_publication_constraints,
    get_export_formats,
    PUBLICATION_SETTINGS
)


class PublicationSettingsWidget:
    """Centralized widget for publication settings across pages"""

    @staticmethod
    def render_publication_settings(session_key_prefix: str = "publication") -> Dict[str, Any]:
        """
        Render publication settings UI and return the settings dictionary

        Args:
            session_key_prefix: Prefix for session state keys to avoid conflicts

        Returns:
            Dictionary containing all publication settings
        """
        publication_mode = st.checkbox(
            "Publication Mode",
            value=True,
            help="Enable high-quality settings for publication",
            key=f"{session_key_prefix}_publication_mode"
        )

        # Get defaults and constraints from config
        pub_defaults = get_publication_settings('publication')
        std_defaults = get_publication_settings('standard')
        text_constraints = get_publication_constraints('textfont_size')
        point_constraints = get_publication_constraints('point_size')

        col1, col2 = st.columns(2)

        # Left column - Text and Point sizes
        with col1:
            if publication_mode:
                textfont_size = st.number_input(
                    "Text Size",
                    min_value=text_constraints.get('min', 6),
                    max_value=text_constraints.get('max', 16),
                    value=pub_defaults['textfont_size'],
                    step=text_constraints.get('step', 1),
                    help="Font size for labels (6-16pt). Suggested: 12pt for publication, 14pt for presentations",
                    key=f"{session_key_prefix}_textfont_size_pub"
                )
                point_size = st.number_input(
                    "Point Size",
                    min_value=point_constraints.get('min', 4),
                    max_value=point_constraints.get('max', 12),
                    value=pub_defaults['point_size'],
                    step=point_constraints.get('step', 1),
                    help="Size of data points (4-12pt). Suggested: 8pt for publication, 10pt for presentations",
                    key=f"{session_key_prefix}_point_size_pub"
                )
            else:
                textfont_size = st.number_input(
                    "Text Size",
                    min_value=text_constraints.get('min', 6),
                    max_value=text_constraints.get('max', 16),
                    value=std_defaults['textfont_size'],
                    step=text_constraints.get('step', 1),
                    help="Font size for labels and annotations",
                    key=f"{session_key_prefix}_textfont_size_std"
                )
                point_size = st.number_input(
                    "Point Size",
                    min_value=point_constraints.get('min', 4),
                    max_value=point_constraints.get('max', 12),
                    value=std_defaults['point_size'],
                    step=point_constraints.get('step', 1),
                    help="Size of data points in the plot",
                    key=f"{session_key_prefix}_point_size_std"
                )

        # Right column - Plot dimensions
        width_constraints = get_publication_constraints('plot_width')
        height_constraints = get_publication_constraints('plot_height')

        with col2:
            if publication_mode:
                plot_width = st.number_input(
                    "Width",
                    min_value=width_constraints.get('min', 800),
                    max_value=width_constraints.get('max', 1600),
                    value=pub_defaults['plot_width'],
                    step=width_constraints.get('step', 50),
                    help="Plot width in pixels (800-1600px). Higher values for publication quality",
                    key=f"{session_key_prefix}_plot_width_pub"
                )
                plot_height = st.number_input(
                    "Height",
                    min_value=height_constraints.get('min', 600),
                    max_value=height_constraints.get('max', 1600),
                    value=pub_defaults['plot_height'],
                    step=height_constraints.get('step', 50),
                    help="Plot height in pixels (600-1600px). Optimized 4:3 aspect ratio for publications",
                    key=f"{session_key_prefix}_plot_height_pub"
                )
            else:
                plot_width = st.number_input(
                    "Width",
                    min_value=width_constraints.get('min', 800),
                    max_value=width_constraints.get('max', 1600),
                    value=std_defaults['plot_width'],
                    step=width_constraints.get('step', 50),
                    help="Plot width in pixels (800-1600px). Standard default width",
                    key=f"{session_key_prefix}_plot_width_std"
                )
                plot_height = st.number_input(
                    "Height",
                    min_value=height_constraints.get('min', 600),
                    max_value=height_constraints.get('max', 1600),
                    value=std_defaults['plot_height'],
                    step=height_constraints.get('step', 50),
                    help="Plot height in pixels (600-1600px). Updated default for better aspect ratio",
                    key=f"{session_key_prefix}_plot_height_std"
                )

        # Export options (only in publication mode)
        if publication_mode:
            st.markdown("**Export Options**")
            col_exp1, col_exp2 = st.columns(2)
            dpi_constraints = get_publication_constraints('export_dpi')
            export_formats = get_export_formats()

            with col_exp1:
                export_format = st.selectbox(
                    "Format",
                    export_formats,
                    index=export_formats.index(pub_defaults['export_format']) if pub_defaults['export_format'] in export_formats else 0,
                    help="Export format: PNG (raster, good for most uses), SVG (vector, scalable), PDF (vector, publication-ready)",
                    key=f"{session_key_prefix}_export_format"
                )

            with col_exp2:
                export_dpi = st.number_input(
                    "DPI",
                    min_value=dpi_constraints.get('min', 150),
                    max_value=dpi_constraints.get('max', 600),
                    value=pub_defaults['export_dpi'],
                    step=dpi_constraints.get('step', 50),
                    help="Dots per inch (150-600). Suggested: 300 DPI for journals, 150-200 for web, 600 for high-quality prints",
                    key=f"{session_key_prefix}_export_dpi"
                )
        else:
            export_format = std_defaults['export_format']
            export_dpi = std_defaults['export_dpi']

        # Return settings dictionary
        return {
            'publication_mode': publication_mode,
            'textfont_size': textfont_size,
            'point_size': point_size,
            'plot_width': plot_width,
            'plot_height': plot_height,
            'export_format': export_format,
            'export_dpi': export_dpi
        }

    @staticmethod
    def get_default_settings() -> Dict[str, Any]:
        """Get default settings for use when publication settings are not available"""
        return get_publication_settings('fallback')