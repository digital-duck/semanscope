import streamlit as st
import pandas as pd
import numpy as np
from streamlit_echarts import st_echarts
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from semanscope.config import (
    PLOT_WIDTH, PLOT_HEIGHT,
    DEFAULT_N_CLUSTERS, DEFAULT_MIN_CLUSTERS, DEFAULT_MAX_CLUSTERS,
    DEFAULT_MAX_WORDS,
    DATA_PATH
)
import json
import shutil
import os
import base64
from datetime import datetime
import time
import tempfile
import html
from semanscope.utils.title_filename_helper import create_title_and_filename, create_chart_title
from pathlib import PosixPath, Path

# Optional imports for automatic PNG export
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.by import By
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


class EChartsPlotManager:
    """Apache ECharts plotting manager for semantic visualizations"""

    def __init__(self):
        self.min_clusters = DEFAULT_MIN_CLUSTERS
        self.max_clusters = DEFAULT_MAX_CLUSTERS
        self.last_chart_config = None  # Store last chart configuration for auto-save
        # ECharts styling configuration with proper language color differentiation
        self.echarts_theme = {
            'background_color': '#ffffff',
            'text_color': '#000000',
            'grid_color': '#e0e0e0',
            'chinese_color': '#dc143c',  # Crimson for Chinese
            'english_color': '#0000ff',  # Blue for English
            'french_color': '#008000',   # Green for French
            'spanish_color': '#ff8c00',  # Orange for Spanish
            'german_color': '#ff0000',   # Red for German
            'arabic_color': '#8b4513',   # Brown for Arabic
            'cluster_colors': [
                '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4',
                '#feca57', '#ff9ff3', '#54a0ff', '#5f27cd'
            ]
        }

    def get_visualization_settings(self):
        """Get visualization settings from global settings"""
        from utils.global_settings import get_global_publication_settings

        default_settings = {
            'plot_width': PLOT_WIDTH,
            'plot_height': PLOT_HEIGHT,
            'text_size': 12,
            'point_size': 8,
            'show_grid': True,
            'animation_enabled': True
        }

        # Get global publication settings for text and point sizes
        try:
            pub_settings = get_global_publication_settings()
            default_settings.update({
                'text_size': pub_settings.get('textfont_size', 12),
                'point_size': pub_settings.get('point_size', 8),
                'plot_height': pub_settings.get('plot_height', PLOT_HEIGHT)
            })
        except:
            pass  # Fallback to defaults if global settings not available

        return st.session_state.get('echarts_settings', default_settings)

    def get_language_color(self, color_name):
        """Map colors - prioritize semantic hex colors, fallback to language colors"""
        # If it's already a hex color (semantic domain color), return as-is
        if isinstance(color_name, str) and color_name.startswith('#'):
            return color_name

        # Simple source/target color scheme for language-based colors
        source_color = '#ff0000'  # Red for source language
        target_color = '#0000ff'  # Blue for target language

        # Map based on the original COLOR_MAP logic:
        # - 'red' typically represents source language (Chinese in original design)
        # - All other colors represent target languages
        if color_name == 'red':
            return source_color  # Red for source
        else:
            return target_color  # Blue for target (English, German, French, Spanish, Arabic, etc.)

    def create_title(self, method_name, model_name, dataset_name="", lang_codes=None):
        """Create standardized plot title using centralized helper"""
        # Convert single values to lists for helper function
        method_names = [method_name] if method_name else []
        model_names = [model_name] if model_name else []

        # Ensure lang_codes is a list
        if lang_codes:
            if isinstance(lang_codes, str):
                lang_codes = [lang_codes]
            elif not isinstance(lang_codes, list):
                lang_codes = list(lang_codes)
        else:
            lang_codes = []

        # Use the centralized helper (only need the title part)
        title = create_chart_title(method_names, model_names, dataset_name or "", lang_codes)
        return title

    def get_highlight_settings(self):
        """Get word search settings from UI"""
        with st.sidebar.expander("🔍 Word Search", expanded=False):
            st.markdown("### Search and highlight specific words in the visualization")

            # Enable/disable highlighting
            enable_highlight = st.checkbox("Enable Word Search", value=False,
                                          help="Search and highlight specific words with custom colors and sizes")

            if not enable_highlight:
                return None

            # Word search input
            keywords_input = st.text_area(
                "Search Words (one per line)",
                placeholder="空\n佛\n觀自在菩薩\n揭諦揭諦波羅揭諦波羅僧揭諦菩提薩婆訶",
                height=150,
                help="Enter words to search and highlight, one per line"
            )

            if not keywords_input.strip():
                st.warning("Please enter at least one word to search")
                return None

            keywords = [kw.strip() for kw in keywords_input.strip().split('\n') if kw.strip()]

            # Color and size settings
            col1, col2 = st.columns(2)
            with col1:
                highlight_color = st.color_picker("Search Result Color", "#FFD700",
                                                  help="Color for highlighted search results")
            with col2:
                highlight_size = st.slider("Search Result Size", 10, 50, 20,
                                          help="Point size for highlighted search results")

            # Font size for highlighted labels
            highlight_font_size = st.slider("Label Font Size", 10, 30, 16,
                                           help="Font size for highlighted search result labels")

            # Multiple keyword groups option
            use_multiple_colors = st.checkbox("Use different colors for each search word", value=False)

            if use_multiple_colors:
                st.markdown("**Predefined color palette for search words:**")
                color_palette = [
                    "#FFD700",  # Gold
                    "#FF6B6B",  # Red
                    "#4ECDC4",  # Teal
                    "#45B7D1",  # Blue
                    "#96CEB4",  # Green
                    "#FECA57",  # Yellow
                    "#FF9FF3",  # Pink
                    "#54A0FF",  # Light Blue
                ]
                st.caption(f"First {len(color_palette)} keywords will use different colors automatically")

            return {
                'keywords': keywords,
                'color': highlight_color,
                'size': highlight_size,
                'font_size': highlight_font_size,
                'use_multiple_colors': use_multiple_colors,
                'color_palette': color_palette if use_multiple_colors else None
            }

    def apply_highlighting(self, data_points, labels, settings, highlight_config):
        """Apply highlighting to data points based on searched words"""
        if not highlight_config:
            return data_points

        keywords = highlight_config['keywords']
        use_multiple_colors = highlight_config.get('use_multiple_colors', False)
        color_palette = highlight_config.get('color_palette', [])

        def normalize_text(text):
            """Normalize text for matching by removing common punctuation"""
            import re
            # Remove common punctuation marks (English and Chinese)
            return re.sub(r'[,，。.、？?！!；;：:""''「」『』（）\(\)\[\]【】]', '', text.strip())

        for i, point in enumerate(data_points):
            label = labels[i]
            normalized_label = normalize_text(label)

            # Check if label matches the keyword (exact or after normalization)
            for idx, keyword in enumerate(keywords):
                normalized_keyword = normalize_text(keyword)

                # Try exact match first, then normalized match
                if label == keyword or normalized_label == normalized_keyword:
                    # Determine color
                    if use_multiple_colors and idx < len(color_palette):
                        color = color_palette[idx]
                    else:
                        color = highlight_config['color']

                    # Update point style
                    point['itemStyle'] = {'color': color}

                    # For 3D charts, symbolSize can be set at point level
                    # For 2D charts it's typically at series level, but point-level works too
                    point['symbolSize'] = highlight_config['size']

                    # Update label style
                    point['label'] = {
                        'show': True,
                        'fontSize': highlight_config['font_size'],
                        'color': color,
                        'fontWeight': 'bold',
                        'formatter': '{b}'  # Show name
                    }
                    break  # Only highlight once per point

        return data_points

    def plot_2d(self, embeddings, labels, colors, title, clustering=False, n_clusters=DEFAULT_N_CLUSTERS,
                max_words=DEFAULT_MAX_WORDS, method_name="", model_name="", dataset_name="", highlight_config=None, display_chart=True, lang_codes=None, chart_key=None, sa_metrics_text=None):
        """Create 2D scatter plot using ECharts"""
        settings = self.get_visualization_settings()

        # Create standardized title
        if method_name and model_name:
            title = self.create_title(method_name, model_name, dataset_name, lang_codes)

        # Generate the plot
        if clustering:
            result = self._plot_2d_cluster_echarts(embeddings, labels, colors, title, n_clusters, settings, method_name, model_name, dataset_name, highlight_config, display_chart, chart_key, sa_metrics_text)
        else:
            result = self._plot_2d_simple_echarts(embeddings, labels, colors, title, settings, method_name, model_name, dataset_name, highlight_config, display_chart, chart_key, sa_metrics_text)

        # Store the ECharts configuration in session state for auto-save
        if result:
            import streamlit as st
            st.session_state.current_echarts_config = result

        # Universal auto-save for all 2D visualizations (using working ECharts approach)
        if display_chart and method_name and model_name and dataset_name:
            try:
                import streamlit as st
                from utils.title_filename_helper import create_title_and_filename

                # Use the same working auto-save approach as ECharts page
                auto_save_status = self.get_auto_save_status()

                # Check if auto-save is enabled in global settings (same as ECharts page)
                global_settings = st.session_state.get('global_settings', {})
                echarts_settings = global_settings.get('echarts', {})
                auto_save_enabled = echarts_settings.get('auto_save_enabled', False)

                # Also check legacy location for backward compatibility
                legacy_auto_save = st.session_state.get('echarts_auto_save', {})
                if not auto_save_enabled and legacy_auto_save.get('enabled', False):
                    auto_save_enabled = True


                # Only proceed if auto-save is enabled and selenium is available
                if (auto_save_enabled and
                    auto_save_status.get('available', False) and
                    'current_echarts_config' in st.session_state):

                    # Determine active languages from lang_codes or fallback
                    active_languages = lang_codes if lang_codes else ['auto']

                    # Use method name as-is without clustering suffix (consistent with original ECharts page)
                    plot_method = method_name

                    # Generate standardized filename using centralized helper (same as ECharts page)
                    _, standardized_filename = create_title_and_filename(
                        [plot_method],
                        [model_name],
                        dataset_name,
                        active_languages,
                        "png"  # Default to PNG
                    )

                    # Determine format and directory based on publication settings (same logic as regular plotting)
                    pub_settings = st.session_state.get('global_settings', {}).get('publication', {})
                    publication_mode = pub_settings.get('publication_mode', False)
                    export_format = pub_settings.get('export_format', 'PNG').upper() if publication_mode else 'PNG'

                    # Create filename and determine output directory (same as working regular plotting)
                    standardized_filename_str = str(standardized_filename)
                    echarts_filename = f"echarts-{standardized_filename_str}"

                    # Use same directory logic as working regular plotting system
                    from pathlib import Path
                    if export_format == 'PDF' and publication_mode:
                        output_dir = Path("../data/images/PDF")
                        echarts_filename = echarts_filename.replace('.png', '.pdf')
                    else:
                        output_dir = Path("../data/images/echarts")
                        echarts_filename = echarts_filename.replace('.pdf', '.png')

                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Save the file and move to correct directory if needed
                    with st.spinner(f"📸 Auto-saving {export_format}..."):
                        auto_save_settings = st.session_state.get('echarts_auto_save', {'width': 1200, 'height': 800})

                        # First save to default location
                        temp_filename = echarts_filename.replace('.pdf', '.png')  # Always save as PNG first
                        saved_png = self.save_echarts_as_png_auto(
                            st.session_state.current_echarts_config,
                            [],  # Empty filename_parts since we're using external_filename
                            "2D",
                            width=auto_save_settings.get('width', 1200),
                            height=auto_save_settings.get('height', 800),
                            external_filename=temp_filename
                        )

                        # If saved successfully and we need to move to PDF directory
                        if saved_png and export_format == 'PDF' and publication_mode:
                            try:

                                # Get the source file path (from default echarts directory)
                                if isinstance(saved_png, dict):
                                    source_path = saved_png.get('filepath')
                                else:
                                    # Construct source path
                                    default_echarts_dir = Path("../data/images/echarts")
                                    source_path = default_echarts_dir / str(saved_png)

                                if source_path and os.path.exists(source_path):
                                    # Move to PDF directory with PNG extension (since it's still a PNG file)
                                    target_path = output_dir / echarts_filename.replace('.pdf', '.png')
                                    shutil.move(str(source_path), str(target_path))

                                    # Update the saved_png result to reflect new location
                                    saved_png = {
                                        'filename': echarts_filename.replace('.pdf', '.png'),
                                        'filepath': str(target_path.absolute())
                                    }
                                    st.info("💡 **Note**: ECharts saved as PNG in PDF directory. Use ECharts download menu (📥) to export as actual PDF.")
                            except Exception as move_error:
                                st.warning(f"File saved but couldn't move to PDF directory: {move_error}")
                        elif export_format == 'PDF':
                            st.info("💡 **Note**: ECharts saved as PNG. Use ECharts download menu (📥) to export as actual PDF.")

                    saved_files = []

                    # Handle success message and create PDF version
                    if saved_png:
                        png_path = None
                        if isinstance(saved_png, dict):
                            # Use the full filepath from the saved result
                            png_path = saved_png.get('filepath', saved_png.get('filename', ''))
                            saved_files.append(f"PNG: {png_path}")
                            # Store the filepath for later display
                            st.session_state['last_echarts_png_path'] = saved_png['filepath']
                        else:
                            # Try to construct full filepath for older format
                            echarts_dir = os.path.join("..", "data", "images", "echarts")
                            png_path = os.path.abspath(os.path.join(echarts_dir, str(saved_png)))
                            saved_files.append(f"PNG: {png_path}")
                            st.session_state['last_echarts_png_path'] = png_path

                        # Also create PDF version from the PNG
                        if png_path and os.path.exists(png_path):
                            try:
                                from PIL import Image

                                # Create PDF directory
                                pdf_dir = Path("../data/images/PDF")
                                pdf_dir.mkdir(parents=True, exist_ok=True)

                                # Generate PDF filename (replace .png with .pdf)
                                png_filename = os.path.basename(png_path)
                                pdf_filename = png_filename.replace('.png', '.pdf')
                                pdf_path = pdf_dir / pdf_filename

                                # Convert PNG to PDF
                                img = Image.open(png_path)
                                # Convert RGBA to RGB if necessary (PDF doesn't support transparency)
                                if img.mode == 'RGBA':
                                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                                    rgb_img.paste(img, mask=img.split()[3])  # Use alpha channel as mask
                                    img = rgb_img

                                img.save(str(pdf_path), 'PDF', resolution=100.0)

                                full_pdf_path = os.path.abspath(pdf_path)
                                saved_files.append(f"PDF: {full_pdf_path}")

                            except ImportError:
                                st.warning("⚠️ Pillow not installed. Install with: pip install Pillow")
                            except Exception as pdf_error:
                                st.warning(f"⚠️ Could not create PDF: {pdf_error}")

                        # Success message with full absolute paths for both formats
                        st.success(f"📊 **Auto-saved in both formats**:\n" + "\n".join([f"• {f}" for f in saved_files]))
                    else:
                        st.warning("❌ **Universal auto-save failed** - save method returned None")
                else:
                    # Show why auto-save didn't run
                    if not auto_save_enabled:
                        st.info("🔧 **Auto-save Debug**: Auto-save disabled in settings")
                    elif not auto_save_status.get('available', False):
                        st.warning("⚠️ **Auto-save Debug**: Selenium WebDriver not found")
                    elif 'current_echarts_config' not in st.session_state:
                        st.info("🔧 **Auto-save Debug**: No ECharts config available for auto-save")

            except Exception as auto_save_error:
                st.error(f"❌ **Universal auto-save error**: {str(auto_save_error)}")
                import traceback
                st.code(traceback.format_exc())

        return result

    def plot_3d(self, embeddings, labels, colors, title, clustering=False, n_clusters=DEFAULT_N_CLUSTERS,
                method_name="", model_name="", dataset_name="", highlight_config=None, display_chart=True, lang_codes=None, chart_key=None):
        """Create 3D scatter plot using ECharts"""
        settings = self.get_visualization_settings()

        # Create standardized title
        if method_name and model_name:
            title = self.create_title(method_name, model_name, dataset_name, lang_codes)

        if clustering:
            result = self._plot_3d_cluster_echarts(embeddings, labels, colors, title, n_clusters, settings, method_name, model_name, dataset_name, highlight_config, display_chart, chart_key)
        else:
            result = self._plot_3d_simple_echarts(embeddings, labels, colors, title, settings, method_name, model_name, dataset_name, highlight_config, display_chart, chart_key)

        # Store the ECharts configuration in session state for consistency
        if result:
            import streamlit as st
            st.session_state.current_echarts_config = result

        return result

    def _perform_clustering(self, embeddings: np.ndarray, n_clusters: int) -> dict:
        """Perform clustering and calculate quality metrics"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)

        metrics = {
            "silhouette": round(silhouette_score(embeddings, clusters), 3),
            "calinski": round(calinski_harabasz_score(embeddings, clusters), 3),
            "inertia": round(kmeans.inertia_, 3),
            "cluster_centers": kmeans.cluster_centers_,
            "cluster_labels": clusters
        }

        return metrics, kmeans

    def _display_cluster_metrics(self, metrics: dict):
        """Display clustering quality metrics"""
        cols = st.columns(3)

        with cols[0]:
            st.metric(
                "Silhouette Score",
                metrics["silhouette"],
                help="Measures how similar an object is to its own cluster compared to other clusters. Range: [-1, 1], higher is better."
            )

        with cols[1]:
            st.metric(
                "Calinski-Harabasz Score",
                metrics["calinski"],
                help="Ratio of between-cluster variance to within-cluster variance. Higher is better."
            )

        with cols[2]:
            st.metric(
                "Inertia",
                metrics["inertia"],
                help="Sum of squared distances to nearest cluster center. Lower is better."
            )

    def _plot_2d_simple_echarts(self, embeddings, labels, colors, title, settings, method_name="", model_name="", dataset_name="", highlight_config=None, display_chart=True, chart_key=None, sa_metrics_text=None):
        """Create simple 2D scatter plot with ECharts"""
        # Prepare data for ECharts
        data_points = []
        for i, (x, y) in enumerate(embeddings):
            color = self.get_language_color(colors[i])
            data_points.append({
                'value': [float(x), float(y)],
                'name': labels[i],
                'itemStyle': {'color': color}
            })

        # Apply highlighting if enabled
        data_points = self.apply_highlighting(data_points, labels, settings, highlight_config)

        option = {
            'title': {
                'text': title,
                'left': 'center',
                'top': '1%',
                'textStyle': {
                    'fontSize': 16,
                    'color': self.echarts_theme['text_color']
                }
            },
            'tooltip': {
                'trigger': 'item',
                'formatter': '{b}: ({c})'
            },
            'grid': {
                'left': '8%',
                'right': '8%',
                'bottom': '12%',
                'top': '8%',
                'containLabel': True
            },
            'xAxis': {
                'type': 'value',
                'name': 'X',
                'nameLocation': 'middle',
                'nameGap': 30,
                'splitLine': {
                    'show': settings['show_grid'],
                    'lineStyle': {'color': self.echarts_theme['grid_color']}
                }
            },
            'yAxis': {
                'type': 'value',
                'name': 'Y',
                'nameLocation': 'middle',
                'nameGap': 40,
                'splitLine': {
                    'show': settings['show_grid'],
                    'lineStyle': {'color': self.echarts_theme['grid_color']}
                }
            },
            'series': [{
                'type': 'scatter',
                'data': data_points,
                'symbolSize': settings['point_size'],
                'label': {
                    'show': True,
                    'position': 'top',
                    'fontSize': settings['text_size'],
                    'color': self.echarts_theme['text_color'],
                    'formatter': '{b}'
                },
                'emphasis': {
                    'itemStyle': {
                        'shadowBlur': 10,
                        'shadowColor': 'rgba(0, 0, 0, 0.5)'
                    }
                }
            }],
            'animation': settings.get('animation_enabled', True),
            'backgroundColor': self.echarts_theme['background_color']
        }

        # Add SA metrics legend if provided
        if sa_metrics_text:
            option = self._add_sa_metrics_graphic(option, sa_metrics_text)

        # Enhance chart with export capabilities
        filename_parts = [method_name, model_name, dataset_name] if method_name and model_name else ["simple", "2d"]
        option = self.enhance_chart_with_export(option, filename_parts, "2D")

        # Store chart configuration for auto-save functionality
        self.last_chart_config = option
        
        # Render ECharts with responsive sizing to prevent cutoff (only if display_chart is True)
        if display_chart:
            chart_height = f"{settings.get('plot_height', 800)}px"
            st_echarts(
                options=option,
                height=chart_height,
                key=chart_key or "echarts_2d_simple"
            )

        return option

    def _plot_2d_cluster_echarts(self, embeddings, labels, colors, title, n_clusters, settings, method_name="", model_name="", dataset_name="", highlight_config=None, display_chart=True, chart_key=None, sa_metrics_text=None):
        """Create 2D scatter plot with clustering using ECharts"""
        # Add clustering controls
        boundary_opacity = st.slider(
            "Cluster Boundary Opacity",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Adjust the visibility of cluster boundaries."
        )

        # Perform clustering
        metrics, kmeans = self._perform_clustering(embeddings, n_clusters)
        self._display_cluster_metrics(metrics)

        # Prepare data for ECharts - group by cluster
        series_data = []
        for cluster_id in range(n_clusters):
            cluster_mask = metrics["cluster_labels"] == cluster_id
            cluster_points = []

            for i, (x, y) in enumerate(embeddings):
                if metrics["cluster_labels"][i] == cluster_id:
                    cluster_points.append({
                        'value': [float(x), float(y)],
                        'name': labels[i]
                    })

            if cluster_points:  # Only add series if it has points
                series_data.append({
                    'name': f'Cluster {cluster_id}',
                    'type': 'scatter',
                    'data': cluster_points,
                    'symbolSize': settings['point_size'],
                    'itemStyle': {
                        'color': self.echarts_theme['cluster_colors'][cluster_id % len(self.echarts_theme['cluster_colors'])]
                    },
                    'label': {
                        'show': True,
                        'position': 'top',
                        'fontSize': settings['text_size'],
                        'color': self.echarts_theme['text_color'],
                        'formatter': '{b}'
                    },
                    'emphasis': {
                        'itemStyle': {
                            'shadowBlur': 10,
                            'shadowColor': 'rgba(0, 0, 0, 0.5)'
                        }
                    }
                })

        option = {
            'title': {
                'text': title,
                'left': 'center',
                'top': '1%',
                'textStyle': {
                    'fontSize': 16,
                    'color': self.echarts_theme['text_color']
                }
            },
            'tooltip': {
                'trigger': 'item',
                'formatter': '{a}: {b} ({c})'
            },
            'legend': {
                'data': [f'Cluster {i}' for i in range(n_clusters)],
                'top': 'bottom',
                'left': 'center'
            },
            'grid': {
                'left': '8%',
                'right': '8%',
                'bottom': '15%',
                'top': '8%',
                'containLabel': True
            },
            'xAxis': {
                'type': 'value',
                'name': 'X',
                'nameLocation': 'middle',
                'nameGap': 30,
                'splitLine': {
                    'show': settings['show_grid'],
                    'lineStyle': {'color': self.echarts_theme['grid_color']}
                }
            },
            'yAxis': {
                'type': 'value',
                'name': 'Y',
                'nameLocation': 'middle',
                'nameGap': 40,
                'splitLine': {
                    'show': settings['show_grid'],
                    'lineStyle': {'color': self.echarts_theme['grid_color']}
                }
            },
            'series': series_data,
            'animation': settings.get('animation_enabled', True),
            'backgroundColor': self.echarts_theme['background_color']
        }

        # Add SA metrics legend if provided
        if sa_metrics_text:
            option = self._add_sa_metrics_graphic(option, sa_metrics_text)

        # Enhance chart with export capabilities
        filename_parts = [method_name, model_name, dataset_name, "cluster"] if method_name and model_name else ["cluster", "2d"]
        option = self.enhance_chart_with_export(option, filename_parts, "2D")

        # Store chart configuration for auto-save functionality
        self.last_chart_config = option
        
        # Render ECharts with responsive sizing to prevent cutoff (only if display_chart is True)
        if display_chart:
            chart_height = f"{settings.get('plot_height', 800)}px"
            st_echarts(
                options=option,
                height=chart_height,
                key="echarts_2d_cluster"
            )

        return option

    def _plot_3d_simple_echarts(self, embeddings, labels, colors, title, settings, method_name="", model_name="", dataset_name="", highlight_config=None, display_chart=True, chart_key=None):
        """Create simple 3D scatter plot with ECharts"""
        # Prepare data for ECharts 3D
        data_points = []
        for i, (x, y, z) in enumerate(embeddings):
            color = self.get_language_color(colors[i])
            data_points.append({
                'value': [float(x), float(y), float(z)],
                'name': labels[i],
                'itemStyle': {'color': color}
            })

        # Apply highlighting if enabled
        data_points = self.apply_highlighting(data_points, labels, settings, highlight_config)

        option = {
            'title': {
                'text': title,
                'left': 'center',
                'textStyle': {
                    'fontSize': 16,
                    'color': self.echarts_theme['text_color']
                }
            },
            'tooltip': {
                'trigger': 'item',
                'formatter': '{b}: ({c})'
            },
            'grid3D': {
                'boxWidth': 200,
                'boxHeight': 200,
                'boxDepth': 200,
                'viewControl': {
                    'projection': 'perspective',
                    'autoRotate': False,
                    'distance': 300
                }
            },
            'xAxis3D': {
                'type': 'value',
                'name': 'X',
                'scale': True
            },
            'yAxis3D': {
                'type': 'value',
                'name': 'Y',
                'scale': True
            },
            'zAxis3D': {
                'type': 'value',
                'name': 'Z',
                'scale': True
            },
            'series': [{
                'type': 'scatter3D',
                'data': data_points,
                'symbolSize': settings['point_size'],
                'label': {
                    'show': True,
                    'fontSize': settings['text_size'],
                    'color': self.echarts_theme['text_color'],
                    'formatter': '{b}'
                },
                'emphasis': {
                    'itemStyle': {
                        'shadowBlur': 10,
                        'shadowColor': 'rgba(0, 0, 0, 0.5)'
                    }
                }
            }],
            'animation': settings.get('animation_enabled', True),
            'backgroundColor': self.echarts_theme['background_color']
        }

        # Enhance chart with export capabilities
        filename_parts = [method_name, model_name, dataset_name] if method_name and model_name else ["simple", "3d"]
        option = self.enhance_chart_with_export(option, filename_parts, "3D")

        # Render ECharts with responsive sizing to prevent cutoff (only if display_chart is True)
        if display_chart:
            chart_height = f"{settings.get('plot_height', 800)}px"
            st_echarts(
                options=option,
                height=chart_height,
                key=chart_key or "echarts_3d_simple"
            )

        return option

    def _plot_3d_cluster_echarts(self, embeddings, labels, colors, title, n_clusters, settings, method_name="", model_name="", dataset_name="", highlight_config=None, display_chart=True, chart_key=None):
        """Create 3D scatter plot with clustering using ECharts"""
        # Perform clustering
        metrics, kmeans = self._perform_clustering(embeddings, n_clusters)
        self._display_cluster_metrics(metrics)

        # Prepare data for ECharts 3D - group by cluster
        series_data = []
        for cluster_id in range(n_clusters):
            cluster_points = []

            for i, (x, y, z) in enumerate(embeddings):
                if metrics["cluster_labels"][i] == cluster_id:
                    cluster_points.append({
                        'value': [float(x), float(y), float(z)],
                        'name': labels[i]
                    })

            if cluster_points:  # Only add series if it has points
                series_data.append({
                    'name': f'Cluster {cluster_id}',
                    'type': 'scatter3D',
                    'data': cluster_points,
                    'symbolSize': settings['point_size'],
                    'itemStyle': {
                        'color': self.echarts_theme['cluster_colors'][cluster_id % len(self.echarts_theme['cluster_colors'])]
                    },
                    'label': {
                        'show': True,
                        'fontSize': settings['text_size'],
                        'color': self.echarts_theme['text_color'],
                        'formatter': '{b}'
                    },
                    'emphasis': {
                        'itemStyle': {
                            'shadowBlur': 10,
                            'shadowColor': 'rgba(0, 0, 0, 0.5)'
                        }
                    }
                })

        option = {
            'title': {
                'text': title,
                'left': 'center',
                'textStyle': {
                    'fontSize': 16,
                    'color': self.echarts_theme['text_color']
                }
            },
            'tooltip': {
                'trigger': 'item',
                'formatter': '{a}: {b} ({c})'
            },
            'legend': {
                'data': [f'Cluster {i}' for i in range(n_clusters)],
                'top': 'bottom',
                'left': 'center'
            },
            'grid3D': {
                'boxWidth': 200,
                'boxHeight': 200,
                'boxDepth': 200,
                'viewControl': {
                    'projection': 'perspective',
                    'autoRotate': False,
                    'distance': 300
                }
            },
            'xAxis3D': {
                'type': 'value',
                'name': 'X',
                'scale': True
            },
            'yAxis3D': {
                'type': 'value',
                'name': 'Y',
                'scale': True
            },
            'zAxis3D': {
                'type': 'value',
                'name': 'Z',
                'scale': True
            },
            'series': series_data,
            'animation': settings.get('animation_enabled', True),
            'backgroundColor': self.echarts_theme['background_color']
        }

        # Enhance chart with export capabilities
        filename_parts = [method_name, model_name, dataset_name, "cluster"] if method_name and model_name else ["cluster", "3d"]
        option = self.enhance_chart_with_export(option, filename_parts, "3D")

        # Render ECharts with responsive sizing to prevent cutoff (only if display_chart is True)
        if display_chart:
            chart_height = f"{settings.get('plot_height', 800)}px"
            st_echarts(
                options=option,
                height=chart_height,
                key="echarts_3d_cluster"
            )

        return option


    def render_settings_controls(self):
        """Render ECharts-specific settings controls in sidebar"""
        with st.sidebar.expander("📊 ECharts Settings", expanded=False):
            settings = self.get_visualization_settings()

            # Visual settings
            settings['text_size'] = st.slider(
                "Text Size",
                min_value=8,
                max_value=20,
                value=settings.get('text_size', 12),
                help="Size of text labels"
            )

            settings['point_size'] = st.slider(
                "Point Size",
                min_value=4,
                max_value=20,
                value=settings.get('point_size', 8),
                help="Size of data points"
            )

            settings['show_grid'] = st.checkbox(
                "Show Grid",
                value=settings.get('show_grid', True),
                help="Display grid lines"
            )

            settings['animation_enabled'] = st.checkbox(
                "Enable Animations",
                value=settings.get('animation_enabled', True),
                help="Enable smooth animations and transitions"
            )

            # Chart dimensions
            col1, col2 = st.columns(2)
            with col1:
                settings['plot_width'] = st.number_input(
                    "Width",
                    min_value=400,
                    max_value=1200,
                    value=settings.get('plot_width', PLOT_WIDTH),
                    step=50
                )

            with col2:
                settings['plot_height'] = st.number_input(
                    "Height",
                    min_value=400,
                    max_value=1200,
                    value=settings.get('plot_height', PLOT_HEIGHT),
                    step=50
                )

            # Save settings to session state
            st.session_state.echarts_settings = settings

            return settings

    def save_echarts_as_pdf(self, chart_config, filename_parts, dimensions="2D", external_filename=None):
        """Save ECharts configuration as PDF for publication quality"""
        try:
            # Use external filename if provided, otherwise use internal logic
            if external_filename:
                filename = str(external_filename).replace('.json', '.pdf').replace('.png', '.pdf')
            else:
                # Create filename without timestamp (keep only latest)
                filename_base = "-".join([
                    str(part).replace(" ", "-").replace("_", "-")
                    for part in filename_parts if part
                ])

                # Add echarts prefix and dimension suffix
                if dimensions == "3D":
                    filename = f"echarts-{filename_base}-3d.pdf"
                else:
                    filename = f"echarts-{filename_base}.pdf"

            # Create echarts images directory if it doesn't exist
            echarts_dir = DATA_PATH / "images" / "echarts"
            echarts_dir.mkdir(parents=True, exist_ok=True)

            filepath = echarts_dir / str(filename).lower()

            # Save chart configuration as JSON for potential future use (keep only latest)
            # COMMENTED OUT: JSON config files not needed
            # config_filename = filename.replace('.pdf', '.json')
            # config_filepath = echarts_dir / config_filename

            # Remove existing files if they exist (keep only latest)
            if os.path.exists(filepath):
                os.remove(filepath)
            # if os.path.exists(config_filepath):
            #     os.remove(config_filepath)

            # Save configuration
            # with open(config_filepath, 'w', encoding='utf-8') as f:
            #     json.dump(chart_config, f, ensure_ascii=False, indent=2)

            # For now, create an empty PDF file as placeholder
            # Note: Full PDF rendering would require additional libraries like playwright
            with open(filepath, 'w') as f:
                f.write(f"% ECharts PDF Export Placeholder\n% Configuration would have been saved as JSON\n% Use manual export for actual PDF\n")

            # st.info(f"💾 Chart configuration saved (PDF placeholder): {filepath}")

            return filepath

        except Exception as e:
            st.error(f"Error saving as PDF: {str(e)}")
            return None

    def save_echarts_as_png(self, chart_config, filename_parts, dimensions="2D", external_filename=None):
        """Save ECharts configuration as PNG image to echarts folder"""
        try:
            # Use external filename if provided, otherwise use internal logic
            if external_filename:
                filename = str(external_filename)
            else:
                # Create filename without timestamp (keep only latest)
                filename_base = "-".join([
                    str(part).replace(" ", "-").replace("_", "-")
                    for part in filename_parts if part
                ])

                # Add echarts prefix and dimension suffix
                if dimensions == "3D":
                    filename = f"echarts-{filename_base}-3d.png"
                else:
                    filename = f"echarts-{filename_base}.png"

            # Create echarts images directory if it doesn't exist
            echarts_dir = DATA_PATH / "images" / "echarts"
            echarts_dir.mkdir(parents=True, exist_ok=True)

            filepath = echarts_dir / str(filename).lower()

            # Save chart configuration as JSON for potential future use (keep only latest)
            # COMMENTED OUT: JSON config files not needed
            # config_filename = filename.replace('.png', '.json')
            # config_filepath = echarts_dir / config_filename

            # Remove existing files if they exist (keep only latest)
            if os.path.exists(filepath):
                os.remove(filepath)
            # if os.path.exists(config_filepath):
            #     os.remove(config_filepath)

            # with open(config_filepath, 'w', encoding='utf-8') as f:
            #     json.dump(chart_config, f, indent=2, ensure_ascii=False)

            # Display combined success message with instructions
            # st.info(f"📊 **ECharts visualization saved**\n\n🖼️ **To save as PNG**: Use your browser's screenshot tool or right-click → 'Save image as...' on the chart")

            # Add download button for the JSON configuration
            # COMMENTED OUT: JSON download not needed
            # with open(config_filepath, 'r', encoding='utf-8') as f:
            #     config_data = f.read()
            #
            # st.download_button(
            #     label="📥 Download ECharts Config (JSON)",
            #     data=config_data,
            #     file_name=config_filename,
            #     mime="application/json",
            #     help="Download the ECharts configuration file"
            # )

            return None  # JSON config functionality commented out

        except Exception as e:
            st.error(f"Failed to save ECharts configuration: {str(e)}")
            return None

    def create_export_instructions(self):
        """Display instructions for exporting ECharts as PNG"""
        with st.expander("📸 How to Save ECharts as PNG", expanded=False):
            st.markdown("""
            **Method 1: Browser Screenshot (Recommended)**
            1. Right-click on the chart
            2. Select "Save image as..." or "Copy image"
            3. Save to your desired location

            **Method 2: Browser Developer Tools**
            1. Right-click on the chart → "Inspect Element"
            2. Find the `<canvas>` element in the DOM
            3. Right-click on the canvas → "Save image as..."

            **Method 3: Browser Extensions**
            - Use screenshot extensions like "Full Page Screen Capture"
            - Many browsers have built-in screenshot tools (Ctrl+Shift+S in some browsers)

            **Tips for High-Quality Images:**
            - Zoom in your browser before taking screenshot for higher resolution
            - Use browser's full-screen mode (F11) for cleaner captures
            - Adjust ECharts settings (text size, point size) for better visibility

            **Note**: ECharts configurations are automatically saved as JSON files
            in `{DATA_PATH}/images/echarts/` for reproducibility.
            """)

    def add_export_controls(self, chart_config, filename_parts, dimensions="2D"):
        """Add export controls and save functionality"""
        col1, col2 = st.columns([3, 1])

        with col1:
            # Show export instructions
            self.create_export_instructions()

        with col2:
            # Save configuration button
            if st.button("💾 Save Config", help="Save ECharts configuration as JSON"):
                saved_file = self.save_echarts_as_png(chart_config, filename_parts, dimensions)
                if saved_file:
                    st.success(f"✅ Saved: {saved_file}")

    def enhance_chart_with_export(self, option, filename_parts, dimensions="2D"):
        """Enhance chart configuration with export capabilities"""
        # Disable toolbox since controls don't work well in Streamlit
        # Export functionality is handled via sidebar buttons instead
        if 'toolbox' not in option:
            option['toolbox'] = {
                'show': False
            }

        # Export controls are now in the sidebar under "PNG Export Settings"
        # self.add_export_controls(option, filename_parts, dimensions)

        return option

    def save_echarts_as_pdf_auto(self, chart_config, filename_parts, dimensions="2D", width=800, height=600, external_filename=None):
        """Automatically save ECharts as PDF for publication quality (for 2D only)"""
        if dimensions == "3D":
            st.warning("⚠️ Automatic PDF export is only available for 2D visualizations (3D depends on viewing angle)")
            return None

        # Use the same logic as PNG but with PDF extension
        return self.save_echarts_as_pdf(chart_config, filename_parts, dimensions, external_filename)

    def save_echarts_as_png_auto(self, chart_config, filename_parts, dimensions="2D", width=800, height=600, external_filename=None):
        """Automatically save ECharts as PNG using headless browser (for 2D only)"""
        if dimensions == "3D":
            st.warning("⚠️ Automatic PNG export is only available for 2D visualizations (3D depends on viewing angle)")
            return None

        if not SELENIUM_AVAILABLE:
            st.warning("⚠️ Automatic PNG export requires selenium. Install with: pip install selenium webdriver-manager")
            return None

        try:
            # Use external filename if provided, otherwise use internal logic
            if external_filename:
                filename = str(external_filename)
            else:
                # Create filename with new convention: echarts-<dataset>-<model>-<method>-<lang_codes>.png
                # filename_parts expected order: [method, model, dataset, lang_codes_joined]
                f_parts = []
                for p in filename_parts:
                    if p is None: continue
                    if type(p) == PosixPath:
                        f_parts.append(p.as_posix())                    
                    else:
                        f_parts.append(p)
                if len(f_parts) >= 4:
                    method, model, dataset, lang_codes = f_parts[0], f_parts[1], f_parts[2], f_parts[3]
                    filename_base = "-".join([
                        str(dataset).replace(" ", "-").replace("_", "-"),
                        str(model).replace(" ", "-").replace("_", "-"),
                        str(method).replace(" ", "-").replace("_", "-"),
                        str(lang_codes).replace(" ", "-").replace("_", "-")
                    ])
                else:
                    # Fallback to original behavior if not enough parts
                    filename_base = "-".join([
                        str(part).replace(" ", "-").replace("_", "-")
                        for part in f_parts if part
                    ])
                filename = f"echarts-{filename_base}.png"

            # Create echarts images directory if it doesn't exist
            echarts_dir = DATA_PATH / "images" / "echarts"
            echarts_dir.mkdir(parents=True, exist_ok=True)
            filepath = echarts_dir / filename

            # Remove existing file if it exists (keep only latest)
            if os.path.exists(filepath):
                os.remove(filepath)

            # Create HTML file with ECharts - add extra padding to prevent cropping
            html_content = self._create_echarts_html(chart_config, width, height + 120)  # Add 120px padding

            # Save to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
                f.write(html_content)
                temp_html_path = f.name

            try:
                # Setup Chrome options for headless mode with better settings
                chrome_options = Options()
                chrome_options.add_argument("--headless")
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")
                chrome_options.add_argument("--disable-gpu")
                chrome_options.add_argument(f"--window-size={width + 160},{height + 220}")  # Extra margin for cropping prevention
                chrome_options.add_argument("--hide-scrollbars")
                chrome_options.add_argument("--force-device-scale-factor=1")

                # Setup Chrome driver
                service = Service(ChromeDriverManager().install())
                driver = webdriver.Chrome(service=service, options=chrome_options)

                try:
                    # Load the HTML file
                    driver.get(f"file://{temp_html_path}")

                    # Wait for chart container to be present
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.ID, "echarts-container"))
                    )

                    # Wait for ECharts library to load and chart to be initialized
                    WebDriverWait(driver, 10).until(
                        lambda driver: driver.execute_script("return typeof echarts !== 'undefined'")
                    )

                    # Wait for render status indicator to show chart is complete
                    WebDriverWait(driver, 15).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "#render-status.render-complete"))
                    )

                    # Wait for canvas element to be created (ECharts creates canvas for rendering)
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "#echarts-container canvas"))
                    )

                    # Additional wait to ensure chart animation is complete
                    time.sleep(2)

                    # Verify chart has content by checking if canvas has been drawn to
                    canvas_has_content = driver.execute_script("""
                        var canvas = document.querySelector('#echarts-container canvas');
                        if (!canvas) return false;
                        var ctx = canvas.getContext('2d');
                        var imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                        var data = imageData.data;
                        // Check if canvas has non-transparent pixels
                        for (var i = 3; i < data.length; i += 4) {
                            if (data[i] > 0) return true;  // Found non-transparent pixel
                        }
                        return false;
                    """)

                    if not canvas_has_content:
                        # Try one more resize and wait
                        driver.execute_script("window.chart && window.chart.resize();")
                        time.sleep(1)

                    # Take screenshot of chart element
                    chart_element = driver.find_element(By.ID, "echarts-container")
                    chart_element.screenshot(str(filepath))  # Convert to string for Selenium

                    # Return filepath as string to avoid PosixPath issues
                    return str(filepath)

                finally:
                    driver.quit()

            finally:
                # Clean up temporary file
                os.unlink(temp_html_path)

        except Exception as e:
            st.error(f"Failed to auto-save PNG: {str(e)}")
            return None

    def _create_echarts_html(self, chart_config, width=800, height=600):
        """Create standalone HTML page for ECharts rendering"""
        # Escape the JSON for safe embedding in HTML
        config_json = json.dumps(chart_config, ensure_ascii=False)

        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>ECharts Auto Export</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 30px 30px 80px 30px;
            background-color: white;
            box-sizing: border-box;
            font-family: Arial, sans-serif;
            overflow: hidden;
            min-height: {height + 160}px;
        }}
        #echarts-container {{
            width: {width}px;
            height: {height}px;
            margin: 40px auto 20px auto;
            padding: 0;
            box-sizing: border-box;
            border: none;
        }}
        .render-indicator {{
            display: none;
        }}
        .render-complete {{
            display: block;
        }}
    </style>
</head>
<body>
    <div id="echarts-container"></div>
    <div id="render-status" class="render-indicator">Chart rendered</div>
    <script>
        // Wait for ECharts library to be fully loaded
        window.addEventListener('load', function() {{
            var chartContainer = document.getElementById('echarts-container');
            var renderStatus = document.getElementById('render-status');

            // Initialize chart and make it globally accessible
            window.chart = echarts.init(chartContainer, null, {{
                renderer: 'canvas',
                width: {width},
                height: {height}
            }});

            var option = {config_json};

            // Set option and wait for render completion
            window.chart.setOption(option, true);

            // Force immediate resize
            window.chart.resize();

            // Listen for render completion
            window.chart.on('rendered', function() {{
                renderStatus.className = 'render-complete';
                console.log('Chart rendered successfully');
            }});

            // Listen for animation finish (more reliable than 'rendered' for complex charts)
            window.chart.on('finished', function() {{
                renderStatus.className = 'render-complete';
                console.log('Chart animation finished');
            }});

            // Additional safety measures with global chart reference
            setTimeout(function() {{
                window.chart.resize();
                renderStatus.className = 'render-complete';
            }}, 500);

            setTimeout(function() {{
                window.chart.resize();
                renderStatus.className = 'render-complete';
            }}, 1500);

            // Final safety timeout to ensure rendering is complete
            setTimeout(function() {{
                renderStatus.className = 'render-complete';
                console.log('Final render confirmation');
            }}, 2000);
        }});
    </script>
</body>
</html>"""
        return html_template

    def _add_sa_metrics_graphic(self, option, sa_metrics_text):
        """Add SA metrics as a graphic text element to the ECharts option"""
        try:
            # Create a compact text graphic element positioned inside chart at top-right
            if 'graphic' not in option:
                option['graphic'] = []

            # Split text into lines for proper formatting (should be 2 lines)
            text_lines = sa_metrics_text.split('\n')
            num_lines = len(text_lines)

            # Calculate box dimensions
            box_width = 200
            box_height = num_lines * 20 + 16

            option['graphic'].append({
                'type': 'group',
                'right': '9%',   # Position inside chart area
                'top': '9%',     # Position inside chart area
                'children': [
                    # Background rectangle
                    {
                        'type': 'rect',
                        'z': 100,
                        'left': 0,
                        'top': 0,
                        'shape': {
                            'width': box_width,
                            'height': box_height
                        },
                        'style': {
                            'fill': 'rgba(255, 255, 255, 0.95)',
                            'stroke': 'rgba(0, 0, 0, 0.4)',
                            'lineWidth': 1.5,
                            'shadowBlur': 4,
                            'shadowColor': 'rgba(0, 0, 0, 0.15)',
                            'shadowOffsetX': 1,
                            'shadowOffsetY': 1
                        }
                    },
                    # Text element (positioned relative to group)
                    {
                        'type': 'text',
                        'z': 101,
                        'left': 8,
                        'top': 8,
                        'style': {
                            'text': sa_metrics_text,
                            'font': '12px Courier New, monospace',
                            'fill': '#000',
                            'textAlign': 'left',
                            'textVerticalAlign': 'top',
                            'lineHeight': 20
                        }
                    }
                ]
            })

            return option
        except Exception as e:
            import streamlit as st
            st.warning(f"Failed to add SA metrics to chart: {e}")
            return option

    def get_auto_save_status(self):
        """Check if automatic PNG saving is available"""
        if SELENIUM_AVAILABLE:
            return {
                'available': True,
                'message': '✅ Automatic PNG export available'
            }
        else:
            return {
                'available': False,
                'message': '❌ Install selenium for automatic PNG export: pip install selenium webdriver-manager'
            }