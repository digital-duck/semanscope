import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from config import (
       PLOT_WIDTH, PLOT_HEIGHT,
       DEFAULT_N_CLUSTERS, DEFAULT_MIN_CLUSTERS, DEFAULT_MAX_CLUSTERS,
       DEFAULT_MAX_WORDS
)
import io
import base64
from utils.title_filename_helper import create_chart_title

class PlotManager:
    def __init__(self):
        self.min_clusters = DEFAULT_MIN_CLUSTERS
        self.max_clusters = DEFAULT_MAX_CLUSTERS
        # Publication-quality settings
        self.publication_settings = {
            'textfont_size': 16,
            'point_size': 12,
            'width': 1400,
            'height': 2520,  # Maximized height for optimal vertical space utilization
            'dpi': 300,  # 300 DPI for publication quality
            'grid_color': '#D0D0D0',
            'grid_width': 1,
            'background_color': 'white',
            'font_family': 'Arial, sans-serif',
            'grid_dash': 'dot'
        }

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
    
    def get_visualization_settings(self):
        """Get visualization settings from global settings"""
        # Get settings from global settings, with defaults if not set
        default_settings = {
            'publication_mode': False,
            'textfont_size': 12,
            'point_size': 8,
            'plot_width': PLOT_WIDTH,
            'plot_height': PLOT_HEIGHT,
            'export_format': 'PNG',
            'export_dpi': 300
        }

        # Try to get from global settings first
        if 'global_settings' in st.session_state and 'publication' in st.session_state.global_settings:
            global_pub_settings = st.session_state.global_settings['publication']
            # Merge with defaults
            settings = default_settings.copy()
            settings.update(global_pub_settings)
            return settings

        # Fallback to old session state key for backward compatibility
        return st.session_state.get('publication_settings', default_settings)
    
    def export_figure(self, fig, filename, settings):
        """Export figure in high quality format with detailed filename"""
        if settings['publication_mode']:
            if settings['export_format'] == 'PNG':
                img_bytes = fig.to_image(format="png", width=settings['plot_width'], 
                                       height=settings['plot_height'], scale=settings['export_dpi']/96)
            elif settings['export_format'] == 'SVG':
                img_bytes = fig.to_image(format="svg", width=settings['plot_width'], 
                                       height=settings['plot_height'])
            elif settings['export_format'] == 'PDF':
                img_bytes = fig.to_image(format="pdf", width=settings['plot_width'], 
                                       height=settings['plot_height'])
            
            # Create detailed filename with all parameters
            detailed_filename = f"{filename}-dpi-{settings['export_dpi']}-text-{settings['textfont_size']}-point-{settings['point_size']}.{settings['export_format'].lower()}"
            
            st.download_button(
                label=f"📥 Download {settings['export_format']} ({settings['export_dpi']} DPI)",
                data=img_bytes,
                file_name=detailed_filename,
                mime=f"image/{settings['export_format'].lower()}"
            )
    
    def plot_2d(self, embeddings, labels, colors, title, clustering=False, n_clusters=DEFAULT_N_CLUSTERS,
                semantic_forces=False, max_words=DEFAULT_MAX_WORDS, method_name="", model_name="", dataset_name="", lang_codes=None, word_search_config=None, sa_metrics_text=None):
        # Get visualization settings
        settings = self.get_visualization_settings()

        # Create standardized title
        if method_name and model_name:
            title = self.create_title(method_name, model_name, dataset_name, lang_codes)

        if semantic_forces:
            fig = self._plot_semantic_forces(embeddings, labels, title, max_words, settings)
        elif clustering:
            fig = self._plot_2d_cluster(embeddings, labels, colors, title, n_clusters, settings)
        else:
            fig = self._plot_2d_simple(embeddings, labels, colors, title, settings)

        # Apply word search highlighting if configured
        if word_search_config:
            fig = self._apply_word_search_highlighting(fig, labels, word_search_config)

        # Add SA metrics legend if provided
        if sa_metrics_text:
            fig = self._add_sa_metrics_legend(fig, sa_metrics_text)

        # Add export functionality
        if settings['publication_mode']:
            # Create standardized lowercase filename
            clean_method = method_name.lower().replace(" ", "-").replace(",", "").replace("_", "-")
            clean_model = model_name.lower().replace(" ", "-").replace(",", "").replace("_", "-")
            clean_dataset = dataset_name.lower().replace(" ", "-").replace(",", "").replace("_", "-")
            filename = f"{clean_method}-{clean_model}-{clean_dataset}"
            self.export_figure(fig, filename, settings)

        # Universal auto-save for regular plotting system (saves in BOTH PNG and PDF formats)
        if method_name and model_name and dataset_name:
            try:
                # Check if auto-save is enabled in global settings (same as ECharts auto-save)
                global_settings = st.session_state.get('global_settings', {})
                echarts_settings = global_settings.get('echarts', {})
                auto_save_enabled = echarts_settings.get('auto_save_enabled', False)

                # Also check legacy location for backward compatibility
                legacy_auto_save = st.session_state.get('echarts_auto_save', {})
                if not auto_save_enabled and legacy_auto_save.get('enabled', False):
                    auto_save_enabled = True

                # Note: Regular plotting doesn't need selenium, so always available=True

                if auto_save_enabled:
                    from utils.title_filename_helper import create_title_and_filename
                    import os
                    from pathlib import Path

                    # Determine active languages from lang_codes or fallback
                    active_languages = lang_codes if lang_codes else ['auto']

                    saved_files = []

                    # Save in BOTH PNG and PDF formats
                    for export_format in ['PNG', 'PDF']:
                        # Generate standardized filename using centralized helper
                        _, standardized_filename = create_title_and_filename(
                            [method_name],
                            [model_name],
                            dataset_name,
                            active_languages,
                            export_format.lower()
                        )

                        # Determine output directory
                        if export_format == 'PDF':
                            output_dir = Path("../data/images/PDF")
                        else:
                            output_dir = Path("../data/images")

                        output_dir.mkdir(parents=True, exist_ok=True)
                        file_path = output_dir / str(standardized_filename)

                        # Save the figure (regular plotting uses plotly)
                        with st.spinner(f"📸 Auto-saving {export_format}..."):
                            if export_format == 'PDF':
                                fig.write_image(str(file_path), format="pdf", width=1400, height=1100)
                            else:
                                fig.write_image(str(file_path), format="png", width=1400, height=1100, scale=2)

                        # Track saved file
                        full_file_path = os.path.abspath(file_path)
                        saved_files.append(f"{export_format}: {full_file_path}")

                    # Success message with all saved files
                    st.success(f"📊 **Auto-saved in both formats**:\n" + "\n".join([f"• {f}" for f in saved_files]))

                else:
                    st.info("🔧 **Auto-save Debug**: Auto-save disabled in settings")

            except Exception as auto_save_error:
                st.error(f"❌ **Universal auto-save error**: {str(auto_save_error)}")
                import traceback
                st.code(traceback.format_exc())

        return fig

    def plot_3d(self, embeddings, labels, colors, title, clustering=False, n_clusters=DEFAULT_N_CLUSTERS,
                method_name="", model_name="", dataset_name="", lang_codes=None):
        # Get visualization settings
        settings = self.get_visualization_settings()

        # Create standardized title
        if method_name and model_name:
            title = self.create_title(method_name, model_name, dataset_name, lang_codes)
        
        if clustering:
            fig = self._plot_3d_cluster(embeddings, labels, colors, title, n_clusters, settings)
        else:
            fig = self._plot_3d_simple(embeddings, labels, colors, title, settings)
        
        # Add export functionality
        if settings['publication_mode']:
            # Create standardized lowercase filename
            clean_method = method_name.lower().replace(" ", "-").replace(",", "").replace("_", "-")
            clean_model = model_name.lower().replace(" ", "-").replace(",", "").replace("_", "-")
            clean_dataset = dataset_name.lower().replace(" ", "-").replace(",", "").replace("_", "-")
            filename = f"{clean_method}-{clean_model}-{clean_dataset}-3d"
            self.export_figure(fig, filename, settings)
        
        return fig

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

    def _plot_2d_cluster(self, embeddings, labels, colors, title, n_clusters, settings):
        # Fixed boundary threshold (UI controls handled in Geometric Analysis section)
        boundary_threshold = 0.5

        # Perform clustering and get metrics (but don't display them here)
        metrics, kmeans = self._perform_clustering(embeddings, n_clusters)

        # Note: Metrics display is handled by Geometric Analysis section to avoid duplication

        # Create figure
        fig = go.Figure()

        # Add scatter plot
        df = pd.DataFrame({
            "x": embeddings[:, 0],
            "y": embeddings[:, 1],
            "label": labels,
            "cluster": metrics["cluster_labels"],
            "semantic_color": colors  # Store semantic colors
        })

        fig.add_trace(go.Scatter(
            x=df["x"],
            y=df["y"],
            mode='markers+text',
            text=df["label"],
            textposition="top center",
            textfont=dict(
                size=settings['textfont_size'],
                family=self.publication_settings['font_family']
            ),
            marker=dict(
                size=settings['point_size'],
                color=colors,  # Use semantic colors instead of cluster IDs
                opacity=0.8,
                line=dict(width=1, color='white') if settings['publication_mode'] else dict()
            )
        ))

        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18 if settings['publication_mode'] else 14, family=self.publication_settings['font_family']),
                x=0.5,  # Center align title
                xanchor='center',
                y=0.99,  # Position very close to top
                yanchor='top'
            ),
            showlegend=True,
            xaxis=dict(
                showgrid=True,
                gridwidth=self.publication_settings['grid_width'],
                gridcolor=self.publication_settings['grid_color'],
                griddash=self.publication_settings['grid_dash'],
                autorange=True
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=self.publication_settings['grid_width'],
                gridcolor=self.publication_settings['grid_color'],
                griddash=self.publication_settings['grid_dash'],
                autorange=True
            ),
            dragmode='pan',
            hovermode='closest',
            width=settings['plot_width'],
            height=settings['plot_height'],
            xaxis_scaleanchor="y",
            xaxis_scaleratio=1,
            plot_bgcolor=self.publication_settings['background_color'],
            font=dict(family=self.publication_settings['font_family']),
            margin=dict(l=60, r=60, t=40, b=60)  # Reduce top margin for closer title
        )

        st.plotly_chart(fig, width='stretch')
        return fig

    def _plot_2d_simple(self, embeddings, labels, colors, title, settings):
        df = pd.DataFrame({"x": embeddings[:, 0], "y": embeddings[:, 1],
                          "label": labels, "color": colors})

        # Color processing (silent) - check for semantic number patterns
        if colors:
            numbers_in_plot = []
            for i, (label, color) in enumerate(zip(labels, colors)):
                if str(label).lower() in ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve']:
                    numbers_in_plot.append(f"{label}→{color}")

        # Use go.Scatter for proper individual color mapping
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df["x"],
            y=df["y"],
            mode='markers+text',
            text=df["label"],
            textposition='top center',
            hoverinfo='text',
            textfont=dict(
                size=settings['textfont_size'],
                family=self.publication_settings['font_family']
            ),
            marker=dict(
                size=settings['point_size'],
                color=colors,  # Direct color assignment
                opacity=0.8,
                line=dict(width=1, color='white') if settings['publication_mode'] else dict()
            )
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18 if settings['publication_mode'] else 14, family=self.publication_settings['font_family']),
                x=0.5,  # Center align title
                xanchor='center',
                y=0.99,  # Position very close to top
                yanchor='top'
            ),
            showlegend=False,
            xaxis=dict(
                showgrid=True,
                gridwidth=self.publication_settings['grid_width'],
                gridcolor=self.publication_settings['grid_color'],
                griddash=self.publication_settings['grid_dash'],
                autorange=True
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=self.publication_settings['grid_width'],
                gridcolor=self.publication_settings['grid_color'],
                griddash=self.publication_settings['grid_dash'],
                autorange=True
            ),
            dragmode='pan',
            hovermode='closest',
            width=settings['plot_width'],
            height=settings['plot_height'],
            xaxis_scaleanchor="y",
            xaxis_scaleratio=1,
            plot_bgcolor=self.publication_settings['background_color'],
            font=dict(family=self.publication_settings['font_family']),
            margin=dict(l=60, r=60, t=40, b=60)  # Reduce top margin for closer title
        )
        
        st.plotly_chart(fig, width='stretch')
        return fig


    def _plot_3d_simple(self, embeddings, labels, colors, title, settings):
        df = pd.DataFrame({
            "x": embeddings[:, 0],
            "y": embeddings[:, 1],
            "z": embeddings[:, 2],
            "label": labels,
            "color": colors
        })

        # Use go.Scatter3d for proper individual color mapping
        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=df["x"],
            y=df["y"],
            z=df["z"],
            mode='markers+text',
            text=df["label"],
            textposition='top center',
            hoverinfo='text',
            textfont=dict(
                size=settings['textfont_size'],
                family=self.publication_settings['font_family']
            ),
            marker=dict(
                size=settings['point_size'],
                color=colors,  # Direct color assignment
                opacity=0.8,
                line=dict(width=1, color='white') if settings['publication_mode'] else dict()
            )
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18 if settings['publication_mode'] else 14, family=self.publication_settings['font_family']),
                x=0.5,  # Center align title
                xanchor='center'
            ),
            showlegend=False,
            scene=dict(
                xaxis=dict(
                    showgrid=True, 
                    gridwidth=self.publication_settings['grid_width'],
                    gridcolor=self.publication_settings['grid_color'],
                    # griddash not supported for 3D scene axes
                ),
                yaxis=dict(
                    showgrid=True, 
                    gridwidth=self.publication_settings['grid_width'],
                    gridcolor=self.publication_settings['grid_color'],
                    # griddash not supported for 3D scene axes
                ),
                zaxis=dict(
                    showgrid=True, 
                    gridwidth=self.publication_settings['grid_width'],
                    gridcolor=self.publication_settings['grid_color'],
                    # griddash not supported for 3D scene axes
                )
            ),
            dragmode='pan',
            hovermode='closest',
            width=settings['plot_width'],
            height=settings['plot_height'],
            plot_bgcolor=self.publication_settings['background_color'],
            font=dict(family=self.publication_settings['font_family'])
        )
        
        st.plotly_chart(fig, width='stretch')
        return fig

    def _plot_3d_cluster(self, embeddings, labels, colors, title, n_clusters, settings):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)

        df = pd.DataFrame({
            "x": embeddings[:, 0],
            "y": embeddings[:, 1],
            "z": embeddings[:, 2],
            "label": labels,
            "color": colors,
            "cluster": clusters
        })

        # Use go.Scatter3d for proper semantic color mapping
        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=df["x"],
            y=df["y"],
            z=df["z"],
            mode='markers+text',
            text=df["label"],
            textposition='top center',
            hoverinfo='text',
            textfont=dict(
                size=settings['textfont_size'],
                family=self.publication_settings['font_family']
            ),
            marker=dict(
                size=settings['point_size'],
                color=colors,  # Use semantic colors instead of cluster colorscale
                opacity=0.8,
                line=dict(width=1, color='white') if settings['publication_mode'] else dict()
            )
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18 if settings['publication_mode'] else 14, family=self.publication_settings['font_family']),
                x=0.5,  # Center align title
                xanchor='center'
            ),
            showlegend=True,
            scene=dict(
                xaxis=dict(
                    showgrid=True, 
                    gridwidth=self.publication_settings['grid_width'],
                    gridcolor=self.publication_settings['grid_color'],
                    # griddash not supported for 3D scene axes
                ),
                yaxis=dict(
                    showgrid=True, 
                    gridwidth=self.publication_settings['grid_width'],
                    gridcolor=self.publication_settings['grid_color'],
                    # griddash not supported for 3D scene axes
                ),
                zaxis=dict(
                    showgrid=True, 
                    gridwidth=self.publication_settings['grid_width'],
                    gridcolor=self.publication_settings['grid_color'],
                    # griddash not supported for 3D scene axes
                )
            ),
            dragmode='pan',
            hovermode='closest',
            width=settings['plot_width'],
            height=settings['plot_height'],
            plot_bgcolor=self.publication_settings['background_color'],
            font=dict(family=self.publication_settings['font_family'])
        )
        
        st.plotly_chart(fig, width='stretch')
        return fig
    
    def _plot_semantic_forces(self, embeddings, labels, title, max_words, settings):
        """Visualize semantic forces between words/phrases using arrows"""
        if len(labels) > max_words:
            st.warning(f"Only showing semantic forces for the first {max_words} words/phrases.")
            embeddings = embeddings[:max_words]
            labels = labels[:max_words]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            mode="markers+text",
            text=labels,
            textposition="top center",
            textfont=dict(
                size=settings['textfont_size'],
                family=self.publication_settings['font_family']
            ),
            marker=dict(
                size=settings['point_size'], 
                color="blue",
                line=dict(width=1, color='white') if settings['publication_mode'] else dict()
            )
        ))

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                fig.add_annotation(
                    x=embeddings[j, 0],
                    y=embeddings[j, 1],
                    ax=embeddings[i, 0],
                    ay=embeddings[i, 1],
                    axref="x",
                    ayref="y",
                    xref="x",
                    yref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="red"
                )

        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18 if settings['publication_mode'] else 14, family=self.publication_settings['font_family']),
                x=0.5,  # Center align title
                xanchor='center',
                y=0.99,  # Position very close to top
                yanchor='top'
            ),
            showlegend=False,
            xaxis=dict(
                showgrid=True,
                gridwidth=self.publication_settings['grid_width'],
                gridcolor=self.publication_settings['grid_color'],
                griddash=self.publication_settings['grid_dash'],
                autorange=True
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=self.publication_settings['grid_width'],
                gridcolor=self.publication_settings['grid_color'],
                griddash=self.publication_settings['grid_dash'],
                autorange=True
            ),
            width=settings['plot_width'],
            height=settings['plot_height'],
            xaxis_scaleanchor="y",
            xaxis_scaleratio=1,
            plot_bgcolor=self.publication_settings['background_color'],
            font=dict(family=self.publication_settings['font_family']),
            margin=dict(l=60, r=60, t=40, b=60)  # Reduce top margin for closer title
        )

        st.plotly_chart(fig, width='stretch')
        return fig
    def _apply_word_search_highlighting(self, fig, labels, word_search_config):
        """Apply word search highlighting to a Plotly figure"""
        try:
            keywords = word_search_config["keywords"]
            use_multiple_colors = word_search_config["use_multiple_colors"]
            base_color = word_search_config["color"]
            color_palette = word_search_config.get("color_palette", [])
            search_size = word_search_config["size"]

            # Find matching indices
            matching_indices = []
            search_colors = []

            for i, label in enumerate(labels):
                label_str = str(label).strip()
                found_keyword_index = None

                # Find which keyword matches this label
                for keyword_idx, keyword in enumerate(keywords):
                    if (label_str.lower() == keyword.lower() or
                        keyword.lower() in label_str.lower()):
                        found_keyword_index = keyword_idx
                        matching_indices.append(i)
                        break

                if found_keyword_index is not None:
                    if use_multiple_colors and color_palette:
                        # Use different color for each keyword
                        color_idx = found_keyword_index % len(color_palette)
                        search_colors.append(color_palette[color_idx])
                    else:
                        # Use single color for all matches
                        search_colors.append(base_color)
                else:
                    search_colors.append(None)

            if matching_indices:
                # Update the figure to highlight matching points
                trace = fig.data[0]

                # Prepare new marker properties
                new_colors = []
                new_sizes = []

                for i in range(len(labels)):
                    if i in matching_indices:
                        # Highlighted point
                        new_colors.append(search_colors[i])
                        new_sizes.append(search_size)
                    else:
                        # Keep original styling for non-matching points
                        if hasattr(trace.marker, "color") and trace.marker.color is not None:
                            if isinstance(trace.marker.color, (list, tuple)):
                                new_colors.append(trace.marker.color[i] if i < len(trace.marker.color) else trace.marker.color[0])
                            else:
                                new_colors.append(trace.marker.color)
                        else:
                            new_colors.append("#1f77b4")  # Default blue

                        if hasattr(trace.marker, "size") and trace.marker.size is not None:
                            if isinstance(trace.marker.size, (list, tuple)):
                                new_sizes.append(trace.marker.size[i] if i < len(trace.marker.size) else trace.marker.size[0])
                            else:
                                new_sizes.append(trace.marker.size)
                        else:
                            new_sizes.append(6)  # Default size

                # Update the trace
                fig.update_traces(
                    marker=dict(
                        color=new_colors,
                        size=new_sizes,
                        line=dict(width=2, color="white")
                    )
                )

                # Add information about search results
                num_matches = len(matching_indices)
                if num_matches > 0:
                    st.info(f"🔍 Found {num_matches} matching word(s): {', '.join(keywords[:5])}")

            return fig

        except Exception as e:
            st.warning(f"Word search highlighting failed: {e}")
            return fig

    def _add_sa_metrics_legend(self, fig, sa_metrics_text):
        """Add SA metrics as a text annotation legend on the chart"""
        try:
            # Add the SA metrics as an annotation box inside the chart at top-right
            fig.add_annotation(
                text=sa_metrics_text,
                xref="x domain",  # Use x domain to position inside chart
                yref="y domain",  # Use y domain to position inside chart
                x=0.96,  # Right side of chart area, slightly inward
                y=0.94,  # Top of chart area, slightly lower
                xanchor="right",
                yanchor="top",
                showarrow=False,
                bgcolor="rgba(255, 255, 255, 0.95)",
                bordercolor="rgba(0, 0, 0, 0.4)",
                borderwidth=1.5,
                borderpad=8,
                font=dict(
                    size=12,
                    family="Courier New, monospace",
                    color="black"
                ),
                align="left"
            )
            return fig
        except Exception as e:
            st.warning(f"Failed to add SA metrics legend: {e}")
            return fig

