import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import plotly.graph_objects as go
from typing import Tuple, Dict

class DynamicClusterer:
    def __init__(self):
        self.min_clusters = 2
        self.max_clusters = 10
        
    def render_controls(self) -> Tuple[int, float]:
        """Render clustering control widgets"""
        n_clusters = st.slider(
            "Number of Clusters",
            min_value=self.min_clusters,
            max_value=self.max_clusters,
            value=5,
            key="cluster_count"
        )
        
        boundary_threshold = st.slider(
            "Cluster Boundary Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Adjust the softness of cluster boundaries. Lower values create larger boundaries.",
            key="boundary_threshold"
        )
        
        return n_clusters, boundary_threshold
    
    def perform_clustering(self, embeddings: np.ndarray, n_clusters: int) -> Dict:
        """Perform clustering and calculate quality metrics"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Calculate cluster quality metrics
        metrics = {
            "silhouette": round(silhouette_score(embeddings, clusters), 3),
            "calinski": round(calinski_harabasz_score(embeddings, clusters), 3),
            "inertia": round(kmeans.inertia_, 3),
            "cluster_centers": kmeans.cluster_centers_,
            "cluster_labels": clusters
        }
        
        return metrics
    
    def display_metrics(self, metrics: Dict):
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
            
    def create_boundary_plot(self, 
                           embeddings: np.ndarray,
                           labels: list,
                           metrics: Dict,
                           boundary_threshold: float,
                           title: str) -> go.Figure:
        """Create plot with cluster boundaries"""
        # Create mesh grid for boundary visualization
        x_min, x_max = embeddings[:, 0].min() - 1, embeddings[:, 0].max() + 1
        y_min, y_max = embeddings[:, 1].min() - 1, embeddings[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))
        
        # Get cluster assignments for mesh grid points
        kmeans = KMeans(n_clusters=len(metrics["cluster_centers"]))
        kmeans.cluster_centers_ = metrics["cluster_centers"]
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Create figure
        fig = go.Figure()
        
        # Add contour plot for cluster boundaries
        fig.add_trace(go.Contour(
            x=np.arange(x_min, x_max, 0.1),
            y=np.arange(y_min, y_max, 0.1),
            z=Z,
            showscale=False,
            opacity=0.3,
            line=dict(width=0),
            colorscale='Viridis'
        ))
        
        # Add scatter plot for points
        fig.add_trace(go.Scatter(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            mode='markers+text',
            text=labels,
            textposition="top center",
            marker=dict(
                size=10,
                color=metrics["cluster_labels"],
                colorscale='Viridis',
                showscale=True,
                opacity=0.8
            ),
            name='Points'
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            showlegend=False,
            width=800,
            height=800,
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
            xaxis_scaleanchor="y",
            xaxis_scaleratio=1
        )
        
        return fig