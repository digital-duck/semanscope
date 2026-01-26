import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from scipy.spatial import ConvexHull, distance_matrix
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from collections import defaultdict
import warnings
import json
import csv
from datetime import datetime
from pathlib import Path
from config import SRC_DIR, DATA_PATH, DEFAULT_GEOMETRIC_ANALYSIS

import re
warnings.filterwarnings('ignore')

class GeometricAnalyzer:
    """
    Comprehensive geometric analysis for semantic embeddings including:
    - Clustering Analysis (silhouette, Davies-Bouldin, neighborhood density)
    - Branching Analysis (pathway linearity, connectivity graphs) 
    - Void Analysis (empty space identification, statistical significance)
    """
    
    def __init__(self):
        self.min_clusters = 2
        self.max_clusters = 15
        self.metrics_dir = DATA_PATH / "metrics"
        
    def render_controls(self) -> Dict[str, Any]:
        """Render geometric analysis control widgets"""
        st.subheader("🔬 Geometric Analysis Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Clustering Analysis:**")
            enable_clustering = st.checkbox(
                "Enable Clustering Analysis", 
                value=DEFAULT_GEOMETRIC_ANALYSIS,
                help="Analyze cluster quality and neighborhood density"
            )
            
            n_clusters = 5
            if enable_clustering:
                n_clusters = st.slider(
                    "Number of Clusters",
                    min_value=self.min_clusters,
                    max_value=self.max_clusters,
                    value=5
                )
                
            density_radius = st.slider(
                "Density Analysis Radius",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.1,
                help="Radius for neighborhood density calculations"
            )
        
        with col2:
            st.write("**Branching Analysis:**")
            enable_branching = st.checkbox(
                "Enable Branching Analysis",
                value=True,
                help="Analyze pathway linearity and connectivity"
            )
            
            connectivity_threshold = st.slider(
                "Connectivity Threshold",
                min_value=0.1,
                max_value=2.0,
                value=0.8,
                step=0.1,
                help="Distance threshold for connectivity graphs"
            )
            
            st.write("**Void Analysis:**")
            enable_void = st.checkbox(
                "Enable Void Analysis",
                value=True,
                help="Identify and analyze empty regions"
            )
            
            void_confidence = st.slider(
                "Void Confidence Level",
                min_value=0.90,
                max_value=0.99,
                value=0.95,
                step=0.01,
                help="Statistical confidence for void region detection"
            )
            
            st.write("**File Options:**")
            save_json_files = st.checkbox(
                "Save JSON files",
                value=False,
                help="Save detailed JSON files in addition to CSV files. JSON files contain complete analysis data but are harder to work with."
            )
        
        return {
            'enable_clustering': enable_clustering,
            'n_clusters': n_clusters,
            'density_radius': density_radius,
            'enable_branching': enable_branching,
            'connectivity_threshold': connectivity_threshold,
            'enable_void': enable_void,
            'void_confidence': void_confidence,
            'save_json_files': save_json_files
        }
    
    def analyze_clustering(self, embeddings: np.ndarray, n_clusters: int, 
                          density_radius: float, labels: List[str]) -> Dict[str, Any]:
        """Comprehensive clustering analysis"""
        results = {}
        
        try:
            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Basic clustering metrics
            silhouette = silhouette_score(embeddings, cluster_labels)
            davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)
            
            results['basic_metrics'] = {
                'silhouette_score': round(silhouette, 4),
                'davies_bouldin_score': round(davies_bouldin, 4),
                'inertia': round(kmeans.inertia_, 4),
                'cluster_centers': kmeans.cluster_centers_,
                'cluster_labels': cluster_labels
            }
            
            # Neighborhood density analysis
            density_metrics = self._calculate_neighborhood_density(embeddings, density_radius)
            results['density_metrics'] = density_metrics
            
            # Cluster boundary detection using convex hulls
            boundary_metrics = self._analyze_cluster_boundaries(embeddings, cluster_labels, kmeans.cluster_centers_)
            results['boundary_metrics'] = boundary_metrics
            
            # Semantic domain coherence (group by language if labels contain language info)
            coherence_metrics = self._calculate_semantic_coherence(embeddings, cluster_labels, labels)
            results['coherence_metrics'] = coherence_metrics

            # ACL Paper specific metrics
            acl_metrics = self._calculate_acl_metrics(embeddings, cluster_labels, labels)
            results['acl_metrics'] = acl_metrics

        except Exception as e:
            st.error(f"Clustering analysis failed: {str(e)}")
            results = {'error': str(e)}
            
        return results
    
    def analyze_branching(self, embeddings: np.ndarray, labels: List[str], 
                         connectivity_threshold: float) -> Dict[str, Any]:
        """Branching and pathway analysis"""
        results = {}
        
        try:
            # Build connectivity graph
            distance_matrix = squareform(pdist(embeddings, metric='euclidean'))
            connectivity_graph = self._build_connectivity_graph(
                distance_matrix, connectivity_threshold, labels
            )
            results['connectivity_graph'] = connectivity_graph
            
            # Pathway linearity scoring
            linearity_scores = self._calculate_pathway_linearity(embeddings, labels)
            results['linearity_scores'] = linearity_scores
            
            # Sequential ordering validation (detect digit sequences, alphabetical patterns)
            ordering_analysis = self._validate_sequential_ordering(embeddings, labels)
            results['ordering_analysis'] = ordering_analysis
            
            # Branch topology characterization
            topology_metrics = self._characterize_branch_topology(connectivity_graph)
            results['topology_metrics'] = topology_metrics
            
        except Exception as e:
            st.error(f"Branching analysis failed: {str(e)}")
            results = {'error': str(e)}
            
        return results
    
    def analyze_voids(self, embeddings: np.ndarray, confidence_level: float) -> Dict[str, Any]:
        """Void analysis - identify empty regions and test statistical significance"""
        results = {}
        
        try:
            # Identify potential void regions using grid sampling
            void_regions = self._identify_void_regions(embeddings)
            results['void_regions'] = void_regions
            
            # Statistical significance testing
            significance_results = self._test_void_significance(
                embeddings, void_regions, confidence_level
            )
            results['significance_results'] = significance_results
            
            # Calculate void density metrics
            void_metrics = self._calculate_void_metrics(embeddings, void_regions)
            results['void_metrics'] = void_metrics
            
        except Exception as e:
            st.error(f"Void analysis failed: {str(e)}")
            results = {'error': str(e)}
            
        return results
    
    def _calculate_neighborhood_density(self, embeddings: np.ndarray, radius: float) -> Dict:
        """Calculate neighborhood density for each point"""
        nbrs = NearestNeighbors(radius=radius).fit(embeddings)
        distances, indices = nbrs.radius_neighbors(embeddings)
        
        densities = [len(idx) - 1 for idx in indices]  # -1 to exclude self
        
        return {
            'mean_density': np.mean(densities),
            'std_density': np.std(densities),
            'min_density': np.min(densities),
            'max_density': np.max(densities),
            'densities': densities
        }
    
    def _analyze_cluster_boundaries(self, embeddings: np.ndarray, 
                                   cluster_labels: np.ndarray, centers: np.ndarray) -> Dict:
        """Analyze cluster boundaries using convex hulls"""
        boundary_info = {}
        
        unique_clusters = np.unique(cluster_labels)
        hull_areas = []
        hull_perimeters = []
        
        for cluster_id in unique_clusters:
            cluster_points = embeddings[cluster_labels == cluster_id]
            
            if len(cluster_points) >= 3:  # Need at least 3 points for a hull
                try:
                    hull = ConvexHull(cluster_points)
                    hull_areas.append(hull.volume)  # In 2D, volume = area
                    hull_perimeters.append(np.sum(
                        [np.linalg.norm(cluster_points[hull.simplices[i, 1]] - 
                                      cluster_points[hull.simplices[i, 0]]) 
                         for i in range(len(hull.simplices))]
                    ))
                except:
                    hull_areas.append(0)
                    hull_perimeters.append(0)
            else:
                hull_areas.append(0)
                hull_perimeters.append(0)
        
        boundary_info['hull_areas'] = hull_areas
        boundary_info['hull_perimeters'] = hull_perimeters
        boundary_info['mean_hull_area'] = np.mean(hull_areas)
        boundary_info['total_hull_area'] = np.sum(hull_areas)
        
        return boundary_info
    
    def _calculate_semantic_coherence(self, embeddings: np.ndarray, 
                                     cluster_labels: np.ndarray, labels: List[str]) -> Dict:
        """Calculate semantic coherence within clusters"""
        # Detect language patterns in labels
        chinese_pattern = any('\u4e00' <= char <= '\u9fff' for label in labels for char in label)
        
        coherence_metrics = {}
        unique_clusters = np.unique(cluster_labels)
        
        language_separation_scores = []
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_labels_list = [labels[i] for i in range(len(labels)) if cluster_mask[i]]
            
            if chinese_pattern:
                # Count Chinese vs English in each cluster
                chinese_count = sum(1 for label in cluster_labels_list 
                                  if any('\u4e00' <= char <= '\u9fff' for char in label))
                english_count = len(cluster_labels_list) - chinese_count
                
                # Calculate language homogeneity (0 = mixed, 1 = pure)
                if len(cluster_labels_list) > 0:
                    homogeneity = max(chinese_count, english_count) / len(cluster_labels_list)
                    language_separation_scores.append(homogeneity)
        
        coherence_metrics['language_separation_scores'] = language_separation_scores
        coherence_metrics['mean_language_coherence'] = np.mean(language_separation_scores) if language_separation_scores else 0
        
        return coherence_metrics
    
    def _build_connectivity_graph(self, distance_matrix: np.ndarray, 
                                 threshold: float, labels: List[str]) -> Dict:
        """Build connectivity graph based on distance threshold"""
        n_points = distance_matrix.shape[0]
        edges = []
        
        for i in range(n_points):
            for j in range(i + 1, n_points):
                if distance_matrix[i, j] <= threshold:
                    edges.append((i, j, distance_matrix[i, j]))
        
        # Create NetworkX graph for analysis
        G = nx.Graph()
        G.add_nodes_from(range(n_points))
        G.add_weighted_edges_from(edges)
        
        # Calculate graph metrics
        connected_components = list(nx.connected_components(G))
        clustering_coefficient = nx.average_clustering(G) if len(edges) > 0 else 0
        
        return {
            'edges': edges,
            'num_edges': len(edges),
            'num_components': len(connected_components),
            'largest_component_size': max(len(comp) for comp in connected_components) if connected_components else 0,
            'clustering_coefficient': clustering_coefficient,
            'graph': G
        }
    
    def _calculate_pathway_linearity(self, embeddings: np.ndarray, labels: List[str]) -> Dict:
        """Calculate linearity scores for potential pathways"""
        from sklearn.decomposition import PCA  # Import at method level to avoid scope issues
        
        linearity_results = {}
        
        # Detect digit sequences (0-9)
        digit_indices = []
        digit_labels = []
        
        for i, label in enumerate(labels):
            # Check for digit patterns
            if any(char.isdigit() for char in str(label)):
                digit_indices.append(i)
                digit_labels.append(label)
        
        if len(digit_indices) >= 3:
            digit_embeddings = embeddings[digit_indices]
            
            # Calculate linearity score using PCA
            pca = PCA(n_components=1)
            pca.fit(digit_embeddings)
            
            # Linearity score = proportion of variance explained by first PC
            linearity_score = pca.explained_variance_ratio_[0]
            linearity_results['digit_sequence_linearity'] = round(linearity_score, 4)
            linearity_results['digit_indices'] = digit_indices
        else:
            linearity_results['digit_sequence_linearity'] = None
            
        # General pathway analysis for all points
        if len(embeddings) >= 3:
            pca_all = PCA(n_components=min(2, embeddings.shape[1]))
            pca_all.fit(embeddings)
            overall_linearity = pca_all.explained_variance_ratio_[0]
            linearity_results['overall_linearity'] = round(overall_linearity, 4)
        
        return linearity_results
    
    def _validate_sequential_ordering(self, embeddings: np.ndarray, labels: List[str]) -> Dict:
        """Validate sequential ordering patterns"""
        ordering_results = {}
        
        # Check for digit sequences
        digit_pattern_analysis = self._analyze_digit_patterns(embeddings, labels)
        ordering_results['digit_patterns'] = digit_pattern_analysis
        
        # Check for alphabetical patterns
        alpha_pattern_analysis = self._analyze_alphabetical_patterns(embeddings, labels)
        ordering_results['alphabetical_patterns'] = alpha_pattern_analysis
        
        return ordering_results
    
    def _analyze_digit_patterns(self, embeddings: np.ndarray, labels: List[str]) -> Dict:
        """Analyze digit sequence patterns (0-9)"""
        digit_info = {}
        digit_map = {}
        
        # Extract digits and their positions
        for i, label in enumerate(labels):
            # Try to extract digits from label
            digits = ''.join(c for c in str(label) if c.isdigit())
            if digits:
                try:
                    digit_val = int(digits)
                    if 0 <= digit_val <= 9:  # Focus on single digits
                        digit_map[digit_val] = i
                except:
                    continue
        
        if len(digit_map) >= 3:  # Need at least 3 digits for meaningful analysis
            # Calculate ordering consistency
            sorted_digits = sorted(digit_map.keys())
            digit_positions = [embeddings[digit_map[d]] for d in sorted_digits]
            
            # Calculate sequential distance consistency
            distances = []
            for i in range(len(digit_positions) - 1):
                dist = np.linalg.norm(digit_positions[i+1] - digit_positions[i])
                distances.append(dist)
            
            consistency_score = 1.0 - (np.std(distances) / np.mean(distances)) if np.mean(distances) > 0 else 0
            
            digit_info = {
                'found_digits': sorted_digits,
                'num_digits': len(digit_map),
                'sequential_consistency': round(max(0, consistency_score), 4),
                'mean_step_distance': round(np.mean(distances), 4),
                'step_distance_std': round(np.std(distances), 4)
            }
        
        return digit_info
    
    def _analyze_alphabetical_patterns(self, embeddings: np.ndarray, labels: List[str]) -> Dict:
        """Analyze alphabetical ordering patterns"""
        alpha_info = {}
        
        # Extract single letters
        letter_map = {}
        for i, label in enumerate(labels):
            clean_label = str(label).lower().strip()
            if len(clean_label) == 1 and clean_label.isalpha():
                letter_map[clean_label] = i
        
        if len(letter_map) >= 3:
            # Calculate alphabetical consistency
            sorted_letters = sorted(letter_map.keys())
            letter_positions = [embeddings[letter_map[letter]] for letter in sorted_letters]
            
            # Similar analysis as digits
            distances = []
            for i in range(len(letter_positions) - 1):
                dist = np.linalg.norm(letter_positions[i+1] - letter_positions[i])
                distances.append(dist)
            
            consistency_score = 1.0 - (np.std(distances) / np.mean(distances)) if np.mean(distances) > 0 else 0
            
            alpha_info = {
                'found_letters': sorted_letters,
                'num_letters': len(letter_map),
                'sequential_consistency': round(max(0, consistency_score), 4),
                'mean_step_distance': round(np.mean(distances), 4)
            }
        
        return alpha_info
    
    def _characterize_branch_topology(self, connectivity_graph: Dict) -> Dict:
        """Characterize the topology of the connectivity graph"""
        topology_metrics = {}
        
        G = connectivity_graph['graph']
        
        if G.number_of_nodes() > 0:
            # Basic topology metrics
            topology_metrics['density'] = nx.density(G)
            topology_metrics['transitivity'] = nx.transitivity(G)
            
            # Node degree statistics
            degrees = [d for n, d in G.degree()]
            topology_metrics['mean_degree'] = np.mean(degrees)
            topology_metrics['max_degree'] = np.max(degrees) if degrees else 0
            
            # Identify hub nodes (high degree nodes)
            mean_degree = np.mean(degrees) if degrees else 0
            hub_threshold = mean_degree * 1.5
            hubs = [n for n, d in G.degree() if d > hub_threshold]
            topology_metrics['num_hubs'] = len(hubs)
            topology_metrics['hub_nodes'] = hubs
        
        return topology_metrics
    
    def _identify_void_regions(self, embeddings: np.ndarray, grid_resolution: int = 20) -> Dict:
        """Identify potential void regions using grid sampling"""
        x_min, x_max = embeddings[:, 0].min(), embeddings[:, 0].max()
        y_min, y_max = embeddings[:, 1].min(), embeddings[:, 1].max()
        
        # Expand boundaries slightly
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= x_range * 0.1
        x_max += x_range * 0.1
        y_min -= y_range * 0.1
        y_max += y_range * 0.1
        
        # Create grid
        x_grid = np.linspace(x_min, x_max, grid_resolution)
        y_grid = np.linspace(y_min, y_max, grid_resolution)
        
        void_candidates = []
        grid_distances = []
        
        for x in x_grid:
            for y in y_grid:
                grid_point = np.array([x, y])
                # Calculate minimum distance to any embedding point
                min_dist = np.min([np.linalg.norm(grid_point - emb) for emb in embeddings])
                grid_distances.append(min_dist)
                
                # If distance is significantly large, consider as void candidate
                if min_dist > np.mean(grid_distances[:-1]) + 2 * np.std(grid_distances[:-1]) if len(grid_distances) > 1 else min_dist > 1.0:
                    void_candidates.append({'position': grid_point, 'distance': min_dist})
        
        return {
            'void_candidates': void_candidates,
            'num_voids': len(void_candidates),
            'grid_distances': grid_distances,
            'mean_void_distance': np.mean([v['distance'] for v in void_candidates]) if void_candidates else 0
        }
    
    def _test_void_significance(self, embeddings: np.ndarray, void_regions: Dict, 
                              confidence_level: float) -> Dict:
        """Test statistical significance of void regions"""
        significance_results = {}
        
        if void_regions['num_voids'] == 0:
            return {'significant_voids': 0, 'total_tested': 0}
        
        # Use chi-square test for spatial randomness
        # Compare observed empty regions vs expected under random distribution
        
        grid_distances = void_regions['grid_distances']
        distance_threshold = np.percentile(grid_distances, 95)  # Top 5% distances
        
        observed_voids = len([d for d in grid_distances if d > distance_threshold])
        expected_voids = len(grid_distances) * 0.05  # Expected 5% under random
        
        # Chi-square goodness of fit test
        if expected_voids > 0:
            chi_stat = (observed_voids - expected_voids) ** 2 / expected_voids
            # Degrees of freedom = 1 for this simple test
            p_value = 1 - chi2.cdf(chi_stat, df=1)
            
            significance_results = {
                'chi_statistic': round(chi_stat, 4),
                'p_value': round(p_value, 4),
                'is_significant': p_value < (1 - confidence_level),
                'observed_voids': observed_voids,
                'expected_voids': round(expected_voids, 2),
                'significance_level': 1 - confidence_level
            }
        
        return significance_results
    
    def _calculate_void_metrics(self, embeddings: np.ndarray, void_regions: Dict) -> Dict:
        """Calculate comprehensive void metrics"""
        void_metrics = {}
        
        if void_regions['num_voids'] > 0:
            void_distances = [v['distance'] for v in void_regions['void_candidates']]
            void_metrics = {
                'total_void_area_estimate': len(void_regions['void_candidates']) * (0.1 ** 2),  # Rough grid-based estimate
                'mean_void_distance': round(np.mean(void_distances), 4),
                'max_void_distance': round(np.max(void_distances), 4),
                'void_distance_std': round(np.std(void_distances), 4)
            }
        else:
            void_metrics = {
                'total_void_area_estimate': 0,
                'mean_void_distance': 0,
                'max_void_distance': 0,
                'void_distance_std': 0
            }
        
        return void_metrics

    def _calculate_acl_metrics(self, embeddings: np.ndarray, cluster_labels: np.ndarray, labels: List[str], debug_flag: bool = False) -> Dict[str, Any]:
        """Calculate ACL paper specific metrics for cross-linguistic validation"""
        acl_metrics = {}

        try:
            # Get domain information by reconstructing from word_color_map in session state
            from config import get_all_domain_colors

            domains = []
            words = []

            # Check if we have word_color_map in session state
            word_color_map = None
            if hasattr(st.session_state, 'visualization_data') and 'word_color_map' in st.session_state.visualization_data:
                word_color_map = st.session_state.visualization_data['word_color_map']
            else:
                # Try to find it in semantic colors session keys
                for key in st.session_state.keys():
                    if key.endswith('_semantic_colors'):
                        word_color_map = st.session_state[key]
                        break

            if word_color_map:
                # Create reverse mapping from color to domain
                domain_colors = get_all_domain_colors()
                color_to_domain = {color: domain for domain, color in domain_colors.items()}

                # Debug: Show some mappings
                if debug_flag: st.info(f"🔍 Color mapping: {len(domain_colors)} domains, {len(color_to_domain)} reverse mappings")

                # Reconstruct domains from colors
                for label in labels:
                    words.append(label)
                    word_color = word_color_map.get(label, '#666666')
                    domain = color_to_domain.get(word_color, 'unknown')
                    domains.append(domain)

                # Debug: Show some word-color-domain examples
                sample_mappings = []
                for i, (word, color, domain) in enumerate(zip(words[:10], [word_color_map.get(w, '#666666') for w in words[:10]], domains[:10])):
                    sample_mappings.append(f"{word}→{color}→{domain}")
                if debug_flag: 
                    st.info(f"🔍 Sample mappings: {', '.join(sample_mappings)}")
                    st.info(f"🔍 ACL Metrics: Reconstructed domains from word_color_map for {len(words)} words")
            else:
                # Fallback: try to parse from comma-separated labels (original logic)
                for label in labels:
                    if ',' in label:
                        parts = label.split(',')
                        words.append(parts[0].strip())
                        domains.append(parts[1].strip() if len(parts) > 1 else 'unknown')
                    else:
                        words.append(label)
                        domains.append('unknown')

                st.warning("⚠️ ACL Metrics: No word_color_map found, using fallback domain parsing")

            # Debug: Show parsing results
            unique_domains = set(domains)
            if debug_flag: st.info(f"🔍 Found domains: {sorted(unique_domains)}")

            # 1. Function Separation: Measure separation between function and content words
            function_separation = self._calculate_function_separation(embeddings, domains)
            if function_separation is not None:
                acl_metrics['function_separation'] = round(function_separation, 3)

            # 2. Morphological Coherence: Measure clustering of morphological families
            morphological_coherence = self._calculate_morphological_coherence(embeddings, domains, words)
            if morphological_coherence is not None:
                acl_metrics['morphological_coherence'] = round(morphological_coherence, 3)

            # 3. Number Linearity (R²): Enhanced R² calculation for number sequences
            number_linearity = self._calculate_number_linearity_r2(embeddings, domains, words)
            if number_linearity is not None:
                acl_metrics['number_linearity_r2'] = round(number_linearity, 3)

        except Exception as e:
            st.error(f"❌ ACL metrics calculation failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            # Return empty dict if calculation fails

        return acl_metrics

    def _calculate_function_separation(self, embeddings: np.ndarray, domains: List[str]) -> Optional[float]:
        """Calculate separation between function words and content words"""
        try:
            # Define function word domains based on ACL dataset structure
            function_domains = {'articles_determiners', 'prepositions', 'conjunctions', 'pronouns'}
            content_domains = {'family_kinship', 'body_parts', 'animals', 'emotions', 'nature_elements',
                             'food', 'actions_verbs', 'abstract_qualities', 'time_temporal', 'spatial_directional'}

            # Find function and content word indices
            function_indices = [i for i, domain in enumerate(domains) if domain in function_domains]
            content_indices = [i for i, domain in enumerate(domains) if domain in content_domains]

            if len(function_indices) < 2 or len(content_indices) < 2:
                return None

            # Calculate centroids
            function_centroid = np.mean(embeddings[function_indices], axis=0)
            content_centroid = np.mean(embeddings[content_indices], axis=0)

            # Calculate inter-group distance
            inter_distance = np.linalg.norm(function_centroid - content_centroid)

            # Calculate intra-group distances
            function_distances = [np.linalg.norm(embeddings[i] - function_centroid) for i in function_indices]
            content_distances = [np.linalg.norm(embeddings[i] - content_centroid) for i in content_indices]

            avg_intra_distance = (np.mean(function_distances) + np.mean(content_distances)) / 2

            # Separation score: ratio of inter to intra distance
            separation_score = inter_distance / (avg_intra_distance + 1e-8)

            # Normalize to 0-1 range (higher is better separation)
            normalized_score = min(separation_score / 10.0, 1.0)

            return normalized_score

        except Exception:
            return None

    def _calculate_morphological_coherence(self, embeddings: np.ndarray, domains: List[str], words: List[str]) -> Optional[float]:
        """Calculate coherence within morphological families"""
        try:
            # Define morphological families based on ACL dataset structure
            morphological_domains = {domain for domain in domains if domain.startswith('morphological_')}

            if len(morphological_domains) < 2:
                return None

            coherence_scores = []

            for morph_domain in morphological_domains:
                # Find words in this morphological family
                morph_indices = [i for i, domain in enumerate(domains) if domain == morph_domain]

                if len(morph_indices) < 2:
                    continue

                # Calculate pairwise distances within family
                morph_embeddings = embeddings[morph_indices]
                pairwise_distances = pdist(morph_embeddings, metric='euclidean')

                # Coherence = 1 / (1 + mean_distance) - closer words = higher coherence
                mean_distance = np.mean(pairwise_distances)
                coherence = 1.0 / (1.0 + mean_distance)
                coherence_scores.append(coherence)

            if coherence_scores:
                return np.mean(coherence_scores)
            else:
                return None

        except Exception:
            return None

    def _calculate_number_linearity_r2(self, embeddings: np.ndarray, domains: List[str], words: List[str]) -> Optional[float]:
        """Calculate R² for linear arrangement of numbers"""
        try:
            # Find number words
            number_indices = []
            number_values = []

            for i, (domain, word) in enumerate(zip(domains, words)):
                if domain == 'numbers':
                    # Try to extract numeric value from word
                    try:
                        if word.lower() in ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']:
                            number_map = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
                                        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
                            number_indices.append(i)
                            number_values.append(number_map[word.lower()])
                        elif word.isdigit():
                            number_indices.append(i)
                            number_values.append(int(word))
                    except (ValueError, KeyError):
                        continue

            if len(number_indices) < 3:
                return None

            # Get number embeddings and perform PCA to find main axis
            number_embeddings = embeddings[number_indices]

            # Use first principal component as the linear axis
            mean_embedding = np.mean(number_embeddings, axis=0)
            centered_embeddings = number_embeddings - mean_embedding

            # SVD to get principal component
            U, s, Vt = np.linalg.svd(centered_embeddings, full_matrices=False)
            pc1 = Vt[0]  # First principal component

            # Project numbers onto first principal component
            projections = np.dot(centered_embeddings, pc1)

            # Calculate R² with actual number values
            number_values = np.array(number_values)

            # Linear regression between projections and number values
            lr = LinearRegression()
            lr.fit(projections.reshape(-1, 1), number_values)
            r2_score = lr.score(projections.reshape(-1, 1), number_values)

            return max(0.0, r2_score)  # Ensure non-negative

        except Exception:
            return None

    def display_clustering_metrics(self, clustering_results: Dict):
        """Display clustering analysis results"""
        if 'error' in clustering_results:
            st.error(f"Clustering analysis error: {clustering_results['error']}")
            return
        
        st.subheader("🔍 Clustering Analysis Results")
        
        # Basic clustering metrics
        basic = clustering_results['basic_metrics']
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Silhouette Score", basic['silhouette_score'], 
                     help="Measures how well-separated clusters are. Range [-1,1]: >0.7 excellent, >0.5 good, >0.25 reasonable, <0 overlapping clusters")
        with col2:
            st.metric("Davies-Bouldin Score", basic['davies_bouldin_score'],
                     help="Measures cluster separation vs compactness. Lower is better: <1.0 indicates well-separated clusters, >2.0 indicates poor clustering")
        with col3:
            st.metric("Inertia", basic['inertia'],
                     help="Sum of squared distances from points to their cluster centers. Lower values indicate more compact clusters")
        
        # Density metrics
        if 'density_metrics' in clustering_results:
            density = clustering_results['density_metrics']
            st.subheader("🌐 Neighborhood Density Analysis")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean Density", f"{density['mean_density']:.2f}",
                         help="Average number of neighboring points within the specified radius. Higher values indicate denser regions")
            with col2:
                st.metric("Density Std", f"{density['std_density']:.2f}",
                         help="Standard deviation of neighborhood density. Higher values indicate more varied density across the space")
            with col3:
                st.metric("Min Density", density['min_density'],
                         help="Minimum neighborhood density found. Low values indicate isolated points or sparse regions")
            with col4:
                st.metric("Max Density", density['max_density'],
                         help="Maximum neighborhood density found. High values indicate very crowded regions or cluster centers")
        
        # Boundary metrics
        if 'boundary_metrics' in clustering_results:
            boundary = clustering_results['boundary_metrics']
            st.subheader("🔷 Cluster Boundary Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Mean Hull Area", f"{boundary['mean_hull_area']:.4f}",
                         help="Average area of convex hulls around clusters. Larger values indicate more spread-out clusters")
            with col2:
                st.metric("Total Hull Area", f"{boundary['total_hull_area']:.4f}",
                         help="Combined area of all cluster convex hulls. Indicates overall space occupation by clusters")
        
        # Coherence metrics
        if 'coherence_metrics' in clustering_results:
            coherence = clustering_results['coherence_metrics']
            st.subheader("🗣️ Semantic Coherence Analysis")
            if coherence['mean_language_coherence'] > 0:
                st.metric("Language Coherence Score",
                         f"{coherence['mean_language_coherence']:.4f}",
                         help="Measures how well languages separate into different clusters. 1.0 = perfect separation, 0.5 = random mixing, 0.0 = completely mixed")

        # ACL Paper Specific Metrics
        if 'acl_metrics' in clustering_results:
            acl = clustering_results['acl_metrics']
            st.subheader("📊 ACL Paper Metrics")
            col1, col2, col3 = st.columns(3)

            with col1:
                if 'function_separation' in acl:
                    st.metric("Function Separation", f"{acl['function_separation']:.3f}",
                             help="Measures separation between function words (articles, prepositions) and content words. Higher values indicate better separation for linguistic analysis")

            with col2:
                if 'morphological_coherence' in acl:
                    st.metric("Morphological Coherence", f"{acl['morphological_coherence']:.3f}",
                             help="Measures how well morphologically related words cluster together (e.g., work/worker/working). Higher values indicate better morphological organization")

            with col3:
                if 'number_linearity_r2' in acl:
                    st.metric("Number Linearity (R²)", f"{acl['number_linearity_r2']:.3f}",
                             help="R-squared value for linear arrangement of numbers in embedding space. Values >0.9 indicate strong linear organization")
    
    def display_branching_metrics(self, branching_results: Dict):
        """Display branching analysis results"""
        if 'error' in branching_results:
            st.error(f"Branching analysis error: {branching_results['error']}")
            return
        
        st.subheader("🌿 Branching Analysis Results")
        
        # Connectivity metrics
        if 'connectivity_graph' in branching_results:
            conn = branching_results['connectivity_graph']
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Connected Components", conn['num_components'],
                         help="Number of separate connected groups in the graph. Fewer components mean more interconnected data")
            with col2:
                st.metric("Total Edges", conn['num_edges'],
                         help="Number of connections between nearby points. More edges indicate denser connectivity")
            with col3:
                st.metric("Largest Component", conn['largest_component_size'],
                         help="Size of the biggest connected group. Larger values indicate most points are interconnected")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Clustering Coefficient", f"{conn['clustering_coefficient']:.4f}",
                         help="Measures how much neighbors of a point are also connected to each other. Range [0,1]: higher values indicate more clustered local neighborhoods")
            with col2:
                if 'topology_metrics' in branching_results:
                    topo = branching_results['topology_metrics']
                    st.metric("Graph Density", f"{topo.get('density', 0):.4f}",
                             help="Ratio of actual edges to possible edges. Range [0,1]: higher values mean more densely connected points")
        
        # Linearity metrics
        if 'linearity_scores' in branching_results:
            linearity = branching_results['linearity_scores']
            st.subheader("📏 Pathway Linearity Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                if linearity.get('digit_sequence_linearity'):
                    st.metric("Digit Sequence Linearity", 
                             f"{linearity['digit_sequence_linearity']:.4f}",
                             help="How linear digit sequences (0-9) are arranged. Range [0,1]: >0.8 indicates strong linear progression, <0.5 indicates scattered arrangement")
                else:
                    st.info("No digit sequences found for linearity analysis")
            
            with col2:
                if linearity.get('overall_linearity'):
                    st.metric("Overall Linearity", 
                             f"{linearity['overall_linearity']:.4f}",
                             help="Overall linear structure of all points. Range [0,1]: higher values indicate more linear/straight arrangements")
        
        # Ordering analysis
        if 'ordering_analysis' in branching_results:
            ordering = branching_results['ordering_analysis']
            st.subheader("🔢 Sequential Ordering Validation")
            
            col1, col2 = st.columns(2)
            with col1:
                digit_patterns = ordering.get('digit_patterns', {})
                if digit_patterns and 'num_digits' in digit_patterns:
                    st.metric("Digits Found", digit_patterns['num_digits'],
                             help="Number of digit sequences (0-9) detected in the data labels")
                    if digit_patterns.get('sequential_consistency'):
                        st.metric("Digit Consistency", 
                                f"{digit_patterns['sequential_consistency']:.4f}",
                                help="How evenly spaced digits are in sequence. Range [0,1]: 1.0 = perfectly uniform spacing, <0.5 = irregular spacing")
            
            with col2:
                alpha_patterns = ordering.get('alphabetical_patterns', {})
                if alpha_patterns and 'num_letters' in alpha_patterns:
                    st.metric("Letters Found", alpha_patterns['num_letters'],
                             help="Number of alphabetical sequences detected in the data labels")
                    if alpha_patterns.get('sequential_consistency'):
                        st.metric("Letter Consistency", 
                                f"{alpha_patterns['sequential_consistency']:.4f}",
                                help="How evenly spaced letters are in alphabetical sequence. Range [0,1]: higher values indicate more uniform spacing")
    
    def display_void_metrics(self, void_results: Dict):
        """Display void analysis results"""
        if 'error' in void_results:
            st.error(f"Void analysis error: {void_results['error']}")
            return
        
        st.subheader("🕳️ Void Analysis Results")
        
        # Basic void metrics
        if 'void_regions' in void_results:
            voids = void_results['void_regions']
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Void Candidates Found", voids['num_voids'],
                         help="Number of potential empty regions detected in the embedding space using grid sampling")
            with col2:
                st.metric("Mean Void Distance", f"{voids['mean_void_distance']:.4f}",
                         help="Average distance from void regions to nearest data points. Higher values indicate larger empty spaces")
        
        # Statistical significance
        if 'significance_results' in void_results:
            sig = void_results['significance_results']
            if sig:  # Check if not empty
                st.subheader("📊 Statistical Significance Testing")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Chi-Square Statistic", f"{sig.get('chi_statistic', 0):.4f}",
                             help="Test statistic comparing observed vs expected void distribution. Higher values suggest non-random void patterns")
                with col2:
                    st.metric("P-Value", f"{sig.get('p_value', 1):.4f}",
                             help="Probability that void pattern occurred by chance. <0.05 suggests significant non-random void structure")
                with col3:
                    significance_text = "Yes" if sig.get('is_significant', False) else "No"
                    st.metric("Statistically Significant", significance_text,
                             help="Whether void regions are statistically different from random distribution at the chosen confidence level")
        
        # Void metrics
        if 'void_metrics' in void_results:
            metrics = void_results['void_metrics']
            st.subheader("📐 Void Geometry Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Estimated Void Area", f"{metrics['total_void_area_estimate']:.6f}",
                         help="Rough estimate of total area occupied by void regions. Based on grid sampling resolution")
            with col2:
                st.metric("Max Void Distance", f"{metrics['max_void_distance']:.4f}",
                         help="Largest distance from any void region to nearest data point. Indicates the size of the biggest empty space")
    
    def create_comprehensive_analysis_plot(self, embeddings: np.ndarray, labels: List[str],
                                         clustering_results: Dict, branching_results: Dict,
                                         void_results: Dict, model_name: str = None, 
                                         method_name: str = None, dataset_name: str = None) -> go.Figure:
        """Create a simplified clustering visualization for summary"""
        
        # Create single plot focusing on clustering
        fig = go.Figure()
        
        # Plot clustering with boundaries if available
        if 'basic_metrics' in clustering_results:
            cluster_labels = clustering_results['basic_metrics']['cluster_labels']
            
            # Add scatter plot with cluster colors
            fig.add_trace(go.Scatter(
                x=embeddings[:, 0], 
                y=embeddings[:, 1],
                mode='markers+text', 
                text=labels,
                marker=dict(
                    size=12, 
                    color=cluster_labels, 
                    colorscale='viridis', 
                    showscale=True,
                    colorbar=dict(
                        title="Cluster",
                        x=1.02,  # Position to the right
                        y=0.8,   # Position towards top
                        len=0.6, # Shorter length to avoid overlap
                        thickness=15  # Thinner colorbar
                    ),
                    line=dict(width=1, color='white')
                ),
                textposition="top center",
                textfont=dict(size=10),
                name="Data Points",
                hovertemplate='<b>%{text}</b><br>Cluster: %{marker.color}<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
            ))
            
            # Add cluster centers if available
            if 'cluster_centers' in clustering_results['basic_metrics']:
                centers = clustering_results['basic_metrics']['cluster_centers']
                fig.add_trace(go.Scatter(
                    x=centers[:, 0],
                    y=centers[:, 1],
                    mode='markers',
                    marker=dict(
                        size=20,
                        color='red',
                        symbol='x',
                        line=dict(width=2, color='white')
                    ),
                    name="Cluster Centers",
                    hovertemplate='Cluster Center<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
                ))
        else:
            # Fallback: just show points without clustering
            fig.add_trace(go.Scatter(
                x=embeddings[:, 0], 
                y=embeddings[:, 1],
                mode='markers+text', 
                text=labels,
                marker=dict(size=10, color='steelblue'),
                textposition="top center",
                name="Data Points"
            ))
        
        # Create standardized title format
        title_parts = []
        if method_name:
            title_parts.append(f"[Method] {method_name}")
        if model_name:
            title_parts.append(f"[Model] {model_name}")
        if dataset_name:
            title_parts.append(f"[Dataset] {dataset_name}")
        clustering_title = ", ".join(title_parts) if title_parts else "Clustering Analysis Visualization"
        
        fig.update_layout(
            title=dict(
                text=clustering_title,
                font=dict(size=18, family='Arial, sans-serif'),
                x=0.5,  # Center align title
                xanchor='center'
            ),
            dragmode='pan',
            hovermode='closest',
            showlegend=True,
            legend=dict(
                x=1.02,  # Position to the right
                y=0.4,   # Position towards bottom to avoid colorbar
                xanchor='left',
                yanchor='middle'
            ),
            height=700,  # More square aspect ratio to match Overview
            width=800,   # Controlled width to reduce white space
            plot_bgcolor='white',
            font=dict(family='Arial, sans-serif'),
            margin=dict(l=60, r=80, t=80, b=60)  # Increased right margin for legend/colorbar
        )
        
        # Update axes with dotted grid lines and "x"/"y" labels to match Overview
        fig.update_xaxes(
            title_text="x",
            showgrid=True, 
            gridwidth=1,
            gridcolor='#D0D0D0',
            griddash='dot'
        )
        fig.update_yaxes(
            title_text="y",
            showgrid=True, 
            gridwidth=1,
            gridcolor='#D0D0D0',
            griddash='dot',
            scaleanchor="x", 
            scaleratio=1
        )
        
        return fig
    
    def save_summary_plot_as_png(self, fig: go.Figure, input_name: str, model_name: str, method_name: str) -> str:
        """Save the summary plot as PNG file"""
        try:
            # Create metrics directory if it doesn't exist
            metrics_dir = DATA_PATH / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            safe_input = self.sanitize_filename(input_name)
            safe_model = self.sanitize_filename(model_name)
            safe_method = self.sanitize_filename(method_name)
            
            filename = f"{safe_input}-{safe_model}-{safe_method}-clustering.png"
            filepath = metrics_dir / filename
            
            # Save as PNG
            fig.write_image(str(filepath), width=800, height=600, scale=2)
            
            return str(filename)
        except Exception as e:
            st.error(f"Error saving plot as PNG: {str(e)}")
            return None

    def generate_cross_dataset_comparison(self, dataset_results: List[Dict], 
                                        dataset_names: List[str]) -> Dict:
        """Generate normalized metrics for cross-dataset comparison"""
        comparison_results = {}
        
        # Extract key metrics for comparison
        metrics_to_compare = [
            'silhouette_score', 'davies_bouldin_score', 'mean_language_coherence',
            'overall_linearity', 'clustering_coefficient', 'num_voids'
        ]
        
        comparison_data = {}
        
        for metric in metrics_to_compare:
            comparison_data[metric] = []
            
            for result_set in dataset_results:
                # Extract metric value from nested results
                value = self._extract_metric_value(result_set, metric)
                comparison_data[metric].append(value)
        
        # Normalize metrics for fair comparison
        normalized_data = {}
        for metric, values in comparison_data.items():
            if any(v is not None for v in values):  # Check if we have any valid values
                valid_values = [v for v in values if v is not None]
                if valid_values:
                    mean_val = np.mean(valid_values)
                    std_val = np.std(valid_values) if len(valid_values) > 1 else 1
                    # Normalize: (value - mean) / std
                    normalized_data[metric] = [
                        (v - mean_val) / std_val if v is not None and std_val > 0 else 0 
                        for v in values
                    ]
                else:
                    normalized_data[metric] = [0] * len(values)
            else:
                normalized_data[metric] = [0] * len(values)
        
        comparison_results['raw_data'] = comparison_data
        comparison_results['normalized_data'] = normalized_data
        comparison_results['dataset_names'] = dataset_names
        
        return comparison_results
    
    def _extract_metric_value(self, result_set: Dict, metric_name: str) -> Optional[float]:
        """Extract specific metric value from nested result structure"""
        # Navigate through nested dictionaries to find the metric
        try:
            if metric_name == 'silhouette_score':
                return result_set.get('clustering', {}).get('basic_metrics', {}).get('silhouette_score')
            elif metric_name == 'davies_bouldin_score':
                return result_set.get('clustering', {}).get('basic_metrics', {}).get('davies_bouldin_score')
            elif metric_name == 'mean_language_coherence':
                return result_set.get('clustering', {}).get('coherence_metrics', {}).get('mean_language_coherence')
            elif metric_name == 'overall_linearity':
                return result_set.get('branching', {}).get('linearity_scores', {}).get('overall_linearity')
            elif metric_name == 'clustering_coefficient':
                return result_set.get('branching', {}).get('connectivity_graph', {}).get('clustering_coefficient')
            elif metric_name == 'num_voids':
                return result_set.get('void', {}).get('void_regions', {}).get('num_voids')
            else:
                return None
        except:
            return None
    
    def display_comparison_results(self, comparison_results: Dict):
        """Display cross-dataset comparison results"""
        st.subheader("📊 Cross-Dataset Comparison")
        
        if not comparison_results:
            st.warning("No comparison data available")
            return
        
        # Create comparison table
        comparison_df = pd.DataFrame(comparison_results['normalized_data'], 
                                   index=comparison_results['dataset_names'])
        
        st.subheader("Normalized Metrics Comparison")
        st.dataframe(comparison_df.round(3))
        
        # Create radar chart for visual comparison
        if len(comparison_results['dataset_names']) <= 3:  # Limit for readability
            fig = go.Figure()
            
            for i, dataset_name in enumerate(comparison_results['dataset_names']):
                values = list(comparison_results['normalized_data'].values())
                metrics = list(comparison_results['normalized_data'].keys())
                
                fig.add_trace(go.Scatterpolar(
                    r=[values[j][i] for j in range(len(values))],
                    theta=metrics,
                    fill='toself',
                    name=dataset_name
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[-2, 2]  # Typical range for normalized values
                    )),
                showlegend=True,
                title="Normalized Metrics Comparison"
            )
            
            st.plotly_chart(fig, width='stretch')
    
    def sanitize_filename(self, text: str) -> str:
        """Sanitize text for use in filenames"""
        # Remove or replace problematic characters
        sanitized = re.sub(r'[^\w\s-]', '', str(text))  # Remove special chars except spaces and hyphens
        sanitized = re.sub(r'[-\s]+', '-', sanitized)  # Replace spaces and multiple hyphens with single hyphen
        return sanitized.strip('-').lower()
    
    def save_metrics_to_files(self, results: Dict, input_name: str, model_name: str, 
                             method_name: str, languages: List[str], save_json: bool = False) -> Dict[str, str]:
        """Save geometric analysis metrics to various file formats"""
        
        # Create metrics directory if it doesn't exist
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize filename components
        safe_input = self.sanitize_filename(input_name)
        safe_model = self.sanitize_filename(model_name)
        safe_method = self.sanitize_filename(method_name)
        
        # Determine language suffix
        if len(languages) > 1:
            lang_suffix = "multilingual"
        elif languages:
            lang_suffix = self.sanitize_filename(languages[0])
        else:
            lang_suffix = "unknown"
        
        # Base filename (no timestamp)
        base_filename = f"{safe_input}-{safe_model}-{safe_method}-{lang_suffix}"
        
        # Generate timestamp for metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_files = {}
        
        try:
            # 1. Save comprehensive JSON with all results (optional)
            if save_json:
                json_file = self.metrics_dir / f"{base_filename}_complete.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    # Convert numpy arrays to lists for JSON serialization
                    json_ready_results = self._prepare_for_json(results)
                    json_ready_results['metadata'] = {
                        'input_name': input_name,
                        'model_name': model_name,
                        'method_name': method_name,
                        'languages': languages,
                        'timestamp': timestamp,
                        'analysis_date': datetime.now().isoformat()
                    }
                    json.dump(json_ready_results, f, indent=2, ensure_ascii=False)
                saved_files['complete_json'] = str(json_file.name)
            
            # 2. Save summary CSV with key metrics
            csv_file = self.metrics_dir / f"{base_filename}_summary.csv"
            summary_data = self._extract_summary_metrics(results)
            summary_data.update({
                'input_name': input_name,
                'model_name': model_name,
                'method_name': method_name,
                'languages': ';'.join(languages),
                'timestamp': timestamp
            })
            
            # Write CSV
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=summary_data.keys())
                writer.writeheader()
                writer.writerow(summary_data)
            saved_files['summary_csv'] = str(csv_file.name)
            
            # 3. Save clustering metrics separately (JSON + CSV)
            if 'clustering' in results:
                # Save clustering JSON (optional)
                if save_json:
                    clustering_file = self.metrics_dir / f"{base_filename}_clustering.json"
                    with open(clustering_file, 'w', encoding='utf-8') as f:
                        clustering_data = self._prepare_for_json(results['clustering'])
                        clustering_data['metadata'] = {
                            'input_name': input_name,
                            'model_name': model_name,
                            'method_name': method_name,
                            'timestamp': timestamp
                        }
                        json.dump(clustering_data, f, indent=2, ensure_ascii=False)
                    saved_files['clustering_json'] = str(clustering_file.name)
                
                # Save clustering CSV
                clustering_csv = self.metrics_dir / f"{base_filename}_clustering.csv"
                clustering_csv_data = self._extract_clustering_csv(results['clustering'], input_name, model_name, method_name, timestamp)
                if clustering_csv_data:
                    # Get all possible fieldnames from all rows
                    all_fieldnames = set()
                    for row in clustering_csv_data:
                        all_fieldnames.update(row.keys())
                    
                    # Ensure all rows have all fields
                    for row in clustering_csv_data:
                        for field in all_fieldnames:
                            if field not in row:
                                row[field] = None
                    
                    with open(clustering_csv, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=sorted(all_fieldnames))
                        writer.writeheader()
                        writer.writerows(clustering_csv_data)
                    saved_files['clustering_csv'] = str(clustering_csv.name)
            
            # 4. Save branching metrics separately (JSON + CSV)
            if 'branching' in results:
                # Save branching JSON (optional)
                if save_json:
                    branching_file = self.metrics_dir / f"{base_filename}_branching.json"
                    with open(branching_file, 'w', encoding='utf-8') as f:
                        branching_data = self._prepare_for_json(results['branching'])
                        branching_data['metadata'] = {
                            'input_name': input_name,
                            'model_name': model_name,
                            'method_name': method_name,
                            'timestamp': timestamp
                        }
                        json.dump(branching_data, f, indent=2, ensure_ascii=False)
                    saved_files['branching_json'] = str(branching_file.name)
                
                # Save branching CSV
                branching_csv = self.metrics_dir / f"{base_filename}_branching.csv"
                branching_csv_data = self._extract_branching_csv(results['branching'], input_name, model_name, method_name, timestamp)
                if branching_csv_data:
                    # Get all possible fieldnames from all rows
                    all_fieldnames = set()
                    for row in branching_csv_data:
                        all_fieldnames.update(row.keys())
                    
                    # Ensure all rows have all fields
                    for row in branching_csv_data:
                        for field in all_fieldnames:
                            if field not in row:
                                row[field] = None
                    
                    with open(branching_csv, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=sorted(all_fieldnames))
                        writer.writeheader()
                        writer.writerows(branching_csv_data)
                    saved_files['branching_csv'] = str(branching_csv.name)
            
            # 5. Save void metrics separately (JSON + CSV)
            if 'void' in results:
                # Save void JSON (optional)
                if save_json:
                    void_file = self.metrics_dir / f"{base_filename}_void.json"
                    with open(void_file, 'w', encoding='utf-8') as f:
                        void_data = self._prepare_for_json(results['void'])
                        void_data['metadata'] = {
                            'input_name': input_name,
                            'model_name': model_name,
                            'method_name': method_name,
                            'timestamp': timestamp
                        }
                        json.dump(void_data, f, indent=2, ensure_ascii=False)
                    saved_files['void_json'] = str(void_file.name)
                
                # Save void CSV
                void_csv = self.metrics_dir / f"{base_filename}_void.csv"
                void_csv_data = self._extract_void_csv(results['void'], input_name, model_name, method_name, timestamp)
                if void_csv_data:
                    # Get all possible fieldnames from all rows
                    all_fieldnames = set()
                    for row in void_csv_data:
                        all_fieldnames.update(row.keys())
                    
                    # Ensure all rows have all fields
                    for row in void_csv_data:
                        for field in all_fieldnames:
                            if field not in row:
                                row[field] = None
                    
                    with open(void_csv, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=sorted(all_fieldnames))
                        writer.writeheader()
                        writer.writerows(void_csv_data)
                    saved_files['void_csv'] = str(void_csv.name)
                
        except Exception as e:
            st.error(f"Error saving metrics: {str(e)}")
            
        return saved_files
    
    def _prepare_for_json(self, data: Any) -> Any:
        """Recursively prepare data for JSON serialization"""
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, dict):
            return {key: self._prepare_for_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_json(item) for item in data]
        elif isinstance(data, nx.Graph):  # NetworkX graphs
            try:
                # Convert to node-link format which is JSON serializable
                return {
                    'graph_type': 'networkx_graph',
                    'nodes': list(data.nodes()),
                    'edges': list(data.edges()),
                    'node_count': data.number_of_nodes(),
                    'edge_count': data.number_of_edges(),
                    'is_directed': data.is_directed()
                }
            except Exception as e:
                return {
                    'graph_type': 'networkx_graph_error',
                    'error': str(e),
                    'node_count': data.number_of_nodes() if hasattr(data, 'number_of_nodes') else 0,
                    'edge_count': data.number_of_edges() if hasattr(data, 'number_of_edges') else 0
                }
        elif hasattr(data, 'tolist'):  # Other array-like objects
            try:
                return data.tolist()
            except:
                return str(data)
        elif hasattr(data, '__dict__'):  # Other objects with attributes
            try:
                return str(data)
            except:
                return 'unserializable_object'
        else:
            return data
    
    def _extract_summary_metrics(self, results: Dict) -> Dict[str, Any]:
        """Extract key metrics for summary CSV"""
        summary = {}
        
        # Clustering metrics
        if 'clustering' in results:
            clustering = results['clustering']
            if 'basic_metrics' in clustering:
                basic = clustering['basic_metrics']
                summary['silhouette_score'] = basic.get('silhouette_score', 0)
                summary['davies_bouldin_score'] = basic.get('davies_bouldin_score', 0)
                summary['inertia'] = basic.get('inertia', 0)
                summary['num_clusters'] = len(np.unique(basic.get('cluster_labels', []))) if 'cluster_labels' in basic else 0
            
            if 'density_metrics' in clustering:
                density = clustering['density_metrics']
                summary['mean_density'] = density.get('mean_density', 0)
                summary['density_std'] = density.get('std_density', 0)
                summary['min_density'] = density.get('min_density', 0)
                summary['max_density'] = density.get('max_density', 0)
            
            if 'boundary_metrics' in clustering:
                boundary = clustering['boundary_metrics']
                summary['mean_hull_area'] = boundary.get('mean_hull_area', 0)
                summary['total_hull_area'] = boundary.get('total_hull_area', 0)
            
            if 'coherence_metrics' in clustering:
                coherence = clustering['coherence_metrics']
                summary['language_coherence'] = coherence.get('mean_language_coherence', 0)
        
        # Branching metrics
        if 'branching' in results:
            branching = results['branching']
            
            if 'connectivity_graph' in branching:
                conn = branching['connectivity_graph']
                summary['connected_components'] = conn.get('num_components', 0)
                summary['total_edges'] = conn.get('num_edges', 0)
                summary['largest_component'] = conn.get('largest_component_size', 0)
                summary['clustering_coefficient'] = conn.get('clustering_coefficient', 0)
            
            if 'linearity_scores' in branching:
                linearity = branching['linearity_scores']
                summary['digit_linearity'] = linearity.get('digit_sequence_linearity', 0) or 0
                summary['overall_linearity'] = linearity.get('overall_linearity', 0)
            
            if 'topology_metrics' in branching:
                topo = branching['topology_metrics']
                summary['graph_density'] = topo.get('density', 0)
                summary['mean_degree'] = topo.get('mean_degree', 0)
                summary['num_hubs'] = topo.get('num_hubs', 0)
            
            if 'ordering_analysis' in branching:
                ordering = branching['ordering_analysis']
                digit_patterns = ordering.get('digit_patterns', {})
                alpha_patterns = ordering.get('alphabetical_patterns', {})
                summary['digits_found'] = digit_patterns.get('num_digits', 0)
                summary['digit_consistency'] = digit_patterns.get('sequential_consistency', 0) or 0
                summary['letters_found'] = alpha_patterns.get('num_letters', 0)
                summary['letter_consistency'] = alpha_patterns.get('sequential_consistency', 0) or 0
        
        # Void metrics
        if 'void' in results:
            void = results['void']
            
            if 'void_regions' in void:
                voids = void['void_regions']
                summary['num_voids'] = voids.get('num_voids', 0)
                summary['mean_void_distance'] = voids.get('mean_void_distance', 0)
            
            if 'significance_results' in void:
                sig = void['significance_results']
                if sig:  # Check if not empty
                    summary['chi_statistic'] = sig.get('chi_statistic', 0)
                    summary['p_value'] = sig.get('p_value', 1)
                    summary['void_significant'] = sig.get('is_significant', False)
            
            if 'void_metrics' in void:
                void_metrics = void['void_metrics']
                summary['void_area_estimate'] = void_metrics.get('total_void_area_estimate', 0)
                summary['max_void_distance'] = void_metrics.get('max_void_distance', 0)
        
        return summary
    
    def _extract_clustering_csv(self, clustering_results: Dict, input_name: str, 
                               model_name: str, method_name: str, timestamp: str) -> List[Dict]:
        """Extract clustering metrics into CSV-friendly format"""
        rows = []
        
        # Basic metrics row
        basic = clustering_results.get('basic_metrics', {})
        if basic:
            row = {
                'input_name': input_name,
                'model_name': model_name,
                'method_name': method_name,
                'timestamp': timestamp,
                'metric_type': 'basic_clustering',
                'silhouette_score': basic.get('silhouette_score', 0),
                'davies_bouldin_score': basic.get('davies_bouldin_score', 0),
                'inertia': basic.get('inertia', 0),
                'num_clusters': len(np.unique(basic.get('cluster_labels', []))) if 'cluster_labels' in basic else 0
            }
            rows.append(row)
        
        # Density metrics row
        density = clustering_results.get('density_metrics', {})
        if density:
            row = {
                'input_name': input_name,
                'model_name': model_name,
                'method_name': method_name,
                'timestamp': timestamp,
                'metric_type': 'density_analysis',
                'mean_density': density.get('mean_density', 0),
                'density_std': density.get('std_density', 0),
                'min_density': density.get('min_density', 0),
                'max_density': density.get('max_density', 0)
            }
            rows.append(row)
        
        # Boundary metrics row
        boundary = clustering_results.get('boundary_metrics', {})
        if boundary:
            row = {
                'input_name': input_name,
                'model_name': model_name,
                'method_name': method_name,
                'timestamp': timestamp,
                'metric_type': 'boundary_analysis',
                'mean_hull_area': boundary.get('mean_hull_area', 0),
                'total_hull_area': boundary.get('total_hull_area', 0),
                'hull_areas': ';'.join(map(str, boundary.get('hull_areas', []))),
                'hull_perimeters': ';'.join(map(str, boundary.get('hull_perimeters', [])))
            }
            rows.append(row)
        
        # Coherence metrics row
        coherence = clustering_results.get('coherence_metrics', {})
        if coherence:
            row = {
                'input_name': input_name,
                'model_name': model_name,
                'method_name': method_name,
                'timestamp': timestamp,
                'metric_type': 'language_coherence',
                'mean_language_coherence': coherence.get('mean_language_coherence', 0),
                'language_separation_scores': ';'.join(map(str, coherence.get('language_separation_scores', [])))
            }
            rows.append(row)
        
        return rows
    
    def _extract_branching_csv(self, branching_results: Dict, input_name: str,
                              model_name: str, method_name: str, timestamp: str) -> List[Dict]:
        """Extract branching metrics into CSV-friendly format"""
        rows = []
        
        # Connectivity graph metrics row
        conn = branching_results.get('connectivity_graph', {})
        if conn:
            row = {
                'input_name': input_name,
                'model_name': model_name,
                'method_name': method_name,
                'timestamp': timestamp,
                'metric_type': 'connectivity_graph',
                'num_components': conn.get('num_components', 0),
                'num_edges': conn.get('num_edges', 0),
                'largest_component_size': conn.get('largest_component_size', 0),
                'clustering_coefficient': conn.get('clustering_coefficient', 0)
            }
            rows.append(row)
        
        # Linearity scores row
        linearity = branching_results.get('linearity_scores', {})
        if linearity:
            row = {
                'input_name': input_name,
                'model_name': model_name,
                'method_name': method_name,
                'timestamp': timestamp,
                'metric_type': 'linearity_analysis',
                'digit_sequence_linearity': linearity.get('digit_sequence_linearity', 0) or 0,
                'overall_linearity': linearity.get('overall_linearity', 0),
                'digit_indices': ';'.join(map(str, linearity.get('digit_indices', [])))
            }
            rows.append(row)
        
        # Topology metrics row
        topo = branching_results.get('topology_metrics', {})
        if topo:
            row = {
                'input_name': input_name,
                'model_name': model_name,
                'method_name': method_name,
                'timestamp': timestamp,
                'metric_type': 'topology_analysis',
                'graph_density': topo.get('density', 0),
                'transitivity': topo.get('transitivity', 0),
                'mean_degree': topo.get('mean_degree', 0),
                'max_degree': topo.get('max_degree', 0),
                'num_hubs': topo.get('num_hubs', 0),
                'hub_nodes': ';'.join(map(str, topo.get('hub_nodes', [])))
            }
            rows.append(row)
        
        # Ordering analysis rows
        ordering = branching_results.get('ordering_analysis', {})
        if ordering:
            # Digit patterns
            digit_patterns = ordering.get('digit_patterns', {})
            if digit_patterns and 'num_digits' in digit_patterns:
                row = {
                    'input_name': input_name,
                    'model_name': model_name,
                    'method_name': method_name,
                    'timestamp': timestamp,
                    'metric_type': 'digit_ordering',
                    'found_digits': ';'.join(map(str, digit_patterns.get('found_digits', []))),
                    'num_digits': digit_patterns.get('num_digits', 0),
                    'sequential_consistency': digit_patterns.get('sequential_consistency', 0),
                    'mean_step_distance': digit_patterns.get('mean_step_distance', 0),
                    'step_distance_std': digit_patterns.get('step_distance_std', 0)
                }
                rows.append(row)
            
            # Alphabetical patterns
            alpha_patterns = ordering.get('alphabetical_patterns', {})
            if alpha_patterns and 'num_letters' in alpha_patterns:
                row = {
                    'input_name': input_name,
                    'model_name': model_name,
                    'method_name': method_name,
                    'timestamp': timestamp,
                    'metric_type': 'alphabetical_ordering',
                    'found_letters': ';'.join(alpha_patterns.get('found_letters', [])),
                    'num_letters': alpha_patterns.get('num_letters', 0),
                    'sequential_consistency': alpha_patterns.get('sequential_consistency', 0),
                    'mean_step_distance': alpha_patterns.get('mean_step_distance', 0)
                }
                rows.append(row)
        
        return rows
    
    def _extract_void_csv(self, void_results: Dict, input_name: str,
                         model_name: str, method_name: str, timestamp: str) -> List[Dict]:
        """Extract void metrics into CSV-friendly format"""
        rows = []
        
        # Void regions row
        voids = void_results.get('void_regions', {})
        if voids:
            row = {
                'input_name': input_name,
                'model_name': model_name,
                'method_name': method_name,
                'timestamp': timestamp,
                'metric_type': 'void_regions',
                'num_voids': voids.get('num_voids', 0),
                'mean_void_distance': voids.get('mean_void_distance', 0)
            }
            rows.append(row)
        
        # Statistical significance row
        sig = void_results.get('significance_results', {})
        if sig and sig:  # Check if not empty
            row = {
                'input_name': input_name,
                'model_name': model_name,
                'method_name': method_name,
                'timestamp': timestamp,
                'metric_type': 'void_significance',
                'chi_statistic': sig.get('chi_statistic', 0),
                'p_value': sig.get('p_value', 1),
                'is_significant': sig.get('is_significant', False),
                'observed_voids': sig.get('observed_voids', 0),
                'expected_voids': sig.get('expected_voids', 0),
                'significance_level': sig.get('significance_level', 0.05)
            }
            rows.append(row)
        
        # Void geometry metrics row
        void_metrics = void_results.get('void_metrics', {})
        if void_metrics:
            row = {
                'input_name': input_name,
                'model_name': model_name,
                'method_name': method_name,
                'timestamp': timestamp,
                'metric_type': 'void_geometry',
                'total_void_area_estimate': void_metrics.get('total_void_area_estimate', 0),
                'max_void_distance': void_metrics.get('max_void_distance', 0),
                'void_distance_std': void_metrics.get('void_distance_std', 0)
            }
            rows.append(row)
        
        return rows
    
    def load_metrics_history(self, input_name: str = None, model_name: str = None) -> List[Dict]:
        """Load historical metrics for comparison"""
        history = []
        
        if not self.metrics_dir.exists():
            return history
        
        # Find all summary CSV files
        pattern = "*_summary.csv"
        if input_name:
            safe_input = self.sanitize_filename(input_name)
            pattern = f"{safe_input}*_summary.csv"
        
        for csv_file in self.metrics_dir.glob(pattern):
            try:
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    if model_name is None or row.get('model_name') == model_name:
                        history.append(row.to_dict())
            except Exception as e:
                st.warning(f"Could not load {csv_file.name}: {str(e)}")
        
        return history
    
    def display_metrics_save_status(self, saved_files: Dict[str, str]):
        """Display information about saved metrics files"""
        if saved_files:
            # st.success("📁 Metrics saved successfully!")
            
            with st.expander("📋 Saved Files Details", expanded=False):
                for file_type, filename in saved_files.items():
                    file_description = {
                        'complete_json': '🗂️ Complete analysis results (JSON)',
                        'summary_csv': '📊 Key metrics summary (CSV)',
                        'clustering_json': '🔍 Clustering analysis detailed (JSON)',
                        'clustering_csv': '🔍 Clustering analysis (CSV)',
                        'branching_json': '🌿 Branching analysis detailed (JSON)',
                        'branching_csv': '🌿 Branching analysis (CSV)',
                        'void_json': '🕳️ Void analysis detailed (JSON)',
                        'void_csv': '🕳️ Void analysis (CSV)'
                    }
                    
                    desc = file_description.get(file_type, f'📄 {file_type}')
                    st.text(f"{desc}: {filename}")
                
                st.info(f"📂 All files saved to: {self.metrics_dir}")
    
    def create_metrics_comparison_plot(self, history: List[Dict], 
                                     current_results: Dict) -> go.Figure:
        """Create a plot comparing current metrics with historical data"""
        if not history:
            st.warning("No historical data available for comparison")
            return None
        
        # Extract key metrics for comparison
        comparison_metrics = [
            'silhouette_score', 'davies_bouldin_score', 'overall_linearity',
            'language_coherence', 'num_voids', 'clustering_coefficient'
        ]
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=comparison_metrics,
            vertical_spacing=0.1
        )
        
        for i, metric in enumerate(comparison_metrics):
            row = (i // 3) + 1
            col = (i % 3) + 1
            
            # Historical values
            hist_values = [item.get(metric, 0) for item in history if metric in item]
            hist_timestamps = [item.get('timestamp', '') for item in history if metric in item]
            
            # Current value
            current_summary = self._extract_summary_metrics(current_results)
            current_value = current_summary.get(metric, 0)
            
            # Add historical data
            if hist_values:
                fig.add_trace(go.Scatter(
                    x=list(range(len(hist_values))),
                    y=hist_values,
                    mode='lines+markers',
                    name=f'Historical {metric}',
                    showlegend=False
                ), row=row, col=col)
            
            # Add current value
            fig.add_trace(go.Scatter(
                x=[len(hist_values)],
                y=[current_value],
                mode='markers',
                marker=dict(size=12, color='red'),
                name=f'Current {metric}',
                showlegend=False
            ), row=row, col=col)
        
        fig.update_layout(
            title="Metrics Comparison: Current vs Historical",
            height=600,
            showlegend=False
        )
        
        return fig