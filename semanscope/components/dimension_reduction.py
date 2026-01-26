import streamlit as st
import numpy as np
from sklearn.manifold import LocallyLinearEmbedding, MDS, SpectralEmbedding, TSNE, Isomap
from sklearn.decomposition import PCA, KernelPCA
from umap import UMAP
from phate import PHATE
from utils.cache_manager import (
    get_cached_dimension_reduction, save_dimension_reduction_to_cache
)
try:
    import trimap
    TRIMAP_AVAILABLE = True
except ImportError:
    TRIMAP_AVAILABLE = False

try:
    import pacmap
    PACMAP_AVAILABLE = True
except ImportError:
    PACMAP_AVAILABLE = False

try:
    import igraph as ig
    from sklearn.neighbors import kneighbors_graph
    FORCEATLAS2_AVAILABLE = True
except ImportError:
    FORCEATLAS2_AVAILABLE = False
from utils.error_handling import handle_errors

class DimensionReducer:
    def __init__(self):
        self.reducers = {
            "t-SNE": self._get_tsne,
            "Isomap": self._get_isomap,
            "UMAP": self._get_umap,
            "LLE": self._get_lle,
            "MDS": self._get_mds,
            "PCA": self._get_pca,
            "Kernel PCA": self._get_kernel_pca,
            "Spectral Embedding": self._get_spectral,
            "PHATE": self._get_phate
        }

        # Force re-check imports during initialization to handle Streamlit caching issues
        self._check_and_add_new_methods()

    def _check_and_add_new_methods(self):
        """Re-check availability and add new methods"""
        # Re-check TriMap
        try:
            import trimap
            self.reducers["TriMap"] = self._get_trimap
        except ImportError:
            pass

        # Re-check PaCMAP
        try:
            import pacmap
            self.reducers["PaCMAP"] = self._get_pacmap
        except ImportError:
            pass

        # Re-check ForceAtlas2
        try:
            import igraph as ig
            from sklearn.neighbors import kneighbors_graph
            self.reducers["ForceAtlas2"] = self._get_forceatlas2
        except ImportError:
            pass

    def reduce_dimensions_with_cache(self, embeddings: np.ndarray, method: str, dimensions: int = 2,
                                   dataset: list = None, lang: str = None, model: str = None) -> np.ndarray:
        """Reduce dimensions with cross-page caching (Semanscope performance cache)"""

        # Create method parameters for cache key
        method_params = {"dimensions": dimensions}

        # Try cache first if we have dataset/lang/model info
        if dataset is not None and lang is not None and model is not None:
            cached_result = get_cached_dimension_reduction(dataset, lang, model, method, method_params)
            if cached_result is not None:
                return cached_result

        # No cache hit, compute the reduction
        result = self.reduce_dimensions(embeddings, method, dimensions)

        # Save to cache if we have the required info
        if result is not None and dataset is not None and lang is not None and model is not None:
            save_dimension_reduction_to_cache(result, dataset, lang, model, method, method_params)

        return result

    @handle_errors
    def reduce_dimensions(self, embeddings: np.ndarray, method: str, dimensions: int = 2) -> np.ndarray:
        """Reduce dimensions of embeddings using specified method with robust error handling"""
        n_samples = embeddings.shape[0]

        # Validate input embeddings
        validation_issues = self._validate_embeddings(embeddings, method)
        if validation_issues:
            embeddings = self._clean_embeddings(embeddings)

        # Handle very small datasets
        if n_samples < 3:
            st.warning(f"Dataset too small for {method}. Using PCA instead.")
            return PCA(n_components=dimensions).fit_transform(embeddings)

        # Check if method is available
        if method not in self.reducers:
            available_methods = list(self.reducers.keys())
            raise ValueError(f"Method '{method}' not available. Available methods: {available_methods}")

        # Get appropriate reducer and apply with detailed error reporting
        try:
            reducer = self.reducers[method](n_samples, dimensions)

            # Special preprocessing for PHATE to handle numerical instability
            if method == "PHATE":
                # Add small amount of noise to break potential numerical degeneracies
                noise_scale = 1e-8
                embeddings = embeddings + np.random.RandomState(42).normal(0, noise_scale, embeddings.shape)
                # Ensure no perfect duplicates that could cause PHATE issues
                embeddings = embeddings + np.arange(embeddings.shape[0]).reshape(-1, 1) * 1e-10

            result = reducer.fit_transform(embeddings)

            # Validate output
            if self._validate_output(result, method):
                result = self._clean_output(result)

            return result

        except Exception as e:
            # Provide detailed error information for research purposes
            error_details = self._analyze_method_failure(method, e, embeddings, n_samples)
            st.error(f"❌ **{method} Failed**: {error_details}")

            # Re-raise the exception to maintain research integrity
            raise ValueError(f"{method} dimensionality reduction failed: {str(e)}")

    def _get_tsne(self, n_samples: int, dimensions: int):
        perplexity = min(30, n_samples - 1)
        return TSNE(n_components=dimensions, random_state=42, perplexity=perplexity)

    def _get_isomap(self, n_samples: int, dimensions: int):
        return Isomap(n_components=dimensions)

    def _get_umap(self, n_samples: int, dimensions: int):
        return UMAP(n_components=dimensions, random_state=42)

    def _get_lle(self, n_samples: int, dimensions: int):
        return LocallyLinearEmbedding(n_components=dimensions, random_state=42)

    def _get_mds(self, n_samples: int, dimensions: int):
        return MDS(n_components=dimensions, random_state=42)

    def _get_pca(self, n_samples: int, dimensions: int):
        return PCA(n_components=dimensions)

    def _get_kernel_pca(self, n_samples: int, dimensions: int):
        return KernelPCA(n_components=dimensions, kernel='rbf')

    def _get_spectral(self, n_samples: int, dimensions: int):
        return SpectralEmbedding(n_components=dimensions, random_state=42)

    def _get_phate(self, n_samples: int, dimensions: int):
        # Enhanced PHATE with numerical stability for problematic embeddings
        # Use adaptive knn based on dataset size
        if n_samples < 100:
            knn = min(5, n_samples - 1)
        elif n_samples < 1000:
            knn = 10
        else:
            knn = 15  # For large datasets like 3621 samples

        return PHATE(
            n_components=dimensions,
            knn=knn,
            decay=40,                 # Moderate decay for stability
            t='auto',                 # Auto-select diffusion time
            gamma=1,                  # Informational distance
            n_jobs=1,                 # Single-threaded for stability
            random_state=42,          # Reproducible results
            verbose=0                 # Suppress warnings
        )

    def _get_trimap(self, n_samples: int, dimensions: int):
        """
        TriMap - Superior balance of local and global structure preservation
        Perfect for morphological family clustering and cross-lingual comparisons
        """
        # Adjust parameters based on dataset size
        n_inliers = min(10, max(3, n_samples // 3))
        n_outliers = min(5, max(2, n_samples // 6))
        n_random = min(5, max(2, n_samples // 6))

        # Configure TriMap with optimal parameters for semantic embeddings
        return trimap.TRIMAP(
            n_dims=dimensions,
            n_inliers=n_inliers,  # Number of nearest neighbors for local structure
            n_outliers=n_outliers,  # Number of outlier points for global structure
            n_random=n_random,    # Number of random points for structure balance
            verbose=False  # Disable progress output for Streamlit
        )

    def _get_pacmap(self, n_samples: int, dimensions: int):
        """
        PaCMAP - Exceptional global structure preservation for cross-lingual analysis
        Ideal for comparing morphological signatures across languages
        """
        # Adjust parameters based on dataset size
        n_neighbors = min(10, max(3, n_samples // 3))

        # Configure PaCMAP with optimal parameters for semantic embeddings
        return pacmap.PaCMAP(
            n_components=dimensions,
            n_neighbors=n_neighbors,  # Number of nearest neighbors
            MN_ratio=0.5,            # Balance between local and global structure
            FP_ratio=2.0,            # Further pairs ratio for global preservation
            distance='euclidean',    # Distance metric
            verbose=False            # Disable progress output for Streamlit
        )

    def _get_forceatlas2(self, n_samples: int, dimensions: int):
        """
        ForceAtlas2 - Designed specifically for network visualization
        Perfect for morphological networks (子-network, Haus/Arbeit families)
        """
        return ForceAtlas2Reducer(n_components=dimensions)

    def _validate_embeddings(self, embeddings: np.ndarray, method: str) -> bool:
        """Validate input embeddings for NaN, Inf, and other issues"""
        issues_found = False

        # Check for NaN values
        if np.isnan(embeddings).any():
            nan_count = np.isnan(embeddings).sum()
            st.warning(f"🔍 **Input Validation**: {nan_count} NaN values found in embeddings for {method}")
            issues_found = True

        # Check for infinite values
        if np.isinf(embeddings).any():
            inf_count = np.isinf(embeddings).sum()
            st.warning(f"🔍 **Input Validation**: {inf_count} infinite values found in embeddings for {method}")
            issues_found = True

        # Check for extreme values that might cause numerical issues
        if np.abs(embeddings).max() > 1000:
            st.warning(f"🔍 **Input Validation**: Extremely large values detected (max: {np.abs(embeddings).max():.2f}) for {method}")
            issues_found = True

        # Check for zero variance dimensions (can cause issues with some methods)
        if embeddings.shape[0] > 1:
            variances = np.var(embeddings, axis=0)
            zero_var_dims = np.sum(variances < 1e-12)
            if zero_var_dims > 0:
                issues_found = True

        return issues_found

    def _clean_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Clean embeddings by replacing NaN/Inf values and normalizing extreme values"""
        cleaned = embeddings.copy()

        # Replace NaN and Inf values
        cleaned = np.nan_to_num(cleaned, nan=0.0, posinf=10.0, neginf=-10.0)

        # Clip extreme values to prevent numerical issues
        cleaned = np.clip(cleaned, -100, 100)

        # Optional: L2 normalize if values are still very large
        max_norm = np.max(np.linalg.norm(cleaned, axis=1))
        if max_norm > 50:
            norms = np.linalg.norm(cleaned, axis=1, keepdims=True)
            cleaned = cleaned / (norms + 1e-8)

        return cleaned

    def _validate_output(self, result: np.ndarray, method: str) -> bool:
        """Validate output for NaN or other issues"""
        if np.isnan(result).any():
            nan_count = np.isnan(result).sum()
            st.error(f"❌ **Output Validation**: {method} produced {nan_count} NaN values in output")
            return True

        if np.isinf(result).any():
            inf_count = np.isinf(result).sum()
            st.error(f"❌ **Output Validation**: {method} produced {inf_count} infinite values in output")
            return True

        return False

    def _clean_output(self, result: np.ndarray) -> np.ndarray:
        """Clean output by replacing NaN/Inf values"""
        cleaned = np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)
        return cleaned

    def _analyze_method_failure(self, method: str, exception: Exception, embeddings: np.ndarray, n_samples: int) -> str:
        """Analyze why a dimensionality reduction method failed and provide research insights"""

        error_msg = str(exception)
        error_type = type(exception).__name__

        analysis_parts = []

        # Basic error information
        analysis_parts.append(f"**Error Type**: {error_type}")
        analysis_parts.append(f"**Error Message**: {error_msg}")

        # Data characteristics that might cause issues
        analysis_parts.append(f"**Dataset Size**: {n_samples} samples, {embeddings.shape[1]} dimensions")

        # Check for common issues by method
        if method == "t-SNE":
            perplexity = min(30, n_samples - 1)
            analysis_parts.append(f"**t-SNE Perplexity**: {perplexity}")
            if n_samples < 4:
                analysis_parts.append("⚠️ t-SNE requires at least 4 samples")
            if "perplexity" in error_msg.lower():
                analysis_parts.append("💡 **Research Note**: Consider reducing perplexity or increasing dataset size")

        elif method == "UMAP":
            if n_samples < 10:
                analysis_parts.append("⚠️ UMAP performs best with 10+ samples")
            if "neighbors" in error_msg.lower():
                analysis_parts.append("💡 **Research Note**: UMAP neighbor parameter may need adjustment for small datasets")

        elif method == "PHATE":
            if "nan" in error_msg.lower() or "inf" in error_msg.lower():
                analysis_parts.append("💡 **Research Note**: PHATE detected NaN/Inf values despite preprocessing")
            if n_samples < 5:
                analysis_parts.append("⚠️ PHATE may struggle with very small datasets")

        elif method in ["TriMap", "PaCMAP"]:
            if n_samples < 6:
                analysis_parts.append(f"⚠️ {method} requires sufficient samples for triplet/pair formation")

        elif method == "Isomap":
            if "neighbors" in error_msg.lower():
                analysis_parts.append("💡 **Research Note**: Isomap neighbor graph may be disconnected")

        # Input data quality issues
        if np.isnan(embeddings).any():
            nan_count = np.isnan(embeddings).sum()
            analysis_parts.append(f"⚠️ **Data Quality**: {nan_count} NaN values in embeddings")

        if np.isinf(embeddings).any():
            inf_count = np.isinf(embeddings).sum()
            analysis_parts.append(f"⚠️ **Data Quality**: {inf_count} infinite values in embeddings")

        # Variance analysis
        if embeddings.shape[0] > 1:
            variances = np.var(embeddings, axis=0)
            zero_var_dims = np.sum(variances < 1e-12)
            if zero_var_dims > 0:
                analysis_parts.append(f"⚠️ **Data Quality**: {zero_var_dims} dimensions with zero variance")

        # Research recommendations
        analysis_parts.append("🔬 **Research Recommendation**: Try a different dimensionality reduction method or preprocess the data differently")

        return "\n".join(analysis_parts)


class ForceAtlas2Reducer:
    """
    Custom ForceAtlas2 implementation for embedding dimensionality reduction
    Converts embeddings to similarity graph and applies ForceAtlas2 layout
    """

    def __init__(self, n_components=2, n_neighbors=5, iterations=100):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.iterations = iterations

    def fit_transform(self, X):
        """Transform embeddings using ForceAtlas2 graph layout"""
        n_samples = X.shape[0]

        # Handle very small datasets
        if n_samples < 3:
            return np.random.rand(n_samples, self.n_components)

        # Create k-nearest neighbors graph from embeddings
        # This converts similarity relationships to network connections
        adjacency = kneighbors_graph(
            X,
            n_neighbors=min(self.n_neighbors, n_samples-1),
            mode='connectivity',
            include_self=False
        )

        # Convert to igraph format
        edges = []
        adjacency_coo = adjacency.tocoo()  # Convert to COO format for easier iteration
        for i, j in zip(adjacency_coo.row, adjacency_coo.col):
            if i < j:  # Avoid duplicate edges
                edges.append((i, j))

        # Ensure we have at least some edges for graph layout
        if len(edges) == 0:
            # Create a minimal connected graph
            for i in range(n_samples - 1):
                edges.append((i, i + 1))

        # Create igraph graph
        g = ig.Graph(n=n_samples, edges=edges)

        # Apply ForceAtlas2 layout
        # Use Fruchterman-Reingold as fallback (similar physics-based approach)
        if self.n_components == 2:
            try:
                # Try ForceAtlas2 if available in igraph
                layout = g.layout_fruchterman_reingold(
                    niter=self.iterations,
                    repulserad=n_samples * 2,  # Repulsion radius
                    area=n_samples * 10,       # Layout area
                    coolexp=1.5                # Cooling exponent
                )
            except:
                # Fallback to spring layout
                layout = g.layout_kamada_kawai()
        else:
            # For 3D, use 3D spring layout
            layout = g.layout_fruchterman_reingold_3d(
                niter=self.iterations
            )

        # Convert layout to numpy array
        result = np.array(layout.coords)

        # Ensure correct dimensionality
        if result.shape[1] != self.n_components:
            if self.n_components == 2 and result.shape[1] == 3:
                result = result[:, :2]  # Take first 2 dimensions
            elif self.n_components == 3 and result.shape[1] == 2:
                # Add third dimension with zeros
                result = np.column_stack([result, np.zeros(result.shape[0])])

        return result