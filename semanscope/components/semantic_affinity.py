"""
Semantic Affinity Metric Computation

Implementation of the Semantic Affinity (SA) metric for benchmarking embedding models
based on cross-lingual translation alignment.

Reference: docs/semantic-affinity.md

Key Concepts:
- Horizontal spread: Semantic scale within each language (normalization baseline)
  * Computed on unique words only (flattened and deduplicated)
  * Averaged across all languages
  * Euclidean: RMS (root mean square) - natural L2 norm
  * Cosine: Mean - cosine distance already well-defined in [0,2]

- Vertical spread: Cross-lingual spread for translation pairs
  * Computed on expanded word pairs (cross-product of multi-meaning translations)
  * Measures how far apart translations are across languages
  * Same spread formula as horizontal (RMS for Euclidean, mean for cosine)

- Semantic Ratio (intermediate): SR = vertical_spread / horizontal_spread
  * Lower SR = stronger alignment (translations closer than baseline)
  * SR ≈ 1 = neutral (translations at vocabulary baseline)
  * Higher SR = weaker alignment (translations farther than baseline)

- Semantic Affinity (final): SA = 1 / (1 + SR)
  * SA ∈ (0, 1] - bounded, normalized metric
  * Higher SA (closer to 1) = stronger cross-lingual affinity
  * SA = 1.0: Perfect alignment (SR=0)
  * SA = 0.5: Neutral (SR=1, translations = baseline)
  * SA < 0.5: Weak affinity (SR>1, translations farther than baseline)
  * SA > 0.5: Strong affinity (SR<1, translations closer than baseline)

- Collapse detection: Flag models where max distance < epsilon

Distance Metric Notes:
- Euclidean uses RMS because it's the natural L2 norm for squared distances
- Cosine uses simple mean because (1 - cos θ) ∈ [0,2] is already a proper metric
  on the unit hypersphere; squaring would introduce non-linear distortion
- SA transformation normalizes both Euclidean and Cosine to [0,1] scale
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from functools import partial


def compute_vertical_spreads_vectorized(
    embeddings_dict: Dict[str, np.ndarray],
    language_codes: List[str],
    metric: str = 'euclidean',
    n_jobs: int = 1,
    max_batch_size: int = 1000
) -> np.ndarray:
    """
    MEMORY-EFFICIENT + PARALLEL computation of vertical spreads (cross-lingual distances).

    Distributes N word pairs across CPUs, processing in batches to avoid memory explosion.

    Memory optimization:
        - Process words in batches to avoid stacking all embeddings at once
        - For 8 languages × 37,481 words × 768 dims, full stack would be ~2GB
        - Batch processing keeps memory under control

    Orthogonal optimization strategy:
        - Horizontal axis: Parallelize across 8 languages (8 independent pdist calls)
        - Vertical axis: Parallelize across N/batch word pair batches (THIS function)

    Args:
        embeddings_dict: Dictionary of embeddings per language
            Example: {'chn': array(600, 768), 'enu': array(600, 768), ...}
        language_codes: List of language codes (e.g., 8 languages)
        metric: 'euclidean' or 'cosine'
        n_jobs: Number of CPUs (1=sequential, -1=all CPUs, 8=distribute 600 words to 8 CPUs)
        max_batch_size: Maximum number of words to process in one batch (default=1000)

    Returns:
        vertical_spreads: array(n_words,) - vertical spread for each word pair

    Performance (600 words, 8 languages):
        - Sequential loop: ~0.02s (600 iterations calling pdist)
        - Parallel (n_jobs=8): ~0.005s (8 CPUs each process 75 words)
        - Speedup: ~4-8× depending on CPU count
    """
    from scipy.spatial.distance import pdist
    from multiprocessing import cpu_count

    n_words = len(embeddings_dict[language_codes[0]])
    n_languages = len(language_codes)

    def process_word_batch(word_indices, embeddings_dict_local, language_codes_local, metric_local):
        """
        Process a batch of word pairs (e.g., words 0-999 on CPU 1)

        For each word pair:
            - Extract embeddings across L=8 languages
            - Compute pairwise distances (L choose 2 = 28 pairs)
            - Calculate spread (RMS or mean)

        Memory-efficient: Only stacks embeddings for this batch, not all words
        """
        batch_spreads = []

        # Stack embeddings for this batch only
        batch_size = len(word_indices)
        batch_embeddings = np.zeros((batch_size, n_languages, embeddings_dict_local[language_codes_local[0]].shape[1]))

        for lang_idx, lang in enumerate(language_codes_local):
            for batch_idx, word_idx in enumerate(word_indices):
                batch_embeddings[batch_idx, lang_idx, :] = embeddings_dict_local[lang][word_idx]

        for i in range(batch_size):
            # Get embeddings for this word across all languages (shape: L, D)
            word_embeddings = batch_embeddings[i]  # (8, 768)

            # Compute pairwise distances across languages
            distances = pdist(word_embeddings, metric=metric_local if metric_local == 'cosine' else 'euclidean')

            # Compute spread
            if metric_local == 'euclidean':
                spread = np.sqrt(np.mean(distances ** 2))  # RMS
            else:  # cosine
                spread = np.mean(distances)

            batch_spreads.append(spread)

        return batch_spreads

    # Determine number of workers
    if n_jobs == -1:
        n_workers = cpu_count()
    elif n_jobs > 1:
        n_workers = min(n_jobs, max(1, n_words // max_batch_size))
    else:
        n_workers = 1

    # Create batches (limit batch size to avoid memory issues)
    actual_batch_size = min(max_batch_size, max(1, (n_words + n_workers - 1) // n_workers))
    word_batches = [
        list(range(i, min(i + actual_batch_size, n_words)))
        for i in range(0, n_words, actual_batch_size)
    ]

    # Parallelize if beneficial
    if n_workers > 1 and n_words > 50 and len(word_batches) > 1:
        from joblib import Parallel, delayed

        # Process batches in parallel (use 'loky' for true multiprocessing, not 'threading')
        results = Parallel(n_jobs=n_workers, backend='loky')(
            delayed(process_word_batch)(batch, embeddings_dict, language_codes, metric)
            for batch in word_batches
        )

        # Flatten results
        vertical_spreads = np.array([spread for batch_result in results for spread in batch_result])
    else:
        # Sequential processing (single batch or small dataset)
        vertical_spreads = np.array(process_word_batch(range(n_words), embeddings_dict, language_codes, metric))

    return vertical_spreads


@dataclass
class SemanticAffinityResult:
    """Results from Semantic Affinity analysis

    Semantic Ratio (SR, intermediate) = vertical_spread / horizontal_spread
    - Lower SR = stronger alignment (translations closer than baseline)
    - SR < 1 = translations closer than vocabulary baseline
    - SR ≈ 1 = translations at vocabulary baseline (neutral)
    - SR > 1 = translations farther than vocabulary baseline

    Semantic Affinity (SA, final) = 1 / (1 + SR)
    - Bounded metric in (0, 1]
    - Higher SA (closer to 1) = stronger cross-lingual affinity
    - SA = 1.0: Perfect alignment (SR=0, translations perfectly aligned)
    - SA = 0.5: Neutral (SR=1, translations at baseline)
    - SA < 0.5: Weak affinity (SR>1, translations farther than baseline)
    - SA > 0.5: Strong affinity (SR<1, translations closer than baseline)

    Error propagation:
    - std: Standard deviation of per-word SA scores (spread of data)
    - sem: Standard error of mean SA (propagated from SR uncertainties)
    """
    score: float  # Semantic Affinity (SA) = 1 / (1 + SR), primary metric in [0, 1]
    semantic_ratio: float  # Semantic Ratio (SR), intermediate value for reference
    status: str  # "OK" or "COLLAPSED"
    horizontal_spread: float  # Semantic scale (normalization baseline)
    vertical_spread: float  # Average cross-lingual spread
    per_word_scores: Dict[str, float]  # Per-word SA scores
    outliers: List[Tuple[str, float]]  # Top outliers by SA score
    std: float  # Standard deviation of per-word SA scores
    sem: float  # Standard error of mean SA (propagated uncertainty)
    n_words: int
    n_languages: int
    per_language_spreads: Dict[str, float]  # Horizontal spread per language
    unique_word_counts: Dict[str, int]  # Unique words per language
    horizontal_spread_sem: float  # Standard error of horizontal spread
    vertical_spread_sem: float  # Standard error of vertical spread


class SemanticAffinityAnalyzer:
    """Compute Semantic Affinity metric for embedding models"""

    def __init__(self, collapse_epsilon: float = 1e-6, n_jobs: int = -1, use_vectorized: bool = True, max_batch_size: int = 1000):
        """
        Args:
            collapse_epsilon: Threshold for detecting collapsed embeddings
            n_jobs: Number of parallel jobs (-1 = all CPUs, 1 = sequential)
            use_vectorized: Use vectorized computation (15-120× faster, default=True)
                           Set to False for cross-validation or debugging
            max_batch_size: Maximum number of words to process in one batch (default=1000)
                           Lower values use less memory but may be slower
        """
        self.collapse_epsilon = collapse_epsilon
        self.n_jobs = cpu_count() if n_jobs == -1 else max(1, n_jobs)
        self.use_vectorized = use_vectorized
        self.max_batch_size = max_batch_size

    def compute_pairwise_distances(self, embeddings: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
        """
        Compute pairwise distances between all embeddings

        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
            metric: 'euclidean' for L2 distance, 'cosine' for cosine distance

        Returns:
            Array of pairwise distances
        """
        if metric == 'euclidean':
            # Compute Euclidean distances
            # d[i,j] = ||e[i] - e[j]||
            from scipy.spatial.distance import pdist
            return pdist(embeddings, metric='euclidean')
        elif metric == 'cosine':
            # Compute cosine similarity, then convert to distance
            # cosine_distance = 1 - cosine_similarity
            from scipy.spatial.distance import pdist
            return pdist(embeddings, metric='cosine')
        else:
            raise ValueError(f"Unsupported metric: {metric}. Use 'euclidean' or 'cosine'")

    def compute_spread(self, distances: np.ndarray, metric: str = 'euclidean') -> float:
        """
        Compute spread of distances

        For Euclidean: Use RMS (root mean square) - natural L2 norm
        For Cosine: Use simple mean - cosine distance already well-defined in [0,2]

        Args:
            distances: Array of pairwise distances
            metric: 'euclidean' or 'cosine'

        Returns:
            Spread value (RMS for Euclidean, mean for cosine)
        """
        if len(distances) == 0:
            return 0.0

        if metric == 'euclidean':
            # RMS: Natural for squared Euclidean distances
            return np.sqrt(np.mean(distances ** 2))
        elif metric == 'cosine':
            # Simple mean: Cosine distance (1 - cos θ) already in [0,2], no squaring needed
            return np.mean(distances)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def check_collapse(self, embeddings: np.ndarray) -> Tuple[bool, float]:
        """
        Check if embedding space has collapsed

        Args:
            embeddings: Array of shape (n_samples, embedding_dim)

        Returns:
            (is_collapsed, max_distance)
        """
        distances = self.compute_pairwise_distances(embeddings)
        max_distance = np.max(distances) if len(distances) > 0 else 0.0
        is_collapsed = max_distance < self.collapse_epsilon
        return is_collapsed, max_distance

    @staticmethod
    def _compute_word_local_spread(word_idx: int,
                                   embeddings_dict: Dict[str, np.ndarray],
                                   word_translations: List[Dict[str, str]],
                                   languages: List[str],
                                   metric: str) -> Tuple[Optional[str], Optional[float]]:
        """
        Helper function for parallel computation of per-word local spread

        This is a static method to be pickle-able for multiprocessing.

        Args:
            word_idx: Index of the word to process
            embeddings_dict: Dictionary of embeddings per language
            word_translations: List of translation dictionaries
            languages: List of language codes
            metric: Distance metric ('euclidean' or 'cosine')

        Returns:
            (reference_word, local_spread) or (None, None) if skipped
        """
        from scipy.spatial.distance import pdist

        translation_group = word_translations[word_idx]

        # Get embeddings for this word across all languages
        word_embeddings = []
        reference_word = None

        for lang in languages:
            if lang in translation_group and lang in embeddings_dict:
                word_embeddings.append(embeddings_dict[lang][word_idx])
                if reference_word is None:
                    reference_word = translation_group[lang]

        if len(word_embeddings) < 2:
            return None, None

        # Compute local spread
        word_embeddings_array = np.array(word_embeddings)

        # Compute pairwise distances
        if metric == 'euclidean':
            local_distances = pdist(word_embeddings_array, metric='euclidean')
            # RMS for Euclidean
            local_spread = np.sqrt(np.mean(local_distances ** 2)) if len(local_distances) > 0 else 0.0
        elif metric == 'cosine':
            local_distances = pdist(word_embeddings_array, metric='cosine')
            # Simple mean for cosine (no squaring)
            local_spread = np.mean(local_distances) if len(local_distances) > 0 else 0.0
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        return reference_word, local_spread

    def compute_semantic_affinity(
        self,
        embeddings_dict: Dict[str, np.ndarray],
        word_translations: List[Dict[str, str]],
        original_embeddings_dict: Dict[str, np.ndarray],
        original_word_translations: List[Dict[str, str]],
        metric: str = 'euclidean'
    ) -> SemanticAffinityResult:
        """
        Compute Semantic Affinity (SA) score using horizontal/vertical spread framework

        Semantic Affinity measures cross-lingual alignment strength as a bounded metric.

        Computation:
        1. Horizontal spread: Vocabulary-level baseline (normalization)
        2. Vertical spread: Cross-lingual translation distances
        3. Semantic Ratio (SR, intermediate): vertical_spread / horizontal_spread
        4. Semantic Affinity (SA, final): 1 / (1 + SR)

        SA Properties:
        - Bounded in (0, 1]
        - Higher = stronger alignment
        - SA = 1.0: Perfect alignment (SR=0)
        - SA = 0.5: Neutral (SR=1, translations at baseline)
        - SA < 0.5: Weak affinity (SR>1)
        - SA > 0.5: Strong affinity (SR<1)

        Args:
            embeddings_dict: Dictionary of EXPANDED embeddings (after cross-product)
                Example: {'chn': array(...), 'enu': array(...)}  # 621 word pairs
            word_translations: List of EXPANDED translation dictionaries
                Example: [{'chn': '十', 'enu': 'ten'}, {'chn': '十', 'enu': 'complete'}, ...]
            original_embeddings_dict: Dictionary of ORIGINAL embeddings (before expansion)
                Used for horizontal spread calculation
            original_word_translations: List of ORIGINAL translation dictionaries
                Example: [{'chn': '十|完整', 'enu': 'ten|complete'}, ...]  # 327 words
            metric: Distance metric to use ('euclidean' or 'cosine')

        Returns:
            SemanticAffinityResult with Semantic Affinity score, SR reference, and diagnostics
        """
        # Validate inputs
        if not embeddings_dict or not word_translations:
            raise ValueError("embeddings_dict and word_translations cannot be empty")

        languages = list(embeddings_dict.keys())
        n_languages = len(languages)
        n_words_expanded = len(word_translations)
        n_words_original = len(original_word_translations)

        # Stage 1: Collapse detection (using expanded embeddings)
        all_embeddings = np.vstack([embeddings_dict[lang] for lang in languages])
        is_collapsed, max_distance = self.check_collapse(all_embeddings)
        if is_collapsed:
            return SemanticAffinityResult(
                score=0.0,  # SA = 0 for collapsed embeddings
                semantic_ratio=float('inf'),  # SR = inf for collapsed
                status='COLLAPSED',
                horizontal_spread=0.0,
                vertical_spread=0.0,
                per_word_scores={},
                outliers=[],
                std=0.0,
                sem=0.0,
                n_words=n_words_expanded,
                n_languages=n_languages,
                per_language_spreads={},
                unique_word_counts={},
                horizontal_spread_sem=0.0,
                vertical_spread_sem=0.0
            )

        # Stage 2: Compute horizontal spread (semantic scale, normalization baseline)
        # Uses ORIGINAL embeddings with unique words only
        # Parallelized across languages (8 independent pdist operations)
        horizontal_spread, per_language_spreads, unique_word_counts, horizontal_spread_sem = compute_horizontal_spread(
            original_embeddings_dict,
            original_word_translations,
            languages,
            metric,
            n_jobs=self.n_jobs  # Parallelize across 8 languages
        )

        if horizontal_spread == 0:
            return SemanticAffinityResult(
                score=0.0,  # SA = 0 for collapsed embeddings
                semantic_ratio=float('inf'),  # SR = inf for collapsed
                status='COLLAPSED',
                horizontal_spread=0.0,
                vertical_spread=0.0,
                per_word_scores={},
                outliers=[],
                std=0.0,
                sem=0.0,
                n_words=n_words_expanded,
                n_languages=n_languages,
                per_language_spreads=per_language_spreads,
                unique_word_counts=unique_word_counts,
                horizontal_spread_sem=0.0,
                vertical_spread_sem=0.0
            )

        # Stage 3: Compute per-word vertical spreads (cross-lingual)
        # Uses EXPANDED embeddings with cross-product word pairs
        per_word_scores = {}
        vertical_spreads = []

        if self.use_vectorized:
            # VECTORIZED + PARALLEL computation (8-100× faster)
            # Compute all vertical spreads using batched vectorization across CPUs
            vertical_spreads_array = compute_vertical_spreads_vectorized(
                embeddings_dict=embeddings_dict,
                language_codes=languages,
                metric=metric,
                n_jobs=self.n_jobs,
                max_batch_size=self.max_batch_size
            )

            # Normalize by horizontal spread
            normalized_vertical_spreads = vertical_spreads_array / horizontal_spread
            vertical_spreads = normalized_vertical_spreads.tolist()

            # Build per-word scores dictionary
            for i, translation_group in enumerate(word_translations):
                # Get reference word (first language with this word)
                reference_word = None
                for lang in languages:
                    if lang in translation_group:
                        reference_word = translation_group[lang]
                        break

                if reference_word:
                    per_word_scores[reference_word] = normalized_vertical_spreads[i]

        else:
            # SEQUENTIAL computation (original code for cross-validation)
            for i, translation_group in enumerate(word_translations):
                # Get embeddings for this word pair across all languages
                word_embeddings = []
                reference_word = None

                for lang in languages:
                    if lang in translation_group and lang in embeddings_dict:
                        # Get the embedding for this word pair in this language
                        word_embeddings.append(embeddings_dict[lang][i])
                        if reference_word is None:
                            reference_word = translation_group[lang]

                if len(word_embeddings) < 2:
                    # Skip word pairs with only one language
                    continue

                # Compute vertical spread for this translation pair (cross-lingual)
                word_embeddings_array = np.array(word_embeddings)
                vertical_distances = self.compute_pairwise_distances(word_embeddings_array, metric=metric)
                vertical_spread = self.compute_spread(vertical_distances, metric=metric)

                # Normalize by horizontal spread (semantic scale)
                normalized_vertical_spread = vertical_spread / horizontal_spread
                per_word_scores[reference_word] = normalized_vertical_spread
                vertical_spreads.append(normalized_vertical_spread)

        # Stage 4: Compute Semantic Affinity (SA)
        if not vertical_spreads:
            return SemanticAffinityResult(
                score=0.0,  # SA = 0 for no data
                semantic_ratio=0.0,  # SR = 0 for no data
                status='NO_DATA',
                horizontal_spread=horizontal_spread,
                vertical_spread=0.0,
                per_word_scores={},
                outliers=[],
                std=0.0,
                sem=0.0,
                n_words=n_words_expanded,
                n_languages=n_languages,
                per_language_spreads=per_language_spreads,
                unique_word_counts=unique_word_counts,
                horizontal_spread_sem=horizontal_spread_sem,
                vertical_spread_sem=0.0
            )

        # Compute average vertical spread (cross-lingual spread)
        mean_vertical_spread = np.mean(vertical_spreads)

        # Semantic Ratio (SR, intermediate) = vertical_spread / horizontal_spread
        # Lower SR = stronger alignment (translations closer than baseline)
        semantic_ratio = mean_vertical_spread  # Already normalized by horizontal_spread

        # Semantic Affinity (SA, final) = 1 / (1 + SR)
        # Higher SA (closer to 1) = stronger cross-lingual affinity
        # Bounded metric in (0, 1]
        semantic_affinity = 1.0 / (1.0 + semantic_ratio)

        # Convert per-word SR scores to SA scores
        per_word_sa_scores = {}
        per_word_sa_values = []
        for word, sr_score in per_word_scores.items():
            sa_score = 1.0 / (1.0 + sr_score)
            per_word_sa_scores[word] = sa_score
            per_word_sa_values.append(sa_score)

        # Compute statistics on SA scores
        sa_std = np.std(per_word_sa_values)

        # Compute standard error of mean for vertical spread (SR-based)
        # Sample size N = number of vertical spread samples (expanded word pairs)
        n_vertical_samples = len(vertical_spreads)
        if n_vertical_samples > 1:
            vertical_spread_std = np.std(vertical_spreads, ddof=1)  # Use sample std
            vertical_spread_sem = vertical_spread_std / np.sqrt(n_vertical_samples)
        else:
            vertical_spread_sem = 0.0

        # Compute average unnormalized vertical spread for reporting
        avg_vertical_spread_unnormalized = mean_vertical_spread * horizontal_spread

        # Compute SEM for unnormalized vertical spread
        vertical_spread_unnormalized_sem = vertical_spread_sem * horizontal_spread

        # Propagate errors into SR using error propagation formula for division
        # For SR = V / H, error: sem_SR = SR * sqrt((sem_V/V)^2 + (sem_H/H)^2)
        if semantic_ratio > 0 and horizontal_spread > 0:
            # Use unnormalized vertical spread for error propagation
            rel_err_vertical = (vertical_spread_unnormalized_sem / avg_vertical_spread_unnormalized) if avg_vertical_spread_unnormalized > 0 else 0.0
            rel_err_horizontal = (horizontal_spread_sem / horizontal_spread) if horizontal_spread > 0 else 0.0
            sem_sr = semantic_ratio * np.sqrt(rel_err_vertical**2 + rel_err_horizontal**2)
        else:
            sem_sr = 0.0

        # Propagate SR error to SA using derivative-based error propagation
        # For SA = 1/(1+SR), derivative: dSA/dSR = -1/(1+SR)^2
        # Error propagation: sem_SA = |dSA/dSR| * sem_SR
        sem_sa = sem_sr / ((1.0 + semantic_ratio) ** 2)

        # Find outliers (word pairs with LOW SA scores = poor alignment)
        # Lower SA = worse alignment, so we want bottom 10%
        threshold = np.percentile(per_word_sa_values, 10)
        outliers = [(word, score) for word, score in per_word_sa_scores.items()
                    if score < threshold]
        outliers.sort(key=lambda x: x[1])  # Sort ascending (worst first)

        return SemanticAffinityResult(
            score=semantic_affinity,  # SA = 1/(1+SR), primary metric
            semantic_ratio=semantic_ratio,  # SR for reference
            status='OK',
            horizontal_spread=horizontal_spread,
            vertical_spread=avg_vertical_spread_unnormalized,
            per_word_scores=per_word_sa_scores,  # SA scores per word
            outliers=outliers[:10],  # Bottom 10 outliers (worst alignment)
            std=sa_std,  # Standard deviation of SA scores
            sem=sem_sa,  # Standard error of mean SA
            n_words=n_words_expanded,
            n_languages=n_languages,
            per_language_spreads=per_language_spreads,
            unique_word_counts=unique_word_counts,
            horizontal_spread_sem=horizontal_spread_sem,
            vertical_spread_sem=vertical_spread_unnormalized_sem
        )


def compute_horizontal_spread(
    embeddings_dict: Dict[str, np.ndarray],
    word_translations: List[Dict[str, str]],
    language_codes: List[str],
    metric: str = 'euclidean',
    n_jobs: int = 1
) -> Tuple[float, Dict[str, float], Dict[str, int], float]:
    """
    Compute horizontal spread (semantic scale) per language using unique words only.

    Horizontal spread represents the overall semantic spread within each language's
    vocabulary, serving as the normalization baseline for vertical (cross-lingual) spreads.

    Process:
    1. Flatten multi-meaning words (split by '|')
    2. Deduplicate using set()
    3. Compute pairwise distances among unique words per language
    4. Calculate spread (RMS for Euclidean, mean for Cosine)
    5. Average across all languages

    Args:
        embeddings_dict: Dictionary mapping language codes to embedding arrays
        word_translations: List of translation dictionaries (NOT expanded, original dataset)
        language_codes: List of language codes
        metric: Distance metric ('euclidean' or 'cosine')

    Returns:
        (average_horizontal_spread, per_language_spreads, unique_word_counts, horizontal_spread_sem)
    """
    from scipy.spatial.distance import pdist

    # Extract unique words per language by flattening and deduplicating
    unique_words_per_lang = {}
    for lang_code in language_codes:
        words = []
        for trans_dict in word_translations:
            if lang_code in trans_dict:
                # Split by '|' and flatten to get all individual words
                word_str = str(trans_dict[lang_code])
                individual_words = [w.strip() for w in word_str.split('|')]
                words.extend(individual_words)

        # Use set() to deduplicate
        unique_words_per_lang[lang_code] = list(set(words))

    # Compute horizontal spread per language (parallelized across languages)
    def compute_spread_for_language(lang_code):
        """Compute horizontal spread for one language (for parallel execution)"""
        unique_words = unique_words_per_lang[lang_code]
        count = len(unique_words)

        # Build word-to-embedding mapping
        word_to_embedding = {}
        for i, trans_dict in enumerate(word_translations):
            if lang_code in trans_dict:
                word_str = str(trans_dict[lang_code])
                individual_words = [w.strip() for w in word_str.split('|')]
                for word in individual_words:
                    if word not in word_to_embedding:
                        # Store the embedding for this unique word
                        word_to_embedding[word] = embeddings_dict[lang_code][i]

        # Extract unique embeddings
        unique_embeddings = np.array([word_to_embedding[word] for word in unique_words])

        # Compute pairwise distances
        if len(unique_embeddings) < 2:
            return lang_code, 0.0, count

        lang_distances = pdist(unique_embeddings, metric=metric if metric == 'cosine' else 'euclidean')

        # Compute spread (RMS for Euclidean, mean for Cosine)
        if metric == 'euclidean':
            spread = np.sqrt(np.mean(lang_distances ** 2))
        else:  # cosine
            spread = np.mean(lang_distances)

        return lang_code, spread, count

    # Execute in parallel or sequential
    if n_jobs != 1 and len(language_codes) > 1:
        from joblib import Parallel, delayed
        from multiprocessing import cpu_count

        n_workers = cpu_count() if n_jobs == -1 else min(n_jobs, len(language_codes))
        # Use 'loky' backend for true multiprocessing (avoid GIL limitations)
        results = Parallel(n_jobs=n_workers, backend='loky')(
            delayed(compute_spread_for_language)(lang_code) for lang_code in language_codes
        )
    else:
        # Sequential execution
        results = [compute_spread_for_language(lang_code) for lang_code in language_codes]

    # Unpack results
    horizontal_spreads = {}
    unique_word_counts = {}
    for lang_code, spread, count in results:
        horizontal_spreads[lang_code] = spread
        unique_word_counts[lang_code] = count

    # Average across languages
    spread_values = list(horizontal_spreads.values())
    avg_horizontal_spread = np.mean(spread_values)

    # Compute standard error of mean for horizontal spread
    # Sample size N = number of languages
    n_languages = len(spread_values)
    if n_languages > 1:
        std_horizontal = np.std(spread_values, ddof=1)  # Use sample std (ddof=1)
        sem_horizontal = std_horizontal / np.sqrt(n_languages)
    else:
        sem_horizontal = 0.0

    return avg_horizontal_spread, horizontal_spreads, unique_word_counts, sem_horizontal


def expand_multi_meaning_translations(word_translations: List[Dict[str, str]],
                                     language_codes: List[str]) -> Tuple[List[Dict[str, str]], List[int]]:
    """
    Expand multi-meaning translations into cross-product word pairs.

    For words with multiple meanings separated by '|', this creates all possible
    cross-product combinations between languages.

    Example:
        Input:  {'chn': '十|完整', 'enu': 'ten|complete'}
        Output: [
            {'chn': '十', 'enu': 'ten'},
            {'chn': '十', 'enu': 'complete'},
            {'chn': '完整', 'enu': 'ten'},
            {'chn': '完整', 'enu': 'complete'}
        ]

    Args:
        word_translations: List of translation dictionaries
        language_codes: List of language codes

    Returns:
        (expanded_translations, original_indices)
        expanded_translations: List with cross-product expansions
        original_indices: Maps each expanded entry back to original word index
    """
    from itertools import product

    expanded_translations = []
    original_indices = []

    for original_idx, trans_dict in enumerate(word_translations):
        # Split each language's translation by '|'
        meaning_lists = {}
        for lang_code in language_codes:
            if lang_code in trans_dict:
                # Split by '|' and strip whitespace
                meanings = [m.strip() for m in str(trans_dict[lang_code]).split('|')]
                meaning_lists[lang_code] = meanings
            else:
                meaning_lists[lang_code] = ['']  # Empty placeholder

        # Generate cross-product of all language meanings
        lang_codes_sorted = sorted(meaning_lists.keys())
        meaning_combinations = list(product(*[meaning_lists[lc] for lc in lang_codes_sorted]))

        # Create a translation dict for each combination
        for combo in meaning_combinations:
            expanded_dict = {}
            for i, lang_code in enumerate(lang_codes_sorted):
                if combo[i]:  # Skip empty placeholders
                    expanded_dict[lang_code] = combo[i]

            if expanded_dict:  # Only add if we have at least one translation
                expanded_translations.append(expanded_dict)
                original_indices.append(original_idx)

    return expanded_translations, original_indices


def load_translation_dataset(csv_path: str, selected_languages: List[str]) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Load translation dataset from CSV file

    Rows starting with '#' are treated as comments and skipped.

    Args:
        csv_path: Path to CSV file with columns: chinese,english,spanish,french,german,russian,korean,arabic
        selected_languages: List of language codes to load (e.g., ['english', 'spanish'])

    Returns:
        (word_translations, language_codes)
        word_translations: List of dictionaries mapping language codes to translations
        language_codes: List of selected language codes in standardized format

    Note:
        Multi-meaning words with '|' delimiter are kept as-is here.
        Use expand_multi_meaning_translations() to create cross-product expansions.
    """
    # Load CSV with comment character support
    df = pd.read_csv(csv_path, comment='#')

    # Language name to code mapping
    lang_name_to_code = {
        'chinese': 'chn',
        'english': 'enu',
        'spanish': 'spa',
        'french': 'fra',
        'german': 'deu',
        'russian': 'rus',
        'korean': 'kor',
        'arabic': 'ara',
        'turkish': 'tur'
    }

    # Convert selected language names to codes
    language_codes = [lang_name_to_code[lang.lower()] for lang in selected_languages
                      if lang.lower() in lang_name_to_code]

    # Build word translations list
    word_translations = []
    for _, row in df.iterrows():
        translation_dict = {}
        for lang_name, lang_code in lang_name_to_code.items():
            if lang_code in language_codes and lang_name in row:
                # Keep multi-meaning words as-is (e.g., "ten|complete")
                # Expansion happens later via expand_multi_meaning_translations()
                translation = str(row[lang_name])
                translation_dict[lang_code] = translation

        if translation_dict:  # Only add if we have at least one translation
            word_translations.append(translation_dict)

    return word_translations, language_codes
