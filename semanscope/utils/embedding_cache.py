"""
Centralized embedding cache for all Semanscope pages

Structure: {model_name: {lang_code: {word: embedding_vector}}}

This allows incremental embedding computation across all pages:
- Semantic Affinity
- Semantics Explorer
- Translator
- etc.

All pages share the same embedding cache, avoiding redundant model calls.

Cache location is configurable via config.SEMANTIC_CACHE_PATH
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import streamlit as st


class EmbeddingCache:
    """
    Centralized embedding cache manager

    Features:
    - Per-word granular caching (not per-dataset)
    - Shared across all Semanscope pages
    - Incremental updates (only compute new words)
    - Persistent storage with pickle
    - Configurable cache location (outside repo via config.SEMANTIC_CACHE_PATH)
    """

    def __init__(self, cache_file_path: Optional[str] = None):
        """
        Initialize embedding cache

        Args:
            cache_file_path: Path to cache file. If None, uses config.SEMANTIC_CACHE_PATH
        """
        if cache_file_path is None:
            # Import here to avoid circular dependency
            from semanscope.config import SEMANTIC_CACHE_PATH
            cache_file_path = SEMANTIC_CACHE_PATH

        # Expand ~ to home directory
        self.master_cache_file = Path(cache_file_path).expanduser()

        # Create parent directory if it doesn't exist
        self.master_cache_file.parent.mkdir(parents=True, exist_ok=True)

        # In-memory cache: {model_name: {lang_code: {word: embedding}}}
        self.cache = self._load_master_cache()
        self.cache_updated = False

    def _load_master_cache(self) -> Dict:
        """Load master cache from disk"""
        if self.master_cache_file.exists():
            try:
                with open(self.master_cache_file, 'rb') as f:
                    cache = pickle.load(f)
                # Show cache stats
                total_words = sum(
                    len(lang_dict)
                    for model_dict in cache.values()
                    for lang_dict in model_dict.values()
                )
                st.info(f"📂 Loaded embedding cache: {total_words} words across {len(cache)} models from {self.master_cache_file}")
                return cache
            except Exception as e:
                st.warning(f"Failed to load cache from {self.master_cache_file}: {e}")
                return {}
        else:
            st.info(f"📂 No existing cache found at {self.master_cache_file}, starting fresh")
            return {}

    def save_master_cache(self):
        """Save master cache to disk"""
        if self.cache_updated:
            try:
                with open(self.master_cache_file, 'wb') as f:
                    pickle.dump(self.cache, f)
                total_words = sum(
                    len(lang_dict)
                    for model_dict in self.cache.values()
                    for lang_dict in model_dict.values()
                )
                st.info(f"💾 Saved embedding cache: {total_words} total words to {self.master_cache_file}")
                self.cache_updated = False
            except Exception as e:
                st.warning(f"Failed to save cache to {self.master_cache_file}: {e}")

    def get_embeddings(
        self,
        words: List[str],
        model_name: str,
        lang_code: str,
        embedding_func: callable,
        force_recompute: bool = False
    ) -> Tuple[np.ndarray, int, int]:
        """
        Get embeddings for words, using cache when possible

        Args:
            words: List of words to embed
            model_name: Model identifier (e.g., "sentence-transformers/LaBSE")
            lang_code: Language code (e.g., "chn", "enu")
            embedding_func: Function to call for computing new embeddings
                           Should accept (words: List[str]) and return np.ndarray
            force_recompute: If True, ignore cache and recompute all embeddings

        Returns:
            (embeddings_array, cached_count, computed_count)
        """
        # Ensure model and language exist in cache
        if model_name not in self.cache:
            self.cache[model_name] = {}
        if lang_code not in self.cache[model_name]:
            self.cache[model_name][lang_code] = {}

        lang_cache = self.cache[model_name][lang_code]

        # Separate cached vs new words
        embeddings_list = []
        words_to_compute = []
        cached_count = 0

        for word in words:
            if not force_recompute and word in lang_cache:
                embeddings_list.append(lang_cache[word])
                cached_count += 1
            else:
                words_to_compute.append(word)
                embeddings_list.append(None)  # Placeholder

        # Compute new embeddings if needed
        computed_count = 0
        if words_to_compute:
            new_embeddings = embedding_func(words_to_compute)

            if new_embeddings is None:
                raise ValueError(f"Embedding function returned None for {len(words_to_compute)} words")

            # Validate embedding shape
            if not isinstance(new_embeddings, np.ndarray):
                raise ValueError(f"Embedding function must return np.ndarray, got {type(new_embeddings)}")

            if new_embeddings.ndim == 0:
                raise ValueError(f"Embedding function returned 0-dimensional array. Model likely failed to load. Check model initialization errors above.")

            if new_embeddings.ndim == 1:
                # Single embedding returned, reshape to 2D
                new_embeddings = new_embeddings.reshape(1, -1)

            if new_embeddings.shape[0] != len(words_to_compute):
                raise ValueError(
                    f"Embedding function returned {new_embeddings.shape[0]} embeddings but "
                    f"{len(words_to_compute)} words were requested. Model may have failed silently."
                )

            # Fill placeholders and update cache
            new_idx = 0
            for i, word in enumerate(words):
                if embeddings_list[i] is None:
                    embedding = new_embeddings[new_idx]
                    embeddings_list[i] = embedding
                    lang_cache[word] = embedding
                    new_idx += 1
                    computed_count += 1
                    self.cache_updated = True

        embeddings_array = np.array(embeddings_list)
        return embeddings_array, cached_count, computed_count

    def clear_cache(self, model_name: Optional[str] = None, lang_code: Optional[str] = None):
        """
        Clear cache (with optional filtering)

        Args:
            model_name: If provided, only clear this model
            lang_code: If provided, only clear this language (requires model_name)
        """
        if model_name is None:
            # Clear all
            self.cache = {}
            st.success("🗑️ Cleared entire embedding cache")
        elif lang_code is None:
            # Clear all languages for this model
            if model_name in self.cache:
                del self.cache[model_name]
                st.success(f"🗑️ Cleared cache for model: {model_name}")
        else:
            # Clear specific language
            if model_name in self.cache and lang_code in self.cache[model_name]:
                del self.cache[model_name][lang_code]
                st.success(f"🗑️ Cleared cache for {model_name}/{lang_code}")

        self.cache_updated = True
        self.save_master_cache()

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        stats = {
            'total_models': len(self.cache),
            'total_languages': sum(len(model_dict) for model_dict in self.cache.values()),
            'total_words': sum(
                len(lang_dict)
                for model_dict in self.cache.values()
                for lang_dict in model_dict.values()
            ),
            'models': {}
        }

        for model_name, model_dict in self.cache.items():
            model_stats = {
                'languages': len(model_dict),
                'words_per_language': {
                    lang: len(word_dict)
                    for lang, word_dict in model_dict.items()
                }
            }
            stats['models'][model_name] = model_stats

        return stats


# Global cache instance (singleton pattern)
_global_cache = None

def get_embedding_cache() -> EmbeddingCache:
    """Get global embedding cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = EmbeddingCache()
    return _global_cache
