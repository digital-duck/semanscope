"""
Semanscope Cross-Page Cache Manager
====================================

Advanced caching system for Semanscope calculations with TTL support.
Caches expensive computations like embedding generation and dimensionality reduction
across different Semanscope pages for improved performance.

Cache Key Format: <dataset_hash>-<lang>-<model>-<method>-<params_hash>
"""

import streamlit as st
import hashlib
import pickle
import time
import os
from typing import Optional, Any, Dict, List
from datetime import datetime, timedelta
from semanscope.config import CACHE_PATH


class SemanscapeCacheManager:
    """Advanced cache manager for Semanscope calculations with TTL support"""

    def __init__(self):
        self.cache_dir = CACHE_PATH
        self.ensure_cache_dir()

    def ensure_cache_dir(self):
        """Ensure cache directory exists"""
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_ttl_hours(self) -> int:
        """Get cache TTL from settings (default 4 hours)"""
        if 'global_settings' not in st.session_state:
            return 4  # Default TTL
        return st.session_state.global_settings.get('cache_ttl_hours', 4)

    def generate_cache_key(self, dataset: List[str], lang: str, model: str,
                          method: str, params: Dict = None) -> str:
        """Generate unique cache key for dataset-lang-model-method combination"""
        # Create deterministic hash of dataset
        dataset_str = '|'.join(sorted(dataset))
        dataset_hash = hashlib.md5(dataset_str.encode()).hexdigest()[:8]

        # Create deterministic hash of parameters
        params_str = str(sorted((params or {}).items()))
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]

        # Format: <dataset_hash>-<lang>-<model>-<method>-<params_hash>
        cache_key = f"{dataset_hash}-{lang}-{model}-{method}-{params_hash}"

        # Clean cache key (remove special characters)
        cache_key = "".join(c for c in cache_key if c.isalnum() or c in '-_')

        return cache_key

    def get_cache_file_path(self, cache_key: str) -> str:
        """Get full path to cache file"""
        return str(self.cache_dir / f"{cache_key}.pkl")

    def is_cache_valid(self, cache_file_path: str) -> bool:
        """Check if cache file exists and is within TTL"""
        if not os.path.exists(cache_file_path):
            return False

        # Check TTL
        ttl_hours = self.get_cache_ttl_hours()
        if ttl_hours == 0:  # No caching
            return False

        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file_path))
        expiry_time = file_time + timedelta(hours=ttl_hours)

        return datetime.now() < expiry_time

    def get_cached_result(self, dataset: List[str], lang: str, model: str,
                         method: str, params: Dict = None) -> Optional[Any]:
        """Retrieve cached result if valid"""
        cache_key = self.generate_cache_key(dataset, lang, model, method, params)
        cache_file_path = self.get_cache_file_path(cache_key)

        if not self.is_cache_valid(cache_file_path):
            return None

        try:
            with open(cache_file_path, 'rb') as f:
                cached_data = pickle.load(f)



            return cached_data

        except Exception as e:
            st.warning(f"Cache read error: {str(e)}")
            # Clean up corrupted cache file
            try:
                os.remove(cache_file_path)
            except:
                pass
            return None

    def save_to_cache(self, result: Any, dataset: List[str], lang: str,
                     model: str, method: str, params: Dict = None):
        """Save result to cache"""
        ttl_hours = self.get_cache_ttl_hours()
        if ttl_hours == 0:  # No caching
            return

        cache_key = self.generate_cache_key(dataset, lang, model, method, params)
        cache_file_path = self.get_cache_file_path(cache_key)

        try:
            with open(cache_file_path, 'wb') as f:
                pickle.dump(result, f)

            st.success(f"💾 Result cached for {ttl_hours}h ({cache_key[:16]}...)")

        except Exception as e:
            st.warning(f"Cache save error: {str(e)}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.cache_dir.exists():
            return {"total_files": 0, "total_size_mb": 0, "valid_files": 0}

        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = 0
        valid_files = 0

        for cache_file_path in cache_files:
            try:
                file_size = cache_file_path.stat().st_size
                total_size += file_size

                if self.is_cache_valid(str(cache_file_path)):
                    valid_files += 1
            except:
                continue

        return {
            "total_files": len(cache_files),
            "valid_files": valid_files,
            "expired_files": len(cache_files) - valid_files,
            "total_size_mb": round(total_size / (1024 * 1024), 2)
        }

    def cleanup_expired_cache(self) -> int:
        """Remove expired cache files"""
        if not self.cache_dir.exists():
            return 0

        cache_files = list(self.cache_dir.glob("*.pkl"))
        removed_count = 0

        for cache_file_path in cache_files:
            if not self.is_cache_valid(str(cache_file_path)):
                try:
                    cache_file_path.unlink()
                    removed_count += 1
                except:
                    continue

        return removed_count

    def clear_all_cache(self) -> int:
        """Clear all cache files"""
        if not self.cache_dir.exists():
            return 0

        cache_files = list(self.cache_dir.glob("*.pkl"))
        removed_count = 0

        for cache_file_path in cache_files:
            try:
                cache_file_path.unlink()
                removed_count += 1
            except:
                continue

        return removed_count


# Global cache manager instance
_cache_manager = None

def get_cache_manager() -> SemanscapeCacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = SemanscapeCacheManager()
    return _cache_manager


# Convenient wrapper functions
def get_cached_embeddings(dataset: List[str], lang: str, model: str):
    """Get cached embeddings for dataset-lang-model combination"""
    return get_cache_manager().get_cached_result(dataset, lang, model, "embeddings")

def save_embeddings_to_cache(embeddings, dataset: List[str], lang: str, model: str):
    """Save embeddings to cache"""
    get_cache_manager().save_to_cache(embeddings, dataset, lang, model, "embeddings")

def get_cached_dimension_reduction(dataset: List[str], lang: str, model: str,
                                  method: str, params: Dict = None):
    """Get cached dimension reduction result"""
    return get_cache_manager().get_cached_result(dataset, lang, model, method, params)

def save_dimension_reduction_to_cache(result, dataset: List[str], lang: str,
                                    model: str, method: str, params: Dict = None):
    """Save dimension reduction result to cache"""
    get_cache_manager().save_to_cache(result, dataset, lang, model, method, params)

def get_cache_stats():
    """Get cache statistics"""
    return get_cache_manager().get_cache_stats()

def cleanup_cache():
    """Cleanup expired cache files"""
    return get_cache_manager().cleanup_expired_cache()

def clear_all_cache():
    """Clear all cache files"""
    return get_cache_manager().clear_all_cache()