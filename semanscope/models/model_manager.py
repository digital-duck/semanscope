# models/model_manager.py
import streamlit as st
from abc import ABC, abstractmethod
import numpy as np
import requests
from typing import List, Optional
from utils.error_handling import (
    handle_errors, ModelNotFoundError, EmbeddingError,
)
from config import (
    OLLAMA_MODELS, MODEL_INFO,
)

# Import v2o optimized models (loaded lazily to avoid circular imports)
_V2O_MODELS_LOADED = False
def _load_v2o_models():
    """Lazy load v2o models to avoid startup overhead"""
    global _V2O_MODELS_LOADED
    if not _V2O_MODELS_LOADED:
        try:
            from models.model_manager_v2o import get_model_v2o
            _V2O_MODELS_LOADED = True
            return get_model_v2o
        except ImportError as e:
            st.warning(f"⚠️ V2O models not available: {e}")
            return None
    else:
        from models.model_manager_v2o import get_model_v2o
        return get_model_v2o
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
except ImportError as e:
    import streamlit as st
    st.error(f"Failed to import transformers: {e}")
    # Fallback - only support Ollama models
    AutoTokenizer = None
    AutoModel = None
    torch = None

# OpenRouter models handled differently to avoid circular import
def get_openrouter_model(model_name: str, model_path: str):
    """Create OpenRouter model instance dynamically"""
    try:
        from .openrouter_model import OpenRouterModel
        return OpenRouterModel(model_path, model_name)
    except ImportError as e:
        st.error(f"OpenRouter integration not available: {e}")
        raise ModelNotFoundError(f"OpenRouter model {model_name} not available")

# Google Gemini models handled differently to avoid circular import
def get_gemini_model(model_name: str, model_path: str):
    """Create Google Gemini model instance dynamically"""
    try:
        # Force reload to pick up code changes during development
        import importlib
        import sys
        module_name = 'models.gemini_model'
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])

        from .gemini_model import GeminiModel
        return GeminiModel(model_path, model_name)
    except ImportError as e:
        st.error(f"Google Gemini integration not available: {e}")
        raise ModelNotFoundError(f"Gemini model {model_name} not available")

# Voyage AI models handled differently to avoid circular import
def get_voyage_model(model_name: str, model_path: str):
    """Create Voyage AI model instance dynamically"""
    try:
        from .voyage_model import VoyageModel
        return VoyageModel(model_path, model_name)
    except ImportError as e:
        st.error(f"Voyage AI integration not available: {e}")
        raise ModelNotFoundError(f"Voyage AI model {model_name} not available")

@st.cache_resource
def get_ollama_session():
    """Create a cached session for Ollama requests to improve performance"""
    session = requests.Session()
    return session

class EmbeddingModel(ABC):
    """Abstract base class for embedding models"""
    @abstractmethod
    def get_embeddings(self, texts: List[str], lang: str = "en") -> np.ndarray:
        pass

class OllamaModel(EmbeddingModel):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.session = get_ollama_session()
        
    @handle_errors
    def get_embeddings(self, texts: List[str], lang: str = "en", debug_flag: bool = False) -> Optional[np.ndarray]:
        embeddings = []
        failed_chars = []

        if debug_flag:
            # Store debug info in session state for persistent display
            if 'debug_logs' not in st.session_state:
                st.session_state.debug_logs = []

            debug_msg = f"🔍 OLLAMA DEBUG: Processing {len(texts)} texts with {self.model_name}"
            st.session_state.debug_logs.append(debug_msg)

            # Show the actual texts being processed  
            text_preview = [f"'{text}'" for text in texts[:10]]
            if len(texts) > 10:
                text_preview.append(f"...and {len(texts)-10} more")

            char_msg = f"📝 Texts to encode: {', '.join(text_preview)}"
            st.session_state.debug_logs.append(char_msg)

            # Also display immediately
            st.info(debug_msg)
            st.info(char_msg)

        for idx, text in enumerate(texts):
            try:
                response = self.session.post(
                    "http://localhost:11434/api/embeddings",
                    json={"model": self.model_name, "prompt": text}
                )
                if response.status_code == 200:
                    embedding = response.json().get("embedding")
                    if embedding:
                        # Check for problematic embeddings
                        import numpy as np
                        emb_array = np.array(embedding)
                        if np.isnan(emb_array).any():
                            failed_chars.append(f"'{text}' - NaN embedding")
                            if debug_flag:
                                st.warning(f"⚠️ NaN embedding for: '{text}'")
                        elif np.allclose(emb_array, 0.0):
                            failed_chars.append(f"'{text}' - Zero embedding")
                            if debug_flag:
                                st.warning(f"⚠️ Zero embedding for: '{text}'")
                        else:
                            embeddings.append(embedding)
                            if debug_flag and idx < 10:  # Show first 10 successful encodings
                                norm = np.linalg.norm(emb_array)
                                st.success(f"✅ Successfully encoded: '{text}' - norm: {norm:.4f}")
                    else:
                        failed_chars.append(f"'{text}' - No embedding returned")
                        if debug_flag:
                            st.warning(f"❌ No embedding returned for: '{text}'")
                else:
                    failed_chars.append(f"'{text}'")
                    if debug_flag:
                        st.warning(f"❌ HTTP {response.status_code} for: '{text}'")
            except Exception as e:
                failed_chars.append(f"'{text}'")
                if debug_flag:
                    st.error(f"💥 Exception for '{text}': {str(e)}")

        # Summary report
        if failed_chars:
            st.warning(f"⚠️ Failed to encode {len(failed_chars)}/{len(texts)} characters: {', '.join(failed_chars[:10])}" +
                      (f" and {len(failed_chars)-10} more..." if len(failed_chars) > 10 else ""))

        if debug_flag:
            st.info(f"📊 Summary: {len(embeddings)}/{len(texts)} characters successfully encoded")
            if failed_chars:
                st.error(f"❌ {len(failed_chars)} failed encodings detected!")
                # Show detailed failure breakdown
                failure_types = {}
                for fail in failed_chars:
                    if "NaN embedding" in fail:
                        failure_types["NaN embeddings"] = failure_types.get("NaN embeddings", 0) + 1
                    elif "Zero embedding" in fail:
                        failure_types["Zero embeddings"] = failure_types.get("Zero embeddings", 0) + 1
                    elif "No embedding returned" in fail:
                        failure_types["No embedding returned"] = failure_types.get("No embedding returned", 0) + 1
                    else:
                        failure_types["HTTP errors"] = failure_types.get("HTTP errors", 0) + 1

                for fail_type, count in failure_types.items():
                    st.warning(f"🔍 {fail_type}: {count} characters")

        return np.array(embeddings) if embeddings else None

class HuggingFaceModel(EmbeddingModel):
    def __init__(self, model_name: str, model_path: str):
        self.model_name = model_name
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        
    def _lazy_load(self):
        if not self.tokenizer:
            if AutoTokenizer is None or AutoModel is None:
                raise ImportError("Transformers library not available due to compatibility issues")
            
            # Special handling for Qwen models that require trust_remote_code
            if "Qwen" in self.model_name:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                    self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
                except Exception as e:
                    # Fallback for older transformers versions
                    st.error(f"Failed to load Qwen model {self.model_name}. Error: {str(e)}")
                    st.info("Try updating transformers: pip install transformers --upgrade")
                    raise e
            else:
                # Standard loading for other models
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModel.from_pretrained(self.model_path)
            
    @handle_errors
    def get_embeddings(self, texts: List[str], lang: str = "en", debug_flag: bool = False) -> Optional[np.ndarray]:
        # LASER support disabled due to torch compatibility issues
        # if self.model_name == "LASER":
        #     try:
        #         from laserembeddings import Laser
        #         laser = Laser()
        #         return laser.embed_sentences(texts, lang=lang)
        #     except Exception as e:
        #         st.error(f"Unsupported model: {self.model_name}")
        #         return None

        self._lazy_load()
        embeddings = []
        if debug_flag:
            st.info(f"🔍 DEBUG: Processing {len(texts)} texts with {self.model_name}")
            # Show the actual texts being processed
            text_preview = [f"'{text}'" for text in texts[:10]]
            if len(texts) > 10:
                text_preview.append(f"...and {len(texts)-10} more")
            st.info(f"📝 Texts to encode: {', '.join(text_preview)}")
            print(f"[DEBUG] Processing {len(texts)} texts with {self.model_name}")
        for i, text in enumerate(texts):
            # Skip empty texts that could cause NaN issues
            if not text or not text.strip():
                st.warning(f"Empty text detected, using zero embedding")
                # Create a zero embedding with the model's hidden size
                dummy_inputs = self.tokenizer("dummy", return_tensors="pt")
                dummy_outputs = self.model(**dummy_inputs)
                zero_embedding = np.zeros_like(dummy_outputs.last_hidden_state.mean(dim=1).detach().numpy())
                embeddings.append(zero_embedding)
                continue

            # E5 model preprocessing
            # 1. Add "query: " prefix for E5-Instruct models (critical for proper embedding)
            if "e5" in self.model_name.lower() and "instruct" in self.model_name.lower():
                text = f"query: {text}"
            # 2. Add space before Chinese text for better E5 tokenization
            elif self.model_name in ["E5-Base-v2"] and any('\u4e00' <= char <= '\u9fff' for char in text):
                text = f" {text}"

            # Tokenize the input text with proper attention masks
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)

            # Debug tokenization issues with Chinese text
            if any('\u4e00' <= char <= '\u9fff' for char in text):
                token_count = inputs['input_ids'].size(1)
                if token_count > 100:  # Very long tokenization might indicate issues
                    st.warning(f"Chinese text '{text[:20]}...' tokenized to {token_count} tokens")

            # Get the encoder outputs
            outputs = self.model(**inputs)

            if debug_flag: print(f"[DEBUG] Text {i}: '{text[:30]}...'")
            if debug_flag: print(f"[DEBUG] Token count: {inputs['input_ids'].size(1)}")

            # Proper mean pooling using attention masks to avoid NaN from padding tokens
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state

            # Debug raw embeddings
            if debug_flag: print(f"[DEBUG] Raw embeddings shape: {token_embeddings.shape}")
            if debug_flag: print(f"[DEBUG] Raw embeddings range: {token_embeddings.min().item():.4f} to {token_embeddings.max().item():.4f}")
            if debug_flag: print(f"[DEBUG] Raw embeddings has NaN: {torch.isnan(token_embeddings).any().item()}")

            # Mask out padding tokens and compute mean
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = (token_embeddings * input_mask_expanded).sum(dim=1)
            sum_mask = input_mask_expanded.sum(dim=1)

            if debug_flag: print(f"[DEBUG] Sum mask: {sum_mask}")
            if debug_flag: print(f"[DEBUG] Sum embeddings has NaN: {torch.isnan(sum_embeddings).any().item()}")

            # Avoid division by zero
            sum_mask = sum_mask.clamp(min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

            if debug_flag: print(f"[DEBUG] Mean pooled shape: {mean_pooled.shape}")
            if debug_flag: print(f"[DEBUG] Mean pooled range: {mean_pooled.min().item():.4f} to {mean_pooled.max().item():.4f}")
            if debug_flag: print(f"[DEBUG] Mean pooled has NaN: {torch.isnan(mean_pooled).any().item()}")

            # L2 normalization (critical for XLM-RoBERTa to prevent embedding collapse)
            # Good practice for all models to ensure embeddings are on unit sphere
            import torch.nn.functional as F
            mean_pooled = F.normalize(mean_pooled, p=2, dim=1)

            if debug_flag: print(f"[DEBUG] After L2 normalization - norm: {torch.norm(mean_pooled, dim=1).mean().item():.4f}")

            # Check for NaN values and extreme values before adding to embeddings
            embedding_array = mean_pooled.detach().numpy()
            if np.isnan(embedding_array).any():
                st.warning(f"NaN detected in embedding for text: '{text[:50]}...', using zero embedding")
                embedding_array = np.zeros_like(embedding_array)
            elif np.isinf(embedding_array).any():
                st.warning(f"Infinite values detected in embedding for text: '{text[:50]}...', clipping values")
                embedding_array = np.clip(embedding_array, -10.0, 10.0)
            elif np.abs(embedding_array).max() > 100:
                st.warning(f"Extreme values detected in embedding for text: '{text[:50]}...', normalizing")
                # L2 normalize to prevent extreme values
                norm = np.linalg.norm(embedding_array, axis=1, keepdims=True)
                embedding_array = embedding_array / (norm + 1e-8)

            embeddings.append(embedding_array)

        # Final debugging before returning
        final_embeddings = np.vstack(embeddings)
        if debug_flag: print(f"[DEBUG] Final embeddings shape: {final_embeddings.shape}")
        if debug_flag: print(f"[DEBUG] Final embeddings range: {final_embeddings.min():.4f} to {final_embeddings.max():.4f}")
        if debug_flag: print(f"[DEBUG] Final embeddings has NaN: {np.isnan(final_embeddings).any()}")
        if debug_flag: print(f"[DEBUG] Final embeddings has Inf: {np.isinf(final_embeddings).any()}")

        # Force replace any remaining NaN or Inf values
        if np.isnan(final_embeddings).any() or np.isinf(final_embeddings).any():
            st.error("Final embeddings still contain NaN/Inf values, replacing with zeros")
            final_embeddings = np.nan_to_num(final_embeddings, nan=0.0, posinf=10.0, neginf=-10.0)

        return final_embeddings

def get_active_models():
    """Get only active models for UI display"""
    active_models = {}

    # Add active Ollama models
    for name, info in OLLAMA_MODELS.items():
        if info.get("is_active", True):  # Default to True for backward compatibility
            active_models[name] = info

    # Add active Hugging Face models
    for name, info in MODEL_INFO.items():
        if info.get("is_active", True):  # Default to True for backward compatibility
            active_models[name] = info

    return active_models

def get_model(model_name: str) -> EmbeddingModel:
    """
    Factory function for creating embedding models

    This is the BASELINE version (CPU, no caching).
    For optimized version, use get_model_with_strategy() which checks global optimization setting.
    """
    if model_name in OLLAMA_MODELS:
        if not OLLAMA_MODELS[model_name].get("is_active", True):
            raise ModelNotFoundError(f"Model {model_name} is currently inactive")
        return OllamaModel(OLLAMA_MODELS[model_name]["path"])
    elif "(OpenRouter)" in model_name:
        # Handle OpenRouter models dynamically
        if model_name not in MODEL_INFO:
            raise ModelNotFoundError(f"Model {model_name} not found")
        if not MODEL_INFO[model_name].get("is_active", True):
            raise ModelNotFoundError(f"Model {model_name} is currently inactive")
        return get_openrouter_model(model_name, MODEL_INFO[model_name]["path"])
    elif "(Google Cloud)" in model_name:
        # Handle Google Cloud Gemini models dynamically
        if model_name not in MODEL_INFO:
            raise ModelNotFoundError(f"Model {model_name} not found")
        if not MODEL_INFO[model_name].get("is_active", True):
            raise ModelNotFoundError(f"Model {model_name} is currently inactive")
        return get_gemini_model(model_name, MODEL_INFO[model_name]["path"])
    elif "(Voyage AI)" in model_name:
        # Handle Voyage AI models dynamically
        if model_name not in MODEL_INFO:
            raise ModelNotFoundError(f"Model {model_name} not found")
        if not MODEL_INFO[model_name].get("is_active", True):
            raise ModelNotFoundError(f"Model {model_name} is currently inactive")
        return get_voyage_model(model_name, MODEL_INFO[model_name]["path"])
    elif model_name in MODEL_INFO:
        if not MODEL_INFO[model_name].get("is_active", True):
            raise ModelNotFoundError(f"Model {model_name} is currently inactive")
        return HuggingFaceModel(model_name, MODEL_INFO[model_name]["path"])


def get_model_with_strategy(model_name: str, optimization_strategy: str = None) -> EmbeddingModel:
    """
    Factory function for creating embedding models with optimization strategy support

    Args:
        model_name: Name of the model (e.g., "Qwen3-Embedding-0.6B")
        optimization_strategy: "baseline" or "v2o"
                              If None, checks st.session_state.optimization_strategy

    Returns:
        EmbeddingModel instance (baseline or v2o depending on strategy)

    User Priority (per request):
        1. HuggingFace models (highest priority for v2o)
        2. OpenRouter models (medium priority)
        3. Ollama models (backup)
    """
    # Get optimization strategy from session state if not provided
    if optimization_strategy is None:
        optimization_strategy = st.session_state.get('optimization_strategy', 'baseline')

    # Baseline strategy - use existing get_model()
    if optimization_strategy == 'baseline':
        return get_model(model_name)

    # V2O strategy - use optimized models
    elif optimization_strategy == 'v2o':
        get_model_v2o_func = _load_v2o_models()

        if get_model_v2o_func is None:
            st.warning("⚠️ V2O models not available, falling back to baseline")
            return get_model(model_name)

        # Determine model type and route to v2o
        if model_name in OLLAMA_MODELS:
            if not OLLAMA_MODELS[model_name].get("is_active", True):
                raise ModelNotFoundError(f"Model {model_name} is currently inactive")
            return get_model_v2o_func(
                model_name=model_name,
                model_type="ollama",
                model_path=OLLAMA_MODELS[model_name]["path"]
            )

        elif "(OpenRouter)" in model_name:
            if model_name not in MODEL_INFO:
                raise ModelNotFoundError(f"Model {model_name} not found")
            if not MODEL_INFO[model_name].get("is_active", True):
                raise ModelNotFoundError(f"Model {model_name} is currently inactive")
            return get_model_v2o_func(
                model_name=model_name,
                model_type="openrouter",
                model_path=MODEL_INFO[model_name]["path"]
            )

        elif model_name in MODEL_INFO:
            if not MODEL_INFO[model_name].get("is_active", True):
                raise ModelNotFoundError(f"Model {model_name} is currently inactive")
            # HuggingFace models (highest priority for v2o)
            return get_model_v2o_func(
                model_name=model_name,
                model_type="huggingface",
                model_path=MODEL_INFO[model_name]["path"]
            )

        else:
            raise ModelNotFoundError(f"Unknown model: {model_name}")

    else:
        st.warning(f"⚠️ Unknown optimization strategy '{optimization_strategy}', using baseline")
        return get_model(model_name)
