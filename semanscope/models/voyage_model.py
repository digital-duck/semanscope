"""
Voyage AI API integration for embedding models
Supports Voyage AI embedding models through their official API
"""

import os
import numpy as np
import streamlit as st
from typing import List, Optional, Any
from abc import ABC, abstractmethod

try:
    import voyageai
    VOYAGEAI_AVAILABLE = True
except ImportError:
    VOYAGEAI_AVAILABLE = False

# Define EmbeddingModel locally to avoid circular import
class EmbeddingModel(ABC):
    """Abstract base class for embedding models"""

    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name

    @abstractmethod
    def get_embeddings(self, texts: List[str], lang: str = "en", debug_flag: bool = False) -> Optional[np.ndarray]:
        pass


class VoyageModel(EmbeddingModel):
    """Voyage AI API-based embedding model"""

    def __init__(self, model_path: str, model_name: str):
        super().__init__(model_path, model_name)
        self.api_key = os.getenv("VOYAGE_API_KEY")

        if not self.api_key:
            error_msg = "VOYAGE_API_KEY environment variable not set. Please set it in ~/.bashrc or .env file."
            st.error(error_msg)
            raise ValueError(error_msg)

        if not VOYAGEAI_AVAILABLE:
            error_msg = "voyageai package not available. Install with: pip install voyageai"
            st.error(error_msg)
            raise ImportError(error_msg)

        # Initialize Voyage AI client
        try:
            # st.info(f"🌊 Initializing Voyage AI model: {model_name} (path: {model_path})")  # Commented out - too verbose
            self.client = voyageai.Client(api_key=self.api_key)
            self.available = True
            # st.success(f"✅ Voyage AI client initialized successfully")  # Commented out - too verbose
        except Exception as e:
            error_msg = f"Failed to initialize Voyage AI client: {str(e)}"
            st.error(error_msg)
            raise RuntimeError(error_msg) from e

    def load_model(self):
        """Voyage AI models don't need local loading"""
        if self.available:
            # st.info(f"🌊 Using Voyage AI API for {self.model_name}")  # Commented out - too verbose
            return True
        return False

    def get_embeddings(self, texts: List[str], lang: str = "en", debug_flag: bool = False) -> Optional[np.ndarray]:
        """Generate embeddings using Voyage AI API

        Args:
            texts: List of texts to embed
            lang: Language code (not used by Voyage AI, included for API compatibility)
            debug_flag: Enable debug logging

        Returns:
            numpy array of embeddings or None on error
        """
        if not self.available:
            error_msg = "Voyage AI client not available. Client initialization may have failed."
            st.error(error_msg)
            raise RuntimeError(error_msg)

        if not texts:
            return np.array([])

        if debug_flag:
            st.info(f"🌊 Voyage AI Debug: Generating embeddings for {len(texts)} texts using {self.model_name}")

        try:
            # Get batch size from session state (set in Advanced Settings) or use default
            batch_size = st.session_state.get('api_batch_size', 128)  # Voyage supports up to 128 per batch
            all_embeddings = []

            if debug_flag:
                st.write(f"🔍 Processing {len(texts)} texts in {(len(texts)-1)//batch_size + 1} batches of {batch_size}")

            total_batches = (len(texts) - 1) // batch_size + 1

            # Commented out verbose API progress logging - enable if debugging needed
            # with st.expander(f"🌊 Voyage AI API Progress ({len(texts)} words in {total_batches} batches)", expanded=False):
            for i in range(0, len(texts), batch_size):
                batch_num = i // batch_size + 1
                batch_texts = texts[i:i + batch_size]

                # Show progress for each batch
                # st.info(f"🌊 Batch {batch_num}/{total_batches}: Generating {len(batch_texts)} embeddings...")  # Commented out - too verbose

                try:
                    # Call Voyage AI API with input_type parameter
                    # input_type is CRITICAL for Voyage AI performance
                    result = self.client.embed(
                        batch_texts,
                        model=self.model_path,
                        input_type="document"  # Critical for Voyage performance
                    )

                    # Check if this batch succeeded
                    if not result or not hasattr(result, 'embeddings'):
                        error_msg = f"❌ Batch {batch_num} failed: Response missing embeddings"
                        st.error(error_msg)
                        raise RuntimeError(error_msg)

                    # Extract embeddings from this batch
                    batch_embeddings = result.embeddings

                    if not batch_embeddings:
                        error_msg = f"❌ Batch {batch_num} failed: Empty embeddings list"
                        st.error(error_msg)
                        raise RuntimeError(error_msg)

                    all_embeddings.extend(batch_embeddings)
                    # st.success(f"✅ Batch {batch_num}/{total_batches}: {len(batch_embeddings)} embeddings complete")  # Commented out - too verbose

                except Exception as e:
                    error_msg = f"❌ Batch {batch_num}/{total_batches} failed: {str(e)}"
                    st.error(error_msg)
                    if "rate limit" in str(e).lower():
                        st.warning("⏱️ Rate limit reached - try reducing batch size or wait a moment")
                    raise RuntimeError(error_msg) from e

            # Combine all embeddings
            embeddings_array = np.array(all_embeddings)

            # Display success message
            # st.success(f"✅ Generated {len(all_embeddings)} embeddings via Voyage AI")  # Commented out - too verbose

            if debug_flag:
                st.info(f"📊 Final shape: {embeddings_array.shape}, range: {embeddings_array.min():.3f} to {embeddings_array.max():.3f}")

            return embeddings_array

        except Exception as e:
            error_msg = f"❌ Voyage AI API error: {str(e)}"
            st.error(error_msg)
            if debug_flag:
                st.write("Full error details:", str(e))
                st.write("Exception type:", type(e))
            raise

    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "name": self.model_name,
            "path": self.model_path,
            "type": "Voyage AI API",
            "api_key_set": bool(self.api_key),
            "client_available": self.available
        }


# Voyage AI model configurations
VOYAGE_MODELS = {
    # Voyage 3 - Latest Generation
    "Voyage-3 (Voyage AI)": {
        "path": "voyage-3",
        "help": "🌊 Voyage AI's latest flagship embedding model. State-of-the-art performance with optimized multilingual capabilities. Best choice for research-grade semantic analysis.",
        "is_active": True,
        "alias": "Voyage-3",
        "provider": "voyage",
        "embedding_dim": 1024,
        "context_length": 32000,
        "pricing": "$0.06/M tokens",
        "research_priority": "critical"
    },
    "Voyage-3-Lite (Voyage AI)": {
        "path": "voyage-3-lite",
        "help": "⚡ Voyage AI's efficient lite model from the 3rd generation. Balanced performance and cost for large-scale experiments.",
        "is_active": True,
        "alias": "Voyage-3-Lite",
        "provider": "voyage",
        "embedding_dim": 512,
        "context_length": 32000,
        "pricing": "$0.02/M tokens",
        "research_priority": "high"
    },

    # Voyage Large 2 - Previous Generation (Instruct & Base)
    "Voyage-Large-2-Instruct (Voyage AI)": {
        "path": "voyage-large-2-instruct",
        "help": "🎯 Voyage AI's instruction-tuned large model. Optimized for complex reasoning and instruction-following embedding tasks.",
        "is_active": True,
        "alias": "Voyage-L2-Instruct",
        "provider": "voyage",
        "embedding_dim": 1024,
        "context_length": 16000,
        "pricing": "$0.12/M tokens",
        "research_priority": "medium"
    },
    "Voyage-Large-2 (Voyage AI)": {
        "path": "voyage-large-2",
        "help": "📊 Voyage AI's large baseline model (2nd gen). Strong general-purpose embeddings for semantic similarity.",
        "is_active": True,
        "alias": "Voyage-L2",
        "provider": "voyage",
        "embedding_dim": 1536,
        "context_length": 16000,
        "pricing": "$0.12/M tokens",
        "research_priority": "medium"
    },

    # Voyage Code 2 - Code-Specialized
    "Voyage-Code-2 (Voyage AI)": {
        "path": "voyage-code-2",
        "help": "💻 Voyage AI's code-specialized embedding model. Optimized for code search, retrieval, and semantic code analysis.",
        "is_active": False,
        "alias": "Voyage-Code-2",
        "provider": "voyage",
        "embedding_dim": 1536,
        "context_length": 16000,
        "pricing": "$0.12/M tokens",
        "research_priority": "low"
    },

    # Voyage Multilingual 2 - Multilingual Specialist
    "Voyage-Multilingual-2 (Voyage AI)": {
        "path": "voyage-multilingual-2",
        "help": "🌍 Voyage AI's multilingual specialist model. Optimized for 100+ languages including Chinese-English research tasks. Perfect for cross-lingual semantic analysis.",
        "is_active": True,
        "alias": "Voyage-Multi-2",
        "provider": "voyage",
        "embedding_dim": 1024,
        "context_length": 32000,
        "pricing": "$0.12/M tokens",
        "research_priority": "high"
    },

    # Voyage Finance 2 - Domain-Specialized
    "Voyage-Finance-2 (Voyage AI)": {
        "path": "voyage-finance-2",
        "help": "💰 Voyage AI's finance-specialized embedding model. Optimized for financial documents and terminology.",
        "is_active": False,
        "alias": "Voyage-Fin-2",
        "provider": "voyage",
        "embedding_dim": 1024,
        "context_length": 32000,
        "pricing": "$0.12/M tokens",
        "research_priority": "low"
    },

    # Voyage Law 2 - Domain-Specialized
    "Voyage-Law-2 (Voyage AI)": {
        "path": "voyage-law-2",
        "help": "⚖️ Voyage AI's legal-specialized embedding model. Optimized for legal documents and terminology.",
        "is_active": False,
        "alias": "Voyage-Law-2",
        "provider": "voyage",
        "embedding_dim": 1024,
        "context_length": 32000,
        "pricing": "$0.12/M tokens",
        "research_priority": "low"
    }
}
