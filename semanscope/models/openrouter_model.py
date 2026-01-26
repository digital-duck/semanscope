"""
OpenRouter API integration for embedding models
Supports Google Gemini and OpenAI embedding models through OpenRouter.ai
"""

import os
import numpy as np
import streamlit as st
from typing import List, Optional, Any
from abc import ABC, abstractmethod

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Define EmbeddingModel locally to avoid circular import
class EmbeddingModel(ABC):
    """Abstract base class for embedding models"""

    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name

    @abstractmethod
    def get_embeddings(self, texts: List[str], lang: str = "en", debug_flag: bool = False) -> Optional[np.ndarray]:
        pass


class OpenRouterModel(EmbeddingModel):
    """OpenRouter API-based embedding model using OpenAI client"""

    def __init__(self, model_path: str, model_name: str):
        super().__init__(model_path, model_name)
        st.info(f"Initializing OpenRouter model: {model_name} at path: {model_path}")
        self.api_key = os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available. Install with: pip install openai")

        # Initialize OpenAI client with OpenRouter endpoint
        try:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
                timeout=60.0,  # 60 second timeout per request
            )
            self.available = True
        except Exception as e:
            st.error(f"Failed to initialize OpenRouter client: {str(e)}")
            self.available = False

    def load_model(self):
        """OpenRouter models don't need local loading"""
        if self.available:
            st.info(f"🌐 Using OpenRouter API for {self.model_name}")
            return True
        return False

    def get_embeddings(self, texts: List[str], lang: str = "en", debug_flag: bool = False) -> Optional[np.ndarray]:
        """Generate embeddings using OpenRouter API via OpenAI client"""
        if not self.available:
            st.error("OpenRouter client not available")
            return None

        if not texts:
            return None

        if debug_flag:
            st.info(f"🌐 OpenRouter Debug: Generating embeddings for {len(texts)} texts using {self.model_name}")

        try:
            # Get batch size from session state (set in Advanced Settings) or use default
            batch_size = st.session_state.get('api_batch_size', 100)
            all_embeddings = []

            if debug_flag:
                st.write(f"🔍 Processing {len(texts)} texts in {(len(texts)-1)//batch_size + 1} batches of {batch_size}")

            total_batches = (len(texts) - 1) // batch_size + 1

            # Use expander for API call details (collapsed by default)
            with st.expander(f"🌐 OpenRouter API Progress ({len(texts)} words in {total_batches} batches)", expanded=False):
                for i in range(0, len(texts), batch_size):
                    batch_num = i // batch_size + 1
                    batch_texts = texts[i:i + batch_size]

                    # Show progress for each batch
                    st.info(f"🌐 Batch {batch_num}/{total_batches}: Generating {len(batch_texts)} embeddings...")

                    try:
                        response = self.client.embeddings.create(
                            model=self.model_path,
                            input=batch_texts,
                            encoding_format="float"
                        )

                        # Check if this batch succeeded
                        if not response.data:
                            st.error(f"❌ Batch {batch_num} failed: Response data is None")
                            return None

                        # Extract embeddings from this batch
                        batch_embeddings = []
                        for item in response.data:
                            if hasattr(item, 'embedding'):
                                batch_embeddings.append(item.embedding)
                            else:
                                st.error(f"❌ Response item missing embedding: {item}")
                                return None

                        all_embeddings.extend(batch_embeddings)
                        st.success(f"✅ Batch {batch_num}/{total_batches}: {len(batch_embeddings)} embeddings complete")

                    except Exception as e:
                        st.error(f"❌ Batch {batch_num}/{total_batches} failed: {str(e)}")
                        if "timeout" in str(e).lower():
                            st.warning("⏱️ API timeout - try reducing batch size or check network connection")
                        return None

            # Combine all embeddings
            embeddings_array = np.array(all_embeddings)

            # Display success message
            st.success(f"✅ Generated {len(all_embeddings)} embeddings via OpenRouter")

            if debug_flag:
                st.info(f"📊 Final shape: {embeddings_array.shape}, range: {embeddings_array.min():.3f} to {embeddings_array.max():.3f}")

            return embeddings_array

        except Exception as e:
            st.error(f"❌ OpenRouter API error: {str(e)}")
            if debug_flag:
                st.write("Full error details:", str(e))
                st.write("Exception type:", type(e))
                # Try to extract more info about the error
                if hasattr(e, 'response'):
                    st.write("Error response:", e.response)
            return None

    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "name": self.model_name,
            "path": self.model_path,
            "type": "OpenRouter API (OpenAI Client)",
            "api_key_set": bool(self.api_key),
            "client_available": self.available
        }


# OpenRouter model configurations - All available embedding models
OPENROUTER_MODELS = {
    # Qwen Series (Priority - Large Models First)
    "Qwen3-Embedding-8B (OpenRouter)": {
        "path": "qwen/qwen3-embedding-8b",
        "help": "🚀 Qwen3 8B embedding model via OpenRouter API. Largest available Qwen embedding model, ideal for investigating parameter scaling effects on geometric structure.",
        "is_active": True,
        "alias": "Qwen3-8B",        
        "provider": "openrouter",
        "embedding_dim": 1024,
        "context_length": 32000,
        "pricing": "$0.02/M tokens",
        "research_priority": "critical"
    },
    "Qwen3-Embedding-4B (OpenRouter)": {
        "path": "qwen/qwen3-embedding-4b",
        "help": "⚡ Qwen3 4B embedding model via OpenRouter API. Mid-scale model for parameter scaling studies without local VRAM constraints.",
        "is_active": True,
        "alias": "Qwen3-4B",        
        "provider": "openrouter",
        "embedding_dim": 1024,
        "context_length": 32000,
        "pricing": "$0.015/M tokens",
        "research_priority": "high"
    },
    "Qwen3-Embedding-0.6B (OpenRouter)": {
        "path": "qwen/qwen3-embedding-0.6b",
        "help": "📊 Qwen3 0.6B embedding model via OpenRouter API. Cloud version of our local baseline for architecture comparison studies.",
        "is_active": True,
        "alias": "Qwen3-06B",        
        "provider": "openrouter",
        "embedding_dim": 1024,
        "context_length": 32000,
        "pricing": "$0.01/M tokens",
        "research_priority": "high"
    },

    # Google Models
    "Gemini-Embedding-001 (OpenRouter)": {
        "path": "google/gemini-embedding-001",
        "help": "🔥 Google's Gemini Embedding model via OpenRouter API. 3072 dimensions, 20K context. Critical for testing Google's newest architecture vs EmbeddingGemma collapse.",
        "is_active": True,
        "alias": "Gemini-001",        
        "provider": "openrouter",
        "context_length": 20000,
        "pricing": "$0.15/M tokens",
        "embedding_dim": 3072,  # Confirmed via API test
        "research_priority": "critical"
    },

    # OpenAI Models
    "OpenAI Text-Embedding-3-Large (OpenRouter)": {
        "path": "openai/text-embedding-3-large",
        "help": "🚀 OpenAI's latest large embedding model via OpenRouter. 3072 dimensions, state-of-the-art performance for comparison studies.",
        "is_active": True,
        "alias": "OpenAI-3-large",
        "provider": "openrouter",
        "embedding_dim": 3072,
        "context_length": 8192,
        "pricing": "$0.13/M tokens",
        "openai_generation": 3,
        "research_priority": "high"
    },
    "OpenAI Text-Embedding-3-Small (OpenRouter)": {
        "path": "openai/text-embedding-3-small",
        "help": "⚡ OpenAI's efficient 3rd gen embedding model via OpenRouter. 1536 dimensions, cost-effective for large-scale analysis.",
        "is_active": True,
        "alias": "OpenAI-3-small",
        "provider": "openrouter",
        "embedding_dim": 1536,
        "context_length": 8192,
        "pricing": "$0.02/M tokens",
        "openai_generation": 3,
        "research_priority": "medium"
    },
    "OpenAI Text-Embedding-Ada-002 (OpenRouter)": {
        "path": "openai/text-embedding-ada-002",
        "help": "🤖 OpenAI's proven embedding model via OpenRouter. 1536 dimensions, reliable baseline with confirmed geometric structure preservation.",
        "is_active": True,
        "alias": "OpenAI-ada-002",
        "provider": "openrouter",
        "embedding_dim": 1536,
        "context_length": 8192,
        "pricing": "$0.10/M tokens",
        "openai_generation": 2,
        "research_priority": "medium"
    },

    # Other Providers
    "Mistral-Embed-2312 (OpenRouter)": {
        "path": "mistral/mistral-embed-2312",
        "help": "🌟 Mistral's embedding model via OpenRouter API. 1024 dimensions optimized for semantic search. European AI alternative for architectural diversity.",
        "is_active": False,
        "alias": "Mistral-2312",
        "provider": "openrouter",
        "embedding_dim": 1024,
        "context_length": 8192,
        "pricing": "$0.10/M tokens",
        "research_priority": "medium"
    },
    "Voyage-Large-2-Instruct (OpenRouter)": {
        "path": "voyage/voyage-large-2-instruct",
        "help": "🌊 Voyage AI's large instruction-tuned embedding model. Specialized for complex reasoning and instruction-following tasks.",
        "is_active": False,
        "alias": "Voyage-Large-2",
        "provider": "openrouter",
        "embedding_dim": 1024,
        "context_length": 16000,
        "pricing": "$0.12/M tokens",
        "research_priority": "medium"
    }
}