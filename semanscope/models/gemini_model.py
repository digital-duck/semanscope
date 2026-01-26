"""
Google Gemini API integration for embedding models
Supports Gemini embedding models through Google AI API
"""

import os
import numpy as np
import streamlit as st
from typing import List, Optional, Any
from abc import ABC, abstractmethod

# Define EmbeddingModel locally to avoid circular import
class EmbeddingModel(ABC):
    """Abstract base class for embedding models"""

    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name

    @abstractmethod
    def get_embeddings(self, texts: List[str], lang: str = "en", debug_flag: bool = False) -> Optional[np.ndarray]:
        pass


class GeminiModel(EmbeddingModel):
    """Google Gemini API-based embedding model"""

    def __init__(self, model_path: str, model_name: str):
        super().__init__(model_path, model_name)
        self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        # Try to import and initialize the client
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai
            self.available = True
        except ImportError:
            st.error("google-generativeai package not installed. Install with: pip install google-generativeai")
            self.available = False
        except Exception as e:
            st.error(f"Failed to initialize Gemini client: {str(e)}")
            self.available = False

    def load_model(self):
        """Gemini models don't need local loading"""
        if self.available:
            st.info(f"🤖 Using Google Gemini API for {self.model_name}")
            return True
        return False

    def get_embeddings(self, texts: List[str], lang: str = "en", debug_flag: bool = False) -> Optional[np.ndarray]:
        """Generate embeddings using Google Gemini API"""
        if not self.available:
            st.error("Gemini API not available")
            return None

        if not texts:
            return None

        if debug_flag:
            st.info(f"🤖 Gemini Debug: Generating embeddings for {len(texts)} texts using {self.model_name}")

        try:
            embeddings = []

            # Process texts in batches (Gemini API handles multiple texts per request)
            batch_size = 100  # Conservative batch size

            with st.spinner(f"🤖 Generating embeddings via Gemini API ({self.model_name})..."):
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]

                    if debug_flag:
                        st.info(f"Processing batch {i//batch_size + 1}: {len(batch_texts)} texts")

                    try:
                        # Use the embed_content method
                        response = self.client.embed_content(
                            model=self.model_path,
                            content=batch_texts,
                            task_type='retrieval_document',  # Good for general semantic analysis
                            title=f"Semanscope Research Dataset Batch {i//batch_size + 1}"
                        )

                        # Extract embeddings from response
                        if hasattr(response, 'embedding'):
                            # Single text response
                            embeddings.append(response.embedding)
                        elif hasattr(response, 'embeddings'):
                            # Multiple texts response
                            embeddings.extend(response.embeddings)
                        else:
                            st.error(f"Unexpected response format from Gemini API")
                            if debug_flag:
                                st.write("Response structure:", dir(response))
                            return None

                    except Exception as e:
                        st.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                        if debug_flag:
                            st.write(f"Batch texts: {batch_texts[:3]}...")  # Show first 3 texts
                        return None

            if embeddings:
                embeddings_array = np.array(embeddings)

                if debug_flag:
                    st.success(f"✅ Generated {len(embeddings)} embeddings via Gemini API")
                    st.info(f"📊 Embedding dimensions: {embeddings_array.shape}")
                    st.info(f"📈 Embedding range: {embeddings_array.min():.4f} to {embeddings_array.max():.4f}")

                # Display usage information (if available)
                st.success(f"✅ Generated {len(embeddings)} embeddings via Gemini API")

                return embeddings_array
            else:
                st.error("No embeddings were generated")
                return None

        except Exception as e:
            st.error(f"❌ Unexpected error with Gemini API: {str(e)}")
            if debug_flag:
                st.write("Full error details:", str(e))
            return None

    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "name": self.model_name,
            "path": self.model_path,
            "type": "Google Gemini API",
            "api_key_set": bool(self.api_key),
            "client_available": self.available
        }


# Google Gemini model configurations
GEMINI_MODELS = {
    "Gemini-Embedding-001 (Google Cloud)": {
        "path": "gemini-embedding-001",
        "help": "🤖 Google's state-of-the-art Gemini embedding model via direct Google API. Up to 3072 dimensions, MTEB leaderboard leader. CRITICAL for testing Google's latest architecture for geometric collapse investigation.",
        "is_active": False,
        "alias": "Gemini-001",        
        "provider": "google",
        "embedding_dim": 3072,
        "context_length": 2048,
        "pricing": "See Google AI pricing",
        "research_priority": "critical"
    },
    "Text-Embedding-005 (Google Cloud)": {
        "path": "text-embedding-005",
        "help": "📝 Google's specialized embedding model for English and code. 768 dimensions, optimized for semantic similarity tasks.",
        "is_active": False,
        "alias": "Google-005",        
        "provider": "google",
        "embedding_dim": 768,
        "context_length": 2048,
        "pricing": "See Google Cloud Vertex AI pricing",
        "research_priority": "medium"
    },
    "Text-Multilingual-Embedding-002 (Google Cloud)": {
        "path": "text-multilingual-embedding-002",
        "help": "🌍 Google's multilingual embedding model. 768 dimensions, optimized for multiple languages including Chinese-English research tasks.",
        "is_active": False,
        "alias": "Google-002",        
        "provider": "google",
        "embedding_dim": 768,
        "context_length": 2048,
        "pricing": "See Google Cloud Vertex AI pricing",
        "research_priority": "high"
    }
}