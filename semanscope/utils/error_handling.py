# utils/error_handling.py
import streamlit as st
import logging
import requests
from functools import wraps
from typing import Callable, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def handle_errors(func: Callable) -> Callable:
    """Decorator for handling errors in functions"""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            return None
    return wrapper

def check_ollama_connection():
    """Check if Ollama service is available"""
    try:
        response = requests.get("http://localhost:11434/api/version")
        return response.status_code == 200
    except:
        return False

class EmbeddingError(Exception):
    """Custom exception for embedding-related errors"""
    pass

class ModelNotFoundError(Exception):
    """Custom exception for model not found errors"""
    pass