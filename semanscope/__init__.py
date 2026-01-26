"""
Semanscope - Multilingual Semantic Embedding Visualization and Analysis Toolkit

A comprehensive toolkit for visualizing and analyzing semantic embeddings across multiple languages,
featuring Semantic Affinity (SA) and Relational Affinity (RA) metrics.

Key Features:
- Multi-model embedding support (LaBSE, SONAR, Gemma, OpenAI, etc.)
- Advanced dimensionality reduction (UMAP, PHATE, t-SNE, PaCMAP, TriMap)
- Semantic Affinity (SA) metric for measuring semantic consistency
- Relational Affinity (RA) metric for measuring relational structure preservation
- Interactive Streamlit UI for exploration
- Batch benchmarking CLI for research
- Multilingual support (70+ languages)

Usage:
    # Launch the UI
    from semanscope.cli import launch_ui
    launch_ui()

    # Or use the command line
    # semanscope-ui

    # For batch benchmarking
    # semanscope-benchmark-sa --help
    # semanscope-benchmark-ra --help

Documentation: https://github.com/semanscope/semanscope
"""

__version__ = "1.0.0"
__author__ = "Semanscope Contributors"
__license__ = "MIT"

# Core imports - make key components easily accessible
from semanscope import config
from semanscope.components import (
    semantic_affinity,
    embedding_viz,
    dimension_reduction,
)
from semanscope.models import model_manager

__all__ = [
    '__version__',
    'config',
    'semantic_affinity',
    'embedding_viz',
    'dimension_reduction',
    'model_manager',
]
