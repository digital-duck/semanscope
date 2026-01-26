#!/usr/bin/env python3
"""
Basic Semanscope Visualization Example

This demo shows how to:
1. Load an embedding model
2. Visualize word embeddings
3. Apply dimensionality reduction
4. Save visualization output

Usage:
    python demo/basic_visualization.py
"""

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from semanscope.models.model_manager import get_model
from semanscope.components.embedding_viz import EmbeddingVisualizer
from semanscope.components.dimension_reduction import reduce_dimensions
import numpy as np


def main():
    print("=" * 60)
    print("Semanscope Basic Visualization Demo")
    print("=" * 60)

    # Step 1: Load a model
    print("\n[1/4] Loading embedding model...")
    model_name = "LaBSE"  # Language-agnostic BERT Sentence Embedding
    print(f"      Model: {model_name}")

    try:
        model = get_model(model_name)
        print(f"      ✓ Model loaded successfully")
    except Exception as e:
        print(f"      ✗ Error loading model: {e}")
        print(f"      Note: Make sure you have installed torch and transformers")
        return

    # Step 2: Define words to visualize
    print("\n[2/4] Preparing words...")
    words = [
        # Greetings
        "hello", "hi", "greetings",
        # Emotions
        "happy", "sad", "angry", "joy", "fear",
        # Animals
        "cat", "dog", "bird", "fish",
        # Nature
        "tree", "flower", "mountain", "river",
        # Abstract
        "love", "peace", "freedom", "truth"
    ]
    print(f"      Words: {len(words)} total")

    # Step 3: Get embeddings
    print("\n[3/4] Computing embeddings...")
    try:
        embeddings = model.encode(words)
        print(f"      ✓ Embeddings computed: shape {embeddings.shape}")
    except Exception as e:
        print(f"      ✗ Error computing embeddings: {e}")
        return

    # Step 4: Apply dimensionality reduction
    print("\n[4/4] Applying dimensionality reduction...")
    methods = ["UMAP", "PCA", "t-SNE"]

    for method in methods:
        try:
            print(f"      • {method}...", end=" ")
            reduced = reduce_dimensions(
                embeddings,
                method=method,
                n_components=2
            )
            print(f"✓ shape {reduced.shape}")

            # Print a few coordinates
            print(f"        Sample: {words[0]} → ({reduced[0, 0]:.3f}, {reduced[0, 1]:.3f})")

        except Exception as e:
            print(f"✗ Error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  • Launch UI: python run_app.py")
    print("  • Try multilingual: add words in different languages")
    print("  • Explore models: check semanscope.config.MODEL_INFO")
    print("  • Run benchmarks: semanscope-benchmark-sa --help")


if __name__ == "__main__":
    main()
