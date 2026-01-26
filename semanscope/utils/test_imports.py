#!/usr/bin/env python3
"""
Test script to check which dimensionality reduction packages are available
"""

print("Testing package imports...")

# Test TriMap
try:
    import trimap
    print("✓ TriMap available:", getattr(trimap, '__version__', 'imported'))
except ImportError as e:
    print("✗ TriMap failed:", e)

# Test PaCMAP
try:
    import pacmap
    print("✓ PaCMAP available:", getattr(pacmap, '__version__', 'imported'))
except ImportError as e:
    print("✗ PaCMAP failed:", e)

# Test igraph
try:
    import igraph as ig
    print("✓ igraph available:", getattr(ig, '__version__', 'imported'))
except ImportError as e:
    print("✗ igraph failed:", e)

# Test sklearn.neighbors
try:
    from sklearn.neighbors import kneighbors_graph
    print("✓ kneighbors_graph available")
except ImportError as e:
    print("✗ kneighbors_graph failed:", e)

# Test the full ForceAtlas2 dependencies
try:
    import igraph as ig
    from sklearn.neighbors import kneighbors_graph
    print("✓ ForceAtlas2 dependencies all available")
except ImportError as e:
    print("✗ ForceAtlas2 dependencies failed:", e)

print("\nTesting DimensionReducer initialization...")
try:
    from semanscope.components.dimension_reduction import DimensionReducer
    reducer = DimensionReducer()
    methods = list(reducer.reducers.keys())
    print("Available methods:", methods)

    new_methods = ['TriMap', 'PaCMAP', 'ForceAtlas2']
    for method in new_methods:
        if method in methods:
            print(f"✓ {method} successfully added")
        else:
            print(f"✗ {method} missing from reducers")

except Exception as e:
    print("✗ DimensionReducer test failed:", e)