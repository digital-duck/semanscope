#!/usr/bin/env python3
"""
Simple test script for ECharts integration
"""

import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from components.plotting_echarts import EChartsPlotManager
    print("✅ EChartsPlotManager imported successfully")
except ImportError as e:
    print(f"❌ Failed to import EChartsPlotManager: {e}")
    sys.exit(1)

# Test basic functionality
def test_echarts_functionality():
    print("\n🧪 Testing ECharts functionality...")

    # Create test data
    embeddings_2d = np.random.randn(10, 2)
    embeddings_3d = np.random.randn(10, 3)
    labels = [f"word_{i}" for i in range(10)]
    colors = ['red' if i < 5 else 'blue' for i in range(10)]

    # Initialize manager
    echarts_manager = EChartsPlotManager()
    print("✅ EChartsPlotManager initialized")

    # Test settings
    settings = echarts_manager.get_visualization_settings()
    print(f"✅ Default settings loaded: {len(settings)} settings")

    # Test title creation
    title = echarts_manager.create_title("UMAP", "SBERT", "Test Dataset")
    expected_title = "[Method] UMAP, [Model] SBERT, [Dataset] Test Dataset"
    assert title == expected_title, f"Title mismatch: got '{title}', expected '{expected_title}'"
    print("✅ Title creation working correctly")

    # Test clustering functionality
    metrics, _ = echarts_manager._perform_clustering(embeddings_2d, 3)
    assert 'silhouette' in metrics, "Silhouette score missing from clustering metrics"
    assert 'calinski' in metrics, "Calinski-Harabasz score missing from clustering metrics"
    assert 'cluster_labels' in metrics, "Cluster labels missing from clustering metrics"
    print("✅ Clustering functionality working correctly")

    print("\n🎉 All ECharts tests passed!")
    return True

def test_streamlit_echarts_import():
    print("\n🧪 Testing streamlit-echarts import...")
    try:
        from streamlit_echarts import st_echarts
        print("✅ streamlit_echarts imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import streamlit_echarts: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting ECharts integration tests...")

    # Test imports
    if not test_streamlit_echarts_import():
        sys.exit(1)

    # Test functionality
    if not test_echarts_functionality():
        sys.exit(1)

    print("\n✨ All tests completed successfully!")
    print("\n📊 ECharts integration is ready to use!")
    print("🌐 You can now run: streamlit run src/Welcome.py")
    print("📋 Then navigate to '4_Semantics_Explorer-ECharts' page")