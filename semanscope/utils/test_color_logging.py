#!/usr/bin/env python3
"""
Test script for color-coding logging functionality
"""
import sys
import json
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from semanscope.components.embedding_viz import EmbeddingVisualizer

def simulate_session_state():
    """Simulate streamlit session state for testing"""
    class MockSessionState:
        def __init__(self):
            self.data = {
                'input_name_selected': 'ACL-word-v2'
            }

        def get(self, key, default=None):
            return self.data.get(key, default)

    return MockSessionState()

def test_color_logging():
    """Test the color logging functionality"""
    print("🧪 Testing color-coding logging system...")

    # Mock streamlit session state
    import streamlit as st
    if not hasattr(st, 'session_state'):
        st.session_state = simulate_session_state()
    else:
        st.session_state.data = {'input_name_selected': 'ACL-word-v2'}

    # Initialize visualizer
    visualizer = EmbeddingVisualizer()

    # Test loading semantic data which should trigger logging
    try:
        words, colors = visualizer.load_semantic_data_from_file('ACL-word-v2', 'enu')

        if words and colors:
            print(f"✅ Loaded {len(words)} words with {len(colors)} colors")

            # Check if log file was created
            log_file = Path("../data/logs/color-coding/ACL-word-v2-enu.json")
            if log_file.exists():
                print(f"✅ Log file created: {log_file}")

                # Read and display sample content
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)

                print(f"📊 Log contains {len(log_data['word_colors'])} word entries")
                print(f"📊 Found {log_data['metadata']['total_domains']} semantic domains")

                # Show sample entries for numbers domain to help debug mixed colors
                print("\n🔍 Sample entries for 'numbers' domain:")
                numbers_words = [entry for entry in log_data['word_colors'] if entry['domain'] == 'numbers']
                for i, entry in enumerate(numbers_words[:10]):  # Show first 10
                    print(f"   {entry['word']} → {entry['color']} (type: {entry['type']})")

                print(f"\n✅ Color logging test completed successfully!")
                return True
            else:
                print(f"❌ Log file not found at {log_file}")
                return False
        else:
            print("❌ No words or colors loaded")
            return False

    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False

if __name__ == "__main__":
    success = test_color_logging()
    sys.exit(0 if success else 1)