#!/usr/bin/env python3
"""
Test script to verify word-color dictionary mapping is working correctly
"""
import sys
import json
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from semanscope.components.embedding_viz import EmbeddingVisualizer

def test_word_color_mapping():
    """Test the word-color mapping functionality directly"""
    print("🧪 Testing word-color dictionary mapping...")

    # Initialize visualizer
    visualizer = EmbeddingVisualizer()

    # Test loading semantic data which should return dictionary
    words, word_color_map = visualizer.load_semantic_data_from_file('ACL-word-v2', 'enu')

    if not words:
        print("❌ No words loaded")
        return False

    if not isinstance(word_color_map, dict):
        print(f"❌ Expected dict, got {type(word_color_map)}")
        return False

    print(f"✅ Loaded {len(words)} words with {len(word_color_map)} color mappings")
    print(f"📊 Dictionary type: {type(word_color_map)}")

    # Check specific number words
    number_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve']

    print("\n🔍 Checking number word colors:")
    numbers_found = []
    for word in number_words:
        if word in word_color_map:
            color = word_color_map[word]
            numbers_found.append(f"{word}→{color}")
            print(f"   {word}: {color}")
        else:
            print(f"   {word}: NOT FOUND")

    # Check if all numbers have the same color
    number_colors = [word_color_map[word] for word in number_words if word in word_color_map]
    unique_colors = set(number_colors)

    print(f"\n📈 Numbers analysis:")
    print(f"   Found {len(numbers_found)} number words")
    print(f"   Unique colors: {unique_colors}")
    print(f"   Expected color: #44FF44")

    if len(unique_colors) == 1 and '#44FF44' in unique_colors:
        print("✅ SUCCESS: All numbers have consistent green color!")
        return True
    else:
        print("❌ FAILED: Numbers have inconsistent colors!")
        print(f"   Numbers with colors: {numbers_found}")
        return False

if __name__ == "__main__":
    success = test_word_color_mapping()
    sys.exit(0 if success else 1)