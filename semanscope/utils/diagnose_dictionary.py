#!/usr/bin/env python3
"""
Diagnose what's actually in the word-color dictionary
"""
import sys
import json
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from components.embedding_viz import EmbeddingVisualizer

def diagnose_dictionary():
    """Diagnose the word-color dictionary content"""
    print("🔍 Diagnosing word-color dictionary...")

    # Initialize visualizer
    visualizer = EmbeddingVisualizer()

    # Load semantic data
    words, word_color_map = visualizer.load_semantic_data_from_file('ACL-word-v2', 'enu')

    if not word_color_map:
        print("❌ No word-color mapping found")
        return

    print(f"📊 Dictionary contains {len(word_color_map)} entries")

    # Check specific number words
    number_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve']

    print("\n🔢 Number words in dictionary:")
    for word in number_words:
        if word in word_color_map:
            color = word_color_map[word]
            print(f"   {word}: {color}")
        else:
            print(f"   {word}: NOT FOUND")

    # Show a sample of all dictionary entries
    print("\n📋 Sample dictionary entries:")
    count = 0
    for word, color in word_color_map.items():
        print(f"   {word}: {color}")
        count += 1
        if count >= 20:  # Show first 20 entries
            break

    # Check if all numbers have the same color
    number_colors = [word_color_map[word] for word in number_words if word in word_color_map]
    unique_colors = set(number_colors)

    print(f"\n📈 Numbers analysis:")
    print(f"   Found {len(number_colors)} number words")
    print(f"   Unique colors: {unique_colors}")

    if len(unique_colors) == 1:
        color = list(unique_colors)[0]
        if color == '#44FF44':
            print("✅ SUCCESS: All numbers have correct green color!")
        else:
            print(f"❌ PROBLEM: All numbers have {color} instead of #44FF44")
    else:
        print("❌ PROBLEM: Numbers have inconsistent colors!")

if __name__ == "__main__":
    diagnose_dictionary()