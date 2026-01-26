#!/usr/bin/env python3
"""
Debug word processing to compare loaded words vs processed words
"""
import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from components.embedding_viz import EmbeddingVisualizer

def debug_word_processing():
    """Debug word processing differences between loading and visualization"""
    print("🔍 Debugging word processing differences...")

    # Initialize visualizer
    visualizer = EmbeddingVisualizer()

    # Load semantic data
    words, word_color_map = visualizer.load_semantic_data_from_file('ACL-word-v2', 'enu')

    if not words:
        print("❌ Failed to load words")
        return

    print(f"\n1. Words loaded from CSV file: {len(words)}")
    print(f"   Dictionary size: {len(word_color_map)}")

    # Show first 20 words as loaded
    print("\n   First 20 words as loaded:")
    for i, word in enumerate(words[:20]):
        color = word_color_map.get(word, 'NOT_FOUND')
        print(f"     {i:2d}: {word} → {color}")

    # Find all number words in loaded data
    number_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve']
    loaded_numbers = {}
    for word in number_words:
        if word in word_color_map:
            loaded_numbers[word] = word_color_map[word]

    print(f"\n2. Number words in loaded dictionary: {len(loaded_numbers)}")
    for word, color in loaded_numbers.items():
        print(f"     {word} → {color}")

    # Simulate text area content
    text_content = '\n'.join(words)
    print(f"\n3. Text area content simulation:")
    print(f"   Content length: {len(text_content)} characters")
    print(f"   First 200 characters: {repr(text_content[:200])}")

    # Process the text using the same method as during visualization
    processed_words = visualizer.process_text(text_content)
    print(f"\n4. Words after process_text(): {len(processed_words)}")

    # Check if processing changes any words
    original_set = set(words)
    processed_set = set(processed_words)

    if original_set == processed_set:
        print("   ✅ No differences between loaded and processed words")
    else:
        added = processed_set - original_set
        removed = original_set - processed_set
        print(f"   ❌ Found differences:")
        if added:
            print(f"     Added during processing: {sorted(added)}")
        if removed:
            print(f"     Removed during processing: {sorted(removed)}")

    # Check number words specifically
    processed_numbers = [word for word in processed_words if word in number_words]
    print(f"\n5. Number words after processing: {len(processed_numbers)}")
    print(f"   Numbers found: {sorted(processed_numbers)}")

    # Test dictionary lookups for processed number words
    print(f"\n6. Dictionary lookups for processed number words:")
    lookup_results = {}
    for word in processed_numbers:
        if word in word_color_map:
            color = word_color_map[word]
            lookup_results[word] = color
            print(f"     {word} → {color}")
        else:
            print(f"     {word} → NOT_FOUND (this would cause fallback)")

    # Summary
    if len(lookup_results) == len(processed_numbers) and len(set(lookup_results.values())) == 1:
        print(f"\n✅ SUCCESS: All {len(processed_numbers)} processed number words found in dictionary with consistent color")
    else:
        print(f"\n❌ PROBLEM: Dictionary lookup issues detected")
        print(f"   Processed numbers: {len(processed_numbers)}")
        print(f"   Found in dictionary: {len(lookup_results)}")
        print(f"   Unique colors: {set(lookup_results.values())}")

if __name__ == "__main__":
    debug_word_processing()