#!/usr/bin/env python3
"""
Debug session state key mapping to understand the exact mismatch
"""
import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from components.embedding_viz import EmbeddingVisualizer

def simulate_user_workflow():
    """Simulate the exact user workflow that causes the color issue"""
    print("🔍 Simulating user workflow...")

    # Initialize visualizer
    visualizer = EmbeddingVisualizer()

    # Simulate loading ACL-word-v2-enu.txt with same language settings
    print("\n1. Simulating 'Load Text' with English-English (same language):")
    input_name = "ACL-word-v2"
    source_lang_code = "enu"  # English as source
    target_lang_code = "enu"  # English as target (same language)
    same_language_loading = source_lang_code == target_lang_code

    print(f"   Input: {input_name}")
    print(f"   Source language: {source_lang_code}")
    print(f"   Target language: {target_lang_code}")
    print(f"   Same language loading: {same_language_loading}")

    # Load semantic data
    words, word_color_map = visualizer.load_semantic_data_from_file(input_name, source_lang_code)

    if not words:
        print("❌ Failed to load words")
        return

    print(f"   ✅ Loaded {len(words)} words with {len(word_color_map)} color mappings")

    # Simulate session state storage (same language path)
    session_state_key = f"{source_lang_code}_semantic_colors"
    print(f"   📝 Would store in session state as: '{session_state_key}'")

    # Show some number words from the dictionary
    number_words = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
    dict_numbers = {word: word_color_map.get(word, 'NOT_FOUND') for word in number_words if word in word_color_map}
    print(f"   📊 Numbers in loaded dictionary: {dict_numbers}")

    print("\n2. Simulating 'Visualize' session state lookup:")

    # Simulate the lookup logic during visualization (same language)
    same_language = source_lang_code == target_lang_code
    print(f"   Same language during visualization: {same_language}")

    if same_language:
        lookup_key = f"{source_lang_code}_semantic_colors"
        print(f"   🔍 Would look for key: '{lookup_key}'")
        print(f"   ✅ Keys match: {session_state_key == lookup_key}")

        # Simulate finding the data
        semantic_color_map = word_color_map  # This would be from session state
        print(f"   📊 Dictionary found: {len(semantic_color_map)} entries")

        # Test specific number word lookups
        test_words = ['nine', 'two', 'five', 'ten', 'six', 'three']
        print(f"\n3. Testing word lookups for: {test_words}")

        for word in test_words:
            if word in semantic_color_map:
                color = semantic_color_map[word]
                print(f"   {word} → {color}")
            else:
                print(f"   {word} → NOT_FOUND")

        # Check if all numbers have consistent color
        found_colors = [semantic_color_map.get(word) for word in test_words if word in semantic_color_map]
        unique_colors = set(found_colors)
        print(f"\n   📈 Found colors: {found_colors}")
        print(f"   📈 Unique colors: {unique_colors}")

        if len(unique_colors) == 1 and '#44FF44' in unique_colors:
            print("   ✅ SUCCESS: All numbers have consistent green color!")
        else:
            print("   ❌ PROBLEM: Numbers have inconsistent colors!")

    else:
        # This shouldn't happen in our test case
        print("   ❌ Different language path (shouldn't happen in this test)")

if __name__ == "__main__":
    simulate_user_workflow()