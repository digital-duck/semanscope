#!/usr/bin/env python3
"""
Fix for word duplication bug in multilingual embedding visualization.

The issue: When combining ENU+KOR datasets, duplicate words like "being"
appear in multiple locations because they're added to all_words multiple times
without deduplication.

This script shows the proper fix for embedding_viz.py lines 873-918.
"""

def fixed_word_combination_logic():
    """
    Proposed fix for the word duplication bug.

    This should replace lines 873-918 in embedding_viz.py
    """

    # ===== FIXED VERSION =====
    # Combine all words and create corresponding colors with semantic support
    # WITH PROPER DEDUPLICATION

    word_to_info = {}  # Dictionary to track: word -> (color, lang_code, first_occurrence)
    all_words = []     # Final deduplicated word list
    all_colors = []    # Colors corresponding to deduplicated words

    # Map language codes to color map keys (fallback for non-semantic data)
    lang_color_map = {
        "chn": "chinese",
        "enu": "english",
        "fra": "french",
        "spa": "spanish",
        "deu": "german",
        "ara": "arabic",
        "kor": "korean",  # Add Korean support
        "jpn": "japanese"  # Add Japanese support
    }

    # Process all selected languages WITH DEDUPLICATION
    for lang_code, data in language_data.items():
        if data['selected'] and data['words']:

            # Get semantic color mapping for this language
            semantic_color_map = st.session_state.get(f"{lang_code}_semantic_colors", {})

            if not semantic_color_map:
                input_name = st.session_state.get('input_name_selected', '')
                if input_name:
                    dataset_words, dataset_color_map = self.load_semantic_data_from_file(input_name, lang_code)
                    if dataset_color_map:
                        semantic_color_map = dataset_color_map

            # Process each word in this language
            for word in data['words']:
                if word not in word_to_info:  # 🔧 FIX: Only add if not already seen

                    # Determine color for this word
                    if semantic_color_map and word in semantic_color_map:
                        color = semantic_color_map[word]
                    else:
                        # Word not found in semantic mapping, use language fallback
                        color_key = lang_color_map.get(lang_code, "english")
                        color = COLOR_MAP[color_key]

                    # Store word info and add to final lists
                    word_to_info[word] = (color, lang_code, len(all_words))
                    all_words.append(word)
                    all_colors.append(color)

                # If word already exists, we could optionally update its color based on priority
                # For now, we keep the first occurrence to maintain consistency

    return all_words, all_colors

def show_bug_explanation():
    """Explain the bug with a clear example"""
    print("🐛 WORD DUPLICATION BUG EXPLANATION:")
    print()
    print("BEFORE FIX (Buggy behavior):")
    print("  ENU data: ['i', 'you', 'being', 'something']")
    print("  KOR data: ['나', '너', 'being', '무언가']")  # Note: 'being' appears in KOR too
    print("  Result: ['i', 'you', 'being', 'something', '나', '너', 'being', '무언가']")
    print("  💥 'being' appears TWICE → Two different points in PHATE embedding!")
    print()
    print("AFTER FIX (Correct behavior):")
    print("  ENU data: ['i', 'you', 'being', 'something']")
    print("  KOR data: ['나', '너', 'being', '무언가']")
    print("  Result: ['i', 'you', 'being', 'something', '나', '너', '무언가']")
    print("  ✅ 'being' appears ONCE → Single point in PHATE embedding!")

if __name__ == "__main__":
    show_bug_explanation()
    print()
    print("To apply the fix:")
    print("1. Open /st_semantics/src/components/embedding_viz.py")
    print("2. Replace lines 873-918 with the fixed_word_combination_logic()")
    print("3. The 'being' duplication will be resolved!")