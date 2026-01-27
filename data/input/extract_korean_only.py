#!/usr/bin/env python3
"""
Script to extract only Korean words (no numbers, no arrows)
"""

def extract_korean_words():
    # Read the file
    input_file = '/home/papagame/projects/Proj-Geometry-of-Meaning/st_semantics/data/input/ICML-NSM-Primes-kor.txt'

    # Read file and extract Korean words only
    korean_words = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '→' in line:
                # Split by arrow and get only the Korean word
                korean_word = line.split('→')[-1].strip()
                if korean_word:
                    korean_words.append(korean_word)

    # Write only Korean words (one per line)
    with open(input_file, 'w', encoding='utf-8') as f:
        for word in korean_words:
            f.write(f"{word}\n")

    print(f"Extracted {len(korean_words)} Korean words")
    print("First 10 words:")
    for i in range(min(10, len(korean_words))):
        print(f"  {korean_words[i]}")

if __name__ == "__main__":
    extract_korean_words()