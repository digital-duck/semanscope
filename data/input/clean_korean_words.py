#!/usr/bin/env python3
"""
Script to clean Korean words file by extracting only Korean words from the second column
"""

import pandas as pd

def clean_korean_file():
    # Read the file
    input_file = '/home/papagame/projects/Proj-Geometry-of-Meaning/st_semantics/data/input/ICML-NSM-Primes-kor.txt'

    # Read file as text lines
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Process each line
    korean_words = []
    line_numbers = []

    for i, line in enumerate(lines):
        line = line.strip()
        if '→' in line:
            # Split by arrow and get the Korean word (second part)
            parts = line.split('→')
            if len(parts) >= 2:
                korean_word = parts[-1].strip()  # Get the last part (Korean word)
                if korean_word:  # Only add non-empty words
                    korean_words.append(korean_word)
                    line_numbers.append(len(korean_words))  # Sequential numbering

    # Write clean file with only Korean words
    output_file = input_file  # Overwrite the original file

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, word in enumerate(korean_words):
            f.write(f"{i+1:>6}→{word}\n")

    print(f"Cleaned file written to: {output_file}")
    print(f"Total Korean words: {len(korean_words)}")
    print("First 10 words:")
    for i in range(min(10, len(korean_words))):
        print(f"  {i+1}→{korean_words[i]}")

if __name__ == "__main__":
    clean_korean_file()