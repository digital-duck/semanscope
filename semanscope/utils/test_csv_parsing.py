#!/usr/bin/env python3
"""
Test script for CSV parsing logic with semantic color coding
"""
import pandas as pd
from io import StringIO
import sys
from pathlib import Path

def test_csv_parsing():
    """Test the CSV parsing logic with ACL-word-v2-enu.txt"""

    # Load the file (adjust path since we're now in src/utils)
    file_path = Path("../data/input/ACL-word-v2-enu.txt")
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False

    try:
        content = file_path.read_text(encoding='utf-8').strip()
        print(f"✅ Successfully loaded file with {len(content)} characters")

        # Check CSV format detection
        lines = content.split('\n')
        first_line = lines[0].strip()
        print(f"📋 First line: {first_line}")

        is_csv = ',' in first_line and first_line.lower().startswith('word')
        print(f"🔍 CSV format detected: {is_csv}")

        if not is_csv:
            print("❌ File doesn't appear to be CSV format")
            return False

        # Parse as CSV
        df = pd.read_csv(StringIO(content))
        print(f"📊 Loaded DataFrame with {len(df)} rows, {len(df.columns)} columns")
        print(f"📊 Columns: {list(df.columns)}")

        # Validate required columns
        if 'word' not in df.columns:
            print("❌ Missing 'word' column")
            return False

        words = df['word'].dropna().tolist()
        print(f"📝 Extracted {len(words)} words")

        # Enhanced semantic domain color scheme with more robust defaults
        domain_color_scheme = {
            # Function words (Red family)
            'articles_determiners': '#FF4444',
            'prepositions': '#FF6666',
            'conjunctions': '#FF8888',
            'pronouns': '#FFAAAA',

            # Abstract sequential (Green/Orange)
            'numbers': '#44FF44',                    # Green
            'colors': '#FF8800',                     # Orange

            # Content words (Blue family)
            'family_kinship': '#4488FF',             # Light Blue
            'body_parts': '#8B4513',                 # Brown
            'animals': '#FF8C00',                    # Dark Orange
            'food': '#FF6347',                       # Tomato
            'actions_verbs': '#4444FF',              # Medium Blue
            'emotions': '#6666FF',                   # Blue
            'nature_elements': '#228B22',            # Dark Green
            'time_temporal': '#4444FF',              # Medium Blue
            'spatial_directional': '#888888',        # Gray
            'abstract_qualities': '#DDA0DD',         # Plum

            # Morphological families (Purple family)
            'morphological_work': '#800080',         # Purple
            'morphological_light': '#9400D3',        # Purple variant
            'morphological_book': '#8A2BE2',         # Purple variant
            'morphological_子': '#800080',            # Purple (Chinese)

            # Fallback colors for unexpected domains
            'unknown': '#CCCCCC',                    # Light Gray
            'default': '#666666',                    # Dark Gray
        }

        # Test color assignment with robust error handling
        colors = []
        domain_stats = {}
        missing_domains = set()

        if 'domain' in df.columns:
            print(f"🎨 Processing domain-based color assignment...")

            for idx, row in df.iterrows():
                if pd.notna(row['word']):
                    domain = row.get('domain', 'unknown')

                    # Handle missing or NaN domains
                    if pd.isna(domain) or domain == '':
                        domain = 'unknown'

                    # Count domain frequencies
                    domain_stats[domain] = domain_stats.get(domain, 0) + 1

                    # Get color with fallback logic
                    if domain in domain_color_scheme:
                        color = domain_color_scheme[domain]
                    else:
                        color = domain_color_scheme['default']
                        missing_domains.add(domain)

                    colors.append(color)

            print(f"📈 Domain statistics:")
            for domain, count in sorted(domain_stats.items(), key=lambda x: x[1], reverse=True):
                color = domain_color_scheme.get(domain, domain_color_scheme['default'])
                print(f"   {domain}: {count} words → {color}")

            if missing_domains:
                print(f"⚠️  Unknown domains (using default color): {sorted(missing_domains)}")

        else:
            print("⚠️  No 'domain' column found, using default blue")
            colors = ['#4444FF'] * len(words)

        print(f"✅ Successfully generated {len(colors)} colors for {len(words)} words")

        # Verify colors match words count
        if len(colors) == len(words):
            print("✅ Color count matches word count")
        else:
            print(f"❌ Color count mismatch: {len(colors)} colors vs {len(words)} words")
            return False

        # Sample output
        print(f"\n📋 Sample words with colors:")
        for i in range(min(10, len(words))):
            domain = df.iloc[i].get('domain', 'unknown') if 'domain' in df.columns else 'default'
            print(f"   '{words[i]}' ({domain}) → {colors[i]}")

        print(f"\n🎉 CSV parsing test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error during parsing: {e}")
        return False

if __name__ == "__main__":
    success = test_csv_parsing()
    sys.exit(0 if success else 1)