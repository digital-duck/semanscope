#!/usr/bin/env python3
"""
Merge language-specific .txt files into a single SA-format CSV.

This script does the reverse of Semanscope's "Split by lang" functionality.
It combines PHATE-format .txt files (per language) into SA-format CSV (multi-column).

Usage:
    python merge_langs_to_sa.py <dataset-name> [--translate] [--auto-translate]

Examples:
    python merge_langs_to_sa.py ACL-2-word-v2
    python merge_langs_to_sa.py ACL-2-word-v2 --translate
    python merge_langs_to_sa.py ACL-1-Alphabets

The script will:
1. Auto-detect all language files: <dataset-name>-{lang}.txt
2. Merge them into: <dataset-name>-SA.csv
3. Use columns: English, Chinese, German, etc. (full language names)
4. Optionally translate missing entries
"""

import csv
import glob
import os
import sys
from pathlib import Path

# Import language code mapping from config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
try:
    from config import LANGUAGE_CODE_TO_NAME_MAP
    HAS_CONFIG = True
except ImportError:
    # Fallback if config.py not available
    HAS_CONFIG = False
    LANGUAGE_CODE_TO_NAME_MAP = {
        "chn": "Chinese",
        "enu": "English",
        "fra": "French",
        "spa": "Spanish",
        "deu": "German",
        "ara": "Arabic",
        "heb": "Hebrew",
        "hin": "Hindi",
        "jpn": "Japanese",
        "kor": "Korean",
        "rus": "Russian",
        "tha": "Thai",
        "grk": "Greek",
        "fas": "Persian",
        "tur": "Turkish",
        "kat": "Georgian",
        "hye": "Armenian",
        "vie": "Vietnamese"
    }


def extract_lang_code(filename):
    """Extract 3-letter language code from filename."""
    stem = Path(filename).stem
    parts = stem.split('-')
    if len(parts) >= 2:
        lang = parts[-1]
        if len(lang) == 3 and lang.isalpha():
            return lang.lower()
    return None


def get_language_name(lang_code):
    """Convert language code to full language name."""
    return LANGUAGE_CODE_TO_NAME_MAP.get(lang_code, lang_code.upper())


def read_language_file(filepath, is_reference=False):
    """
    Read a language-specific word file.

    Args:
        filepath: Path to the .txt file
        is_reference: If True, this is the reference language (no english_original column)

    Returns:
        dict: Mapping from reference_word -> translated_word
    """
    words_map = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            word = row['word'].strip()

            # Skip comments and empty lines
            if not word or word.startswith('#'):
                continue

            if is_reference:
                # For reference language, word maps to itself
                words_map[word] = word
            else:
                # For other languages, use english_original column if available
                if 'english_original' in row:
                    english_original = row['english_original'].strip()
                    if english_original:
                        # Handle duplicates: prefer first occurrence
                        if english_original not in words_map:
                            words_map[english_original] = word

    return words_map


def auto_translate_missing(english_words, target_lang_code, use_auto=False):
    """
    Attempt to translate missing words.

    Args:
        english_words: List of English words to translate
        target_lang_code: Target language code (e.g., 'deu', 'chn')
        use_auto: If True, attempt automatic translation (requires external service)

    Returns:
        dict: Mapping from English word to translated word
    """
    translations = {}

    if not use_auto or not english_words:
        # Return empty translations if auto-translate not requested
        return translations

    # Try using deep_translator if available
    try:
        from deep_translator import GoogleTranslator

        # Map internal codes to Google Translate codes
        lang_map = {
            'chn': 'zh-CN',
            'enu': 'en',
            'deu': 'de',
            'fra': 'fr',
            'spa': 'es',
            'ara': 'ar',
            'jpn': 'ja',
            'kor': 'ko',
            'rus': 'ru'
        }

        target = lang_map.get(target_lang_code, 'en')
        translator = GoogleTranslator(source='en', target=target)

        print(f"\n⚡ Auto-translating {len(english_words)} words to {get_language_name(target_lang_code)}...")

        for word in english_words:
            try:
                translation = translator.translate(word)
                translations[word] = translation
                print(f"  ✓ {word} → {translation}")
            except Exception as e:
                print(f"  ✗ Failed to translate '{word}': {e}")

    except ImportError:
        print(f"\n⚠️  Auto-translation not available. Install deep-translator:")
        print(f"   pip install deep-translator")
    except Exception as e:
        print(f"\n⚠️  Auto-translation failed: {e}")

    return translations


def merge_languages(dataset_name, input_dir='.', auto_translate=False):
    """
    Merge all language files for a dataset into SA format.

    Args:
        dataset_name: Base name of the dataset (e.g., "ACL-2-word-v2")
        input_dir: Directory containing the .txt files
        auto_translate: If True, attempt to auto-translate missing entries
    """
    # Find all language files for this dataset
    pattern = os.path.join(input_dir, f"{dataset_name}-*.txt")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"❌ No language files found for dataset: {dataset_name}")
        print(f"   Searched pattern: {pattern}")
        return

    print(f"Found {len(files)} language file(s) for '{dataset_name}':")

    # Extract language codes and identify reference language (English)
    lang_files = {}
    reference_lang = None

    for filepath in files:
        lang_code = extract_lang_code(filepath)
        if lang_code:
            lang_files[lang_code] = filepath
            lang_name = get_language_name(lang_code)
            print(f"  - {lang_name} ({lang_code}): {os.path.basename(filepath)}")
            if lang_code == 'enu':
                reference_lang = 'enu'

    if not lang_files:
        print("❌ Could not extract language codes from filenames")
        return

    # Use English as reference if available, otherwise first language
    if reference_lang is None:
        reference_lang = sorted(lang_files.keys())[0]
        print(f"\n⚠️  No English file found, using {get_language_name(reference_lang)} as reference")
    else:
        print(f"\n✓ Using {get_language_name(reference_lang)} as reference language")

    # Load all language data
    lang_data = {}
    reference_words = []

    for lang_code, filepath in sorted(lang_files.items()):
        is_ref = (lang_code == reference_lang)
        words_map = read_language_file(filepath, is_reference=is_ref)
        lang_data[lang_code] = words_map

        if is_ref:
            reference_words = sorted(words_map.keys())

        lang_name = get_language_name(lang_code)
        print(f"  {lang_name}: {len(words_map)} words")

    # Auto-translate missing words if requested
    if auto_translate:
        for lang_code in lang_files.keys():
            if lang_code == reference_lang:
                continue

            # Find missing translations
            missing_words = [word for word in reference_words if word not in lang_data[lang_code]]

            if missing_words:
                print(f"\n🔍 {get_language_name(lang_code)}: {len(missing_words)} missing translations")
                translations = auto_translate_missing(missing_words, lang_code, use_auto=True)

                # Add translations to lang_data
                for eng_word, translation in translations.items():
                    lang_data[lang_code][eng_word] = translation

    # Create SA format CSV
    output_file = os.path.join(input_dir, f"{dataset_name}-SA.csv")

    # Sort language codes for consistent column order (reference first)
    sorted_langs = [reference_lang] + sorted([l for l in lang_files.keys() if l != reference_lang])

    # Convert codes to names for column headers (lowercase for Semanscope compatibility)
    column_headers = [get_language_name(lang_code).lower() for lang_code in sorted_langs]

    matched_count = 0
    total_count = len(reference_words)
    partial_counts = {lang: 0 for lang in sorted_langs}

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)

        # Write header with language names
        writer.writerow(column_headers)

        # Write data rows
        for ref_word in reference_words:
            row = []
            has_all_langs = True

            for lang_code in sorted_langs:
                if ref_word in lang_data[lang_code]:
                    row.append(lang_data[lang_code][ref_word])
                    partial_counts[lang_code] += 1
                else:
                    # Missing translation
                    row.append('')
                    has_all_langs = False

            writer.writerow(row)

            if has_all_langs:
                matched_count += 1

    print(f"\n{'='*70}")
    print(f"✓ Created: {output_file}")
    print(f"{'='*70}")
    print(f"  Total rows: {total_count}")
    print(f"  Complete translations (all languages): {matched_count}")
    print(f"  Partial translations: {total_count - matched_count}")
    print(f"\n  Translations per language:")
    for lang_code in sorted_langs:
        lang_name = get_language_name(lang_code)
        count = partial_counts[lang_code]
        percentage = (count / total_count * 100) if total_count > 0 else 0
        print(f"    {lang_name:15} {count:4}/{total_count} ({percentage:5.1f}%)")
    print(f"\n  Columns: {', '.join(column_headers)}")

    return output_file


def main():
    if len(sys.argv) < 2:
        print("Usage: python merge_langs_to_sa.py <dataset-name> [--auto-translate]")
        print("\nExamples:")
        print("  python merge_langs_to_sa.py ACL-2-word-v2")
        print("  python merge_langs_to_sa.py ACL-2-word-v2 --auto-translate")
        print("  python merge_langs_to_sa.py ACL-1-Alphabets")
        print("\nOptions:")
        print("  --auto-translate    Automatically translate missing entries using Google Translate")
        print("                      (requires: pip install deep-translator)")
        print("\nAvailable datasets in current directory:")

        # List available datasets
        txt_files = glob.glob("*-*.txt")
        datasets = set()
        for f in txt_files:
            stem = Path(f).stem
            parts = stem.split('-')
            if len(parts) >= 2:
                lang = parts[-1]
                if len(lang) == 3 and lang.isalpha():
                    dataset = '-'.join(parts[:-1])
                    datasets.add(dataset)

        for dataset in sorted(datasets):
            print(f"  - {dataset}")

        sys.exit(1)

    dataset_name = sys.argv[1]
    auto_translate = '--auto-translate' in sys.argv
    input_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"{'='*70}")
    print(f"Merging language files for: {dataset_name}")
    if auto_translate:
        print(f"Mode: AUTO-TRANSLATE (will attempt to translate missing entries)")
    else:
        print(f"Mode: MERGE ONLY (no automatic translation)")
    print(f"{'='*70}\n")

    merge_languages(dataset_name, input_dir, auto_translate)


if __name__ == "__main__":
    main()
