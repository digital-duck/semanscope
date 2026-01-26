#!/usr/bin/env python3
"""
Text Decomposer Tool

Decomposes text (poems, passages, phrases) into hierarchical components for semantic geometry analysis:
- Individual characters (Chinese) or words (English)
- Multi-character phrases / word combinations
- Complete lines / sentences
- Full text

Saves output to a single text file for use in Semanscope.
"""

import click
import re
from pathlib import Path
from typing import List


class TextDecomposer:
    """Decomposes text into hierarchical semantic components"""

    def __init__(self, language: str = 'auto'):
        """
        Initialize decomposer

        Args:
            language: 'chn' for Chinese, 'enu' for English, 'auto' for auto-detect
        """
        self.language = language

    def is_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters"""
        return bool(re.search(r'[\u4e00-\u9fff]', text))

    def decompose_chinese_text(self, text_lines: List[str]) -> List[str]:
        """
        Decompose Chinese text into hierarchical components

        Returns list of components in order:
        1. Individual characters
        2. 2-character phrases
        3. Multi-character phrases (3+)
        4. Complete lines
        5. Full text
        """
        components = []
        all_chars = []

        # Clean and validate lines
        text_lines = [line.strip() for line in text_lines if line.strip()]

        # Level 1: Individual characters
        for line in text_lines:
            for char in line:
                if self.is_chinese(char):
                    all_chars.append(char)

        components.extend(all_chars)
        components.append("")  # Separator

        # Level 2: Generate n-character phrases
        two_char_phrases = set()
        three_char_phrases = set()
        longer_phrases = set()

        for line in text_lines:
            chars = [c for c in line if self.is_chinese(c)]

            # 2-character phrases
            for i in range(len(chars) - 1):
                two_char_phrases.add(chars[i] + chars[i+1])

            # 3-character phrases
            for i in range(len(chars) - 2):
                three_char_phrases.add(chars[i] + chars[i+1] + chars[i+2])

            # Longer phrases (up to line length - 1)
            for length in range(4, len(chars)):
                for i in range(len(chars) - length + 1):
                    phrase = ''.join(chars[i:i+length])
                    longer_phrases.add(phrase)

        # Add phrases in order: 2-char, 3-char, longer
        components.extend(sorted(two_char_phrases))
        if three_char_phrases:
            components.append("")
            components.extend(sorted(three_char_phrases))
        if longer_phrases:
            components.append("")
            components.extend(sorted(longer_phrases))

        # Level 3: Complete lines
        components.append("")
        components.extend(text_lines)

        # Level 4: Full text (all lines concatenated)
        components.append("")
        full_text = ''.join(text_lines)
        components.append(full_text)

        return components

    def decompose_english_text(self, text_lines: List[str]) -> List[str]:
        """
        Decompose English text into hierarchical components

        Returns list of components in order:
        1. Individual words
        2. 2-word phrases
        3. 3-word phrases
        4. Complete lines
        5. Full text
        """
        components = []
        all_words = []

        # Clean and validate lines
        text_lines = [line.strip() for line in text_lines if line.strip()]

        # Level 1: Individual words (remove punctuation)
        for line in text_lines:
            words = re.findall(r'\b[a-zA-Z]+\b', line.lower())
            all_words.extend(words)

        # Remove duplicates while preserving order
        seen = set()
        unique_words = []
        for word in all_words:
            if word not in seen:
                seen.add(word)
                unique_words.append(word)

        components.extend(unique_words)
        components.append("")  # Separator

        # Level 2: Generate n-word phrases
        two_word_phrases = set()
        three_word_phrases = set()

        for line in text_lines:
            words = re.findall(r'\b[a-zA-Z]+\b', line.lower())

            # 2-word phrases
            for i in range(len(words) - 1):
                two_word_phrases.add(f"{words[i]} {words[i+1]}")

            # 3-word phrases
            for i in range(len(words) - 2):
                three_word_phrases.add(f"{words[i]} {words[i+1]} {words[i+2]}")

        components.extend(sorted(two_word_phrases))
        if three_word_phrases:
            components.append("")
            components.extend(sorted(three_word_phrases))

        # Level 3: Complete lines (preserve original case and punctuation)
        components.append("")
        components.extend(text_lines)

        # Level 4: Full text (all lines joined)
        components.append("")
        full_text = ' '.join(text_lines)
        components.append(full_text)

        return components

    def decompose_text(self, text: str) -> List[str]:
        """
        Main decomposition method - automatically detects language if needed

        Args:
            text: Multi-line text

        Returns:
            List of hierarchical components
        """
        lines = text.strip().split('\n')

        # Determine language
        if self.language == 'auto':
            use_chinese = self.is_chinese(text)
        elif self.language == 'chn':
            use_chinese = True
        else:
            use_chinese = False

        if use_chinese:
            return self.decompose_chinese_text(lines)
        else:
            return self.decompose_english_text(lines)


@click.command()
@click.option(
    '-i', '--input',
    'input_file',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Input text file (plain text, one line per sentence/verse)'
)
@click.option(
    '-o', '--output',
    'output_file',
    type=click.Path(path_type=Path),
    required=True,
    help='Output file path (e.g., text-name-chn.txt or text-name-enu.txt)'
)
@click.option(
    '--lang',
    type=click.Choice(['chn', 'enu', 'auto'], case_sensitive=False),
    default='auto',
    help='Language: chn (Chinese), enu (English), auto (auto-detect)'
)
@click.option(
    '--preview',
    is_flag=True,
    help='Preview decomposition without saving'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Show detailed output'
)
def main(input_file: Path, output_file: Path, lang: str, preview: bool, verbose: bool):
    """
    Decompose text into hierarchical components for semantic geometry analysis.

    \b
    Examples:
      # Chinese poem
      text_decomposer.py -i jingye.txt -o poem-li-bai-moonlight-chn.txt

      # English poem
      text_decomposer.py -i sonnet.txt -o poem-shakespeare-sonnet18-enu.txt

      # Preview mode
      text_decomposer.py -i text.txt -o output.txt --preview

      # Specify language explicitly
      text_decomposer.py -i text.txt -o output.txt --lang chn
    """

    # Initialize decomposer
    decomposer = TextDecomposer(language=lang)

    # Read input
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    if not text.strip():
        click.echo(click.style("❌ Error: Input file is empty", fg='red'))
        return 1

    # Decompose
    components = decomposer.decompose_text(text)

    # Determine detected language
    detected_lang = 'Chinese' if decomposer.is_chinese(text) else 'English'

    if preview:
        click.echo(click.style("\n📋 Preview of decomposed components:\n", fg='cyan', bold=True))
        for i, component in enumerate(components[:40], 1):  # Show first 40
            if component:
                click.echo(f"{i:3d}. {component}")
            else:
                click.echo(click.style("     ---", fg='bright_black'))

        if len(components) > 40:
            remaining = len(components) - 40
            click.echo(click.style(f"\n... and {remaining} more components", fg='yellow'))
    else:
        # Save output
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for component in components:
                f.write(component + '\n')

        click.echo(click.style(f"✅ Saved {len([c for c in components if c])} components to: {output_file}", fg='green'))

    # Print summary
    if verbose or preview:
        click.echo(click.style("\n📊 Decomposition Summary:", fg='cyan', bold=True))
        click.echo(f"   Language: {detected_lang}")
        click.echo(f"   Total components: {len([c for c in components if c])}")
        click.echo(f"   Empty separators: {len([c for c in components if not c])}")
        click.echo(f"   Input: {input_file}")
        if not preview:
            click.echo(f"   Output: {output_file}")

    return 0


if __name__ == '__main__':
    main()
