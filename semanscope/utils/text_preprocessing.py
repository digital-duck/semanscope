"""
Text preprocessing utilities for handling Chinese radicals and other problematic characters
that may cause NaN issues in embedding models.
"""
import re
import unicodedata
from typing import List, Tuple
import streamlit as st


class ChineseTextPreprocessor:
    """Preprocessor for Chinese text to handle radicals and special characters"""

    def __init__(self):
        # Common problematic Unicode ranges that may cause embedding issues
        self.problematic_ranges = [
            (0x2E80, 0x2EFF),  # CJK Radicals Supplement
            (0x2F00, 0x2FDF),  # Kangxi Radicals
            (0x2FF0, 0x2FFF),  # Ideographic Description Characters
            (0x3400, 0x4DBF),  # CJK Extension A (rare characters)
            (0x20000, 0x2A6DF), # CJK Extension B (very rare characters)
            (0x2A700, 0x2B73F), # CJK Extension C
            (0x2B740, 0x2B81F), # CJK Extension D
            (0x2B820, 0x2CEAF), # CJK Extension E
        ]

        # Safe Chinese character ranges
        self.safe_ranges = [
            (0x4E00, 0x9FFF),   # CJK Unified Ideographs (main block)
            (0x3400, 0x4DBF),   # CJK Extension A (commonly used)
        ]

        # Common safe radicals (Kangxi radicals that are typically well-supported)
        self.safe_radicals = {
            '一', '丨', '丶', '丿', '乙', '亅', '二', '亠', '人', '儿',
            '入', '八', '冂', '冖', '冫', '几', '凵', '刀', '力', '勹',
            '匕', '匚', '匸', '十', '卜', '卩', '厂', '厶', '又', '口',
            '囗', '土', '士', '夂', '夊', '夕', '大', '女', '子', '宀',
            '寸', '小', '尢', '尸', '屮', '山', '川', '工', '己', '巾',
            '干', '幺', '广', '廴', '廾', '弋', '弓', '彐', '彡', '彳',
            '心', '戈', '戶', '手', '支', '攴', '文', '斗', '斤', '方',
            '无', '日', '曰', '月', '木', '欠', '止', '歹', '殳', '毋',
            '比', '毛', '氏', '气', '水', '火', '爪', '父', '爻', '爿',
            '片', '牙', '牛', '犬', '玄', '玉', '瓜', '瓦', '甘', '生',
            '用', '田', '疋', '疒', '癶', '白', '皮', '皿', '目', '矛',
            '矢', '石', '示', '禸', '禾', '穴', '立', '竹', '米', '糸',
            '缶', '网', '羊', '羽', '老', '而', '耒', '耳', '聿', '肉',
            '臣', '自', '至', '臼', '舌', '舛', '舟', '艮', '色', '艸',
            '虍', '虫', '血', '行', '衣', '襾', '見', '角', '言', '谷',
            '豆', '豕', '豸', '貝', '赤', '走', '足', '身', '車', '辛',
            '辰', '辵', '邑', '酉', '釆', '里', '金', '長', '門', '阜',
            '隶', '隹', '雨', '青', '非', '面', '革', '韋', '韭', '音',
            '頁', '風', '飛', '食', '首', '香', '馬', '骨', '高', '髟',
            '鬥', '鬯', '鬲', '鬼', '魚', '鳥', '鹵', '鹿', '麥', '麻',
            '黃', '黍', '黑', '黹', '黽', '鼎', '鼓', '鼠', '鼻', '齊',
            '齒', '龍', '龜', '龠'
        }

    def is_problematic_character(self, char: str) -> bool:
        """Check if a character is in a problematic Unicode range"""
        code = ord(char)

        # Check against problematic ranges
        for start, end in self.problematic_ranges:
            if start <= code <= end:
                # Exception: if it's a commonly safe radical, allow it
                if char in self.safe_radicals:
                    return False
                return True

        return False

    def is_safe_chinese_character(self, char: str) -> bool:
        """Check if a character is in a safe Chinese character range"""
        code = ord(char)

        # Check against safe ranges
        for start, end in self.safe_ranges:
            if start <= code <= end:
                return True

        # Also allow safe radicals
        if char in self.safe_radicals:
            return True

        return False

    def preprocess_text_list(self, texts: List[str], warn_on_changes: bool = True) -> Tuple[List[str], List[str]]:
        """
        Preprocess a list of texts to handle problematic characters

        Returns:
            tuple: (processed_texts, warnings)
        """
        processed_texts = []
        warnings = []

        for i, text in enumerate(texts):
            processed_text, text_warnings = self.preprocess_single_text(text, warn_on_changes)
            processed_texts.append(processed_text)

            if text_warnings:
                warnings.extend([f"Text {i+1}: {w}" for w in text_warnings])

        return processed_texts, warnings

    def preprocess_single_text(self, text: str, warn_on_changes: bool = True) -> Tuple[str, List[str]]:
        """
        Preprocess a single text to handle problematic characters

        Returns:
            tuple: (processed_text, warnings)
        """
        if not text or not text.strip():
            return text, []

        original_text = text
        warnings = []

        # Remove or replace problematic characters
        processed_chars = []
        removed_chars = set()

        for char in text:
            if self.is_problematic_character(char):
                # Try to find a replacement or skip
                replacement = self._get_character_replacement(char)
                if replacement:
                    processed_chars.append(replacement)
                    if char not in removed_chars:
                        removed_chars.add(char)
                else:
                    # Skip problematic character
                    if char not in removed_chars:
                        removed_chars.add(char)
            else:
                processed_chars.append(char)

        processed_text = ''.join(processed_chars)

        # Generate warnings
        if removed_chars and warn_on_changes:
            removed_list = list(removed_chars)
            warnings.append(f"Removed/replaced problematic characters: {', '.join(removed_list)}")

        # Check if text became empty or too short
        if len(processed_text.strip()) == 0 and len(original_text.strip()) > 0:
            # warnings.append("Text became empty after preprocessing")
            processed_text = "□"  # Use a safe placeholder
        elif len(processed_text.strip()) < len(original_text.strip()) * 0.5:
            warnings.append(f"Text significantly shortened: '{original_text[:20]}...' → '{processed_text[:20]}...'")

        return processed_text, warnings

    def _get_character_replacement(self, char: str) -> str:
        """Get a replacement for a problematic character"""
        # Common radical replacements
        radical_replacements = {
            # Ideographic Description Characters -> safe alternatives
            '⿰': '',  # Left-right composition
            '⿱': '',  # Top-bottom composition
            '⿲': '',  # Left-middle-right composition
            '⿳': '',  # Top-middle-bottom composition
            '⿴': '',  # Surround composition
            '⿵': '',  # Surround from above
            '⿶': '',  # Surround from below
            '⿷': '',  # Surround from left
            '⿸': '',  # Surround from upper left
            '⿹': '',  # Surround from upper right
            '⿺': '',  # Surround from lower left
            '⿻': '',  # Overlaid composition
        }

        return radical_replacements.get(char, '')

    def analyze_text_list(self, texts: List[str]) -> dict:
        """Analyze a list of texts for potential issues"""
        analysis = {
            'total_texts': len(texts),
            'problematic_texts': 0,
            'problematic_characters': set(),
            'empty_texts': 0,
            'very_short_texts': 0,
            'recommendations': []
        }

        for text in texts:
            if not text or not text.strip():
                analysis['empty_texts'] += 1
                continue

            if len(text.strip()) <= 1:
                analysis['very_short_texts'] += 1

            has_problematic = False
            for char in text:
                if self.is_problematic_character(char):
                    analysis['problematic_characters'].add(char)
                    has_problematic = True

            if has_problematic:
                analysis['problematic_texts'] += 1

        # Generate recommendations
        if analysis['problematic_texts'] > 0:
            analysis['recommendations'].append(
                f"⚠️ {analysis['problematic_texts']} texts contain potentially problematic characters"
            )

        if analysis['empty_texts'] > 0:
            analysis['recommendations'].append(
                f"❌ {analysis['empty_texts']} texts are empty or whitespace-only"
            )

        if analysis['very_short_texts'] > 0:
            analysis['recommendations'].append(
                f"⚠️ {analysis['very_short_texts']} texts are very short (≤1 character)"
            )

        if len(analysis['problematic_characters']) > 0:
            char_list = ', '.join(sorted(analysis['problematic_characters']))[:100]
            if len(char_list) == 100:
                char_list += "..."
            analysis['recommendations'].append(
                f"🔍 Problematic characters found: {char_list}"
            )

        return analysis


def preprocess_texts_for_embedding(texts: List[str], show_warnings: bool = True) -> List[str]:
    """
    Convenience function to preprocess texts before embedding generation

    Args:
        texts: List of input texts
        show_warnings: Whether to show preprocessing warnings in Streamlit

    Returns:
        List of preprocessed texts
    """
    preprocessor = ChineseTextPreprocessor()

    # Analyze texts first
    analysis = preprocessor.analyze_text_list(texts)

    if show_warnings and analysis['recommendations']:
        with st.expander("🔍 Text Preprocessing Analysis", expanded=False):
            st.markdown("**Potential Issues Detected:**")
            for rec in analysis['recommendations']:
                st.markdown(f"- {rec}")

    # Preprocess texts
    processed_texts, warnings = preprocessor.preprocess_text_list(texts, warn_on_changes=show_warnings)

    if show_warnings and warnings:
        with st.expander("🔧 Text Preprocessing Changes", expanded=False):
            st.markdown("**Characters Modified/Removed:**")
            for warning in warnings:  
                st.markdown(f"- {warning}")

    return processed_texts