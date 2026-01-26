"""
Title and Filename Helper for Semanscope
Centralized utility for creating consistent chart titles and image filenames across all Semanscope pages.
"""

import re
from pathlib import Path
from typing import List, Tuple, Union
from config import MODEL_INFO, METHOD_INFO, DATASET_INFO


def create_title_and_filename(
    method_names: List[str],
    model_names: List[str],
    dataset_name: str,
    lang_codes: List[str],
    img_format: Union[str, Path]
) -> Tuple[str, str]:
    """
    Create standardized chart title and filename for Semanscope visualizations.

    Args:
        method_names: List of method names (e.g., ["PHATE", "t-SNE"])
        model_names: List of model names (e.g., ["sentence-bert-multilingual", "Qwen3-Embedding"])
        dataset_name: Dataset name (e.g., "sample-1")
        lang_codes: List of language codes (e.g., ["CHN", "ENU"])
        img_format: Image format ("png" or "pdf")

    Returns:
        Tuple of (chart_title, filename)

    Examples:
        >>> create_title_and_filename(["PHATE"], ["sentence-bert-multilingual"], "sample-1", ["CHN", "ENU"], "png")
        ("[Dataset] sample-1, [Method] PHATE, [Model] SBERT, [Lang] CHN+ENU", "sample-1-phate-sbert-chn+enu.png")
    """

    # Get aliases for methods, models, and dataset
    method_aliases = [_get_alias_from_info(method, METHOD_INFO) for method in method_names]
    model_aliases = [_get_alias_from_info(model, MODEL_INFO) for model in model_names]
    dataset_alias = _get_alias_from_info(dataset_name, DATASET_INFO)

    # Create chart title components
    dataset_part = f"[Dataset] {dataset_alias}"

    # Handle singular vs plural for methods and models
    if len(method_aliases) == 1:
        method_part = f"[Method] {method_aliases[0]}"
    else:
        method_part = f"[Methods] {'+'.join(method_aliases)}"

    if len(model_aliases) == 1:
        model_part = f"[Model] {model_aliases[0]}"
    else:
        model_part = f"[Models] {'+'.join(model_aliases)}"

    # Handle singular vs plural for languages
    if len(lang_codes) == 1:
        lang_part = f"[Language] {'+'.join(lang_codes)}"
    else:
        lang_part = f"[Languages] {'+'.join(lang_codes)}"

    # Combine into final chart title (Languages after Dataset)
    chart_title = f"{dataset_part}, {lang_part}, {method_part}, {model_part}"

    # Create normalized filename
    filename_parts = [
        _normalize_for_filename(dataset_alias),
        _normalize_for_filename('+'.join(method_aliases)),
        _normalize_for_filename('+'.join(model_aliases)),
        _normalize_for_filename('+'.join(lang_codes))
    ]

    # Remove any empty parts
    filename_parts = [part for part in filename_parts if part]

    # Combine with format (handle both strings and Path objects)
    if isinstance(img_format, Path):
        format_str = str(img_format).lower()
    elif img_format:
        format_str = str(img_format).lower()
    else:
        format_str = "png"
    filename = f"{'-'.join(filename_parts)}.{format_str}"

    return chart_title, filename


def _get_alias_from_info(name: str, info_dict: dict) -> str:
    """
    Get alias from info dictionary, fallback to original name.

    Args:
        name: Original name to look up
        info_dict: Dictionary like MODEL_INFO or METHOD_INFO

    Returns:
        Alias if available, otherwise original name
    """
    if name in info_dict:
        item_info = info_dict[name]
        if isinstance(item_info, dict) and 'alias' in item_info:
            return item_info['alias']

    # Fallback to original name
    return name


def _normalize_for_filename(text: Union[str, Path]) -> str:
    """
    Normalize text for filename: lowercase, strip special chars, replace spaces with hyphens.

    Args:
        text: Text to normalize (string or Path object)

    Returns:
        Normalized text safe for filename
    """
    if not text:
        return ""

    # Handle PosixPath objects by converting to string first
    if isinstance(text, Path):
        text = str(text)

    # Ensure we have a string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Replace spaces and multiple separators with single hyphen
    text = re.sub(r'[\s\-_+]+', '-', text)

    # Remove special characters except hyphens and alphanumeric
    text = re.sub(r'[^a-z0-9\-+]', '', text)

    # Clean up multiple hyphens and leading/trailing hyphens
    text = re.sub(r'-+', '-', text)
    text = text.strip('-')

    return text


# Backwards compatibility: create individual title/filename functions
def create_chart_title(method_names: List[str], model_names: List[str], dataset_name: str, lang_codes: List[str]) -> str:
    """Create chart title only (for backwards compatibility)."""
    title, _ = create_title_and_filename(method_names, model_names, dataset_name, lang_codes, "png")
    return title


def create_filename(method_names: List[str], model_names: List[str], dataset_name: str, lang_codes: List[str], img_format: str = "png") -> str:
    """Create filename only (for backwards compatibility)."""
    _, filename = create_title_and_filename(method_names, model_names, dataset_name, lang_codes, img_format)
    return filename


# Quick test function for development
def _test_helper():
    """Test the helper function with sample data."""
    print("Testing title_filename_helper...")

    # Test single values
    title1, filename1 = create_title_and_filename(
        ["PHATE"], ["sentence-bert-multilingual"], "sample-1", ["CHN"], "png"
    )
    print(f"Single: '{title1}' -> '{filename1}'")

    # Test multiple values (Compare page scenario)
    title2, filename2 = create_title_and_filename(
        ["PHATE", "t-SNE"], ["sentence-bert-multilingual", "Qwen3-Embedding"], "sample-1", ["CHN", "ENU"], "png"
    )
    print(f"Multiple: '{title2}' -> '{filename2}'")


if __name__ == "__main__":
    _test_helper()