#!/usr/bin/env python3
"""
Fix for 'PosixPath' object has no attribute 'lower' error in ECharts auto-save PNG functionality.

The error occurs when a PosixPath object is passed to the create_title_and_filename function
instead of a string for the img_format parameter.
"""

def fixed_normalize_for_filename(text):
    """
    Fixed version of _normalize_for_filename that handles PosixPath objects.

    This should replace the function in title_filename_helper.py
    """
    import re
    from pathlib import Path

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

def fixed_create_title_and_filename(
    method_names,
    model_names,
    dataset_name,
    lang_codes,
    img_format
):
    """
    Fixed version that handles PosixPath objects properly.

    This should replace the function in title_filename_helper.py
    """
    from pathlib import Path

    # ... (keep existing code until line 75) ...

    # FIX: Handle PosixPath objects and ensure img_format is a string
    if isinstance(img_format, Path):
        format_str = str(img_format).lower()
    elif img_format:
        format_str = str(img_format).lower()
    else:
        format_str = "png"

    filename = f"{'-'.join(filename_parts)}.{format_str}"

    return chart_title, filename

def show_bug_explanation():
    """Explain the PosixPath bug"""
    print("🐛 POSIXPATH .lower() BUG EXPLANATION:")
    print()
    print("ERROR: 'PosixPath' object has no attribute 'lower'")
    print()
    print("CAUSE:")
    print("  • A pathlib.Path object was passed as img_format parameter")
    print("  • str(PosixPath).lower() fails because Path objects don't have .lower()")
    print("  • This happens when file paths are passed instead of format strings")
    print()
    print("BEFORE FIX (Buggy code):")
    print("  format_str = str(img_format).lower() if img_format else 'png'")
    print("  💥 Fails if img_format is a PosixPath object!")
    print()
    print("AFTER FIX (Robust code):")
    print("  if isinstance(img_format, Path):")
    print("      format_str = str(img_format).lower()")
    print("  elif img_format:")
    print("      format_str = str(img_format).lower()")
    print("  else:")
    print("      format_str = 'png'")
    print("  ✅ Handles both strings and Path objects correctly!")

if __name__ == "__main__":
    show_bug_explanation()
    print()
    print("To apply the fix:")
    print("1. Open /st_semantics/src/utils/title_filename_helper.py")
    print("2. Replace _normalize_for_filename() with the fixed version")
    print("3. Add Path import: from pathlib import Path")
    print("4. The PosixPath error will be resolved!")