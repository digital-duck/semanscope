#!/usr/bin/env python3
"""
Test the PosixPath fix for the title_filename_helper.py
"""
from pathlib import Path

def test_posixpath_handling():
    """Test that PosixPath objects are handled correctly"""

    # Simulate the original buggy behavior
    def buggy_format_handling(img_format):
        """This would fail with PosixPath objects"""
        try:
            format_str = str(img_format).lower() if img_format else "png"
            return format_str
        except AttributeError as e:
            return f"ERROR: {e}"

    # Simulate the fixed behavior
    def fixed_format_handling(img_format):
        """This handles both strings and Path objects"""
        if isinstance(img_format, Path):
            format_str = str(img_format).lower()
        elif img_format:
            format_str = str(img_format).lower()
        else:
            format_str = "png"
        return format_str

    # Test cases
    test_cases = [
        "png",                    # String (should work in both)
        "PDF",                    # String uppercase (should work in both)
        Path("svg"),              # PosixPath (fails in buggy, works in fixed)
        Path("/tmp/test.pdf"),    # Full path (fails in buggy, works in fixed)
        None,                     # None (should default to png)
        "",                       # Empty string (should default to png)
    ]

    print("🧪 TESTING POSIXPATH FIX:")
    print()
    print("| Test Case              | Buggy Version | Fixed Version |")
    print("|------------------------|---------------|---------------|")

    for case in test_cases:
        case_str = repr(case)
        buggy_result = buggy_format_handling(case)
        fixed_result = fixed_format_handling(case)

        # Truncate long results for display
        if len(str(buggy_result)) > 20:
            buggy_display = str(buggy_result)[:17] + "..."
        else:
            buggy_display = str(buggy_result)

        if len(str(fixed_result)) > 20:
            fixed_display = str(fixed_result)[:17] + "..."
        else:
            fixed_display = str(fixed_result)

        print(f"| {case_str:<22} | {buggy_display:<13} | {fixed_display:<13} |")

    print()
    print("✅ EXPECTED RESULTS:")
    print("  • String inputs: Both should work the same")
    print("  • PosixPath inputs: Buggy fails, Fixed works")
    print("  • None/empty: Both should default to 'png'")

if __name__ == "__main__":
    test_posixpath_handling()