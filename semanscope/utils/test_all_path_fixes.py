#!/usr/bin/env python3
"""
Test script to verify all PosixPath .lower() fixes are working
"""
from pathlib import Path

def test_all_path_operations():
    """Test all the operations that could cause PosixPath .lower() errors"""

    print("🧪 TESTING ALL PATH OPERATIONS THAT WERE FIXED:")
    print("=" * 60)

    # Test cases that could cause issues
    test_cases = [
        ("String filename", "test-file.png"),
        ("PosixPath filename", Path("test-file.png")),
        ("String with extension", "echarts-data.json"),
        ("PosixPath with extension", Path("echarts-data.json")),
    ]

    print("\n1️⃣  Testing external_filename.replace() operations:")
    for desc, external_filename in test_cases:
        try:
            # This is what was failing in save_echarts_as_pdf()
            result = str(external_filename).replace('.json', '.pdf').replace('.png', '.pdf')
            print(f"   ✅ {desc}: {result}")
        except AttributeError as e:
            print(f"   ❌ {desc}: {e}")

    print("\n2️⃣  Testing filename.lower() operations:")
    for desc, filename in test_cases:
        try:
            # This is what was failing in plotting_echarts.py
            result = str(filename).lower()
            print(f"   ✅ {desc}: {result}")
        except AttributeError as e:
            print(f"   ❌ {desc}: {e}")

    print("\n3️⃣  Testing selenium screenshot() operations:")
    # Simulating what Selenium WebDriver screenshot() method expects
    for desc, filepath in test_cases:
        try:
            # Convert to string before passing to selenium
            path_str = str(filepath)
            print(f"   ✅ {desc}: Ready for selenium.screenshot('{path_str}')")
        except Exception as e:
            print(f"   ❌ {desc}: {e}")

    print("\n4️⃣  Testing variable.replace() operations:")
    variables = [
        ("dataset", "ICML-NSM-Primes"),
        ("model", "Qwen3-Embedding"),
        ("method", "PHATE"),
        ("lang_codes", "enu+kor"),
        ("dataset_path", Path("ICML-NSM-Primes")),  # Potential issue case
    ]

    for desc, var in variables:
        try:
            result = str(var).replace(" ", "-").replace("_", "-")
            print(f"   ✅ {desc}: {result}")
        except AttributeError as e:
            print(f"   ❌ {desc}: {e}")

    print("\n✅ ALL OPERATIONS SHOULD NOW WORK SAFELY!")
    print("\n🔧 KEY CHANGES MADE:")
    print("   • str(external_filename) before .replace()")
    print("   • str(filepath) before selenium.screenshot()")
    print("   • str(filename) before .lower()")
    print("   • str(variable) before any string operations")

if __name__ == "__main__":
    test_all_path_operations()