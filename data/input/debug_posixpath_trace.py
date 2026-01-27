#!/usr/bin/env python3
"""
Debug script to trace exactly where PosixPath .lower() might still be happening
"""
import sys
from pathlib import Path
import traceback

def analyze_potential_posixpath_issues():
    print("🔍 DEBUGGING POSIXPATH ISSUES - COMPREHENSIVE ANALYSIS:")
    print("=" * 70)
    print()

    print("✅ VERIFIED FIXES IN PLACE:")
    print("   1. plotting_echarts.py:799 - str(filename).lower()")
    print("   2. plotting_echarts.py:851 - str(filename).lower()")
    print("   3. plotting_echarts.py:781 - str(external_filename).replace()")
    print("   4. plotting_echarts.py:1072 - chart_element.screenshot(str(filepath))")
    print("   5. plotting_echarts.py:1075 - return str(filepath)")
    print("   6. title_filename_helper.py:78,80 - str(img_format).lower()")
    print("   7. Multiple str() conversions for Path operations")
    print()

    print("🕵️ POTENTIAL REMAINING SOURCES OF ERROR:")
    print()

    print("   1️⃣  INDIRECT PATH OBJECT USAGE:")
    print("      • Function returns Path object that gets .lower() called elsewhere")
    print("      • Variable assignments where Path gets treated as string")
    print("      • Method chaining where intermediate result is Path")
    print()

    print("   2️⃣  CACHING OR MEMOIZATION:")
    print("      • Streamlit session state might cache old Path objects")
    print("      • Function results cached before fixes were applied")
    print("      • Browser cache of old JavaScript/HTML state")
    print()

    print("   3️⃣  STREAMLIT RERUN BEHAVIOR:")
    print("      • Previous execution state persisting in memory")
    print("      • Widget state carrying over Path objects")
    print("      • File system watcher events with Path objects")
    print()

    print("   4️⃣  CONCURRENT EXECUTION:")
    print("      • Multiple threads/processes using different code versions")
    print("      • Background tasks still running old code")
    print("      • Jupyter notebook kernel state")
    print()

    print("🧪 RECOMMENDED DEBUGGING STEPS:")
    print()
    print("   1. RESTART EVERYTHING:")
    print("      • Stop Streamlit server")
    print("      • Clear browser cache")
    print("      • Restart Python kernel/environment")
    print("      • Fresh run with updated code")
    print()

    print("   2. ADD TRACING:")
    print("      • Add debug prints before .lower() calls")
    print("      • Check type() of variables before string operations")
    print("      • Log full stack trace when error occurs")
    print()

    print("   3. SEARCH FOR MISSED LOCATIONS:")
    print("      • grep -r \"\.lower()\" across entire codebase")
    print("      • Look for f-strings or format() calls with Path objects")
    print("      • Check for lambda functions or list comprehensions")
    print()

    print("🎯 SPECIFIC DEBUGGING CODE TO ADD:")
    print()

    debugging_code = '''
# Add this before any suspected .lower() calls:
if hasattr(variable_name, 'lower'):
    if str(type(variable_name)) == "<class 'pathlib.PosixPath'>":
        print(f"🚨 FOUND POSIXPATH: {variable_name} at {traceback.format_stack()[-2]}")
        variable_name = str(variable_name)
    result = variable_name.lower()
else:
    print(f"🚨 NO .lower() ATTR: {type(variable_name)} = {variable_name}")
'''

    print(debugging_code)

    print("🏁 NEXT STEPS IF ERROR PERSISTS:")
    print("   1. Completely restart Streamlit application")
    print("   2. Test with minimal reproduction case")
    print("   3. Add type checking debug code at suspected locations")
    print("   4. Check if error occurs in Publication mode vs normal mode")
    print("   5. Test with different datasets/models to isolate trigger")
    print()

    print("💡 The fact that our test script passes but the error persists")
    print("   suggests the issue may be in the live application state,")
    print("   caching, or a code path we haven't identified yet.")

if __name__ == "__main__":
    analyze_potential_posixpath_issues()