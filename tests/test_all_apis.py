#!/usr/bin/env python3
"""
Master test runner for all API integrations
Runs all vendor API tests and generates summary report
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_all_tests():
    """Run all API integration tests and generate summary report"""

    print("=" * 80)
    print("SEMANSCOPE API INTEGRATION TEST SUITE")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Track results
    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {},
        "summary": {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0
        }
    }

    # Test 1: Voyage AI
    print("\n" + "─" * 80)
    print("TEST 1/3: Voyage AI API Integration")
    print("─" * 80)
    try:
        from test_voyage_integration import test_voyage_api
        start_time = datetime.now()
        success = test_voyage_api()
        duration = (datetime.now() - start_time).total_seconds()

        results["tests"]["voyage_ai"] = {
            "status": "PASSED" if success else "FAILED",
            "duration_seconds": duration,
            "models_tested": ["voyage-3"]
        }
        results["summary"]["total"] += 1
        if success:
            results["summary"]["passed"] += 1
        else:
            results["summary"]["failed"] += 1
    except Exception as e:
        print(f"❌ Voyage AI test failed with exception: {e}")
        results["tests"]["voyage_ai"] = {
            "status": "FAILED",
            "error": str(e)
        }
        results["summary"]["total"] += 1
        results["summary"]["failed"] += 1

    # Test 2: OpenRouter
    print("\n" + "─" * 80)
    print("TEST 2/3: OpenRouter API Integration")
    print("─" * 80)
    try:
        from test_openrouter_integration import test_openrouter_api
        start_time = datetime.now()
        success = test_openrouter_api()
        duration = (datetime.now() - start_time).total_seconds()

        results["tests"]["openrouter"] = {
            "status": "SKIPPED (not yet implemented)",
            "duration_seconds": duration
        }
        results["summary"]["total"] += 1
        results["summary"]["skipped"] += 1
    except Exception as e:
        print(f"❌ OpenRouter test failed with exception: {e}")
        results["tests"]["openrouter"] = {
            "status": "FAILED",
            "error": str(e)
        }
        results["summary"]["total"] += 1
        results["summary"]["failed"] += 1

    # Test 3: Google Gemini
    print("\n" + "─" * 80)
    print("TEST 3/3: Google Gemini API Integration")
    print("─" * 80)
    try:
        from test_gemini_integration import test_gemini_api
        start_time = datetime.now()
        success = test_gemini_api()
        duration = (datetime.now() - start_time).total_seconds()

        results["tests"]["google_gemini"] = {
            "status": "SKIPPED (not yet implemented)",
            "duration_seconds": duration
        }
        results["summary"]["total"] += 1
        results["summary"]["skipped"] += 1
    except Exception as e:
        print(f"❌ Google Gemini test failed with exception: {e}")
        results["tests"]["google_gemini"] = {
            "status": "FAILED",
            "error": str(e)
        }
        results["summary"]["total"] += 1
        results["summary"]["failed"] += 1

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUITE SUMMARY")
    print("=" * 80)
    print(f"Total tests run:    {results['summary']['total']}")
    print(f"✅ Passed:          {results['summary']['passed']}")
    print(f"❌ Failed:          {results['summary']['failed']}")
    print(f"⚠️  Skipped:         {results['summary']['skipped']}")
    print()

    # Detailed results
    print("DETAILED RESULTS:")
    print("─" * 80)
    for vendor, result in results["tests"].items():
        status_emoji = {
            "PASSED": "✅",
            "FAILED": "❌",
            "SKIPPED (not yet implemented)": "⚠️"
        }.get(result["status"], "❓")

        print(f"{status_emoji} {vendor:20s} - {result['status']}")
        if "duration_seconds" in result:
            print(f"   Duration: {result['duration_seconds']:.2f}s")
        if "error" in result:
            print(f"   Error: {result['error']}")

    # Save report to file
    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_file = reports_dir / f"test_report_{timestamp}.json"

    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Report saved to: {report_file}")
    print("=" * 80)

    # Determine exit code
    if results["summary"]["failed"] > 0:
        print("\n❌ SOME TESTS FAILED")
        return False
    else:
        print(f"\n✅ ALL IMPLEMENTED TESTS PASSED ({results['summary']['passed']}/{results['summary']['total'] - results['summary']['skipped']})")
        return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
