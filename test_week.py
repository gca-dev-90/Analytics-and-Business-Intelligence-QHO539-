"""Test script to diagnose week module loading issues."""

import sys
from pathlib import Path

print("Testing week module loading...")
print(f"Python version: {sys.version}")
print(f"Working directory: {Path.cwd()}")

# Test importing
try:
    print("\n1. Importing week1...")
    from weeks import week1
    print("   [OK] Week 1 imported successfully")

    print("\n2. Checking build_widget function...")
    if hasattr(week1, 'build_widget'):
        print("   [OK] build_widget function found")

        print("\n3. Testing build_widget with sample config...")
        config = {
            "data_path": "data/Gross disposable household income (GDHI) per head for NUTS3 local areasUK1997 to 2016.csv"
        }

        result = week1.build_widget(config)
        print(f"   [OK] build_widget returned: {type(result)}")
        print(f"   Result keys: {result.keys() if isinstance(result, dict) else 'N/A'}")

        if isinstance(result, dict):
            if 'figures' in result:
                print(f"   Figures count: {len(result['figures'])}")
            if 'text' in result:
                print(f"   Text count: {len(result['text'])}")

        print("\n[OK] ALL TESTS PASSED!")

    else:
        print("   [ERROR] build_widget function NOT found")

except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()
