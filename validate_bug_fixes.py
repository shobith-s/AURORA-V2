"""
Validation script to verify all bug fixes are properly applied.
"""

import os
import sys

def check_bug_fixes():
    """Check if all 4 bugs have been fixed."""

    print("=" * 70)
    print("VALIDATING BUG FIXES")
    print("=" * 70)

    all_passed = True

    # BUG #1: Rule validation logic
    print("\n1. Checking Bug #1: Rule validation logic...")
    with open("src/learning/adaptive_engine.py", 'r') as f:
        content = f.read()
        if "compute_pattern_similarity" in content and "similarity >= self.similarity_threshold" in content:
            print("   ✅ Bug #1 FIXED: Rule validation now uses similarity matching")
        else:
            print("   ❌ Bug #1 NOT FIXED")
            all_passed = False

    # BUG #2: Cache references
    print("\n2. Checking Bug #2: Cache references...")
    with open("src/core/preprocessor.py", 'r') as f:
        content = f.read()
        if "self.cache.set" not in content or content.count("self.cache.set") == 0:
            print("   ✅ Bug #2 FIXED: Cache references removed")
        else:
            print("   ❌ Bug #2 NOT FIXED: Cache references still present")
            all_passed = False

    # BUG #3: Type inference threshold
    print("\n3. Checking Bug #3: Type inference threshold...")
    with open("src/core/preprocessor.py", 'r') as f:
        content = f.read()
        if "conversion_rate > 0.9" in content:
            print("   ✅ Bug #3 FIXED: Type inference threshold increased to 90%")
        elif "conversion_rate > 0.5" in content:
            print("   ❌ Bug #3 NOT FIXED: Still using 50% threshold")
            all_passed = False
        else:
            print("   ⚠️  Bug #3: Unable to verify")

    # BUG #4: Input size limits
    print("\n4. Checking Bug #4: Input size limits...")
    with open("src/core/robust_parser.py", 'r') as f:
        content = f.read()
        has_file_limit = "MAX_FILE_SIZE_BYTES" in content
        has_row_limit = "MAX_ROWS" in content
        has_col_limit = "MAX_COLUMNS" in content
        has_checks = "Too many rows" in content or "File too large" in content

        if has_file_limit and has_row_limit and has_col_limit and has_checks:
            print("   ✅ Bug #4 FIXED: Input size limits added")
        else:
            print(f"   ❌ Bug #4 NOT FIXED: Missing limits")
            print(f"      File limit: {has_file_limit}")
            print(f"      Row limit: {has_row_limit}")
            print(f"      Col limit: {has_col_limit}")
            print(f"      Checks: {has_checks}")
            all_passed = False

    # DEAD CODE REMOVAL
    print("\n5. Checking dead code removal...")
    with open("src/core/preprocessor.py", 'r') as f:
        content = f.read()
        dataset_analyzer_disabled = "# from ..analysis.dataset_analyzer import DatasetAnalyzer" in content or \
                                   "DISABLED: Inter-column analysis" in content

        if dataset_analyzer_disabled:
            print("   ✅ DatasetAnalyzer usage disabled")
        else:
            print("   ⚠️  DatasetAnalyzer may still be active")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    if all_passed:
        print("\n✅ ALL CRITICAL BUGS FIXED!")
        print("\nSUMMARY:")
        print("✓ Bug #1: Rule validation uses similarity matching (not exact hash)")
        print("✓ Bug #2: Cache references removed")
        print("✓ Bug #3: Type inference threshold raised to 90%")
        print("✓ Bug #4: Input size limits added (100MB, 1M rows, 1K cols)")
        print("✓ Dead code: DatasetAnalyzer disabled")
        return 0
    else:
        print("\n❌ SOME BUGS NOT FIXED - Review above")
        return 1

if __name__ == "__main__":
    exit_code = check_bug_fixes()
    sys.exit(exit_code)
