"""
Verify that everything is ready for hybrid training.
Checks datasets, dependencies, and paths.

Run this before train_hybrid.py to ensure smooth training.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_dependencies():
    """Check that all required packages are installed."""
    print("1. Checking dependencies...")

    missing = []

    try:
        import pandas
        print("   ✓ pandas installed")
    except ImportError:
        missing.append("pandas")

    try:
        import numpy
        print("   ✓ numpy installed")
    except ImportError:
        missing.append("numpy")

    try:
        import xgboost
        print(f"   ✓ xgboost installed (version {xgboost.__version__})")
    except ImportError:
        missing.append("xgboost")

    try:
        import sklearn
        print("   ✓ scikit-learn installed")
    except ImportError:
        missing.append("scikit-learn")

    if missing:
        print(f"\n   ⚠ Missing packages: {', '.join(missing)}")
        print(f"   Install with: pip install {' '.join(missing)}")
        return False

    return True


def check_datasets(datasets_dir: str = "data/open_datasets"):
    """Check that datasets exist."""
    print(f"\n2. Checking datasets in {datasets_dir}...")

    # Use Path to handle both forward and back slashes
    datasets_path = Path(datasets_dir).resolve()

    print(f"   Looking for: {datasets_path}")

    if not datasets_path.exists():
        print(f"   ✗ Directory not found!")
        print(f"\n   Run this first: python scripts/collect_open_datasets.py")
        return False

    # Count CSV files
    csv_files = list(datasets_path.rglob("*.csv"))

    if not csv_files:
        print(f"   ✗ No CSV files found!")
        print(f"\n   Run this first: python scripts/collect_open_datasets.py")
        return False

    print(f"   ✓ Found {len(csv_files)} CSV files")

    # Show breakdown by subdirectory
    subdirs = {}
    for csv in csv_files:
        subdir = csv.parent.name
        subdirs[subdir] = subdirs.get(subdir, 0) + 1

    for subdir, count in sorted(subdirs.items()):
        print(f"     • {subdir}: {count} file(s)")

    return True


def check_database():
    """Check database connection."""
    print("\n3. Checking database...")

    try:
        from src.database.connection import SessionLocal
        from src.database.models import CorrectionRecord

        db = SessionLocal()
        correction_count = db.query(CorrectionRecord).count()
        db.close()

        print(f"   ✓ Database connected")
        print(f"   ✓ Found {correction_count} user corrections")

        if correction_count == 0:
            print(f"     ⚠ No corrections yet (training will use synthetic + open data only)")

        return True

    except Exception as e:
        print(f"   ⚠ Database check failed: {e}")
        print(f"     (Training can still proceed with synthetic + open data)")
        return True  # Non-critical


def check_output_directory():
    """Check that output directory exists or can be created."""
    print("\n4. Checking output directory...")

    models_dir = Path("models")

    if not models_dir.exists():
        print(f"   Creating models/ directory...")
        models_dir.mkdir(parents=True, exist_ok=True)

    print(f"   ✓ Output directory ready: {models_dir.resolve()}")
    return True


def main():
    print("=" * 70)
    print("AURORA Hybrid Training Setup Verification")
    print("=" * 70 + "\n")

    all_checks_passed = True

    # Run all checks
    all_checks_passed &= check_dependencies()
    all_checks_passed &= check_datasets()
    all_checks_passed &= check_database()
    all_checks_passed &= check_output_directory()

    print("\n" + "=" * 70)

    if all_checks_passed:
        print("✓ ALL CHECKS PASSED - Ready for training!")
        print("=" * 70)
        print("\nRun training with:")
        print("  python scripts/train_hybrid.py --datasets-dir data/open_datasets")
        print("\nOr use the Windows wrapper:")
        print("  scripts/train_hybrid_windows.bat")
    else:
        print("✗ SOME CHECKS FAILED - Please fix issues above")
        print("=" * 70)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
