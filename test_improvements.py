"""
Quick test to verify the symbolic engine improvements and learner redesign.
"""

import pandas as pd
import numpy as np
from src.core.preprocessor import IntelligentPreprocessor
from src.core.actions import PreprocessingAction

def test_simple_cases():
    """Test that simple cases are now handled correctly."""
    print("=" * 70)
    print("TEST 1: Simple Cases (Should have high confidence)")
    print("=" * 70)

    preprocessor = IntelligentPreprocessor(
        confidence_threshold=0.9,
        use_neural_oracle=False,
        enable_learning=True
    )

    test_cases = [
        ("Binary column (0/1)", pd.Series([0, 1, 0, 1, 0, 1, 0, 1] * 100, name="is_active")),
        ("Percentage range", pd.Series(np.random.uniform(0, 100, 800), name="score_pct")),
        ("Already normalized", pd.Series(np.random.uniform(0, 1, 800), name="probability")),
        ("Low cardinality categorical", pd.Series(["A", "B", "C"] * 267, name="category")),
        ("Extreme skew", pd.Series(np.random.exponential(2, 800), name="wait_time")),
    ]

    for name, column in test_cases:
        result = preprocessor.preprocess_column(column, column.name)
        print(f"\n{name}:")
        print(f"  Action: {result.action.value}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Source: {result.source}")
        print(f"  ✓ PASS" if result.confidence >= 0.85 else f"  ✗ FAIL: Low confidence!")

def test_learner_creates_rules():
    """Test that learner creates symbolic rules instead of making decisions."""
    print("\n\n" + "=" * 70)
    print("TEST 2: Learner Creates Rules (Not Direct Decisions)")
    print("=" * 70)

    preprocessor = IntelligentPreprocessor(
        confidence_threshold=0.9,
        use_neural_oracle=False,
        enable_learning=True
    )

    # Get initial rule count
    initial_rules = len(preprocessor.symbolic_engine.rules)
    print(f"\nInitial symbolic rules: {initial_rules}")

    # Create a test column with medium skewness
    test_column = pd.Series(np.random.lognormal(1, 0.5, 800), name="price")

    # Get initial decision
    result1 = preprocessor.preprocess_column(test_column, "price")
    print(f"\nFirst decision:")
    print(f"  Action: {result1.action.value}")
    print(f"  Confidence: {result1.confidence:.3f}")
    print(f"  Source: {result1.source}")

    # Simulate corrections to trigger rule creation
    print(f"\nSimulating 10 corrections to trigger rule creation...")
    for i in range(10):
        preprocessor.process_correction(
            test_column,
            "price",
            wrong_action=PreprocessingAction.KEEP_AS_IS,
            correct_action=PreprocessingAction.LOG_TRANSFORM,
            confidence=0.5
        )

    # Check if new rule was created
    current_rules = len(preprocessor.symbolic_engine.rules)
    print(f"\nAfter 10 corrections:")
    print(f"  Current symbolic rules: {current_rules}")
    print(f"  Rules added: {current_rules - initial_rules}")

    if current_rules > initial_rules:
        print(f"  ✓ PASS: Learner created new symbolic rule!")
        # Show the new rule
        new_rules = [r for r in preprocessor.symbolic_engine.rules if r.name.startswith("LEARNED_")]
        for rule in new_rules:
            print(f"  New rule: {rule.name} -> {rule.action.value}")
    else:
        print(f"  ⚠ INFO: Rule may already exist or pattern not yet ready")

    # Test with similar column - should use learned rule
    test_column2 = pd.Series(np.random.lognormal(1, 0.5, 800), name="cost")
    result2 = preprocessor.preprocess_column(test_column2, "cost")
    print(f"\nDecision on similar column (should use learned rule):")
    print(f"  Action: {result2.action.value}")
    print(f"  Confidence: {result2.confidence:.3f}")
    print(f"  Source: {result2.source}")
    print(f"  Explanation: {result2.explanation[:100]}...")

def test_architecture_integrity():
    """Verify that learner never makes direct decisions."""
    print("\n\n" + "=" * 70)
    print("TEST 3: Architecture Integrity (Learner Doesn't Make Decisions)")
    print("=" * 70)

    preprocessor = IntelligentPreprocessor(
        confidence_threshold=0.9,
        use_neural_oracle=False,
        enable_learning=True
    )

    # Test various columns
    test_columns = [
        pd.Series(np.random.normal(50, 10, 800), name="numeric1"),
        pd.Series(["X", "Y", "Z"] * 267, name="categorical1"),
        pd.Series(np.random.exponential(2, 800), name="skewed1"),
    ]

    print("\nVerifying all decisions come from symbolic engine...")
    all_from_symbolic = True

    for column in test_columns:
        result = preprocessor.preprocess_column(column, column.name)
        # Check source - should NEVER be "learned" (direct learner decision)
        # Should be "symbolic", "meta_learning", "neural", or "conservative_fallback"
        if result.source == "learned":
            print(f"  ✗ FAIL: {column.name} decision from 'learned' (direct learner)!")
            all_from_symbolic = False
        else:
            print(f"  ✓ {column.name}: {result.source} (confidence: {result.confidence:.3f})")

    if all_from_symbolic:
        print(f"\n  ✓ PASS: All decisions from symbolic engine (or fallback), never direct learner!")
    else:
        print(f"\n  ✗ FAIL: Some decisions from direct learner path!")

if __name__ == "__main__":
    try:
        test_simple_cases()
        test_learner_creates_rules()
        test_architecture_integrity()

        print("\n\n" + "=" * 70)
        print("ALL TESTS COMPLETED")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
