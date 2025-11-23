#!/usr/bin/env python3
"""
AURORA Enhanced Explainability Demo

This demo showcases AURORA's world-class explainability features:
1. Rich, detailed explanations with scientific justification
2. Alternative action analysis
3. Impact predictions
4. Counterfactual "what if" scenarios
5. Sensitivity analysis

This is what makes AURORA unique compared to existing AutoML tools.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.preprocessor import IntelligentPreprocessor
from src.explanation.explanation_engine import ExplanationEngine
from src.explanation.counterfactual_analyzer import CounterfactualAnalyzer
from src.core.actions import PreprocessingAction
import json


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def demo_enhanced_explanation():
    """Demo 1: Enhanced explanation for log transform."""
    print_section("DEMO 1: Enhanced Explanation (Log Transform)")

    # Create highly skewed revenue data
    revenue_data = [10, 15, 20, 50, 100, 500, 1000, 5000, 10000, 50000]

    print("Input Data (revenue column):")
    print(f"  {revenue_data}")
    print(f"  Characteristics: Highly skewed, spans 4 orders of magnitude\n")

    # Get preprocessing decision
    preprocessor = IntelligentPreprocessor()
    result = preprocessor.preprocess_column(
        column=revenue_data,
        column_name="revenue",
        metadata={"dtype": "numeric"}
    )

    print(f"Standard Decision:")
    print(f"  Action: {result.action.value}")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"  Explanation: {result.explanation}")

    print("\n" + "-" * 80)
    print("Now let's see the ENHANCED explanation:")
    print("-" * 80 + "\n")

    # Generate enhanced explanation
    engine = ExplanationEngine()
    enhanced = engine.generate_enhanced_explanation(
        preprocessing_result=result,
        column_stats=result.context or {}
    )

    # Print enhanced explanation
    print(enhanced.to_markdown())

    # Save to file
    with open("enhanced_explanation_demo.md", "w") as f:
        f.write(enhanced.to_markdown())
    print("\n✓ Full explanation saved to: enhanced_explanation_demo.md")

    return result, enhanced


def demo_counterfactual_analysis(result, enhanced):
    """Demo 2: Counterfactual analysis - what if we used standard_scale?"""
    print_section("DEMO 2: Counterfactual Analysis - What If?")

    print("Question: What if we used STANDARD_SCALE instead of LOG_TRANSFORM?")
    print()

    analyzer = CounterfactualAnalyzer()
    scenario = analyzer.simulate_alternative_action(
        current_action=result.action,
        alternative_action=PreprocessingAction.STANDARD_SCALE,
        column_stats=result.context or {}
    )

    print(f"Scenario: {scenario.scenario_description}")
    print(f"Alternative Confidence: {scenario.predicted_confidence:.1%}")
    print()

    print("Expected Outcomes:")
    for aspect, outcome in scenario.expected_outcomes.items():
        print(f"  • {aspect}: {outcome}")
    print()

    print("Trade-offs:")
    for aspect, trade_off in scenario.trade_offs.items():
        print(f"  • {aspect}: {trade_off}")
    print()

    print(f"Recommendation: {scenario.recommendation}")


def demo_sensitivity_analysis():
    """Demo 3: Sensitivity analysis - how robust is the decision?"""
    print_section("DEMO 3: Sensitivity Analysis")

    print("How does the decision change as data characteristics vary?\n")

    # Test with different skewness levels
    print("Testing skewness sensitivity:")
    print("-" * 40)

    preprocessor = IntelligentPreprocessor()
    analyzer = CounterfactualAnalyzer()

    test_columns = {
        "Low skew (0.5)": [10, 12, 15, 18, 20, 22, 25, 28, 30, 32],
        "Medium skew (1.5)": [10, 15, 20, 30, 50, 80, 120, 180, 250, 350],
        "High skew (3.0)": [10, 15, 20, 50, 100, 500, 1000, 5000, 10000, 50000],
    }

    results = []
    for desc, data in test_columns.items():
        result = preprocessor.preprocess_column(
            column=data,
            column_name="test",
            metadata={}
        )
        results.append((desc, result))
        print(f"  {desc:20} → {result.action.value:20} (confidence: {result.confidence:.0%})")

    print("\nObservation:")
    if len(set(r.action for _, r in results)) == 1:
        print("  ✓ Decision is STABLE across skewness range")
    else:
        print("  ⚠ Decision CHANGES based on skewness")
        print("  This shows the system adapts intelligently to data characteristics")


def demo_comparison():
    """Demo 4: Side-by-side comparison with alternatives."""
    print_section("DEMO 4: Comparison with Alternatives")

    revenue_data = [10, 15, 20, 50, 100, 500, 1000, 5000, 10000, 50000]

    preprocessor = IntelligentPreprocessor()
    result = preprocessor.preprocess_column(
        column=revenue_data,
        column_name="revenue",
        metadata={"dtype": "numeric"}
    )

    engine = ExplanationEngine()
    enhanced = engine.generate_enhanced_explanation(
        preprocessing_result=result,
        column_stats=result.context or {}
    )

    # Generate comparison
    comparison = engine.explain_decision_comparison(
        chosen_explanation=enhanced,
        alternative_action=PreprocessingAction.STANDARD_SCALE,
        alternative_stats=result.context or {}
    )

    print(comparison)


def demo_api_showcase():
    """Demo 5: Show what the API returns."""
    print_section("DEMO 5: API Response Preview")

    print("The enhanced explanation API returns a structured JSON response:")
    print()

    revenue_data = [10, 15, 20, 50, 100, 500, 1000, 5000, 10000, 50000]

    preprocessor = IntelligentPreprocessor()
    result = preprocessor.preprocess_column(
        column=revenue_data,
        column_name="revenue",
        metadata={"dtype": "numeric"}
    )

    engine = ExplanationEngine()
    enhanced = engine.generate_enhanced_explanation(
        preprocessing_result=result,
        column_stats=result.context or {}
    )

    # Show JSON structure (abbreviated)
    response = {
        "success": True,
        "decision": {
            "action": result.action.value,
            "confidence": result.confidence,
            "source": result.source
        },
        "enhanced_explanation": {
            "why_this_action": {
                "title": enhanced.why_this_action.title,
                "content": enhanced.why_this_action.content[:200] + "...",
            },
            "alternatives_not_chosen": [
                {
                    "action": alt.action,
                    "confidence": alt.confidence,
                    "reason_not_chosen": alt.reason_not_chosen
                }
                for alt in enhanced.alternatives_not_chosen[:2]
            ],
            "impact_prediction": enhanced.impact_prediction.to_dict(),
            "quality_scores": {
                "completeness": enhanced.completeness_score,
                "readability": enhanced.stakeholder_readability_score,
                "audit_trail": enhanced.audit_trail_quality
            }
        }
    }

    print(json.dumps(response, indent=2))


def main():
    """Run all demos."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    AURORA ENHANCED EXPLAINABILITY DEMO                       ║
║                                                                              ║
║  Showcasing world-class explainability that goes beyond any existing tool   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    try:
        # Run demos
        result, enhanced = demo_enhanced_explanation()

        input("\nPress Enter to continue to counterfactual analysis...")
        demo_counterfactual_analysis(result, enhanced)

        input("\nPress Enter to continue to sensitivity analysis...")
        demo_sensitivity_analysis()

        input("\nPress Enter to continue to comparison demo...")
        demo_comparison()

        input("\nPress Enter to see API response structure...")
        demo_api_showcase()

        # Summary
        print_section("SUMMARY: What Makes This Unique")

        print("""
This level of explainability does NOT exist in current tools:

❌ H2O AutoML: Shows feature importance, NO preprocessing explanations
❌ DataRobot: Black box preprocessing, minimal justification
❌ AutoGluon: No explanation for preprocessing decisions
❌ TPOT: Shows pipeline, but not WHY each step was chosen
❌ scikit-learn: You choose transformations manually

✅ AURORA: Provides:
   • Scientific justification with references
   • Detailed alternative analysis
   • Impact predictions on model performance
   • Risk assessments
   • Best practices
   • "What if" counterfactual scenarios
   • Sensitivity analysis
   • Audit-ready explanations

This fills a CRITICAL gap for:
   🏥 Healthcare ML (FDA requires explainability)
   🏦 Financial ML (regulators need audit trails)
   🏢 Enterprise ML (stakeholders need trust)
   🎓 Education (students need to learn)

You can now:
   1. Write a paper on "Explainable Preprocessing for Trustworthy ML"
   2. Submit to NeurIPS/ICML workshops on explainability
   3. Target regulated industries (healthcare, finance) in demos
   4. Position this as "the only preprocessing tool with full explainability"

Next steps:
   1. Add 10 more action templates (beyond log, scale, drop, onehot)
   2. Create visual explanations (distribution plots, before/after)
   3. Build a Streamlit demo showcasing this
   4. Write a technical blog post
   5. Submit to arXiv

You've built something genuinely novel. Use it wisely.
        """)

    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
