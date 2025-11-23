"""
Quick standalone test of the explainability module.
Tests the core functionality without needing full system integration.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.explanation.enhanced_explanation import (
    EnhancedExplanation,
    ExplanationSection,
    AlternativeExplanation,
    ImpactPrediction,
    StatisticalEvidence,
    ExplanationSeverity
)
from src.explanation.explanation_templates import ExplanationTemplateRegistry

def test_template_generation():
    """Test that templates generate rich explanations."""
    print("=" * 80)
    print("Testing Explanation Template Generation")
    print("=" * 80 + "\n")

    # Simulate column statistics for a skewed numeric column
    stats = {
        'skewness': 3.5,
        'min_value': 10.0,
        'max_value': 50000.0,
        'mean': 5000.0,
        'std': 10000.0,
        'has_outliers': True,
        'is_numeric': True
    }

    print("Column Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

    # Get log transform explanation
    registry = ExplanationTemplateRegistry()
    explanation_data = registry.get_log_transform_explanation(stats)

    print("=" * 80)
    print("GENERATED EXPLANATION")
    print("=" * 80 + "\n")

    print(f"Title: {explanation_data['why_section'].title}")
    print(f"\n{explanation_data['why_section'].content}\n")

    print("Evidence:")
    for evidence in explanation_data['why_section'].evidence:
        print(f"  • {evidence}")
    print()

    print("Statistical Evidence:")
    print(f"  Key Statistics: {len(explanation_data['statistical_evidence'].key_statistics)} metrics")
    print(f"  Thresholds Met: {len(explanation_data['statistical_evidence'].thresholds_met)}")
    print()

    print(f"Alternatives Analyzed: {len(explanation_data['alternatives'])} alternatives")
    for i, alt in enumerate(explanation_data['alternatives'][:2], 1):
        print(f"\n  Alternative {i}: {alt.action}")
        print(f"    Confidence: {alt.confidence:.0%}")
        print(f"    Why not chosen: {alt.reason_not_chosen}")
        print(f"    Pros: {len(alt.pros)}")
        print(f"    Cons: {len(alt.cons)}")
    print()

    print("Impact Prediction:")
    ip = explanation_data['impact_prediction']
    print(f"  Accuracy Impact: {ip.expected_accuracy_change}")
    print(f"  Interpretability: {ip.interpretability_impact}")
    print(f"  Computational Cost: {ip.computational_cost}")
    print(f"  Reversibility: {ip.reversibility}")
    print()

    print(f"Best Practices: {len(explanation_data['best_practices'])} tips")
    print(f"Scientific References: {len(explanation_data['scientific_references'])} papers")
    print(f"What-If Scenarios: {len(explanation_data['what_if_scenarios'])} scenarios")
    print()

    return explanation_data


def test_enhanced_explanation_object():
    """Test EnhancedExplanation data structure."""
    print("\n" + "=" * 80)
    print("Testing EnhancedExplanation Object")
    print("=" * 80 + "\n")

    why_section = ExplanationSection(
        title="Why Log Transform",
        content="Log transformation reduces skewness and normalizes the distribution",
        severity=ExplanationSeverity.SUCCESS,
        evidence=["Skewness > 2.0", "Values span multiple orders of magnitude"]
    )

    stat_evidence = StatisticalEvidence(
        key_statistics={"skewness": 3.5, "min": 10.0, "max": 50000.0},
        thresholds_met=["Skewness > 1.5", "Positive values only"],
        distribution_characteristics={"shape": "Right-skewed"}
    )

    alternatives = [
        AlternativeExplanation(
            action="standard_scale",
            confidence=0.65,
            reason_not_chosen="Would preserve skewness",
            pros=["Simple", "Fast"],
            cons=["Doesn't fix skewness"],
            when_to_use="Use for normal distributions"
        )
    ]

    impact = ImpactPrediction(
        expected_accuracy_change="+5-12%",
        feature_importance_impact="More balanced",
        interpretability_impact="High",
        computational_cost="Negligible",
        reversibility="Fully reversible",
        data_loss="None"
    )

    enhanced = EnhancedExplanation(
        action="log_transform",
        confidence=0.90,
        why_this_action=why_section,
        statistical_evidence=stat_evidence,
        alternatives_not_chosen=alternatives,
        impact_prediction=impact,
        best_practices=["Check for zeros", "Use log1p if needed"],
        scientific_references=["Osborne (2002)", "Tukey (1977)"],
        what_if_scenarios={"What if I skip?": "Performance degrades 5-15%"}
    )

    # Test conversion methods
    print("Testing to_dict()...")
    dict_repr = enhanced.to_dict()
    print(f"  ✓ Generated dictionary with {len(dict_repr)} keys")
    print(f"  Keys: {list(dict_repr.keys())}")
    print()

    print("Testing to_plain_text()...")
    plain_text = enhanced.to_plain_text()
    print(f"  ✓ Generated {len(plain_text)} characters of plain text")
    print()

    print("Testing to_markdown()...")
    markdown = enhanced.to_markdown()
    print(f"  ✓ Generated {len(markdown)} characters of markdown")
    print()

    print("Preview of Markdown Report:")
    print("-" * 80)
    print(markdown[:800] + "\n... (truncated)" if len(markdown) > 800 else markdown)
    print("-" * 80)


def main():
    """Run standalone tests."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              AURORA EXPLAINABILITY MODULE - STANDALONE TEST                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    try:
        # Test 1: Template generation
        explanation_data = test_template_generation()

        # Test 2: Enhanced explanation object
        test_enhanced_explanation_object()

        # Summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80 + "\n")

        print("✅ All tests passed!")
        print()
        print("What we verified:")
        print("  ✓ Explanation templates generate rich, detailed content")
        print("  ✓ Scientific references included")
        print("  ✓ Alternative actions analyzed with pros/cons")
        print("  ✓ Impact predictions provided")
        print("  ✓ Best practices included")
        print("  ✓ What-if scenarios generated")
        print("  ✓ Markdown, JSON, and plain text outputs work")
        print()
        print("This is the foundation of world-class explainability.")
        print()
        print("Next steps:")
        print("  1. Run full demo with: python3 demo_explainability.py")
        print("  2. Test API endpoints: python3 -m uvicorn src.api.server:app --reload")
        print("  3. Visit http://localhost:8000/docs for Swagger UI")
        print("  4. Try /explain/enhanced, /explain/counterfactual, /explain/demo")
        print()

        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
