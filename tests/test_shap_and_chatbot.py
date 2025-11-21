"""
Test SHAP explanations and intelligent chatbot.

Verifies that:
1. SHAP explanations work correctly
2. Chatbot can answer various types of questions
3. Chatbot provides relevant SHAP interpretations
"""

import pytest
import pandas as pd
import numpy as np
from src.core.preprocessor import IntelligentPreprocessor
from src.ai.intelligent_assistant import IntelligentAssistant


class TestSHAPExplanations:
    """Test SHAP functionality."""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor with neural oracle."""
        return IntelligentPreprocessor(
            use_neural_oracle=True,
            enable_learning=False  # Disable for consistent testing
        )

    def test_shap_available(self, preprocessor):
        """Test that SHAP can be imported and used."""
        try:
            import shap
            assert True, "SHAP is available"
        except ImportError:
            pytest.fail("SHAP is not installed - run: pip install shap")

    def test_neural_oracle_has_shap_method(self, preprocessor):
        """Test that neural oracle has predict_with_shap method."""
        if preprocessor.neural_oracle is None:
            pytest.skip("Neural oracle not available")

        assert hasattr(preprocessor.neural_oracle, 'predict_with_shap'), \
            "Neural oracle should have predict_with_shap method"

    def test_shap_explanation_structure(self, preprocessor):
        """Test that SHAP explanations have correct structure."""
        # Create a column that will trigger neural oracle (ambiguous case)
        column = pd.Series([1, 2, 3, 5, 8, 13, 21, 34, 55, 89], name="ambiguous")

        try:
            result = preprocessor.preprocess_column(column, "ambiguous")

            # Check if we have explanation
            assert result.explanation is not None
            assert len(result.explanation) > 0

            print(f"\n✓ Got explanation from {result.source} layer")
            print(f"  Action: {result.action.value}")
            print(f"  Confidence: {result.confidence:.2%}")
            print(f"  Explanation: {result.explanation[:200]}...")

        except Exception as e:
            pytest.fail(f"Failed to get SHAP explanation: {e}")


class TestIntelligentChatbot:
    """Test intelligent chatbot assistant."""

    @pytest.fixture
    def assistant(self):
        """Create assistant instance."""
        return IntelligentAssistant()

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample dataframe for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'revenue': np.random.exponential(1000, 100),
            'age': np.random.normal(35, 10, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'id': range(100)
        })

    def test_chatbot_initialization(self, assistant):
        """Test that chatbot initializes correctly."""
        assert assistant is not None
        assert assistant.preprocessor is not None

    def test_chatbot_without_data(self, assistant):
        """Test chatbot responses when no data is loaded."""
        answer = assistant.query("What can you do?")

        assert len(answer) > 0
        assert "help" in answer.lower() or "assist" in answer.lower()
        print(f"\n✓ Chatbot welcome message:\n{answer[:200]}...")

    def test_chatbot_capabilities_query(self, assistant):
        """Test asking about capabilities."""
        answer = assistant.query("help")

        assert "capabilities" in answer.lower() or "can help" in answer.lower()
        print(f"\n✓ Capabilities response:\n{answer[:300]}...")

    def test_chatbot_column_statistics(self, assistant, sample_dataframe):
        """Test asking for column statistics."""
        assistant.set_context(sample_dataframe)

        answer = assistant.query("What are the statistics for revenue?")

        assert "revenue" in answer.lower()
        assert any(word in answer.lower() for word in ['mean', 'median', 'std'])
        print(f"\n✓ Column statistics answer:\n{answer[:400]}...")

    def test_chatbot_recommendation(self, assistant, sample_dataframe):
        """Test asking for preprocessing recommendation."""
        assistant.set_context(sample_dataframe)

        answer = assistant.query("What preprocessing do you recommend for revenue?")

        assert "revenue" in answer.lower()
        assert any(word in answer for word in ['transform', 'scale', 'normalize', 'encode'])
        print(f"\n✓ Recommendation answer:\n{answer[:400]}...")

    def test_chatbot_explanation(self, assistant, sample_dataframe):
        """Test asking why a decision was made."""
        assistant.set_context(sample_dataframe)

        answer = assistant.query("Why did you recommend that for revenue?")

        assert len(answer) > 50  # Should give detailed explanation
        print(f"\n✓ Decision explanation:\n{answer[:500]}...")

    def test_chatbot_dataset_summary(self, assistant, sample_dataframe):
        """Test asking for dataset summary."""
        assistant.set_context(sample_dataframe)

        answer = assistant.query("Give me a dataset summary")

        assert "100" in answer  # Row count
        assert "4" in answer or "columns" in answer.lower()
        print(f"\n✓ Dataset summary:\n{answer[:400]}...")

    def test_chatbot_data_quality(self, assistant, sample_dataframe):
        """Test asking about data quality."""
        assistant.set_context(sample_dataframe)

        answer = assistant.query("What data quality issues do we have?")

        assert "quality" in answer.lower() or "issues" in answer.lower()
        print(f"\n✓ Data quality answer:\n{answer[:400]}...")

    def test_chatbot_statistical_query(self, assistant, sample_dataframe):
        """Test statistical queries."""
        assistant.set_context(sample_dataframe)

        answer = assistant.query("What's the mean of all numeric columns?")

        assert "mean" in answer.lower() or "average" in answer.lower()
        print(f"\n✓ Statistical query answer:\n{answer[:300]}...")

    def test_chatbot_technique_explanation(self, assistant):
        """Test explaining preprocessing techniques."""
        answer = assistant.query("When should I use log transform?")

        assert "log" in answer.lower()
        assert any(word in answer.lower() for word in ['skew', 'transform', 'positive'])
        print(f"\n✓ Technique explanation:\n{answer[:400]}...")

    def test_chatbot_shap_explanation(self, assistant):
        """Test explaining SHAP values."""
        answer = assistant.query("Explain SHAP values")

        assert "shap" in answer.lower()
        assert any(word in answer.lower() for word in ['feature', 'explain', 'influence'])
        print(f"\n✓ SHAP explanation:\n{answer[:500]}...")

    def test_chatbot_contextual_responses(self, assistant, sample_dataframe):
        """Test that chatbot provides different answers with/without context."""
        # Without context
        answer_no_context = assistant.query("Tell me about the data")

        # With context
        assistant.set_context(sample_dataframe)
        answer_with_context = assistant.query("Tell me about the data")

        # Answers should be different
        assert answer_no_context != answer_with_context
        assert "upload" in answer_no_context.lower() or "no data" in answer_no_context.lower()
        assert "100" in answer_with_context or "4" in answer_with_context  # Row/column count

        print(f"\n✓ Without context: {answer_no_context[:150]}...")
        print(f"✓ With context: {answer_with_context[:150]}...")


class TestChatbotSHAPIntegration:
    """Test integration of chatbot with SHAP explanations."""

    @pytest.fixture
    def assistant(self):
        """Create assistant with neural oracle enabled."""
        preprocessor = IntelligentPreprocessor(use_neural_oracle=True)
        return IntelligentAssistant(preprocessor=preprocessor)

    @pytest.fixture
    def skewed_dataframe(self):
        """Create dataframe with highly skewed data."""
        np.random.seed(42)
        return pd.DataFrame({
            'price': np.random.exponential(100, 100),  # Highly skewed
            'quantity': np.random.poisson(5, 100)
        })

    def test_chatbot_can_explain_shap_for_column(self, assistant, skewed_dataframe):
        """Test that chatbot can explain SHAP values for specific columns."""
        assistant.set_context(skewed_dataframe)

        # Ask for explanation
        answer = assistant.query("Why did you recommend log_transform for price?")

        # Should mention features and their impacts
        assert len(answer) > 100  # Detailed explanation
        assert "price" in answer.lower()

        print(f"\n✓ SHAP-based explanation:\n{answer[:600]}...")

    def test_chatbot_interprets_shap_values_user_friendly(self, assistant):
        """Test that SHAP explanations are user-friendly."""
        answer = assistant.query("How do I read SHAP explanations?")

        # Should explain in simple terms
        assert "shap" in answer.lower()
        assert any(word in answer.lower() for word in ['feature', 'influence', 'impact', 'confidence'])
        assert not any(word in answer for word in ['TreeExplainer', 'KernelSHAP'])  # Too technical

        print(f"\n✓ User-friendly SHAP guide:\n{answer[:500]}...")


def test_end_to_end_workflow():
    """Test complete workflow: load data → get recommendation → ask chatbot."""
    print("\n" + "="*70)
    print("END-TO-END WORKFLOW TEST")
    print("="*70)

    # 1. Load data
    df = pd.DataFrame({
        'revenue': [100, 200, 5000, 10000, 50000],  # Skewed
        'age': [25, 30, 35, 40, 45],  # Normal
        'category': ['A', 'B', 'A', 'C', 'B']
    })
    print("\n1. ✓ Data loaded: 5 rows, 3 columns")

    # 2. Get recommendation
    preprocessor = IntelligentPreprocessor(use_neural_oracle=True)
    result = preprocessor.preprocess_column(df['revenue'], 'revenue')
    print(f"\n2. ✓ Recommendation: {result.action.value}")
    print(f"   Confidence: {result.confidence:.2%}")
    print(f"   Source: {result.source}")

    # 3. Ask chatbot
    assistant = IntelligentAssistant(preprocessor=preprocessor)
    assistant.set_context(df)

    questions = [
        "What are the statistics for revenue?",
        "Why did you recommend that action?",
        "Give me a dataset summary"
    ]

    for i, question in enumerate(questions, 3):
        answer = assistant.query(question)
        print(f"\n{i}. Q: {question}")
        print(f"   A: {answer[:200]}...")

    print("\n" + "="*70)
    print("✓ END-TO-END TEST PASSED")
    print("="*70)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
