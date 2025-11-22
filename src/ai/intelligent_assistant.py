"""
Intelligent AI Assistant for AURORA.

Provides conversational interface to query data, explain decisions,
and answer user questions about preprocessing recommendations.

Capabilities:
- Column-level analysis and statistics
- Dataset-level insights
- SHAP explanation interpretation
- User-friendly explanations of technical concepts
- Contextual help based on current data
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from ..core.preprocessor import IntelligentPreprocessor
from ..core.actions import PreprocessingAction
from ..symbolic.engine import SymbolicEngine


class IntelligentAssistant:
    """
    AI Assistant that can answer questions about data and preprocessing.

    Unlike a simple keyword-based chatbot, this assistant:
    - Analyzes actual data
    - Provides real statistics
    - Explains SHAP values
    - Gives contextual recommendations
    """

    def __init__(self, preprocessor: Optional[IntelligentPreprocessor] = None):
        """
        Initialize the assistant.

        Args:
            preprocessor: Reference to the preprocessor for making predictions
        """
        self.preprocessor = preprocessor or IntelligentPreprocessor()
        self.symbolic_engine = SymbolicEngine()
        self.current_dataframe: Optional[pd.DataFrame] = None
        self.current_results: Dict[str, Any] = {}

    def set_context(self, df: Optional[pd.DataFrame], results: Optional[Dict[str, Any]] = None):
        """
        Set the current context (dataframe and analysis results).

        Args:
            df: Current dataframe being analyzed
            results: Recent preprocessing results
        """
        self.current_dataframe = df
        self.current_results = results or {}

    def query(self, user_question: str) -> str:
        """
        Answer a user question intelligently.

        Args:
            user_question: Natural language question

        Returns:
            User-friendly answer with actual data
        """
        q = user_question.lower()

        # Column-specific queries
        if any(word in q for word in ['column', 'feature', 'variable']):
            if 'statistics' in q or 'stats' in q or 'summary' in q:
                return self._get_column_statistics_answer(q)
            elif 'recommend' in q or 'suggest' in q:
                return self._get_column_recommendation_answer(q)
            elif 'why' in q or 'explain' in q:
                return self._explain_column_decision(q)

        # Dataset-level queries
        if any(word in q for word in ['dataset', 'data', 'dataframe', 'table']):
            if 'summary' in q or 'overview' in q:
                return self._get_dataset_summary()
            elif 'quality' in q or 'issues' in q:
                return self._get_data_quality_report()
            elif 'columns' in q or 'features' in q:
                return self._get_columns_overview()

        # SHAP explanations
        if 'shap' in q or 'explanation' in q or 'interpret' in q:
            return self._explain_shap()

        # Statistical queries
        if any(word in q for word in ['mean', 'median', 'std', 'variance', 'distribution']):
            return self._answer_statistical_query(q)

        # Preprocessing technique explanations
        if any(word in q for word in ['transform', 'scale', 'encode', 'impute', 'normalize']):
            return self._explain_preprocessing_technique(q)

        # General help
        if any(word in q for word in ['help', 'can you', 'what can']):
            return self._get_capabilities()

        # Default: Try to give helpful response
        return self._generate_contextual_response(q)

    def _get_column_statistics_answer(self, query: str) -> str:
        """Get statistics for a specific column."""
        if self.current_dataframe is None:
            return "❌ No dataset loaded. Please upload a dataset first to get column statistics."

        # Try to extract column name from query
        column_name = self._extract_column_name(query)

        if column_name and column_name in self.current_dataframe.columns:
            col = self.current_dataframe[column_name]
            stats = self.symbolic_engine.compute_column_statistics(col, column_name)

            response = f"📊 **Statistics for '{column_name}'**\n\n"
            response += f"**Basic Info:**\n"
            response += f"• Type: {stats.detected_type}\n"
            response += f"• Total values: {stats.row_count:,}\n"
            response += f"• Missing: {stats.null_count:,} ({stats.null_percentage*100:.1f}%)\n"
            response += f"• Unique: {stats.unique_count:,} ({stats.unique_ratio*100:.1f}%)\n\n"

            if stats.is_numeric:
                response += f"**Numeric Statistics:**\n"
                response += f"• Mean: {stats.mean:.2f}\n"
                response += f"• Median: {stats.median:.2f}\n"
                response += f"• Std Dev: {stats.std:.2f}\n"
                response += f"• Range: [{stats.min:.2f}, {stats.max:.2f}]\n"
                response += f"• Skewness: {stats.skewness:.2f}\n"
                response += f"• Kurtosis: {stats.kurtosis:.2f}\n"
                response += f"• Outliers: {stats.outlier_count} ({stats.outlier_percentage*100:.1f}%)\n"

            return response
        else:
            # List available columns
            if self.current_dataframe is not None:
                cols = ", ".join(self.current_dataframe.columns[:10].tolist())
                if len(self.current_dataframe.columns) > 10:
                    cols += f" ... ({len(self.current_dataframe.columns)} total)"
                return f"❓ I couldn't identify which column you're asking about.\n\n**Available columns:** {cols}\n\nTry: \"What are the statistics for [column_name]?\""
            return "❌ No dataset loaded yet."

    def _get_column_recommendation_answer(self, query: str) -> str:
        """Get preprocessing recommendation for a column."""
        if self.current_dataframe is None:
            return "❌ No dataset loaded. Please upload a dataset first."

        column_name = self._extract_column_name(query)

        if column_name and column_name in self.current_dataframe.columns:
            col = self.current_dataframe[column_name]
            result = self.preprocessor.preprocess_column(col, column_name)

            response = f"💡 **Recommendation for '{column_name}'**\n\n"
            response += f"**Action:** {result.action.value.replace('_', ' ').title()}\n"
            response += f"**Confidence:** {result.confidence*100:.1f}%\n"
            response += f"**Source:** {result.source.title()}\n\n"
            response += f"**Explanation:**\n{result.explanation}\n\n"

            if result.alternatives:
                response += "**Alternatives:**\n"
                for alt_action, alt_conf in result.alternatives[:3]:
                    response += f"• {alt_action.value.replace('_', ' ').title()} ({alt_conf*100:.0f}%)\n"

            if result.warning:
                response += f"\n⚠️ {result.warning}"

            return response

        return "❓ Please specify which column you want a recommendation for."

    def _explain_column_decision(self, query: str) -> str:
        """Explain why a particular decision was made for a column."""
        if self.current_dataframe is None:
            return "❌ No dataset loaded yet."

        column_name = self._extract_column_name(query)

        if column_name and column_name in self.current_dataframe.columns:
            col = self.current_dataframe[column_name]

            # Get decision with SHAP if available
            try:
                result = self.preprocessor.preprocess_column(col, column_name)

                # Try to get SHAP explanation if neural oracle was used
                if result.source == 'neural' and self.preprocessor.neural_oracle:
                    from ..features.minimal_extractor import get_feature_extractor
                    extractor = get_feature_extractor()
                    features = extractor.extract(col, column_name)

                    shap_result = self.preprocessor.neural_oracle.predict_with_shap(features, top_k=5)

                    response = f"🔍 **Why {result.action.value.replace('_', ' ').title()}?**\n\n"
                    response += f"**Decision Source:** Neural Oracle (Symbolic confidence was {result.confidence:.2%})\n\n"
                    response += f"**Top 5 Contributing Factors:**\n"

                    for i, (feature, impact) in enumerate(shap_result['top_features'], 1):
                        direction = "increases" if impact['impact'] > 0 else "decreases"
                        response += f"{i}. **{feature.replace('_', ' ').title()}** {direction} confidence by {abs(impact['impact']):.3f}\n"

                    response += f"\n**What this means:**\n"
                    for exp in shap_result['explanation']:
                        response += f"• {exp}\n"

                    return response
                else:
                    response = f"🔍 **Why {result.action.value.replace('_', ' ').title()}?**\n\n"
                    response += f"**Decision Source:** {result.source.title()}\n\n"
                    response += f"**Reasoning:**\n{result.explanation}\n\n"

                    # Add statistical context
                    stats = self.symbolic_engine.compute_column_statistics(col, column_name)
                    response += f"**Key Statistics:**\n"

                    if stats.is_numeric:
                        response += f"• Skewness: {stats.skewness:.2f} {'(highly skewed)' if abs(stats.skewness) > 2 else '(moderate)'}\n"
                        response += f"• Outliers: {stats.outlier_percentage*100:.1f}%\n"
                        response += f"• Missing: {stats.null_percentage*100:.1f}%\n"
                    else:
                        response += f"• Unique values: {stats.unique_count:,} ({stats.unique_ratio*100:.1f}%)\n"
                        response += f"• Missing: {stats.null_percentage*100:.1f}%\n"

                    return response

            except Exception as e:
                return f"❌ Error analyzing column: {str(e)}"

        return "❓ Please specify which column you want explained."

    def _get_dataset_summary(self) -> str:
        """Get overall dataset summary."""
        if self.current_dataframe is None:
            return "❌ No dataset loaded. Please upload a CSV file first."

        df = self.current_dataframe

        response = f"📋 **Dataset Summary**\n\n"
        response += f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]:,} columns\n\n"

        # Column types breakdown
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        object_cols = df.select_dtypes(include=['object']).columns
        datetime_cols = df.select_dtypes(include=['datetime']).columns

        response += f"**Column Types:**\n"
        response += f"• Numeric: {len(numeric_cols)}\n"
        response += f"• Categorical: {len(object_cols)}\n"
        response += f"• Datetime: {len(datetime_cols)}\n\n"

        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        response += f"**Memory:** {memory_mb:.2f} MB\n\n"

        # Missing data summary
        missing = df.isnull().sum()
        cols_with_missing = missing[missing > 0]

        if len(cols_with_missing) > 0:
            response += f"**Missing Data:** {len(cols_with_missing)} columns have missing values\n"
            for col in cols_with_missing.head(5).index:
                pct = (missing[col] / len(df)) * 100
                response += f"• {col}: {missing[col]:,} ({pct:.1f}%)\n"

            if len(cols_with_missing) > 5:
                response += f"... and {len(cols_with_missing) - 5} more\n"
        else:
            response += "**Missing Data:** ✓ No missing values!\n"

        return response

    def _get_data_quality_report(self) -> str:
        """Get data quality issues report."""
        if self.current_dataframe is None:
            return "❌ No dataset loaded."

        df = self.current_dataframe
        issues = []

        # Check for high missing rates
        missing_pct = (df.isnull().sum() / len(df)) * 100
        high_missing = missing_pct[missing_pct > 50]
        if len(high_missing) > 0:
            issues.append(f"⚠️ {len(high_missing)} columns have >50% missing data")

        # Check for duplicates
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            issues.append(f"⚠️ {dup_count:,} duplicate rows ({dup_count/len(df)*100:.1f}%)")

        # Check for constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                issues.append(f"⚠️ '{col}' has only one unique value (constant)")

        # Check for high cardinality
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) > 0.9:
                issues.append(f"⚠️ '{col}' has {df[col].nunique():,} unique values (possible ID)")

        response = "🔍 **Data Quality Report**\n\n"

        if len(issues) == 0:
            response += "✅ **No major issues detected!**\n\nYour data looks good."
        else:
            response += f"**Found {len(issues)} potential issues:**\n\n"
            for issue in issues:
                response += f"{issue}\n"

        return response

    def _get_columns_overview(self) -> str:
        """Get overview of all columns."""
        if self.current_dataframe is None:
            return "❌ No dataset loaded."

        df = self.current_dataframe

        response = f"📊 **Columns Overview** ({len(df.columns)} total)\n\n"

        for col in df.columns[:20]:  # Show first 20
            dtype = df[col].dtype
            nulls = df[col].isnull().sum()
            unique = df[col].nunique()

            response += f"**{col}**\n"
            response += f"  Type: {dtype}, Unique: {unique:,}, Missing: {nulls:,}\n\n"

        if len(df.columns) > 20:
            response += f"... and {len(df.columns) - 20} more columns"

        return response

    def _explain_shap(self) -> str:
        """Explain SHAP values in general."""
        return """🔬 **Understanding SHAP Explanations**

**What is SHAP?**
SHAP (SHapley Additive exPlanations) tells you which features influenced the AI's decision and by how much.

**How to read it:**
• **Positive values** (+0.15): This feature made the AI MORE confident in its recommendation
• **Negative values** (-0.08): This feature made the AI LESS confident

**Example:**
```
Top features for LOG_TRANSFORM:
1. skewness (+0.22) - High skew strongly suggests log transform
2. has_outliers (+0.15) - Outliers support transformation
3. null_percentage (-0.08) - Missing data reduces confidence slightly
```

**In plain English:**
The AI is confident about LOG_TRANSFORM mainly because:
1. The data is highly skewed (most important factor)
2. There are outliers present (supporting factor)
3. But some missing data adds slight uncertainty

**Want to see SHAP for a specific column?**
Ask: "Why did you recommend [action] for [column]?" """

    def _answer_statistical_query(self, query: str) -> str:
        """Answer statistical questions about data."""
        if self.current_dataframe is None:
            return "❌ No dataset loaded."

        df = self.current_dataframe
        numeric_cols = df.select_dtypes(include=[np.number])

        if 'mean' in query or 'average' in query:
            response = "📊 **Mean Values:**\n\n"
            for col in numeric_cols.columns[:10]:
                response += f"• {col}: {numeric_cols[col].mean():.2f}\n"

        elif 'median' in query:
            response = "📊 **Median Values:**\n\n"
            for col in numeric_cols.columns[:10]:
                response += f"• {col}: {numeric_cols[col].median():.2f}\n"

        elif 'std' in query or 'standard deviation' in query:
            response = "📊 **Standard Deviations:**\n\n"
            for col in numeric_cols.columns[:10]:
                response += f"• {col}: {numeric_cols[col].std():.2f}\n"

        elif 'distribution' in query:
            response = "📊 **Distribution Analysis:**\n\n"
            for col in numeric_cols.columns[:5]:
                skew = numeric_cols[col].skew()
                kurt = numeric_cols[col].kurtosis()
                response += f"**{col}:**\n"
                response += f"  Skewness: {skew:.2f} "
                response += f"{'(right-skewed)' if skew > 1 else '(left-skewed)' if skew < -1 else '(symmetric)'}\n"
                response += f"  Kurtosis: {kurt:.2f} "
                response += f"{'(heavy-tailed)' if kurt > 3 else '(light-tailed)' if kurt < 3 else '(normal)'}\n\n"

        else:
            response = "📊 **Quick Statistics:**\n\n"
            response += f"Numeric columns: {len(numeric_cols.columns)}\n"
            response += f"Overall mean: {numeric_cols.mean().mean():.2f}\n"
            response += f"Overall std: {numeric_cols.std().mean():.2f}\n"

        return response

    def _explain_preprocessing_technique(self, query: str) -> str:
        """Explain a preprocessing technique."""
        q = query.lower()

        techniques = {
            'log_transform': """📊 **Log Transform**

**When to use:** Highly skewed positive data (skewness > 2.0)

**What it does:** Compresses large values and spreads small values
Example: [1, 10, 100, 1000] → [0, 1, 2, 3]

**Benefits:**
• Makes distribution more normal
• Reduces impact of outliers
• Better for ML algorithms

**Requirements:** All values must be positive

**Use cases:** Revenue, prices, populations""",

            'standard_scale': """📏 **Standard Scaling**

**When to use:** Normal/symmetric distributions

**What it does:** Centers data at 0 with standard deviation of 1
Formula: (x - mean) / std

**Benefits:**
• Features on same scale
• Preserves distribution shape
• Works with negative values

**Use cases:** Well-behaved numeric features""",

            'onehot_encode': """🎯 **One-Hot Encoding**

**When to use:** Low cardinality categorical (<10 categories)

**What it does:** Creates binary column for each category
Example: Color [red, blue, red] → Color_red [1,0,1], Color_blue [0,1,0]

**Benefits:**
• No ordinal relationship implied
• Works with most ML algorithms

**Drawbacks:** Increases dimensions

**Use cases:** Categories, types, labels""",

            'impute': """🔧 **Imputation (Filling Missing Values)**

**Strategies:**
• **Mean/Median:** For numeric data
• **Mode:** For categorical data
• **Forward/Backward Fill:** For time series
• **Interpolation:** For smooth sequences

**When to use:** <30% missing data

**When NOT to use:** >50% missing (consider dropping)

**Impact:** Can introduce bias, always report imputed values"""
        }

        for key, explanation in techniques.items():
            if key.replace('_', '') in q.replace(' ', '').replace('_', ''):
                return explanation

        return "❓ I can explain: log_transform, standard_scale, onehot_encode, imputation, and more. Which technique are you interested in?"

    def _get_capabilities(self) -> str:
        """Explain what the assistant can do."""
        return """🤖 **I'm AURORA's Intelligent Assistant!**

**I can help you with:**

📊 **Column Analysis:**
• "What are the statistics for revenue?"
• "Why did you recommend log_transform for price?"
• "Explain the decision for customer_age"

📈 **Dataset Insights:**
• "Give me a dataset summary"
• "What data quality issues do we have?"
• "Show me all columns"

🔬 **SHAP Explanations:**
• "Explain SHAP values"
• "Why was skewness important?"
• "What features influenced this decision?"

📐 **Statistical Queries:**
• "What's the mean of all numeric columns?"
• "Show me distribution analysis"
• "Calculate standard deviations"

🎓 **Learn Techniques:**
• "When should I use log transform?"
• "Explain standard scaling"
• "What is one-hot encoding?"

**Try asking me something specific about your data!** """

    def _generate_contextual_response(self, query: str) -> str:
        """Generate helpful response when query doesn't match patterns."""
        if self.current_dataframe is not None:
            return f"""🤔 I'm not sure about that specific question, but I can help with:

**Your Current Dataset:**
• {self.current_dataframe.shape[0]:,} rows × {self.current_dataframe.shape[1]:,} columns
• {len(self.current_dataframe.select_dtypes(include=[np.number]).columns)} numeric columns

**Try asking:**
• "What are the statistics for [column_name]?"
• "Give me a dataset summary"
• "Why did you recommend [action] for [column]?"
• "What data quality issues do we have?"

**Or pick a topic:**
• Column analysis
• Dataset insights
• SHAP explanations
• Statistical queries
• Preprocessing techniques"""
        else:
            return """👋 **Welcome to AURORA!**

I'm here to help you understand your data and preprocessing decisions.

**First, upload a dataset**, then ask me questions like:
• "What are the statistics for revenue?"
• "Why should I use log transform?"
• "Give me a dataset summary"
• "Explain SHAP values"

**I can provide:**
✓ Real-time data analysis
✓ Statistical insights
✓ SHAP explanations
✓ Preprocessing recommendations
✓ User-friendly interpretations

Start by uploading a CSV file! 📊"""

    def _extract_column_name(self, query: str) -> Optional[str]:
        """Try to extract column name from query."""
        if self.current_dataframe is None:
            return None

        # Try to find column name in query
        for col in self.current_dataframe.columns:
            if col.lower() in query.lower():
                return col

        # Try quoted strings
        import re
        quoted = re.findall(r'["\']([^"\']+)["\']', query)
        if quoted:
            for q in quoted:
                if q in self.current_dataframe.columns:
                    return q

        return None
