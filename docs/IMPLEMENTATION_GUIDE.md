# 🚀 AURORA Implementation Guide

**Version:** 1.0
**Last Updated:** November 20, 2024
**Purpose:** Step-by-step guide to implement the complete AURORA system with explainability

---

## 📑 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Phase 1: Add SHAP Explainability](#phase-1-add-shap-explainability)
4. [Phase 2: Train on Real-World Data](#phase-2-train-on-real-world-data)
5. [Phase 3: Implement Confidence Thresholds](#phase-3-implement-confidence-thresholds)
6. [Phase 4: Add Layer-by-Layer Metrics](#phase-4-add-layer-by-layer-metrics)
7. [Phase 5: Testing & Validation](#phase-5-testing--validation)
8. [Existing Scripts Reference](#existing-scripts-reference)
9. [Verification Checklist](#verification-checklist)

---

## 1. Architecture Overview

### Final Architecture

```
User Request (CSV column)
        ↓
┌──────────────────────────────────────────┐
│  Layer 1: Learned Patterns               │
│  Source: User corrections (database)     │
│  Explainability: ✅ "Similar to N cases" │
│  Confidence: > 0.9 → DONE                │
└──────────────┬───────────────────────────┘
               ↓ (if confidence < 0.9)
┌──────────────────────────────────────────┐
│  Layer 2: Symbolic Engine                │
│  Source: 165+ deterministic rules        │
│  Explainability: ✅ Rule name + reasoning│
│  Confidence: > 0.9 → DONE                │
└──────────────┬───────────────────────────┘
               ↓ (if confidence < 0.9)
┌──────────────────────────────────────────┐
│  Layer 3: Neural Oracle (XGBoost)        │
│  Training: Synthetic + Open Source       │
│  Explainability: ✅ SHAP values          │
│  Confidence: Always shown with warning   │
└──────────────┬───────────────────────────┘
               ↓
    Final Decision with Full Explanation
```

### Key Principles

✅ **Explainability First** - Every decision must be explainable
✅ **Privacy Preserved** - No raw data stored, only statistical fingerprints
✅ **Measurable** - Track accuracy at each layer
✅ **Simple** - Don't over-engineer

---

## 2. Prerequisites

### 2.1 Dependencies

Add to `requirements.txt`:

```txt
# Already in requirements.txt:
xgboost>=1.7.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0

# NEW - Add these:
shap>=0.42.0  # For explainability
```

Install:
```bash
pip install shap
```

### 2.2 Environment Setup

Ensure `.env` is configured:
```bash
cp .env.example .env

# Edit .env:
DATABASE_URL=sqlite:///./aurora.db
JWT_SECRET_KEY=<generate-with-openssl-rand-hex-32>
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000
```

### 2.3 Verify Existing Scripts

Check that these scripts exist:
```bash
ls scripts/
# Should see:
# - train_neural_oracle.py       (trains on synthetic data)
# - train_from_corrections.py    (trains on user corrections)
# - train_with_dataset.py        (trains on custom datasets)
# - evaluate_system.py           (evaluates accuracy)
# - benchmark_performance.py     (measures latency)
# - generate_synthetic_data.py   (creates synthetic data)
```

---

## 3. Phase 1: Add SHAP Explainability

### 3.1 Goal

Make neural oracle decisions explainable by showing which features contributed to the decision.

### 3.2 Implementation

**File:** `src/neural/oracle.py`

Add this method to the `NeuralOracle` class:

```python
def predict_with_shap(
    self,
    features: MinimalFeatures,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Predict with SHAP explanation.

    SHAP (SHapley Additive exPlanations) shows which features
    contributed to this specific prediction and by how much.

    Args:
        features: Extracted features from column
        top_k: Number of top contributing features to return

    Returns:
        Dictionary with:
        - action: Predicted preprocessing action
        - confidence: Prediction confidence (0-1)
        - explanation: Human-readable explanation
        - shap_values: Feature contributions
        - top_features: Top K contributing features
    """
    import shap

    if self.model is None:
        raise ValueError("Model not trained or loaded.")

    # Get base prediction
    prediction = self.predict(features, return_probabilities=True)

    # Calculate SHAP values
    X = features.to_array().reshape(1, -1)
    explainer = shap.TreeExplainer(self.model)

    # Get SHAP values for the predicted class
    shap_values = explainer.shap_values(X)

    # Handle multi-class output
    if isinstance(shap_values, list):
        # Get SHAP values for predicted class
        predicted_idx = np.argmax(self.model.predict(
            xgb.DMatrix(X, feature_names=self.feature_names)
        )[0])
        class_shap_values = shap_values[predicted_idx][0]
    else:
        class_shap_values = shap_values[0]

    # Create feature contribution dictionary
    contributions = {
        name: float(class_shap_values[idx])
        for idx, name in enumerate(self.feature_names)
    }

    # Get top contributing features
    top_features = sorted(
        contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_k]

    # Build human-readable explanation
    explanation_parts = []
    for feature, impact in top_features:
        direction = "increases" if impact > 0 else "decreases"
        explanation_parts.append(
            f"{feature.replace('_', ' ')} {direction} confidence "
            f"(impact: {impact:+.2f})"
        )

    return {
        'action': prediction.action,
        'confidence': prediction.confidence,
        'explanation': explanation_parts,
        'shap_values': contributions,
        'top_features': [
            {'feature': name, 'impact': impact}
            for name, impact in top_features
        ],
        'action_probabilities': prediction.action_probabilities
    }
```

### 3.3 Integration into Preprocessor

**File:** `src/core/preprocessor.py`

Update the neural oracle decision section:

```python
# Find this section in preprocess_column():
if symbolic_result.confidence < self.neural_fallback_threshold:
    # Use neural oracle
    try:
        # OLD:
        # neural_prediction = self.neural_oracle.predict(features)

        # NEW - Use SHAP-enabled prediction:
        neural_prediction = self.neural_oracle.predict_with_shap(features, top_k=3)

        result = PreprocessingResult(
            action=neural_prediction['action'],
            confidence=neural_prediction['confidence'],
            source='neural',
            explanation=(
                f"Neural oracle prediction based on:\n" +
                "\n".join(f"  • {exp}" for exp in neural_prediction['explanation'])
            ),
            alternatives=self._get_alternatives(
                neural_prediction['action_probabilities']
            ),
            metadata={
                'shap_values': neural_prediction['shap_values'],
                'top_features': neural_prediction['top_features'],
                'symbolic_fallback': {
                    'action': symbolic_result.action.value,
                    'confidence': symbolic_result.confidence,
                    'reasoning': symbolic_result.explanation
                }
            }
        )

        # ... rest of code
```

### 3.4 Test SHAP Implementation

Create `tests/test_shap_explainability.py`:

```python
import pytest
import numpy as np
import pandas as pd
from src.neural.oracle import NeuralOracle
from src.features.minimal_extractor import MinimalFeatureExtractor
from pathlib import Path

def test_shap_explanation_exists():
    """Test that SHAP explanations are generated."""

    # Load trained model
    model_path = Path("models/neural_oracle_v1.pkl")
    if not model_path.exists():
        pytest.skip("No trained model found")

    oracle = NeuralOracle(model_path)
    extractor = MinimalFeatureExtractor()

    # Create test column (skewed data)
    test_column = pd.Series([1, 2, 3, 100, 200, 500])
    features = extractor.extract(test_column)

    # Get SHAP prediction
    result = oracle.predict_with_shap(features)

    # Verify structure
    assert 'action' in result
    assert 'confidence' in result
    assert 'explanation' in result
    assert 'shap_values' in result
    assert 'top_features' in result

    # Verify explanation is non-empty
    assert len(result['explanation']) > 0

    # Verify SHAP values exist for all features
    assert len(result['shap_values']) == len(oracle.feature_names)

    # Verify top features are sorted by impact
    impacts = [abs(f['impact']) for f in result['top_features']]
    assert impacts == sorted(impacts, reverse=True)

    print("\n✅ SHAP explanation generated successfully:")
    print(f"   Action: {result['action'].value}")
    print(f"   Confidence: {result['confidence']:.2%}")
    print("   Top contributing features:")
    for feat in result['top_features']:
        print(f"     • {feat['feature']}: {feat['impact']:+.2f}")
```

Run test:
```bash
pytest tests/test_shap_explainability.py -v
```

---

## 4. Phase 2: Train on Real-World Data

### 4.1 Existing Script: `train_with_dataset.py`

**Status:** ✅ Already exists and functional

This script trains on custom datasets (CSV, JSON, Pickle):

```bash
# Train on a single dataset
python scripts/train_with_dataset.py --file data/titanic.csv --type csv

# Train on multiple datasets
python scripts/train_with_dataset.py \
    --file data/housing.csv \
    --file data/iris.csv \
    --output models/multi_dataset_oracle.pkl
```

### 4.2 Create Dataset Collection Script

**New File:** `scripts/collect_open_datasets.py`

```python
"""
Collect open-source datasets for training.

Downloads popular datasets from:
- Scikit-learn built-in datasets
- UCI ML Repository (manual download)
- Kaggle (requires API key)
"""

import pandas as pd
from sklearn import datasets
from pathlib import Path
import requests


def collect_sklearn_datasets(output_dir: Path):
    """Collect all scikit-learn built-in datasets."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Regression datasets
    datasets_to_fetch = [
        ('boston', datasets.load_diabetes),  # Boston housing deprecated
        ('diabetes', datasets.load_diabetes),
        ('california_housing', datasets.fetch_california_housing),
    ]

    # Classification datasets
    datasets_to_fetch += [
        ('iris', datasets.load_iris),
        ('wine', datasets.load_wine),
        ('breast_cancer', datasets.load_breast_cancer),
        ('digits', datasets.load_digits),
    ]

    collected = []

    for name, loader in datasets_to_fetch:
        try:
            print(f"Fetching {name}...")
            data = loader()

            # Convert to DataFrame
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target

            # Save
            output_file = output_dir / f"{name}.csv"
            df.to_csv(output_file, index=False)

            collected.append(name)
            print(f"  ✓ Saved to {output_file}")

        except Exception as e:
            print(f"  ✗ Failed: {e}")

    return collected


def collect_uci_datasets(output_dir: Path):
    """
    Download popular UCI datasets.

    Note: Some UCI datasets require manual download due to licensing.
    This function downloads the publicly available ones.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # UCI datasets with direct download links
    uci_datasets = [
        {
            'name': 'adult',
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
            'columns': ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                       'marital-status', 'occupation', 'relationship', 'race', 'sex',
                       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
        },
        {
            'name': 'heart_disease',
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',
            'columns': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        }
    ]

    collected = []

    for dataset in uci_datasets:
        try:
            print(f"Downloading {dataset['name']}...")
            response = requests.get(dataset['url'], timeout=30)
            response.raise_for_status()

            # Save raw data
            output_file = output_dir / f"{dataset['name']}.csv"

            with open(output_file, 'w') as f:
                f.write(','.join(dataset['columns']) + '\n')
                f.write(response.text)

            collected.append(dataset['name'])
            print(f"  ✓ Downloaded to {output_file}")

        except Exception as e:
            print(f"  ✗ Failed: {e}")

    return collected


def main():
    """Main collection workflow."""

    print("="*70)
    print("AURORA Open Dataset Collection")
    print("="*70 + "\n")

    # Create datasets directory
    datasets_dir = Path("data/open_datasets")
    datasets_dir.mkdir(parents=True, exist_ok=True)

    # Collect datasets
    print("\n1. Collecting scikit-learn datasets...")
    sklearn_datasets = collect_sklearn_datasets(datasets_dir / "sklearn")

    print(f"\n2. Collecting UCI datasets...")
    uci_datasets = collect_uci_datasets(datasets_dir / "uci")

    # Summary
    print("\n" + "="*70)
    print("COLLECTION COMPLETE")
    print("="*70)
    print(f"\nCollected {len(sklearn_datasets) + len(uci_datasets)} datasets:")
    print(f"  • Scikit-learn: {len(sklearn_datasets)}")
    print(f"  • UCI ML: {len(uci_datasets)}")

    print(f"\nDatasets saved to: {datasets_dir}")
    print("\nNext steps:")
    print("  1. Review datasets for quality")
    print("  2. Train neural oracle:")
    print(f"     python scripts/train_with_dataset.py --dir {datasets_dir}")


if __name__ == "__main__":
    main()
```

### 4.3 Train on Collected Datasets

```bash
# Step 1: Collect datasets
python scripts/collect_open_datasets.py

# Step 2: Train on collected datasets
python scripts/train_with_dataset.py \
    --dir data/open_datasets/sklearn \
    --output models/neural_oracle_sklearn.pkl

# Step 3: Evaluate
python scripts/evaluate_system.py --model models/neural_oracle_sklearn.pkl
```

### 4.4 Hybrid Training (Recommended)

Create `scripts/train_hybrid.py`:

```python
"""
Hybrid training: Synthetic + Open Datasets + User Corrections.

This combines all three data sources for optimal performance.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from train_neural_oracle import load_synthetic_data
from train_from_corrections import load_correction_data
from train_with_dataset import load_and_label_datasets


def main(
    synthetic_samples: int = 1000,
    open_datasets_dir: str = "data/open_datasets",
    min_corrections: int = 20,
    output: str = "models/neural_oracle_hybrid.pkl"
):
    """
    Train hybrid model combining all data sources.

    Data weights:
    - User corrections: 3.0x (most important - domain-specific)
    - Open datasets: 2.0x (real-world patterns)
    - Synthetic: 1.0x (baseline coverage)
    """

    print("="*70)
    print("AURORA Hybrid Neural Oracle Training")
    print("="*70 + "\n")

    all_features = []
    all_labels = []
    all_weights = []

    # 1. Load user corrections
    print("1. Loading user corrections...")
    correction_features, correction_labels, metadata = load_correction_data()

    if len(correction_features) >= min_corrections:
        all_features.extend(correction_features)
        all_labels.extend(correction_labels)
        all_weights.extend([3.0] * len(correction_features))
        print(f"   ✓ Loaded {len(correction_features)} corrections (weight: 3.0x)")
    else:
        print(f"   ⚠ Only {len(correction_features)} corrections (min: {min_corrections})")

    # 2. Load open datasets
    print("\n2. Loading open datasets...")
    if Path(open_datasets_dir).exists():
        open_features, open_labels = load_and_label_datasets(open_datasets_dir)
        all_features.extend(open_features)
        all_labels.extend(open_labels)
        all_weights.extend([2.0] * len(open_features))
        print(f"   ✓ Loaded {len(open_features)} examples (weight: 2.0x)")
    else:
        print(f"   ⚠ Directory not found: {open_datasets_dir}")

    # 3. Load synthetic data
    print("\n3. Generating synthetic data...")
    synthetic_features, synthetic_labels = load_synthetic_data(synthetic_samples)
    all_features.extend(synthetic_features)
    all_labels.extend(synthetic_labels)
    all_weights.extend([1.0] * len(synthetic_features))
    print(f"   ✓ Generated {len(synthetic_features)} samples (weight: 1.0x)")

    # 4. Train model
    print(f"\n4. Training hybrid model...")
    print(f"   Total samples: {len(all_features)}")
    print(f"   Breakdown:")
    print(f"     • Corrections: {len(correction_features)} (weight: 3.0x)")
    print(f"     • Open datasets: {len(open_features)} (weight: 2.0x)")
    print(f"     • Synthetic: {len(synthetic_features)} (weight: 1.0x)")

    # Train with sample weights
    oracle = NeuralOracle()
    metrics = oracle.train(
        features=all_features,
        labels=all_labels,
        validation_split=0.2,
        sample_weights=all_weights  # Weighted training
    )

    print(f"\n   Training complete!")
    print(f"   • Training accuracy: {metrics['train_accuracy']:.2%}")
    print(f"   • Validation accuracy: {metrics['val_accuracy']:.2%}")

    # 5. Save model
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    oracle.save(output_path)

    print(f"\n✓ Hybrid model saved to: {output_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic', type=int, default=1000)
    parser.add_argument('--datasets-dir', default="data/open_datasets")
    parser.add_argument('--min-corrections', type=int, default=20)
    parser.add_argument('--output', default="models/neural_oracle_hybrid.pkl")

    args = parser.parse_args()

    main(
        synthetic_samples=args.synthetic,
        open_datasets_dir=args.datasets_dir,
        min_corrections=args.min_corrections,
        output=args.output
    )
```

Usage:
```bash
python scripts/train_hybrid.py \
    --synthetic 1000 \
    --datasets-dir data/open_datasets \
    --output models/neural_oracle_hybrid.pkl
```

---

## 5. Phase 3: Implement Confidence Thresholds

### 5.1 Add Confidence Constants

**File:** `src/core/preprocessor.py`

Add at the top of the file:

```python
# Confidence thresholds
CONFIDENCE_HIGH = 0.9      # Auto-apply decision
CONFIDENCE_MEDIUM = 0.7    # Show warning
CONFIDENCE_LOW = 0.5       # Require manual review

class PreprocessingResult:
    """Extended preprocessing result with warnings."""

    def __init__(
        self,
        action: PreprocessingAction,
        confidence: float,
        source: str,
        explanation: str,
        alternatives: List[AlternativeAction] = None,
        metadata: Dict[str, Any] = None,
        warning: Optional[str] = None,
        require_manual_review: bool = False
    ):
        self.action = action
        self.confidence = confidence
        self.source = source
        self.explanation = explanation
        self.alternatives = alternatives or []
        self.metadata = metadata or {}
        self.warning = warning
        self.require_manual_review = require_manual_review
```

### 5.2 Add Confidence Checking

Update the decision logic:

```python
def _add_confidence_warnings(self, result: PreprocessingResult) -> PreprocessingResult:
    """Add warnings based on confidence level."""

    if result.confidence < CONFIDENCE_LOW:
        result.warning = "⚠️ Very low confidence - manual review strongly recommended"
        result.require_manual_review = True
    elif result.confidence < CONFIDENCE_MEDIUM:
        result.warning = "⚠️ Low confidence - consider reviewing this decision"

    return result


def preprocess_column(self, column_data, column_name, metadata=None):
    """Process column with confidence warnings."""

    # ... existing code ...

    # After getting result from any layer:
    result = self._add_confidence_warnings(result)

    return result
```

### 5.3 Update API Response

**File:** `src/api/schemas.py`

Add warning fields:

```python
class PreprocessResponse(BaseModel):
    """Preprocessing recommendation response."""

    action: str
    confidence: float
    source: str
    explanation: str
    alternatives: List[AlternativeAction] = []
    metadata: Dict[str, Any] = {}

    # NEW fields:
    warning: Optional[str] = None
    require_manual_review: bool = False
```

---

## 6. Phase 4: Add Layer-by-Layer Metrics

### 6.1 Create Metrics Tracker

**New File:** `src/utils/layer_metrics.py`

```python
"""
Track accuracy and usage by decision layer.

This proves which layers work best and identifies improvement areas.
"""

from typing import Dict, Optional
from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class LayerStats:
    """Statistics for a single decision layer."""

    total_decisions: int = 0
    correct_decisions: int = 0
    total_confidence: float = 0.0

    @property
    def accuracy(self) -> float:
        """Calculate accuracy percentage."""
        return (
            self.correct_decisions / self.total_decisions * 100
            if self.total_decisions > 0 else 0.0
        )

    @property
    def avg_confidence(self) -> float:
        """Calculate average confidence."""
        return (
            self.total_confidence / self.total_decisions
            if self.total_decisions > 0 else 0.0
        )

    @property
    def usage_percentage(self, total: int) -> float:
        """Calculate usage percentage."""
        return (
            self.total_decisions / total * 100
            if total > 0 else 0.0
        )


class LayerMetrics:
    """
    Track metrics for each decision layer.

    Layers:
    - learned: Learned patterns from corrections
    - symbolic: Symbolic rule engine
    - neural: Neural oracle
    """

    def __init__(self, persistence_file: Optional[Path] = None):
        self.stats = {
            'learned': LayerStats(),
            'symbolic': LayerStats(),
            'neural': LayerStats()
        }
        self.persistence_file = persistence_file

        # Load existing stats if available
        if persistence_file and persistence_file.exists():
            self.load()

    def record_decision(
        self,
        layer: str,
        confidence: float,
        was_correct: Optional[bool] = None
    ):
        """
        Record a decision from a layer.

        Args:
            layer: Which layer made the decision
            confidence: Decision confidence
            was_correct: Whether it was correct (None if unknown)
        """
        if layer not in self.stats:
            raise ValueError(f"Unknown layer: {layer}")

        stats = self.stats[layer]
        stats.total_decisions += 1
        stats.total_confidence += confidence

        if was_correct is not None and was_correct:
            stats.correct_decisions += 1

    def record_correction(self, layer: str):
        """
        Record that a decision was corrected (was wrong).

        Args:
            layer: Which layer made the wrong decision
        """
        # Correction means it was wrong, so don't increment correct_decisions
        pass

    def get_summary(self) -> Dict:
        """Get summary of all layer metrics."""

        total_decisions = sum(s.total_decisions for s in self.stats.values())

        return {
            'total_decisions': total_decisions,
            'by_layer': {
                layer: {
                    'decisions': stats.total_decisions,
                    'usage_pct': (
                        stats.total_decisions / total_decisions * 100
                        if total_decisions > 0 else 0
                    ),
                    'accuracy_pct': stats.accuracy,
                    'avg_confidence': stats.avg_confidence
                }
                for layer, stats in self.stats.items()
            },
            'overall_accuracy': (
                sum(s.correct_decisions for s in self.stats.values()) /
                total_decisions * 100
                if total_decisions > 0 else 0
            )
        }

    def save(self):
        """Save metrics to file."""
        if not self.persistence_file:
            return

        self.persistence_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            layer: asdict(stats)
            for layer, stats in self.stats.items()
        }

        with open(self.persistence_file, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load metrics from file."""
        if not self.persistence_file or not self.persistence_file.exists():
            return

        with open(self.persistence_file, 'r') as f:
            data = json.load(f)

        for layer, stats_dict in data.items():
            self.stats[layer] = LayerStats(**stats_dict)
```

### 6.2 Integrate into Preprocessor

**File:** `src/core/preprocessor.py`

```python
from ..utils.layer_metrics import LayerMetrics

class IntelligentPreprocessor:
    def __init__(self):
        # ... existing code ...

        # NEW: Add metrics tracker
        self.layer_metrics = LayerMetrics(
            persistence_file=Path("data/layer_metrics.json")
        )

    def preprocess_column(self, column_data, column_name, metadata=None):
        """Preprocess with metrics tracking."""

        # ... existing decision logic ...

        # After getting result:
        self.layer_metrics.record_decision(
            layer=result.source,
            confidence=result.confidence
        )

        # Save metrics periodically
        if self.layer_metrics.stats['learned'].total_decisions % 100 == 0:
            self.layer_metrics.save()

        return result

    def submit_correction(self, decision_id, correct_action):
        """Record correction and update metrics."""

        # ... existing correction logic ...

        # Update metrics - decision was wrong
        original_decision = self.get_decision(decision_id)
        self.layer_metrics.record_correction(original_decision.source)

        return result
```

### 6.3 Add Metrics API Endpoint

**File:** `src/api/server.py`

```python
@app.get("/metrics/layers")
async def get_layer_metrics():
    """
    Get layer-by-layer performance metrics.

    Shows which layers are used most and their accuracy.
    """
    try:
        summary = preprocessor.layer_metrics.get_summary()

        return {
            "total_decisions": summary['total_decisions'],
            "overall_accuracy": f"{summary['overall_accuracy']:.1f}%",
            "layers": summary['by_layer'],
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"Error getting layer metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get layer metrics: {str(e)}"
        )
```

---

## 7. Phase 5: Testing & Validation

### 7.1 Create Test Suite

**File:** `tests/test_complete_system.py`

```python
"""
Complete system test suite.

Tests:
1. All three layers work
2. Explainability at each layer
3. Confidence thresholds trigger correctly
4. Metrics are tracked
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.core.preprocessor import IntelligentPreprocessor


class TestCompleteSystem:

    @pytest.fixture
    def preprocessor(self):
        """Get preprocessor instance."""
        return IntelligentPreprocessor()

    def test_symbolic_layer_explainability(self, preprocessor):
        """Test that symbolic decisions are explainable."""

        # High skewness -> should trigger symbolic rule
        data = pd.Series([1, 2, 3, 1000, 2000, 5000])

        result = preprocessor.preprocess_column(data, "revenue")

        assert result.source == "symbolic"
        assert result.explanation is not None
        assert len(result.explanation) > 0
        assert result.confidence > 0.9
        print(f"\n✓ Symbolic explanation: {result.explanation}")

    def test_neural_oracle_shap_explainability(self, preprocessor):
        """Test that neural oracle provides SHAP explanations."""

        # Borderline case -> should trigger neural oracle
        data = pd.Series([10, 20, 30, 40, 50, 100])  # Moderate skew

        result = preprocessor.preprocess_column(data, "score")

        if result.source == "neural":
            assert 'shap_values' in result.metadata
            assert result.explanation is not None
            print(f"\n✓ Neural explanation: {result.explanation}")
            print(f"  SHAP values available: {len(result.metadata['shap_values'])} features")

    def test_confidence_warnings(self, preprocessor):
        """Test that low confidence triggers warnings."""

        # Ambiguous data
        data = pd.Series([1, 2, 3, 4, 5])

        result = preprocessor.preprocess_column(data, "unknown")

        if result.confidence < 0.7:
            assert result.warning is not None
            print(f"\n✓ Warning triggered: {result.warning}")

        if result.confidence < 0.5:
            assert result.require_manual_review is True
            print(f"  Manual review required")

    def test_metrics_tracking(self, preprocessor):
        """Test that metrics are tracked."""

        # Make several decisions
        for i in range(10):
            data = pd.Series(np.random.randn(100))
            preprocessor.preprocess_column(data, f"col_{i}")

        # Check metrics
        summary = preprocessor.layer_metrics.get_summary()

        assert summary['total_decisions'] >= 10
        assert 'by_layer' in summary

        print("\n✓ Metrics tracked:")
        for layer, stats in summary['by_layer'].items():
            print(f"  {layer}: {stats['decisions']} decisions "
                  f"({stats['usage_pct']:.1f}% usage)")

    def test_all_layers_accessible(self, preprocessor):
        """Test that all layers can be triggered."""

        layers_triggered = set()

        test_cases = [
            # Learned pattern (if any exist)
            pd.Series([1, 2, 3, 4, 5]),

            # Symbolic (high skewness)
            pd.Series([1, 2, 3, 1000, 2000]),

            # Neural (borderline)
            pd.Series([10, 20, 30, 40, 80])
        ]

        for data in test_cases:
            result = preprocessor.preprocess_column(data, "test")
            layers_triggered.add(result.source)

        print(f"\n✓ Layers triggered: {layers_triggered}")

        # At least symbolic and neural should be accessible
        assert 'symbolic' in layers_triggered or 'neural' in layers_triggered
```

Run tests:
```bash
pytest tests/test_complete_system.py -v -s
```

---

## 8. Existing Scripts Reference

### 8.1 Training Scripts

| Script | Purpose | Usage | Status |
|--------|---------|-------|--------|
| `train_neural_oracle.py` | Train on synthetic data | `python scripts/train_neural_oracle.py` | ✅ Works |
| `train_from_corrections.py` | Train on user corrections | `python scripts/train_from_corrections.py` | ✅ Works |
| `train_with_dataset.py` | Train on custom CSV/JSON | `python scripts/train_with_dataset.py --file data.csv` | ✅ Works |
| `train_hybrid.py` | **NEW** Train on all sources | `python scripts/train_hybrid.py` | 🆕 Create this |

### 8.2 Evaluation Scripts

| Script | Purpose | Usage | Status |
|--------|---------|-------|--------|
| `evaluate_system.py` | Evaluate accuracy | `python scripts/evaluate_system.py` | ✅ Works |
| `benchmark_performance.py` | Measure latency | `python scripts/benchmark_performance.py` | ✅ Works |

### 8.3 Data Scripts

| Script | Purpose | Usage | Status |
|--------|---------|-------|--------|
| `generate_synthetic_data.py` | Generate synthetic data | `python scripts/generate_synthetic_data.py` | ✅ Works |
| `collect_open_datasets.py` | **NEW** Download open datasets | `python scripts/collect_open_datasets.py` | 🆕 Create this |

### 8.4 Integration Scripts

| Script | Purpose | Usage | Status |
|--------|---------|-------|--------|
| `integrate_phase1.py` | Phase 1 integration demo | `python scripts/integrate_phase1.py --demo` | ✅ Works |

---

## 9. Verification Checklist

### 9.1 Phase 1: SHAP Explainability

- [ ] SHAP installed (`pip install shap`)
- [ ] `predict_with_shap()` added to `src/neural/oracle.py`
- [ ] Preprocessor uses SHAP-enabled predictions
- [ ] Test passes: `pytest tests/test_shap_explainability.py`
- [ ] Manual verification: SHAP explanations are human-readable

### 9.2 Phase 2: Real-World Data

- [ ] Open datasets collected (at least 10)
- [ ] `train_with_dataset.py` tested on one dataset
- [ ] Hybrid training script created
- [ ] Hybrid model trained successfully
- [ ] Accuracy measured and documented

### 9.3 Phase 3: Confidence Thresholds

- [ ] Confidence constants defined (0.9, 0.7, 0.5)
- [ ] Warning messages added to responses
- [ ] API schema updated with warning fields
- [ ] Frontend displays warnings (if applicable)
- [ ] Manual test: Low confidence triggers warning

### 9.4 Phase 4: Layer Metrics

- [ ] `LayerMetrics` class created
- [ ] Integrated into preprocessor
- [ ] Metrics API endpoint added (`/metrics/layers`)
- [ ] Metrics persist to file
- [ ] Dashboard shows layer breakdown (if applicable)

### 9.5 Phase 5: Testing

- [ ] Test suite created (`tests/test_complete_system.py`)
- [ ] All tests pass
- [ ] Manual end-to-end test performed
- [ ] Documentation updated

---

## 10. Quick Start Commands

### Complete Implementation (2 hours)

```bash
# Phase 1: Add SHAP (30 min)
pip install shap
# Edit src/neural/oracle.py - add predict_with_shap()
# Edit src/core/preprocessor.py - use SHAP predictions
pytest tests/test_shap_explainability.py -v

# Phase 2: Train on Real Data (45 min)
python scripts/collect_open_datasets.py
python scripts/train_hybrid.py
python scripts/evaluate_system.py

# Phase 3: Add Confidence Thresholds (15 min)
# Edit src/core/preprocessor.py - add constants and warnings
# Edit src/api/schemas.py - add warning fields

# Phase 4: Add Metrics (30 min)
# Create src/utils/layer_metrics.py
# Edit src/core/preprocessor.py - integrate metrics
# Add /metrics/layers endpoint

# Phase 5: Test Everything
pytest tests/test_complete_system.py -v -s
```

### Verify Everything Works

```bash
# Start server
uvicorn src.api.server:app --reload

# Test endpoints
curl http://localhost:8000/metrics/layers
curl http://localhost:8000/metrics/neural_oracle
curl http://localhost:8000/metrics/learning

# Check frontend
npm run dev
# Open http://localhost:3000
# Toggle metrics dashboard
```

---

## 11. Troubleshooting

### SHAP Installation Issues

```bash
# If SHAP install fails:
pip install --upgrade pip setuptools wheel
pip install shap --no-cache-dir

# On M1/M2 Mac:
conda install -c conda-forge shap
```

### Dataset Collection Fails

```bash
# If UCI downloads fail, manually download:
mkdir -p data/open_datasets/uci
# Visit https://archive.ics.uci.edu/ml/
# Download CSVs manually

# For Kaggle datasets:
pip install kaggle
# Set up API key: https://www.kaggle.com/docs/api
kaggle datasets download -d <dataset-id>
```

### Training Fails

```bash
# Check XGBoost version:
pip install --upgrade xgboost

# Verify data:
python -c "from src.neural.oracle import NeuralOracle; print('OK')"

# Check logs:
tail -f logs/training.log
```

---

## 12. Next Steps After Implementation

### Immediate (This Week)

1. **Measure Accuracy** - Run evaluation on 100+ test cases
2. **Document Results** - Update README with real accuracy numbers
3. **User Testing** - Get 5 users to try the system
4. **Collect Feedback** - What explanations make sense?

### Short Term (Next Month)

1. **Write Unit Tests** - Achieve 70% coverage
2. **Set Up CI/CD** - GitHub Actions for automated testing
3. **Add Monitoring** - Prometheus + Grafana
4. **Database Migrations** - Set up Alembic properly

### Medium Term (Next Quarter)

1. **Production Deployment** - Docker + Kubernetes
2. **A/B Testing** - Compare model versions
3. **Automated Retraining** - Weekly model updates
4. **Scale Testing** - 1000 req/s load test

---

**Implementation Guide Version:** 1.0
**Last Updated:** November 20, 2024
**Estimated Time to Complete:** 2-4 hours
**Difficulty:** Intermediate

**Ready to implement? Start with Phase 1 (SHAP) and work your way through!** 🚀
