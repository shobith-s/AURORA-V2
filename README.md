# AURORA V2 - Intelligent Data Preprocessing System

**AI-powered data preprocessing with symbolic rules, neural oracle, and adaptive learning**

---

## Overview

AURORA V2 is an intelligent data preprocessing system that combines:
- **Symbolic Engine**: 185+ expert-crafted rules for common preprocessing patterns
- **Neural Oracle**: Pre-trained ensemble (XGBoost + LightGBM) with 89.4% accuracy for edge cases
- **Adaptive Learning**: Learns from user corrections to improve over time
- **LLM Validation**: Uses AI to validate and improve preprocessing decisions

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AURORA V2 System                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   Symbolic   │───▶│    Neural    │───▶│   Adaptive   │ │
│  │    Engine    │    │    Oracle    │    │   Learning   │ │
│  │  (185 rules) │    │  (89.4% acc) │    │   (User      │ │
│  │              │    │   Ensemble   │    │  Corrections)│ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         │                    │                    │        │
│         └────────────────────┴────────────────────┘        │
│                              │                             │
│                    ┌─────────▼─────────┐                   │
│                    │  Preprocessing    │                   │
│                    │     Decision      │                   │
│                    └───────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Features

### 1. **Three-Layer Decision System**

**Layer 1: Symbolic Engine (Primary)**
- 185+ hand-crafted rules
- Handles 85% of cases with high confidence (>0.7)
- Conservative thresholds (80% null, 99.5% unique)
- Fast (<1ms per column)

**Layer 2: Neural Oracle (Ensemble - Edge Cases)**
- **Architecture:** XGBoost + LightGBM ensemble with soft voting
- **Accuracy:** 89.4% on test data (400+ validated examples)
- **Inference Only:** Uses pre-trained model, no runtime training
- **Model File:** `models/neural_oracle_v2_improved_20251129_150244.pkl`
- **Trained:** November 2025 on 40 diverse OpenML datasets with LLM validation
- **Handles:** Ambiguous preprocessing decisions (when symbolic confidence < 0.75)

**Layer 3: Adaptive Learning (Continuous Improvement)**
- Learns from user corrections
- Creates new rules after 10 consistent corrections
- Validates patterns with LLM before deployment

### 2. **Supported Actions**

- `keep_as_is` - Preserve column unchanged
- `drop_column` - Remove low-value columns
- `standard_scale` - Normalize numeric data
- `robust_scale` - Scale with outlier resistance
- `log_transform` - Handle skewed distributions
- `onehot_encode` - Categorical to binary
- `label_encode` - Categorical to numeric
- `hash_encode` - High-cardinality categoricals
- `fill_null_*` - Various null handling strategies

### 3. **Quality Assurance**

- Statistical validation (normality, variance)
- Consistency validation (correlation preservation)
- Action-specific validators
- Confidence scoring (0.0-1.0)
- Explainable decisions

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/shobith-s/AURORA-V2.git
cd AURORA-V2

# Install dependencies
pip install -r requirements.txt

# Start backend
uvicorn src.api.server:app --reload

# Start frontend (in another terminal)
cd frontend
npm install
npm run dev
```

### Usage

```python
from src.core.preprocessor import IntelligentPreprocessor
import pandas as pd

# Initialize
preprocessor = IntelligentPreprocessor()

# Preprocess a column
df = pd.read_csv('data.csv')
result = preprocessor.preprocess_column(
    column=df['price'],
    column_name='price'
)

print(f"Action: {result.action}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Explanation: {result.explanation}")
```

---

## Documentation

- [Architecture](./ARCHITECTURE.md) - System design and components
- [Neural Oracle Training](./NEURAL_ORACLE.md) - How to train the ML model
- [API Reference](./API.md) - Backend API documentation
- [Development Guide](./DEVELOPMENT.md) - Contributing guidelines

---

## Performance

**Symbolic Engine:**
- Accuracy: 85-95% on common patterns
- Speed: <1ms per column
- Coverage: ~85% of cases

**Neural Oracle (Ensemble):**
- Validation accuracy: 89.4%
- Architecture: XGBoost + LightGBM ensemble
- Inference only: <5ms per column
- Pre-trained on 400+ LLM-validated examples

**Hybrid System:**
- Overall accuracy: ~92%
- Handles both common and edge cases
- Continuous improvement via learning

---

## Tech Stack

**Backend:**
- Python 3.10+
- FastAPI
- XGBoost
- Pandas, NumPy, Scikit-learn

**Frontend:**
- Next.js 14
- TypeScript
- TailwindCSS
- Zustand (state management)

**ML/AI:**
- XGBoost + LightGBM ensemble (neural oracle)
- Groq API (LLM validation)
- SHAP (explainability)

---

## Project Structure

```
AURORA-V2/
├── src/
│   ├── core/           # Core preprocessing logic
│   ├── symbolic/       # Symbolic rule engine
│   ├── neural/         # Neural oracle
│   ├── learning/       # Adaptive learning
│   ├── validation/     # Quality assurance
│   └── api/            # FastAPI backend
├── frontend/           # Next.js UI
├── validator/          # Neural oracle training
├── models/             # Trained models
├── docs/               # Documentation
└── tests/              # Test suite
```

---

## License

MIT License - See [LICENSE](../LICENSE) for details

---

## Contributors

- Shobith S - Creator and maintainer

---

**Version:** 2.0  
**Last Updated:** 2024-11-29  
**Status:** Production Ready ✅
