# AURORA V2 - Intelligent Data Preprocessing System

**AI-powered data preprocessing with symbolic rules, neural oracle, and adaptive learning**

---

## Overview

AURORA V2 is an intelligent data preprocessing system that combines:
- **Symbolic Engine**: 185+ expert-crafted rules for common preprocessing patterns
- **Neural Oracle**: ML model (75.9% accuracy) for handling edge cases
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
│  │  (185 rules) │    │  (75.9% acc) │    │   (User      │ │
│  │              │    │              │    │  Corrections)│ │
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

**Layer 2: Neural Oracle (Edge Cases)**
- XGBoost model trained on 149 validated examples
- 75.9% validation accuracy
- Handles ambiguous cases where symbolic is uncertain
- Uses 10 engineered features

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

**Neural Oracle:**
- Validation accuracy: 75.9%
- Helps with edge cases (symbolic confidence <0.7)
- Trained on 149 LLM-validated examples

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
- XGBoost (neural oracle)
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
