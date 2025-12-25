# AURORA V2 - Intelligent Data Preprocessing System

**AI-powered data preprocessing with symbolic rules, neural oracle, and adaptive learning**

[![Production Ready](https://img.shields.io/badge/status-production--ready-brightgreen)]()
[![Score](https://img.shields.io/badge/evaluation-83.2%2F100-blue)]()
[![Datasets](https://img.shields.io/badge/datasets-10%20real--world-orange)]()

---

## 🎯 Overview

AURORA V2 is an intelligent data preprocessing system that achieved **83.2/100 (B+ Good)** on comprehensive real-world evaluation across 10 diverse datasets.

### Core Components

- **Symbolic Engine**: 230+ expert-crafted rules for intelligent preprocessing
- **Neural Oracle**: Pre-trained ensemble (XGBoost + LightGBM) for edge cases
- **Adaptive Learning**: Learns from user corrections to improve over time
- **Pipeline Export**: Generate standalone preprocessing code
- **Visual Profiling**: Interactive column analysis and visualization

---

## 🏆 Evaluation Results

**Comprehensive 10-Dataset Evaluation:**
- **Overall Score:** 83.2/100 (B+ Good)
- **Datasets:** Titanic, Iris, Wine, Breast Cancer, Adult Income, Diabetes, Digits, California Housing, Credit Card, Housing Prices
- **Total Columns:** 133 analyzed
- **Total Rows:** 57,728 processed
- **Expert Alignment:** 85% match with expert decisions

**Top Performing Datasets:**
1. Digits (UCI): 90.0/100 ⭐
2. Adult Income (UCI): 89.7/100 ⭐
3. Titanic (Kaggle): 88.8/100 ⭐

See [`evaluation/FINAL_COMPREHENSIVE_REPORT.md`](evaluation/FINAL_COMPREHENSIVE_REPORT.md) for detailed analysis.

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/shobith-s/AURORA-V2.git
cd AURORA-V2

# Install dependencies
pip install -r requirements.txt

# Start backend
python -m uvicorn src.api.server:app --reload

# Start frontend (in another terminal)
cd frontend
npm install
npm run dev
```

### Basic Usage

```python
from src.core.preprocessor import IntelligentPreprocessor
import pandas as pd

# Initialize preprocessor
preprocessor = IntelligentPreprocessor(
    use_neural_oracle=False,  # Symbolic engine only
    enable_learning=True       # Enable adaptive learning
)

# Preprocess a column
df = pd.read_csv('your_data.csv')
result = preprocessor.preprocess_column(
    df['column_name'],
    column_name='column_name',
    apply_action=True
)

print(f"Action: {result.action}")
print(f"Confidence: {result.confidence}")
print(f"Explanation: {result.explanation}")
```

---

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AURORA V2 System                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   Symbolic   │───▶│    Neural    │───▶│   Adaptive   │ │
│  │    Engine    │    │    Oracle    │    │   Learning   │ │
│  │  (230 rules) │    │   Ensemble   │    │   (User      │ │
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

## ✨ Key Features

### 1. **Intelligent Preprocessing**

- **Type Detection**: Automatically identifies numeric, categorical, text, temporal data
- **Distribution-Aware**: Applies appropriate transformations based on data distribution
- **Domain Intelligence**: Recognizes domain patterns (IDs, emails, phone numbers, etc.)
- **High Confidence**: Average 0.81 confidence across decisions

### 2. **Pipeline Export**

Export preprocessing pipelines as standalone Python code:

```python
from src.core.pipeline_exporter import PipelineExporter

exporter = PipelineExporter()
code = exporter.export_pipeline(
    preprocessing_results=results,
    dataset_name='my_dataset'
)

# Save to file
with open('preprocessing_pipeline.py', 'w') as f:
    f.write(code)
```

### 3. **Visual Profiling**

Interactive column analysis with statistical insights:

```python
from src.core.column_profiler import ColumnProfiler

profiler = ColumnProfiler()
profile = profiler.profile_column(df['column_name'], 'column_name')

# Returns: distribution, outliers, missing values, statistics
```

### 4. **Adaptive Learning**

System learns from user corrections:

```python
# Submit correction
preprocessor.submit_correction(
    column_data=df['column'],
    column_name='column',
    wrong_action='standard_scale',
    correct_action='log_transform',
    confidence=0.75
)

# System learns and applies to similar columns
```

---

## 📁 Project Structure

```
AURORA-V2/
├── src/
│   ├── core/              # Core preprocessing logic
│   │   ├── preprocessor.py
│   │   ├── pipeline_exporter.py
│   │   └── column_profiler.py
│   ├── symbolic/          # Symbolic engine
│   │   ├── engine.py
│   │   └── rules.py       # 230+ preprocessing rules
│   ├── neural/            # Neural oracle
│   ├── learning/          # Adaptive learning
│   ├── api/               # FastAPI backend
│   └── explanation/       # Explanation generation
├── frontend/              # React frontend
├── evaluation/            # Evaluation scripts and reports
│   ├── FINAL_COMPREHENSIVE_REPORT.md
│   ├── real_world_evaluation.py
│   └── comprehensive_10dataset_results.json
├── scripts/               # Training scripts
├── docs/                  # Documentation
├── tests/                 # Test suite
└── colab/                 # Jupyter notebooks
```

---

## 📚 Documentation

- **[API Documentation](docs/API.md)** - REST API endpoints
- **[Architecture](docs/ARCHITECTURE.md)** - System architecture
- **[Meta Learning Guide](docs/META_LEARNING_GUIDE.md)** - Training neural oracle
- **[Neural Oracle](docs/NEURAL_ORACLE.md)** - Neural oracle details
- **[Smart Preprocessing Guide](docs/SMART_PREPROCESSING_GUIDE.md)** - Usage guide
- **[Transformation Decisions](docs/TRANSFORMATION_DECISIONS.md)** - Decision logic
- **[Universal Preprocessing Vision](docs/UNIVERSAL_PREPROCESSING_VISION.md)** - Project vision

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_symbolic_engine.py

# Run evaluation
python evaluation/real_world_evaluation.py
```

---

## 🎯 Performance Metrics

| Metric | Value | Status |
|--------|-------|--------|
| **Overall Score** | 83.2/100 | ✅ B+ Good |
| **Expert Alignment** | 85% | ✅ High |
| **Average Confidence** | 0.81 | ✅ High |
| **Inference Time** | <5ms | ✅ Fast |
| **High Confidence Decisions** | 42% | ✅ Good |

---

## 🔧 Configuration

Configuration files in `configs/`:
- `preprocessing_config.yaml` - Preprocessing settings
- `neural_oracle_config.yaml` - Neural oracle settings
- `learning_config.yaml` - Adaptive learning settings

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Datasets:** Kaggle, UCI Machine Learning Repository
- **Libraries:** scikit-learn, pandas, numpy, FastAPI, React
- **Evaluation:** 10 real-world datasets, 133 columns, 57K+ rows

---

## 📧 Contact

**Shobith S** - shobi7196@gmail.com

**Project Link:** https://github.com/shobith-s/AURORA-V2

---

## 🎓 Citation

If you use AURORA V2 in your research, please cite:

```bibtex
@software{aurora_v2_2025,
  author = {Shobith S},
  title = {AURORA V2: Intelligent Data Preprocessing System},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/shobith-s/AURORA-V2}
}
```

---

**Status:** Production-Ready ✅  
**Last Updated:** December 2025  
**Version:** 2.0
