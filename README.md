# AURORA v2 - Intelligent Data Preprocessing System

> **Latest Update (v2.1.0):** Neural Oracle integration, Light Theme UI, Enhanced Type Detection

## Overview

AURORA is an intelligent data preprocessing system that combines symbolic rules, meta-learning, and neural networks to automatically recommend optimal preprocessing strategies for your data.

## 🚀 New Features (v2.1.0)

### 1. Neural Oracle
- **Real-world trained ML model** on 6 diverse datasets
- **<0.5ms inference time** for instant recommendations
- **Handles ambiguous cases** that symbolic rules miss
- **Confidence scores** for every decision

### 2. Light Theme UI
- Modern, clean interface optimized for readability
- Consistent design system across all components
- Better visibility in well-lit environments

### 3. Enhanced Type Detection
- **Intelligent type inference** from JSON data
- Correctly identifies numeric vs categorical columns
- Proper health metrics for all data types

### 4. Explanation System
- **Detailed markdown reports** for every decision
- Shows confidence, source, and reasoning
- Alternative approaches with trade-offs
- Metadata insights (skewness, outliers, missing values)

## 🎯 Key Features

- **185+ Symbolic Rules** - Expert knowledge encoded
- **Neural Oracle** - ML-powered decision making
- **Meta-Learning** - Statistical heuristics for edge cases
- **Adaptive Learning** - Learns from user corrections
- **Real-time Processing** - Instant recommendations
- **Explainable AI** - Transparent decision process

## 📊 Architecture

```
┌─────────────────────────────────────────┐
│         User Upload CSV                 │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│     Intelligent Preprocessor            │
│  ┌────────────────────────────────────┐ │
│  │  1. Symbolic Engine (185+ rules)   │ │
│  │     ↓ (if confidence < 0.75)       │ │
│  │  2. Meta-Learning (heuristics)     │ │
│  │     ↓ (if still uncertain)         │ │
│  │  3. Neural Oracle (XGBoost)        │ │
│  │     ↓ (if all fail)                │ │
│  │  4. Conservative Fallback          │ │
│  └────────────────────────────────────┘ │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   Preprocessing Recommendations         │
│   + Explanations + Confidence Scores    │
└─────────────────────────────────────────┘
```

## 🛠️ Installation

### Prerequisites
```bash
# Python 3.8+
# Node.js 14+
```

### Backend Setup
```bash
cd AURORA-V2
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install xgboost scikit-learn shap numba
```

### Frontend Setup
```bash
cd frontend
npm install
npm run build
```

### Train Neural Oracle (Optional)
```bash
# Train on real-world datasets
python scripts/train_realworld.py

# Or use synthetic data
python scripts/train_neural_oracle.py
```

## 🚀 Quick Start

### Start Backend
```bash
uvicorn src.api.server:app --reload
```

### Start Frontend
```bash
cd frontend
npm start
```

### Access Application
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 📖 Usage

1. **Upload CSV** - Drag and drop or click to upload
2. **Analyze** - Click "Analyze" to get recommendations
3. **Review** - Check data health metrics and preprocessing suggestions
4. **Explain** - Click "Explain" on any column for detailed reasoning
5. **Override** - Manually adjust recommendations if needed
6. **Export** - Download preprocessed data or Python script

## 🧠 How It Works

### Decision Process

1. **Symbolic Engine** (Primary)
   - 185+ hand-crafted rules
   - Pattern matching on data characteristics
   - High confidence for known patterns

2. **Meta-Learning** (Bridge)
   - Statistical heuristics
   - Covers edge cases
   - Universal coverage

3. **Neural Oracle** (ML)
   - XGBoost model trained on real data
   - Handles ambiguous cases
   - Provides confidence scores

4. **Conservative Fallback** (Safety)
   - Ultra-safe defaults
   - Preserves data integrity
   - Reversible operations

### Type Detection

```python
# Intelligent type inference
if column.dtype == 'object':
    numeric_column = pd.to_numeric(column, errors='coerce')
    if numeric_column.notna().sum() / len(column) > 0.5:
        # Treat as numeric
        return "numeric"
    else:
        # Treat as categorical
        return "categorical"
```

## 📊 Supported Preprocessing Actions

### Data Quality
- Drop column, Remove duplicates, Fill nulls (mean/median/mode)

### Type Conversion
- Parse datetime, Parse numeric, Parse boolean, Parse categorical

### Scaling
- Standard scale, MinMax scale, Robust scale, MaxAbs scale

### Transformation
- Log transform, Box-Cox, Yeo-Johnson, Quantile transform

### Encoding
- One-hot encode, Label encode, Target encode, Hash encode

### Outlier Handling
- Clip outliers, Winsorize, Remove outliers

## 🔧 Configuration

### Confidence Threshold
```python
# In src/api/server.py
preprocessor = get_preprocessor(
    confidence_threshold=0.75,  # Adjust for more/less neural participation
    use_neural_oracle=True,
    enable_learning=True
)
```

### Neural Oracle Training
```python
# In scripts/train_realworld.py
# Add your own datasets
datasets.append(("MyDataset", pd.read_csv("my_data.csv")))
```

## 📈 Performance

- **Symbolic Engine:** <1ms per column
- **Neural Oracle:** <0.5ms per column
- **Total Pipeline:** <5ms per column
- **Batch Processing:** ~100 columns/second

## 🧪 Testing

### Run Diagnostic Check
```bash
python diagnostic_check.py
```

### Test Neural Oracle
```bash
python -c "from src.neural.oracle import get_neural_oracle; oracle = get_neural_oracle(); print('✓ Neural oracle loaded')"
```

## 📝 API Documentation

### Preprocess Endpoint
```bash
POST /api/preprocess
Content-Type: application/json

{
  "column_data": [1, 2, 3, 4, 5],
  "column_name": "age",
  "target_available": false
}
```

### Explanation Endpoint
```bash
POST /api/explain/enhanced
Content-Type: application/json

{
  "column_data": [1, 2, 3, 4, 5],
  "column_name": "age"
}
```

## 🐛 Troubleshooting

### Neural Oracle Not Participating
```bash
# Check if model exists
ls models/neural_oracle_v1.pkl

# Check server logs for errors
# Look for: "Neural oracle loaded successfully"

# Restart server
uvicorn src.api.server:app --reload
```

### Explanation Modal Empty
```bash
# Hard refresh browser
Ctrl + Shift + R

# Check backend endpoint
curl -X POST http://localhost:8000/api/explain/enhanced \
  -H "Content-Type: application/json" \
  -d '{"column_data": [1,2,3], "column_name": "test"}'
```

### Type Detection Issues
```bash
# Check server logs for type inference messages
# Should see: "Type inference: 'column_name' converted from object to numeric"
```

## 📚 Documentation

- [CHANGELOG.md](./CHANGELOG.md) - Recent updates and changes
- [API Documentation](http://localhost:8000/docs) - Interactive API docs
- [Architecture Guide](./docs/architecture.md) - System design
- [Training Guide](./docs/training.md) - Neural oracle training

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Built with FastAPI, Next.js, XGBoost
- Inspired by AutoML and data preprocessing best practices
- Trained on publicly available datasets

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Run diagnostic tools

---

**Version:** 2.1.0  
**Last Updated:** November 24, 2024  
**Status:** Production Ready ✅
