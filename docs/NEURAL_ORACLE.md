# Neural Oracle Training Guide

## Overview

The neural oracle is an XGBoost model that handles edge cases where the symbolic engine is uncertain (confidence <0.7). It achieves **75.9% validation accuracy** on real-world data.

---

## Training Pipeline

```
Datasets → Symbolic Labels → LLM Validation → XGBoost Training → Deployment
```

### **Step 1: Download Datasets**

```bash
cd validator
python scripts/download_datasets.py
```

**Output:** 18 diverse datasets in `validator/data/`

### **Step 2: Generate Symbolic Labels**

```bash
python scripts/generate_symbolic_labels.py
```

**Output:** Labels categorized by confidence:
- `high_confidence.json` (≥0.90) - Trusted
- `medium_confidence.json` (0.70-0.90) - Needs validation
- `low_confidence.json` (<0.70) - LLM decides

### **Step 3: LLM Validation**

```bash
# Get Groq API key from https://console.groq.com
python scripts/llm_validator.py --mode groq --api-key YOUR_KEY
```

**What it does:**
- Trusts high-confidence labels (no validation needed)
- Validates medium-confidence with LLM
- Only accepts corrections with confidence ≥0.85
- Outputs: `validator/validated/validated_labels.json`

**Expected:** ~7 high-confidence corrections out of 150 labels

### **Step 4: Train XGBoost**

```bash
python scripts/train_neural_oracle.py
```

**Output:**
- Model: `validator/models/neural_oracle_v2_TIMESTAMP.pkl`
- Training history: `validator/models/training_history.json`

**Expected results:**
- Train accuracy: 90-97%
- Validation accuracy: 70-80%

### **Step 5: Deploy**

```bash
# Copy model to production
cp validator/models/neural_oracle_v2_*.pkl models/

# Restart backend
uvicorn src.api.server:app --reload
```

---

## Training in Google Colab (Recommended)

**Why Colab:**
- Free GPU
- No local setup
- Faster training

**Setup:**

```python
# 1. Clone repo
!git clone -b polishing https://github.com/shobith-s/AURORA-V2.git
%cd AURORA-V2/validator

# 2. Install dependencies
!pip install -q xgboost pandas numpy scikit-learn pyyaml tqdm groq

# 3. Set API key
GROQ_API_KEY = "gsk_..."  # Your Groq key

# 4. Run pipeline
!python scripts/download_datasets.py
!python scripts/generate_symbolic_labels.py
!python scripts/llm_validator.py --mode groq --api-key $GROQ_API_KEY
!python scripts/train_neural_oracle.py

# 5. Download model
from google.colab import files
import glob
model_file = glob.glob('models/neural_oracle_v2_*.pkl')[0]
files.download(model_file)
```

**Timeline:** ~2 hours total

---

## Model Details

### Architecture

**Algorithm:** XGBoost Classifier
- 50 boosters
- Max depth: 6
- Learning rate: 0.1
- Subsample: 0.8

### Features (10)

1. `null_percentage` - % of null values
2. `unique_ratio` - Unique values / total rows
3. `skewness` - Distribution skewness
4. `outlier_percentage` - % of outliers (IQR method)
5. `entropy` - Information entropy
6. `pattern_complexity` - Regex pattern complexity
7. `multimodality_score` - Distribution modes
8. `cardinality_bucket` - Categorical cardinality
9. `detected_dtype` - Inferred data type
10. `column_name_signal` - Name-based hints

### Training Data

**Sources:**
- 18 diverse datasets (e-commerce, healthcare, finance, etc.)
- 149 validated examples
- 7 LLM corrections (high confidence only)

**Distribution:**
- `keep_as_is`: ~20%
- `standard_scale`: ~30%
- `robust_scale`: ~25%
- `log_transform`: ~15%
- `onehot_encode`: ~10%

---

## LLM Validation

### Why LLM?

**Problem:** Symbolic engine can make mistakes
**Solution:** Use LLM to validate uncertain decisions

### Validation Process

```python
For each medium-confidence label:
    1. Send to LLM with column info + symbolic decision
    2. LLM evaluates: "Is this correct?"
    3. If LLM confidence ≥0.85: Accept correction
    4. Else: Keep original symbolic decision
```

### LLM Options

| Provider | Model | Speed | Cost | Rate Limit |
|----------|-------|-------|------|------------|
| **Groq** | Llama 3.3 70B | **Fast** | Free | 14,400/day |
| Hugging Face | Qwen 2.5 14B | Slow | Free | 1,000/day |
| Gemini | 1.5 Flash | Medium | Free* | 15/min |

**Recommended:** Groq (fastest + free)

---

## Troubleshooting

### Low Validation Accuracy (<70%)

**Causes:**
- Too many low-confidence corrections
- Noisy training data
- Class imbalance

**Solutions:**
1. Increase LLM confidence threshold (0.85 → 0.90)
2. Use fewer datasets (quality > quantity)
3. Check action distribution

### Overfitting (High train, low val)

**Symptoms:**
- Train accuracy >95%
- Val accuracy <70%

**Solutions:**
1. More training data
2. Reduce model complexity (fewer boosters)
3. Increase regularization

### Rate Limits

**Groq:**
- 14,400 requests/day
- Use friend's API key if needed

**Hugging Face:**
- 1,000 requests/day
- Switch to Groq

---

## Performance Benchmarks

**Current Model (75.9% val accuracy):**
- Training time: 30 seconds
- Inference: ~5ms per column
- Model size: 269 KB

**Comparison:**
- Symbolic only: 85% accuracy (fast)
- Neural only: 76% accuracy (slow)
- **Hybrid: ~92% accuracy** (best)

---

## Future Improvements

1. **More data:** 500+ examples → 80-85% accuracy
2. **Better features:** Add domain-specific features
3. **Ensemble:** Combine multiple models
4. **Active learning:** Prioritize uncertain cases
5. **Online learning:** Update model with user corrections

---

**Version:** 2.0  
**Last Updated:** 2024-11-29  
**Model:** `neural_oracle_v2_20251129_125158.pkl`  
**Accuracy:** 75.9% validation
