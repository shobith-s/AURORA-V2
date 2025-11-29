# Cloud Training Guide for Neural Oracle
**Train neural oracle using Google Colab + Gemini API (100% Free)**

---

## Overview

This guide shows you how to train the neural oracle in the cloud using:
- **Google Colab**: Free GPU for training (T4 GPU, 12GB RAM)
- **Gemini API**: Free LLM for validation (better than local models)
- **Total Cost**: $0

**Estimated Time**: 3-4 hours (mostly automated)

---

## Prerequisites

### Required Accounts (All Free)
1. **Google Account** (for Colab and Gemini)
2. **GitHub Account** (optional, for code upload)

### What You'll Get
- Free T4 GPU (12GB VRAM)
- 12GB RAM
- 100GB disk space
- Gemini 1.5 Flash API (15 requests/min, free)

---

## Step 1: Get Gemini API Key (5 minutes)

### 1.1 Create API Key
```
1. Go to: https://aistudio.google.com/app/apikey
2. Click "Get API Key"
3. Click "Create API Key in new project"
4. Copy the API key (starts with "AIza...")
5. Save it securely
```

### 1.2 Test API Key
```python
# Test in browser console or Python
import google.generativeai as genai

genai.configure(api_key='YOUR_API_KEY')
model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content("Hello!")
print(response.text)  # Should respond with greeting
```

**Free Tier Limits:**
- 15 requests per minute
- 1500 requests per day
- More than enough for our use case!

---

## Step 2: Setup Google Colab (10 minutes)

### 2.1 Create New Notebook
```
1. Go to: https://colab.research.google.com
2. Click "New Notebook"
3. Rename: "AURORA_Neural_Oracle_Training"
```

### 2.2 Enable GPU
```
1. Click "Runtime" menu
2. Select "Change runtime type"
3. Hardware accelerator: "T4 GPU"
4. Click "Save"
```

### 2.3 Verify GPU
```python
# Run this in first cell
!nvidia-smi

# Should show:
# Tesla T4
# 12GB memory
```

---

## Step 3: Upload Code to Colab (15 minutes)

### Option A: Upload Validator Folder Directly

```python
# Cell 1: Upload files
from google.colab import files
import zipfile
import os

# Create validator directory
!mkdir -p validator

# Upload validator.zip (zip your validator folder first)
uploaded = files.upload()

# Extract
!unzip validator.zip -d .
```

### Option B: Clone from GitHub (Recommended)

```python
# Cell 1: Clone repository
!git clone https://github.com/YOUR_USERNAME/AURORA-V2.git
%cd AURORA-V2/validator

# Verify structure
!ls -la
# Should show: scripts/, utils/, config.yaml, etc.
```

### Option C: Manual File Upload

```python
# Upload each file individually
from google.colab import files

# Create structure
!mkdir -p validator/scripts
!mkdir -p validator/utils
!mkdir -p validator/data
!mkdir -p validator/labels
!mkdir -p validator/validated
!mkdir -p validator/models

# Upload files one by one
# Then move to correct directories
```

---

## Step 4: Install Dependencies (5 minutes)

```python
# Cell 2: Install packages
!pip install -q xgboost pandas numpy scikit-learn pyyaml tqdm google-generativeai

# Verify installations
import xgboost as xgb
import google.generativeai as genai
print(f"✅ XGBoost: {xgb.__version__}")
print(f"✅ Gemini API: Ready")
```

---

## Step 5: Configure Gemini API (2 minutes)

```python
# Cell 3: Setup API key
import os

# Set your API key
GEMINI_API_KEY = "AIza..."  # Replace with your key

# Configure
import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)

# Test connection
model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content("Are you working?")
print(f"✅ Gemini response: {response.text}")
```

---

## Step 6: Download Datasets (30 minutes)

```python
# Cell 4: Download datasets
%cd /content/AURORA-V2/validator

# Run download script
!python scripts/download_datasets.py

# Expected output:
# ✅ Iris: 150 rows, 5 columns
# ✅ Wine: 178 rows, 14 columns
# ✅ Breast Cancer: 569 rows, 31 columns
# ✅ Diabetes: 442 rows, 11 columns
# ✅ California Housing: 20640 rows, 9 columns
# ✅ E-commerce: 1000 rows, 8 columns
# ✅ Healthcare: 500 rows, 7 columns
# ✅ Finance: 800 rows, 6 columns
# TOTAL: 8 datasets, ~60 columns
```

---

## Step 7: Generate Symbolic Labels (30 minutes)

```python
# Cell 5: Generate labels with symbolic engine
!python scripts/generate_symbolic_labels.py

# Expected output:
# Processing iris...
#   ✅ 5 columns labeled
# Processing wine...
#   ✅ 14 columns labeled
# ...
# TOTAL: ~60 columns labeled
# High confidence (>0.90): ~35 columns
# Medium confidence (0.70-0.90): ~20 columns
# Low confidence (<0.70): ~5 columns
```

---

## Step 8: LLM Validation (1-2 hours)

```python
# Cell 6: Validate with Gemini
!python scripts/llm_validator.py \
    --api-key $GEMINI_API_KEY \
    --validate-medium \
    --validate-low \
    --batch-size 10

# Expected output:
# Validating medium-confidence cases...
#   [1/20] Year-Of-Publication: ❌ Corrected (drop → keep_as_is)
#   [2/20] Publisher: ✅ Confirmed (hash_encode)
#   ...
# Validating low-confidence cases...
#   [1/5] ISBN: ✅ Confirmed (keep_as_is)
#   ...
# 
# VALIDATION COMPLETE:
# - High confidence: 35 (trusted)
# - Medium validated: 20 (3 corrected)
# - Low decided: 5 (1 corrected)
# TOTAL: 60 examples, 4 LLM corrections
```

**Note**: This takes 1-2 hours due to Gemini rate limits (15 req/min)

---

## Step 9: Train Neural Oracle (30 minutes)

```python
# Cell 7: Train model
!python scripts/train_neural_oracle.py

# Expected output:
# Loading validated labels...
# ✅ 60 training examples loaded
# 
# Train/Val split:
#   Train: 48 examples (80%)
#   Val: 12 examples (20%)
# 
# Training XGBoost...
# [0]  train-mlogloss:1.45  val-mlogloss:1.62
# [10] train-mlogloss:0.68  val-mlogloss:0.85
# [20] train-mlogloss:0.42  val-mlogloss:0.71
# [30] train-mlogloss:0.28  val-mlogloss:0.68
# [40] train-mlogloss:0.19  val-mlogloss:0.66
# [49] train-mlogloss:0.14  val-mlogloss:0.65
# 
# TRAINING COMPLETE:
# Train Accuracy: 92.5%
# Val Accuracy: 75.0%
# 
# ✅ Model saved: models/neural_oracle_v2_20241129.pkl
```

---

## Step 10: Test & Validate (30 minutes)

```python
# Cell 8: Comprehensive testing
!python scripts/test_comprehensive.py

# Expected output:
# Testing on Books.csv...
#   Symbolic accuracy: 85%
#   Neural accuracy: 78%
#   Hybrid accuracy: 87%
# 
# Testing on edge cases (symbolic conf < 0.70)...
#   Symbolic accuracy: 60%
#   Neural accuracy: 75%
#   ✅ Neural HELPS on edge cases!
# 
# FINAL VERDICT:
# ✅ Neural oracle improves edge case handling by +15%
# ✅ Hybrid (symbolic + neural) achieves 87% overall
# 
# RECOMMENDATION: KEEP neural oracle
```

---

## Step 11: Download Trained Model (5 minutes)

```python
# Cell 9: Download model to local machine
from google.colab import files

# Download trained model
files.download('models/neural_oracle_v2_20241129.pkl')

# Download training history
files.download('models/training_history.json')

# Download validation report
files.download('reports/validation_report.html')
```

---

## Step 12: Deploy to Production (10 minutes)

### 12.1 Copy Model to AURORA

```bash
# On your local machine
# Copy downloaded model to AURORA models directory
cp ~/Downloads/neural_oracle_v2_20241129.pkl \
   C:/Users/shobi/Desktop/AURORA/AURORA-V2/models/
```

### 12.2 Update Model Path

```python
# Edit: src/core/preprocessor.py
# Update model path to new model
neural_model_path = Path('models/neural_oracle_v2_20241129.pkl')
```

### 12.3 Restart Backend

```bash
# Stop current server (Ctrl+C)
# Restart
uvicorn src.api.server:app --reload
```

### 12.4 Test in UI

```
1. Upload Books.csv
2. Check decisions
3. Verify neural oracle is being used
4. Check accuracy improved
```

---

## Complete Colab Notebook Template

```python
# ============================================
# AURORA Neural Oracle Training - Google Colab
# ============================================

# Cell 1: Setup
!nvidia-smi  # Verify GPU
!git clone https://github.com/YOUR_USERNAME/AURORA-V2.git
%cd AURORA-V2/validator

# Cell 2: Install dependencies
!pip install -q xgboost pandas numpy scikit-learn pyyaml tqdm google-generativeai

# Cell 3: Configure Gemini
import google.generativeai as genai
GEMINI_API_KEY = "YOUR_KEY_HERE"
genai.configure(api_key=GEMINI_API_KEY)

# Cell 4: Download datasets
!python scripts/download_datasets.py

# Cell 5: Generate symbolic labels
!python scripts/generate_symbolic_labels.py

# Cell 6: LLM validation (1-2 hours)
!python scripts/llm_validator.py --api-key $GEMINI_API_KEY

# Cell 7: Train neural oracle
!python scripts/train_neural_oracle.py

# Cell 8: Test & validate
!python scripts/test_comprehensive.py

# Cell 9: Download model
from google.colab import files
files.download('models/neural_oracle_v2_*.pkl')
```

---

## Troubleshooting

### Issue: "GPU not available"
```python
# Check GPU status
!nvidia-smi

# If not available:
# 1. Runtime → Change runtime type → T4 GPU
# 2. Runtime → Restart runtime
```

### Issue: "Gemini API quota exceeded"
```python
# Check quota
# Free tier: 15 req/min, 1500 req/day

# Solution: Add delays
!python scripts/llm_validator.py --batch-size 5 --delay 5
```

### Issue: "Session disconnected"
```python
# Colab sessions timeout after 12 hours
# Save progress regularly:
!zip -r checkpoint.zip models/ labels/ validated/

# Download checkpoint
from google.colab import files
files.download('checkpoint.zip')

# Resume later: Upload checkpoint.zip and extract
```

### Issue: "Out of memory"
```python
# Reduce batch size
!python scripts/llm_validator.py --batch-size 5

# Or use smaller datasets
!python scripts/download_datasets.py --count 5
```

---

## Cost Breakdown

| Resource | Cost | Notes |
|----------|------|-------|
| Google Colab GPU | $0 | Free T4 GPU |
| Gemini API | $0 | Free tier (1500 req/day) |
| Storage | $0 | 100GB free in Colab |
| **TOTAL** | **$0** | Completely free! |

---

## Expected Results

### Training Metrics
- **Training examples**: 50-70
- **Train accuracy**: 85-95%
- **Validation accuracy**: 70-80%
- **Training time**: 3-4 hours

### Quality Metrics
- **Overall accuracy**: 75-85%
- **Edge case accuracy**: 70-80%
- **Improvement over symbolic**: +5-10%

### Decision Criteria
```
IF neural_accuracy >= 70% AND neural_helps_edge_cases:
    KEEP neural oracle ✅
ELSE:
    REMOVE neural oracle ❌
```

---

## Next Steps After Training

1. **Download model** from Colab
2. **Copy to AURORA** models directory
3. **Update model path** in code
4. **Restart backend**
5. **Test in UI** with Books.csv
6. **Monitor performance** with user corrections
7. **Retrain periodically** (every 3-6 months)

---

## FAQ

**Q: How long does training take?**
A: 3-4 hours total (mostly LLM validation due to rate limits)

**Q: Can I close my browser?**
A: No, Colab requires browser open. Use "Keep Colab Alive" extension.

**Q: What if I run out of GPU time?**
A: Free tier: 12 hours/session. Just restart and resume.

**Q: Is Gemini better than local Qwen?**
A: Yes, Gemini 1.5 Flash is more accurate than Qwen 2.5 7B.

**Q: Can I use GPT-4 instead?**
A: Yes, but costs ~$1-2. Gemini is free and good enough.

**Q: How often should I retrain?**
A: Every 3-6 months, or after 100+ user corrections.

---

## Support

**Issues?**
1. Check Colab runtime is active
2. Verify GPU is enabled
3. Check Gemini API key is valid
4. Review logs in Colab output

**Still stuck?**
- Check GitHub issues
- Review error messages carefully
- Try restarting runtime

---

**Last Updated**: 2024-11-29
**Version**: 1.0
**Status**: Production Ready
