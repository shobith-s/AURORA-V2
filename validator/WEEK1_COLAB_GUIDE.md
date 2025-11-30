# Week 1: Improve Neural Oracle (Colab Guide)

**Goal:** Improve neural oracle from 75.9% to 80%+ accuracy

**What we're doing:**
1. Expand datasets (18 → 40)
2. Add features (10 → 20)
3. Simple ensemble (XGBoost + LightGBM)
4. Retrain & measure

**Time:** ~2 hours

---

## Setup (5 minutes)

```python
# 1. Clone repo
!git clone -b polishing https://github.com/shobith-s/AURORA-V2.git
%cd AURORA-V2/validator

# 2. Install dependencies
!pip install -q xgboost lightgbm pandas numpy scikit-learn pyyaml tqdm groq

# 3. Set API key
GROQ_API_KEY = "gsk_..."  # Your Groq key from console.groq.com
```

---

## Step 1: Expand Datasets (30 minutes)

```python
# Generate 40 datasets (18 existing + 22 new)
print("📊 Generating 40 diverse datasets...")
!python scripts/expand_datasets.py

# Expected output:
# ✅ Loaded 18 existing datasets
# ✅ Created 22 new datasets
# ✅ TOTAL: 40 datasets
```

---

## Step 2: Generate Labels (30 minutes)

```python
# Generate symbolic labels for all 40 datasets
print("🏷️ Generating symbolic labels...")
!python scripts/generate_symbolic_labels.py

# Expected: ~500-600 labels (vs 149)
```

---

## Step 3: LLM Validation (20 minutes)

```python
# Validate with Groq (FREE, fast)
print("🤖 Validating with Groq LLM...")
!python scripts/llm_validator.py --mode groq --api-key $GROQ_API_KEY

# Expected: 
# - High confidence: ~450 labels (trusted)
# - Medium confidence: ~100 labels (validated)
# - Corrections: ~10-20 (confidence ≥0.85)
```

---

## Step 4: Train Improved Model (10 minutes)

```python
# Train with improvements:
# - More data (500+ examples vs 149)
# - Better features (20 vs 10)
# - Ensemble (XGBoost + LightGBM)

print("🎯 Training improved neural oracle...")
!python scripts/train_neural_oracle_v2.py

# Expected results:
# - Train accuracy: 92-95%
# - Val accuracy: 80-85% (vs 75.9%)
# - Model size: ~600KB
```

**Wait, we need to create the improved training script first!**

---

## Create Improved Training Script

```python
# Create train_neural_oracle_v2.py with improvements
%%writefile scripts/train_neural_oracle_v2.py
"""
Improved Neural Oracle Training (Week 1)
- More data (40 datasets)
- Better features (20 vs 10)
- Ensemble (XGBoost + LightGBM)
"""
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

print("="*70)
print("IMPROVED NEURAL ORACLE TRAINING (Week 1)")
print("="*70)

# Load validated labels
labels_file = Path('validated/validated_labels.json')
with open(labels_file, 'r') as f:
    labels = json.load(f)

print(f"\n✅ Loaded {len(labels)} validated labels")

# Extract features (20 features vs 10)
def extract_features_v2(label):
    """Extract 20 features (10 original + 10 new)"""
    features = label['features']
    
    return {
        # Original 10 features
        'null_percentage': features.get('null_pct', 0),
        'unique_ratio': features.get('unique_ratio', 0),
        'skewness': features.get('skewness', 0),
        'outlier_percentage': features.get('outlier_pct', 0),
        'entropy': features.get('entropy', 0),
        'pattern_complexity': features.get('pattern_complexity', 0),
        'multimodality_score': features.get('multimodality', 0),
        'cardinality_bucket': features.get('cardinality_bucket', 0),
        'detected_dtype': features.get('dtype_numeric', 0),
        'column_name_signal': features.get('name_signal', 0),
        
        # NEW: 10 additional features
        'kurtosis': features.get('kurtosis', 0),
        'coefficient_of_variation': features.get('cv', 0),
        'zero_ratio': features.get('zero_ratio', 0),
        'has_negative': features.get('has_negative', 0),
        'has_decimal': features.get('has_decimal', 0),
        'name_contains_id': 1 if 'id' in label.get('column', '').lower() else 0,
        'name_contains_date': 1 if any(x in label.get('column', '').lower() for x in ['date', 'time', 'year']) else 0,
        'name_contains_price': 1 if any(x in label.get('column', '').lower() for x in ['price', 'cost', 'amount']) else 0,
        'range_ratio': features.get('range_ratio', 0),
        'iqr_ratio': features.get('iqr_ratio', 0),
    }

# Prepare data
X = []
y = []

for label in labels:
    features_dict = extract_features_v2(label)
    X.append(list(features_dict.values()))
    y.append(label['action'])

X = np.array(X)
y = np.array(y)

print(f"\n📊 Dataset shape: {X.shape}")
print(f"   Features: {X.shape[1]} (vs 10 original)")
print(f"   Examples: {X.shape[0]} (vs 149 original)")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✂️ Split:")
print(f"   Train: {len(X_train)} examples")
print(f"   Test: {len(X_test)} examples")

# Train ensemble (XGBoost + LightGBM)
print(f"\n🎯 Training ensemble...")

# Model 1: XGBoost
xgb = XGBClassifier(
    n_estimators=50,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)

# Model 2: LightGBM
lgb = LGBMClassifier(
    n_estimators=50,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42,
    verbose=-1
)

# Ensemble (voting)
ensemble = VotingClassifier(
    estimators=[('xgb', xgb), ('lgb', lgb)],
    voting='soft'
)

ensemble.fit(X_train, y_train)

# Evaluate
train_pred = ensemble.predict(X_train)
test_pred = ensemble.predict(X_test)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print(f"\n{'='*70}")
print("RESULTS")
print(f"{'='*70}")
print(f"Train accuracy: {train_acc:.1%}")
print(f"Test accuracy:  {test_acc:.1%}")
print(f"\n🎯 Target: ≥80% test accuracy")

if test_acc >= 0.80:
    print(f"✅ SUCCESS! Achieved {test_acc:.1%}")
else:
    print(f"⚠️  Close! Got {test_acc:.1%}, need ≥80%")

# Save model
output_dir = Path('models')
output_dir.mkdir(exist_ok=True)

from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = output_dir / f'neural_oracle_v2_improved_{timestamp}.pkl'

with open(model_path, 'wb') as f:
    pickle.dump(ensemble, f)

print(f"\n💾 Model saved: {model_path}")
print(f"   Size: {model_path.stat().st_size / 1024:.0f} KB")

# Save training history
history = {
    'train_accuracy': float(train_acc),
    'test_accuracy': float(test_acc),
    'num_features': X.shape[1],
    'num_examples': X.shape[0],
    'model_type': 'ensemble_xgb_lgb',
    'timestamp': timestamp
}

history_path = output_dir / 'training_history_v2.json'
with open(history_path, 'w') as f:
    json.dump(history, f, indent=2)

print(f"📊 History saved: {history_path}")

print(f"\n{'='*70}")
print("COMPARISON")
print(f"{'='*70}")
print(f"Original model:")
print(f"  - Features: 10")
print(f"  - Examples: 149")
print(f"  - Accuracy: 75.9%")
print(f"\nImproved model:")
print(f"  - Features: {X.shape[1]}")
print(f"  - Examples: {X.shape[0]}")
print(f"  - Accuracy: {test_acc:.1%}")
print(f"\nImprovement: +{(test_acc - 0.759)*100:.1f}%")
```

---

## Step 5: Download Model (2 minutes)

```python
# Download improved model
from google.colab import files
import glob

model_file = glob.glob('models/neural_oracle_v2_improved_*.pkl')[0]
files.download(model_file)

print(f"✅ Downloaded: {model_file}")
print(f"\n📋 Next steps:")
print(f"1. Copy to local: C:/Users/shobi/Desktop/AURORA/AURORA-V2/models/")
print(f"2. Restart backend: uvicorn src.api.server:app --reload")
print(f"3. Test in UI with Books.csv")
```

---

## Expected Timeline

| Step | Time | Status |
|------|------|--------|
| Setup | 5 min | ⏱️ |
| Expand datasets | 30 min | ⏱️ |
| Generate labels | 30 min | ⏱️ |
| LLM validation | 20 min | ⏱️ |
| Train improved model | 10 min | ⏱️ |
| Download | 2 min | ⏱️ |
| **Total** | **~2 hours** | |

---

## Expected Results

**Before (Current):**
- Features: 10
- Examples: 149
- Accuracy: 75.9%

**After (Improved):**
- Features: 20
- Examples: 500-600
- Accuracy: **80-85%** ✅

**Improvement: +4-9%**

---

## Troubleshooting

**If accuracy < 80%:**
1. Check if all 40 datasets were generated
2. Verify LLM validation ran correctly
3. Ensure features extracted properly
4. Try adjusting ensemble weights

**If Groq rate limit hit:**
- Wait 24 hours, or
- Use friend's API key, or
- Switch to Hugging Face (slower)

---

**Ready to run this in Colab?** Just copy-paste each code block! 🚀
