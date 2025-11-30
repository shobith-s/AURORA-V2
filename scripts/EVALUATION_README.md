# Scripts Directory - Quick Reference

## Available Scripts (6 Total)

### 1. Neural Oracle Training
**File:** `train_neural_oracle_real_data.py`

Train the neural oracle on real-world datasets with comprehensive safety validation.

**Usage:**
```bash
python scripts/train_neural_oracle_real_data.py
```

**Features:**
- Collects 10+ datasets from sklearn and OpenML
- Trains XGBoost with 30 enhanced features
- 3-layer architecture validation (prevents catastrophic data loss)
- Saves model with comprehensive metadata

**Output:** `models/neural_oracle_v1.pkl`

---

### 2. Data Understanding Diagnostic
**File:** `diagnostic_data_understanding.py`

Measures how well AURORA understands different data types.

**Usage:**
```bash
python scripts/diagnostic_data_understanding.py
```

**Tests:**
- 18 diverse column types (numeric, categorical, temporal, imbalanced, etc.)
- Feature extraction validation
- Pattern detection accuracy
- Type detection correctness

**Output:** Understanding score (currently 84%, target 92%)

---

### 3. Enhanced Features Test
**File:** `test_enhanced_features.py`

Validates the enhanced 30-feature extractor.

**Usage:**
```bash
python scripts/test_enhanced_features.py
```

**Tests:**
- Backward compatibility (30 ↔ 10 features)
- Distribution features (quantiles, gaps, dispersion)
- Text features (length, diversity, numeric ratio)
- Semantic features (type tags, key detection)
- Temporal features (autocorrelation, monotonicity)
- Advanced patterns (embedded nulls, URL/email diversity)

**Output:** Test results (all 7 suites should pass)

---

### 4. Feature Extractor Comparison
**File:** `compare_extractors.py`

Side-by-side comparison of minimal (10) vs enhanced (30) features.

**Usage:**
```bash
python scripts/compare_extractors.py
```

**Shows:**
- Feature differences for ID, metric, email, target columns
- Semantic type detection
- Domain classification
- Key candidate detection

**Output:** Visual comparison of both extractors

---

### 5. Training Setup Verification
**File:** `verify_training_setup.py`

Validates that the environment is ready for training.

**Usage:**
```bash
python scripts/verify_training_setup.py
```

**Checks:**
- Dependencies installed
- Database accessible
- Model directory exists
- Import paths correct

**Output:** Setup validation status

---

### 6. Documentation
**File:** `EVALUATION_README.md` (this file)

Reference documentation for all scripts.

---

## Quick Start Workflow

### First Time Setup
```bash
# 1. Verify environment
python scripts/verify_training_setup.py

# 2. Test enhanced features
python scripts/test_enhanced_features.py

# 3. Run diagnostic
python scripts/diagnostic_data_understanding.py

# 4. Train neural oracle
python scripts/train_neural_oracle_real_data.py
```

### After Training
```bash
# Compare feature extractors
python scripts/compare_extractors.py

# Re-run diagnostic to measure improvement
python scripts/diagnostic_data_understanding.py
```

---

## System Architecture

### Feature Extraction
- **Minimal (10 features)**: Fast, backward compatible, current neural oracle
- **Enhanced (30 features)**: Deep understanding, semantic awareness, future-ready

### Decision Pipeline
1. **Symbolic Engine** (165+ rules) → 75% coverage
2. **Neural Oracle** (XGBoost) → Ambiguous cases
3. **Conservative Fallback** → Ultimate safety net

---

## Performance Targets

### Training
- Training time: < 5 minutes
- Model size: < 5 MB
- Safety validation: 100% pass
- Test accuracy: > 70%

### Understanding
- Overall score: 84% → 92% (Phase 1 target)
- Semantic detection: 95%+
- Pattern recognition: 100%

### Inference
- Column preprocessing: < 50ms
- Feature extraction: < 10ms
- Decision confidence: > 0.75 average

---

## Troubleshooting

### "Module not found"
```bash
pip install -r requirements.txt
```

### "No trained model"
```bash
python scripts/train_neural_oracle_real_data.py
```

### "Safety validation failed"
- Review the specific check that failed
- Increase training data diversity
- Adjust thresholds if needed

### "Low understanding score"
- Normal for current 10-feature system (84%)
- Upgrade to 30 features for 92%+ score
- Collection more real-world datasets

---

## Recent Updates

**2024-11-24: Phase 1 Implementation**
- ✅ Enhanced feature extractor (30 features)
- ✅ Semantic type detection
- ✅ Domain classification
- ✅ Primary/Foreign key detection
- ✅ Comprehensive training script with safety validation

**2024-11-24: Scripts Cleanup**
- ❌ Removed 11 redundant/deprecated files
- ✅ Kept 6 essential scripts
- 📉 Reduced size by 68%
- 📈 Improved clarity significantly

---

## Next Steps

1. **Train enhanced oracle**: Update training script to use 30 features
2. **A/B testing**: Compare 10 vs 30 feature performance
3. **Phase 2**: Inter-column relationship analysis
4. **Schema detection**: Automatic table type classification

---

For more information:
- System architecture: `docs/IMPLEMENTATION_GUIDE.md`
- Feature documentation: `phase1_enhanced_features.md` (artifact)
- Cleanup analysis: `scripts_cleanup_analysis.md` (artifact)
