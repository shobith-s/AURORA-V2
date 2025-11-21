# Evaluation Scripts Guide

## AURORA V2 Evaluation

### Quick Start

**To evaluate the trained neural oracle:**
```bash
python scripts/evaluate_neural_oracle.py
```

**To run complete system tests:**
```bash
pytest tests/test_complete_system.py -v -s
```

---

## Available Evaluation Tools

### 1. Neural Oracle Evaluation
**Script:** `evaluate_neural_oracle.py`

Tests the trained neural oracle with 12 diverse test cases:
- High skewness
- High null rates
- Categorical encodings (low/high cardinality)
- Normal distributions
- Outliers
- Low variance
- Bimodal distributions
- Ambiguous cases

**Usage:**
```bash
# Use default model
python scripts/evaluate_neural_oracle.py

# Specify model path
python scripts/evaluate_neural_oracle.py --model models/neural_oracle_v1.pkl

# Save results to JSON
python scripts/evaluate_neural_oracle.py --output results.json
```

**Output:**
- Layer usage breakdown (learned, symbolic, neural, meta_learning)
- Confidence statistics and distribution
- Warning and manual review counts
- Action diversity
- Detailed results for each test case

---

### 2. Complete System Tests
**Script:** `pytest tests/test_complete_system.py`

Comprehensive integration tests covering all 5 phases:

**Phase 1: SHAP Explainability**
- `test_symbolic_layer_explainability` - Symbolic decisions are explainable
- `test_neural_oracle_shap_explainability` - Neural oracle provides SHAP values

**Phase 2: Training Integration**
- Tested via successful training with `train_hybrid.py`

**Phase 3: Confidence Warnings**
- `test_confidence_warnings_low` - Very low confidence triggers manual review
- `test_confidence_warnings_medium` - Low confidence shows warnings

**Phase 4: Layer Metrics**
- `test_metrics_tracking` - Metrics tracked per layer
- `test_metrics_persistence` - Metrics saved and loaded correctly

**Phase 5: Complete Integration**
- `test_all_phases_integrated` - All phases working together
- `test_all_layers_accessible` - All layers can be triggered
- `test_response_schema_compliance` - API schema compliance
- `test_error_handling` - Graceful error handling

**Usage:**
```bash
# Run all tests
pytest tests/test_complete_system.py -v -s

# Run specific test class
pytest tests/test_complete_system.py::TestCompleteSystem -v -s

# Run specific test
pytest tests/test_complete_system.py::TestCompleteSystem::test_all_phases_integrated -v -s
```

---

### 3. SHAP Explainability Tests
**Script:** `pytest tests/test_shap_explainability.py`

Detailed tests for SHAP functionality:
- SHAP explanation format
- Top contributing features
- Feature impact calculations
- Integration with preprocessor
- Batch predictions

**Usage:**
```bash
pytest tests/test_shap_explainability.py -v -s
```

---

## Deprecated Scripts

### ⚠️ `evaluate_system.py` - DEPRECATED

This script was designed for AURORA V1 and uses the old API:
- References `AuroraPreprocessor` (now `IntelligentPreprocessor`)
- Uses `analyze(dataframe)` API (now `preprocess_column(series, name)`)
- Not compatible with V2 architecture

**Do not use this script.** Use the alternatives above instead.

---

## Understanding Evaluation Results

### Layer Usage

The system uses a 4-layer decision hierarchy:

1. **learned** - Learned patterns from user corrections (highest priority)
2. **symbolic** - Hand-crafted rules for clear cases
3. **neural** - Neural oracle for ambiguous cases
4. **meta_learning** - Heuristics for edge cases

**Expected Distribution:**
- Symbolic: 40-60% (handles most clear cases)
- Learned: 10-30% (grows with user corrections)
- Neural: 10-20% (ambiguous cases only)
- Meta_learning: 5-15% (fallback for edge cases)

### Confidence Scores

- **High (≥0.9)**: Auto-apply decision
- **Medium (0.7-0.9)**: Show warning
- **Low (0.5-0.7)**: Require manual review
- **Very Low (<0.5)**: Strong manual review required

**Healthy System:**
- 60%+ high confidence
- <20% requiring manual review
- Mean confidence ≥0.75

### Neural Oracle Usage

Neural oracle should activate when:
- Symbolic confidence < 0.9
- No learned patterns match
- Data characteristics are ambiguous

**Note:** It's normal for neural oracle to be used 10-20% of the time.
If it's used 0%, symbolic rules are handling everything (also fine).
If it's used >50%, symbolic rules may need improvement.

---

## Continuous Evaluation

### After Training

1. **Immediate:** Run neural oracle evaluation
   ```bash
   python scripts/evaluate_neural_oracle.py
   ```

2. **Integration:** Run system tests
   ```bash
   pytest tests/test_complete_system.py -v -s
   ```

3. **Deployment:** Monitor metrics via API
   ```bash
   curl http://localhost:8000/api/metrics/layers
   ```

### Ongoing Monitoring

1. **Collect corrections** via the UI
2. **Track layer metrics** in production
3. **Retrain periodically** (every 100+ corrections)
4. **Re-evaluate** after retraining

---

## Performance Benchmarks

### Target Metrics

**Model Training:**
- Training time: <2 minutes
- Model size: <5 MB
- Training accuracy: >85%
- Validation accuracy: >80%

**Inference:**
- Column preprocessing: <50ms
- Batch processing (100 columns): <2s
- API response time: <100ms

**System Quality:**
- Mean confidence: >0.75
- High confidence decisions: >60%
- Manual review rate: <20%
- Overall accuracy: >85%

---

## Troubleshooting

### "Neural oracle not used in tests"

**Normal behavior.** Neural oracle only activates when:
- Symbolic confidence < 0.9
- No learned patterns match

If symbolic rules are working well, neural oracle won't be needed.

### "Low validation accuracy (<70%)"

**Possible causes:**
1. Too few training samples
2. Imbalanced action distribution
3. Need more user corrections

**Solutions:**
- Collect more datasets: `python scripts/collect_open_datasets.py`
- Increase synthetic samples: `--synthetic 5000`
- Collect user corrections via UI

### "Test failures"

Check that:
1. Model trained successfully
2. Dependencies installed: `pip install -r requirements.txt`
3. Database accessible: `aurora.db` exists
4. No import errors

Run setup verification:
```bash
python scripts/verify_training_setup.py
```

---

## Next Steps

After successful evaluation:

1. **Start API server:**
   ```bash
   uvicorn src.api.server:app --reload
   ```

2. **Open UI:**
   ```
   http://localhost:8000
   ```

3. **Monitor metrics:**
   ```
   http://localhost:8000/api/metrics/layers
   ```

4. **Collect corrections** to improve the system

5. **Retrain periodically** with accumulated corrections

---

## Questions?

- System architecture: `docs/IMPLEMENTATION_GUIDE.md`
- Training guide: `docs/TRAINING_GUIDE.md`
- System handover: `docs/HANDOVER.md`
- Test coverage: Check `tests/` directory
