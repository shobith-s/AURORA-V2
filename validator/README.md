# LLM-Based Neural Oracle Training System

## Overview

This system uses a **local LLM (Qwen 2.5 7B)** to validate preprocessing decisions and train a high-quality neural oracle. The LLM acts as an expert validator, reviewing symbolic engine decisions and providing ground truth labels.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              LLM Validation Pipeline                     │
└─────────────────────────────────────────────────────────┘

1. Download Real Datasets (UCI, Kaggle, OpenML)
         ↓
2. Run Symbolic Engine on All Columns
         ↓
3. Filter by Confidence:
   - High (>0.90): Trust symbolic (no LLM)
   - Medium (0.70-0.90): LLM validates
   - Low (<0.70): LLM decides
         ↓
4. Train Neural Oracle on Mixed Labels:
   - High-confidence symbolic labels
   - LLM-validated/corrected labels
         ↓
5. Validate on Held-Out Datasets
         ↓
6. Deploy New Model
```

---

## Prerequisites

### System Requirements
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB free space
- **OS**: Windows, Linux, or macOS
- **Python**: 3.8+

### Software Requirements
- Python packages (see `requirements.txt`)
- Ollama (for local LLM)
- Internet connection (for dataset download)

---

## Installation

### Step 1: Install Ollama

**Windows:**
```bash
# Download from: https://ollama.ai/download
# Run the installer
# Verify installation
ollama --version
```

**Linux/macOS:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama --version
```

### Step 2: Download Qwen 2.5 Model

```bash
# Pull the model (4GB download)
ollama pull qwen2.5:7b

# Test it
ollama run qwen2.5:7b "Hello, are you working?"
# Should respond with a greeting

# Exit with: /bye
```

### Step 3: Install Python Dependencies

```bash
# Navigate to validator directory
cd validator

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Quick Start (Automated)

```bash
# Run the complete pipeline
python run_validation_pipeline.py

# This will:
# 1. Download 30 datasets (~1 hour)
# 2. Run symbolic engine (~30 min)
# 3. Validate with LLM (~2-3 hours)
# 4. Train neural oracle (~30 min)
# 5. Test on Books.csv
# Total: ~5 hours (fully automated)
```

### Step-by-Step (Manual)

#### Step 1: Download Datasets

```bash
python download_datasets.py --count 30 --output data/

# Options:
#   --count: Number of datasets (default: 30)
#   --sources: uci,kaggle,openml (default: all)
#   --output: Output directory
```

#### Step 2: Generate Symbolic Labels

```bash
python generate_symbolic_labels.py --data data/ --output labels/

# This creates:
#   labels/symbolic_decisions.json
#   labels/high_confidence.json (>0.90)
#   labels/medium_confidence.json (0.70-0.90)
#   labels/low_confidence.json (<0.70)
```

#### Step 3: LLM Validation

```bash
python llm_validator.py --input labels/ --output validated/

# Options:
#   --model: qwen2.5:7b (default)
#   --validate-medium: Validate medium-confidence cases
#   --validate-low: Let LLM decide low-confidence cases
#   --batch-size: 10 (process 10 at a time)
```

#### Step 4: Train Neural Oracle

```bash
python train_neural_oracle.py --data validated/ --output models/

# This creates:
#   models/neural_oracle_v2_TIMESTAMP.pkl
#   models/training_history.json
```

#### Step 5: Validate Model

```bash
python test_model.py --model models/neural_oracle_v2_*.pkl --test-data data/books.csv

# Expected output:
#   Accuracy: >70% (target)
```

---

## Configuration

### `config.yaml`

```yaml
# LLM Configuration
llm:
  model: "qwen2.5:7b"
  api_url: "http://localhost:11434/api/generate"
  temperature: 0.1  # Low temperature for consistency
  max_tokens: 500

# Validation Strategy
validation:
  high_confidence_threshold: 0.90  # Trust symbolic
  medium_confidence_threshold: 0.70  # Validate with LLM
  validate_medium: true
  validate_low: true

# Dataset Configuration
datasets:
  count: 30
  sources:
    - uci
    - kaggle
    - openml
  max_columns_per_dataset: 20

# Training Configuration
training:
  train_val_split: 0.8
  num_boost_rounds: 50
  max_depth: 6
  learning_rate: 0.1
```

---

## File Structure

```
validator/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── config.yaml                        # Configuration
│
├── scripts/
│   ├── download_datasets.py           # Download real datasets
│   ├── generate_symbolic_labels.py    # Run symbolic engine
│   ├── llm_validator.py               # LLM validation
│   ├── train_neural_oracle.py         # Train model
│   ├── test_model.py                  # Test model
│   └── run_validation_pipeline.py     # Full pipeline
│
├── utils/
│   ├── dataset_loader.py              # Dataset loading utilities
│   ├── llm_client.py                  # LLM API client
│   ├── prompt_templates.py            # LLM prompts
│   └── metrics.py                     # Evaluation metrics
│
├── data/                              # Downloaded datasets
├── labels/                            # Symbolic labels
├── validated/                         # LLM-validated labels
└── models/                            # Trained models
```

---

## Expected Results

### Training Data Quality
- **Total examples**: 450-550
- **High-confidence symbolic**: 300-350 (trusted)
- **LLM-validated**: 150-200 (corrected if needed)

### Model Performance
- **Training accuracy**: 85-95%
- **Validation accuracy**: 70-80%
- **Books.csv accuracy**: >70% (target)

### Time & Cost
- **Total time**: 5-6 hours (automated)
- **Manual time**: 30 min (setup only)
- **Cost**: $0 (local LLM)

---

## Troubleshooting

### Issue: Ollama not found
```bash
# Check if Ollama is running
ollama list

# Start Ollama service (if needed)
ollama serve
```

### Issue: Model download fails
```bash
# Check internet connection
# Retry download
ollama pull qwen2.5:7b --insecure
```

### Issue: LLM validation is slow
```bash
# Use GPU acceleration (if available)
# Check CUDA installation
nvidia-smi

# Or use smaller model
ollama pull qwen2.5:3b
```

### Issue: Out of memory
```bash
# Use smaller model
ollama pull qwen2.5:3b

# Or reduce batch size
python llm_validator.py --batch-size 5
```

---

## Advanced Usage

### Using Different LLMs

```bash
# Llama 3.1 8B (alternative)
ollama pull llama3.1:8b
python llm_validator.py --model llama3.1:8b

# Mistral 7B (faster)
ollama pull mistral:7b
python llm_validator.py --model mistral:7b
```

### Custom Prompts

Edit `utils/prompt_templates.py` to customize LLM prompts:

```python
VALIDATION_PROMPT = """
You are a data preprocessing expert...
[Custom instructions]
"""
```

### Ensemble Validation

```bash
# Use multiple LLMs and vote
python llm_validator.py --models qwen2.5:7b,llama3.1:8b --vote
```

---

## Monitoring

### View Progress

```bash
# Watch validation progress
tail -f logs/validation.log

# Check LLM API calls
tail -f logs/llm_calls.log
```

### Metrics Dashboard

```bash
# Generate validation report
python utils/generate_report.py --input validated/ --output report.html

# Open in browser
start report.html
```

---

## Next Steps

After successful training:

1. **Test on Books.csv** (should get >70% accuracy)
2. **Compare with old model** (should be better)
3. **Deploy to production** (update model path)
4. **Monitor performance** (collect user corrections)
5. **Retrain periodically** (with new corrections)

---

## FAQ

**Q: Why local LLM instead of GPT-4?**
A: Free, private, no API costs, works offline.

**Q: Can I use GPT-4 instead?**
A: Yes, modify `llm_client.py` to use OpenAI API.

**Q: How long does validation take?**
A: 2-3 hours for 500 validations on CPU, 30-45 min on GPU.

**Q: What if LLM makes mistakes?**
A: LLM only validates medium/low confidence cases. High-confidence symbolic decisions are trusted.

**Q: Can I skip LLM validation?**
A: Yes, but model quality will be lower (only symbolic labels).

---

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review configuration in `config.yaml`
3. Test LLM with `ollama run qwen2.5:7b`
4. Check dataset downloads in `data/` directory

---

## License

MIT License - See LICENSE file for details.

---

**Last Updated**: 2024-11-27
**Version**: 1.0.0
**Status**: Ready for Production
