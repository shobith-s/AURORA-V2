# Quick Start: Cloud Training

**Train neural oracle in Google Colab (100% Free)**

## 1. Get Gemini API Key (5 min)
```
https://aistudio.google.com/app/apikey
→ Create API Key
→ Copy key (starts with "AIza...")
```

## 2. Open Google Colab
```
https://colab.research.google.com
→ New Notebook
→ Runtime → Change runtime type → T4 GPU
```

## 3. Run This Code

```python
# Setup
!git clone https://github.com/YOUR_USERNAME/AURORA-V2.git
%cd AURORA-V2/validator
!pip install -q xgboost pandas numpy scikit-learn google-generativeai

# Configure Gemini
import google.generativeai as genai
GEMINI_API_KEY = "YOUR_KEY_HERE"  # Replace!
genai.configure(api_key=GEMINI_API_KEY)

# Download datasets
!python scripts/download_datasets.py

# Generate labels
!python scripts/generate_symbolic_labels.py

# Validate with LLM (1-2 hours)
!python scripts/llm_validator.py --mode gemini --api-key $GEMINI_API_KEY

# Train
!python scripts/train_neural_oracle.py

# Test
!python scripts/test_comprehensive.py

# Download model
from google.colab import files
files.download('models/neural_oracle_v2_*.pkl')
```

## 4. Deploy
```bash
# Copy model to AURORA
cp neural_oracle_v2_*.pkl C:/Users/shobi/Desktop/AURORA/AURORA-V2/models/

# Restart backend
uvicorn src.api.server:app --reload
```

**Total Time**: 3-4 hours (automated)
**Total Cost**: $0

**Full guide**: See `CLOUD_TRAINING_GUIDE.md`
