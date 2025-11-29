# Quick Start: Neural Oracle Training with Hugging Face (FREE)

**100% Free - No Billing Account Required!**

---

## Step 1: Get Hugging Face API Token (5 minutes)

### 1.1 Sign Up
```
1. Go to: https://huggingface.co/join
2. Sign up with email (free, no credit card)
3. Verify email
```

### 1.2 Create Token
```
1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: "aurora-training"
4. Type: "Read"
5. Click "Generate"
6. Copy the token (starts with "hf_...")
```

**That's it! No billing account needed!** ✅

---

## Step 2: Open Google Colab

```
1. Go to: https://colab.research.google.com
2. Click "New Notebook"
3. Runtime → Change runtime type → T4 GPU → Save
```

---

## Step 3: Run Training Pipeline

**Copy-paste this entire code block into Colab:**

```python
# ============================================
# AURORA Neural Oracle Training (Hugging Face)
# ============================================

# Cell 1: Setup
!nvidia-smi  # Verify GPU
!git clone -b polishing https://github.com/shobith-s/AURORA-V2.git
%cd AURORA-V2/validator

# Cell 2: Install dependencies
!pip install -q xgboost pandas numpy scikit-learn pyyaml tqdm huggingface_hub

# Cell 3: Set your Hugging Face token
HF_TOKEN = "hf_..."  # ⚠️ REPLACE WITH YOUR TOKEN!

# Test connection
from huggingface_hub import InferenceClient
client = InferenceClient(token=HF_TOKEN)
response = client.text_generation("Hello!", model="meta-llama/Llama-3.1-8B-Instruct", max_new_tokens=50)
print(f"✅ Hugging Face connected: {response[:50]}...")

# Cell 4: Download datasets (30 min)
!python scripts/download_datasets.py

# Cell 5: Generate symbolic labels (30 min)
!python scripts/generate_symbolic_labels.py

# Cell 6: LLM validation (1-2 hours)
!python scripts/llm_validator.py --mode huggingface --api-key $HF_TOKEN

# Cell 7: Train neural oracle (30 min)
!python scripts/train_neural_oracle.py

# Cell 8: Download model
from google.colab import files
import glob
model_file = glob.glob('models/neural_oracle_v2_*.pkl')[0]
files.download(model_file)
```

---

## Step 4: Deploy Model

**After downloading the model:**

```bash
# On your local machine
# Copy to AURORA models directory
cp ~/Downloads/neural_oracle_v2_*.pkl \
   C:/Users/shobi/Desktop/AURORA/AURORA-V2/models/

# Restart backend
cd C:/Users/shobi/Desktop/AURORA/AURORA-V2
uvicorn src.api.server:app --reload
```

---

## Troubleshooting

### "Invalid token"
```
- Check token starts with "hf_"
- Make sure it's "Read" type
- Regenerate if needed
```

### "Rate limit exceeded"
```
- Free tier: 1000 requests/day
- Wait 24 hours or reduce batch size
- Script auto-handles rate limits
```

### "Model not found"
```
- Using: meta-llama/Llama-3.1-8B-Instruct
- Alternative: mistralai/Mistral-7B-Instruct-v0.2
- Change in llm_client.py if needed
```

---

## Expected Timeline

- Setup: 5 min
- Download datasets: 30 min
- Generate labels: 30 min
- LLM validation: 1-2 hours (rate limits)
- Training: 30 min
- **Total: 3-4 hours**

---

## Cost Breakdown

| Resource | Cost |
|----------|------|
| Google Colab GPU | $0 |
| Hugging Face API | $0 |
| Storage | $0 |
| **TOTAL** | **$0** |

**Truly free!** ✅

---

## What You'll Get

- Trained neural oracle model
- Validation accuracy: 70-80% (expected)
- Training history
- Ready to deploy

---

**Questions?** Check the full guide: `CLOUD_TRAINING_GUIDE.md`
