# Quick Start: Neural Oracle Training with Groq (FASTEST!)

**10x Faster Than Hugging Face - Still 100% Free!**

---

## Why Groq?

- ✅ **10x FASTER** - Validation: 2 hours → 20 minutes!
- ✅ **Better Model** - Llama 3.1 70B (vs 8B)
- ✅ **Higher Limits** - 14,400 requests/day (vs 1,000)
- ✅ **100% FREE** - No billing required

---

## Step 1: Get Groq API Key (2 minutes)

```
1. Go to: https://console.groq.com
2. Sign up (free, no credit card)
3. Click "API Keys"
4. Create new key
5. Copy key (starts with "gsk_...")
```

---

## Step 2: Run Training in Colab

```python
# ============================================
# AURORA Neural Oracle Training (Groq - FAST!)
# ============================================

# Cell 1: Setup
!nvidia-smi
!git clone -b polishing https://github.com/shobith-s/AURORA-V2.git
%cd AURORA-V2/validator

# Cell 2: Install dependencies
!pip install -q xgboost pandas numpy scikit-learn pyyaml tqdm groq

# Cell 3: Set Groq API key
GROQ_API_KEY = "gsk_..."  # ⚠️ REPLACE WITH YOUR KEY!

# Test connection
from groq import Groq
client = Groq(api_key=GROQ_API_KEY)
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=50
)
print(f"✅ Groq connected: {response.choices[0].message.content[:50]}...")

# Cell 4: Download datasets (30 min)
!python scripts/download_datasets.py

# Cell 5: Generate labels (45 min)
!python scripts/generate_symbolic_labels.py

# Cell 6: LLM validation (20 min with Groq! 🚀)
!python scripts/llm_validator.py --mode groq --api-key $GROQ_API_KEY

# Cell 7: Train (30 min)
!python scripts/train_neural_oracle.py

# Cell 8: Download model
from google.colab import files
import glob
model_file = glob.glob('validator/models/neural_oracle_v2_*.pkl')[0]
files.download(model_file)
```

---

## Timeline Comparison

| Step | Hugging Face | Groq |
|------|-------------|------|
| Download | 30 min | 30 min |
| Generate labels | 45 min | 45 min |
| **LLM validation** | **2 hours** | **20 min** ⚡ |
| Training | 30 min | 30 min |
| **TOTAL** | **3.5 hours** | **2 hours** |

**Groq is 10x faster for validation!**

---

## Expected Results

**With 18 datasets + Groq validation:**
- Training examples: 180-250
- Train accuracy: 85-95%
- **Val accuracy: 75-80%** (vs 71.3%)
- Better quality (70B model)

---

## Troubleshooting

### "Invalid API key"
```
- Check key starts with "gsk_"
- Regenerate if needed at console.groq.com
```

### "Rate limit exceeded"
```
- Free tier: 14,400 requests/day
- Very unlikely to hit this!
- Much higher than HF (1,000/day)
```

---

## Cost Breakdown

| Resource | Cost |
|----------|------|
| Google Colab GPU | $0 |
| Groq API | $0 |
| Storage | $0 |
| **TOTAL** | **$0** |

**100% Free + 10x Faster!** 🎉

---

**Ready to train!** 🚀
