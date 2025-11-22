# AURORA V2 - Quick Start Guide

Get AURORA running in **5 minutes**.

---

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Start the Server

```bash
uvicorn src.api.server:app --reload
```

Server runs at: **http://localhost:8000**

---

## 3. Verify It Works

```bash
# Health check
curl http://localhost:8000/health

# Should return:
# {"status": "healthy", "version": "1.0.0", ...}
```

---

## 4. Use the API

### Option A: Swagger UI (Easiest)

Open in browser: **http://localhost:8000/docs**

Click `/preprocess` → Try it out → Enter data → Execute

### Option B: curl

```bash
curl -X POST "http://localhost:8000/preprocess" \
  -H "Content-Type: application/json" \
  -d '{
    "column_data": [1.5, 2.3, 100.0, 3.2, 4.5, 1000.0],
    "column_name": "revenue",
    "target_available": false
  }'
```

### Option C: Python

```python
import requests

response = requests.post(
    "http://localhost:8000/preprocess",
    json={
        "column_data": [1.5, 2.3, 100.0, 3.2, 4.5, 1000.0],
        "column_name": "revenue",
        "target_available": False
    }
)

result = response.json()
print(f"Action: {result['action']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Explanation: {result['explanation']}")
```

---

## 5. Open the Frontend (Optional)

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: **http://localhost:3000**

---

## Common Commands

```bash
# Run tests
pytest

# Run specific test file
pytest tests/test_shap_and_chatbot.py -v

# Check system status
curl http://localhost:8000/health

# View API documentation
open http://localhost:8000/docs
```

---

## What Works

✅ Single column preprocessing recommendations
✅ Adaptive learning from corrections
✅ SHAP explanations (if neural oracle available)
✅ Intelligent assistant chatbot
✅ Health monitoring

---

## Quick Troubleshooting

**Server won't start?**
```bash
# Check if port 8000 is in use
lsof -i :8000
kill -9 <PID>

# Or use different port
uvicorn src.api.server:app --port 8080
```

**"Neural oracle not found" warning?**
- Normal - system works without it
- Uses symbolic rules + meta-learning only
- To train neural oracle: `python scripts/train_hybrid.py`

**Need more help?**
- Full docs: `docs/PRODUCTION_READY_STATUS.md`
- API docs: http://localhost:8000/docs
- Report issues: GitHub Issues

---

**You're ready to go!** 🚀
