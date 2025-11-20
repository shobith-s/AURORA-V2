# 🚀 AURORA Complete Setup Guide

## Overview

This guide will help you set up the complete AURORA system including:
- ✅ Backend API (FastAPI)
- ✅ Frontend UI (Next.js + React + TailwindCSS)
- ✅ Performance Monitoring
- ✅ Interactive Chatbot

---

## 📋 Prerequisites

- Python 3.9+
- Node.js 18+ and npm
- Git
- 4GB+ RAM
- (Optional) Docker

---

## 🎯 Quick Start (5 minutes)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd AURORA-V2
```

### 2. Backend Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the API server
python -m uvicorn src.api.server:app --reload --port 8000
```

✅ Backend running at: http://localhost:8000
📚 API Docs: http://localhost:8000/docs

### 3. Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

✅ Frontend running at: http://localhost:3000

---

## 🐳 Docker Setup (Alternative)

### Backend Only

```bash
# Build image
docker build -t aurora-backend .

# Run container
docker run -p 8000:8000 aurora-backend
```

### Full Stack (Backend + Frontend)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8000
    depends_on:
      - backend
```

Run:
```bash
docker-compose up
```

---

## 🎨 UI Features

### Main Interface (60% Left Panel)

**Data Preprocessing Panel**
- Upload or paste column data
- Sample data templates (skewed, categorical, dates, currency)
- Real-time preprocessing analysis
- Action recommendations with confidence scores
- Alternative suggestions
- Feedback & correction mechanism

**Key Features:**
- 🚀 Instant analysis (<1ms for most cases)
- 📊 Visual confidence indicators
- 🔄 Interactive correction system
- 💡 Sample data templates

### Chatbot Interface (40% Right Panel)

**AI Assistant Features:**
- Real-time Q&A about preprocessing
- Explains technical concepts
- Guided troubleshooting
- Quick question templates
- Context-aware responses

**Example Questions:**
- "What is symbolic preprocessing?"
- "How does the neural oracle work?"
- "When should I use log transform?"
- "Explain privacy-preserving learning"

### Performance Metrics Dashboard (Collapsible)

**Real-time Metrics:**
- CPU & Memory usage (live updates)
- Decision latency (p50, p95, p99)
- Success rate tracking
- Decision source breakdown (symbolic/neural/learned)

**Visualizations:**
- 📈 Latency charts
- 🥧 Decision source pie charts
- 📊 Component performance bars
- 🎯 Confidence distribution

---

## 📡 API Endpoints

### Core Endpoints

#### 1. Preprocess Column
```bash
POST /preprocess
Content-Type: application/json

{
  "column_data": [10, 20, 30, 100, 200],
  "column_name": "revenue",
  "target_available": false
}
```

**Response:**
```json
{
  "action": "log_transform",
  "confidence": 0.92,
  "source": "symbolic",
  "explanation": "High positive skewness (3.45) in positive data",
  "alternatives": [
    {"action": "box_cox", "confidence": 0.85}
  ],
  "decision_id": "550e8400-..."
}
```

#### 2. Submit Correction
```bash
POST /correct

{
  "column_data": [10, 20, 30, ...],
  "column_name": "revenue",
  "wrong_action": "log_transform",
  "correct_action": "robust_scale",
  "confidence": 0.92
}
```

#### 3. Performance Metrics
```bash
GET /metrics/performance
GET /metrics/decisions?limit=100
GET /metrics/realtime
```

#### 4. Batch Processing
```bash
POST /batch

{
  "columns": {
    "age": [25, 30, 35, ...],
    "revenue": [1000, 2000, ...]
  },
  "target_column": null
}
```

---

## 🧪 Testing the System

### 1. Quick Test (Backend)

```python
import requests

# Test preprocessing
response = requests.post('http://localhost:8000/preprocess', json={
    'column_data': [10, 15, 20, 100, 200, 500, 1000],
    'column_name': 'revenue'
})

print(response.json())
```

### 2. UI Test Flow

1. **Open UI**: http://localhost:3000
2. **Click "Skewed Data"** sample button
3. **Click "Analyze & Recommend"**
4. **Review results** with confidence scores
5. **Test correction** by clicking thumbs down
6. **Ask chatbot**: "Why log transform?"
7. **Toggle metrics** to see performance

### 3. Run Test Suite

```bash
# Backend tests
pytest tests/ -v --cov=src

# Frontend tests (if added)
cd frontend
npm test
```

---

## 🔧 Configuration

### Backend Configuration

Edit `configs/rules.yaml`:
```yaml
data_quality:
  null_threshold: 0.6
  constant_threshold: 1

statistical:
  high_skewness: 2.0
  outlier_clip_threshold: 0.1
```

Edit `configs/privacy.yaml`:
```yaml
differential_privacy:
  epsilon: 1.0
  delta: 1.0e-5
```

### Frontend Configuration

Edit `frontend/next.config.js`:
```javascript
async rewrites() {
  return [
    {
      source: '/api/:path*',
      destination: 'http://your-backend-url:8000/:path*',
    },
  ]
}
```

---

## 📊 Performance Benchmarks

### Expected Performance

| Component | Target | Typical |
|-----------|--------|---------|
| Symbolic Engine | <100μs | 80μs |
| Neural Oracle | <5ms | 4.2ms |
| Pattern Learner | <1ms | 0.5ms |
| Full Pipeline | <1ms | 0.4ms |
| Memory Usage | <50MB | 35MB |

### Monitoring Commands

```bash
# Backend metrics
curl http://localhost:8000/metrics/performance | jq

# Realtime stats
watch -n 1 'curl -s http://localhost:8000/metrics/realtime | jq'

# Statistics
curl http://localhost:8000/statistics | jq
```

---

## 🐛 Troubleshooting

### Backend Issues

**Issue: ModuleNotFoundError**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

**Issue: Port 8000 already in use**
```bash
# Change port
uvicorn src.api.server:app --port 8001
```

**Issue: NumPy/Numba errors**
```bash
# Reinstall with specific versions
pip install numpy==1.24.0 numba==0.57.0
```

### Frontend Issues

**Issue: Module not found**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

**Issue: API connection failed**
```bash
# Check backend is running
curl http://localhost:8000/health

# Check CORS settings in src/api/server.py
```

**Issue: Styles not loading**
```bash
# Rebuild Tailwind
npm run dev
```

---

## 🚀 Production Deployment

### Backend (Gunicorn)

```bash
# Install gunicorn
pip install gunicorn

# Run with workers
gunicorn src.api.server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Frontend (Next.js Build)

```bash
cd frontend

# Build for production
npm run build

# Start production server
npm start
```

### Environment Variables

Create `.env`:
```bash
# Backend
PYTHONPATH=/app
LOG_LEVEL=INFO
CORS_ORIGINS=https://your-domain.com

# Frontend
NEXT_PUBLIC_API_URL=https://api.your-domain.com
```

### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location /api/ {
        proxy_pass http://localhost:8000/;
    }

    location / {
        proxy_pass http://localhost:3000;
    }
}
```

---

## 📚 Next Steps

1. **Customize Rules**: Edit `configs/rules.yaml` for domain-specific rules
2. **Train Neural Oracle**: Generate synthetic data and train on your use cases
3. **Add Custom Actions**: Extend `src/core/actions.py` with new preprocessing actions
4. **Integrate Chatbot**: Connect to OpenAI API or custom LLM
5. **Set Up Monitoring**: Use Prometheus + Grafana for production monitoring

---

## 🤝 Support

- 📖 Documentation: `README.md`
- 🐛 Issues: GitHub Issues
- 💬 Discussions: GitHub Discussions
- 📧 Email: support@aurora-system.com

---

## 🎉 Success!

You should now have:
- ✅ Backend API running on http://localhost:8000
- ✅ Frontend UI on http://localhost:3000
- ✅ Performance metrics dashboard
- ✅ Interactive chatbot
- ✅ Real-time preprocessing capabilities

**Try it out:**
1. Open http://localhost:3000
2. Click a sample data button
3. Analyze & enjoy! 🚀
