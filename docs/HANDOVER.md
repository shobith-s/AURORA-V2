# 📋 AURORA V2 - Technical Handover Document

**Document Version:** 1.0
**Last Updated:** November 20, 2024
**Project Status:** Advanced Prototype (60% to Production)
**Current Branch:** `claude/review-repo-status-01QTfuSiB2p1qNFErugtCGxv`

---

## 📑 Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Current Implementation Status](#current-implementation-status)
4. [API Endpoints](#api-endpoints)
5. [Database Schema](#database-schema)
6. [Frontend/UI](#frontend-ui)
7. [Neural Oracle & Learning System](#neural-oracle--learning-system)
8. [Current Limitations](#current-limitations)
9. [Future Roadmap](#future-roadmap)
10. [Development Workflow](#development-workflow)
11. [Deployment Guide](#deployment-guide)
12. [Troubleshooting](#troubleshooting)

---

## 1. Executive Summary

### What is AURORA?

AURORA (Intelligent Data Preprocessing System) is an adaptive preprocessing engine that combines symbolic rules, neural intelligence, and privacy-preserving learning to automate data preprocessing decisions for tabular data.

### Core Innovation

**Privacy-Preserving Learning Loop** - The system learns from user corrections without storing raw data, using only statistical fingerprints. This is the genuinely novel aspect of the project.

### Current State

- **Architecture:** 3-layer adaptive system (Learned → Symbolic → Neural)
- **Backend:** FastAPI + SQLAlchemy + XGBoost
- **Frontend:** Next.js + React + Recharts
- **Security:** JWT authentication, CORS whitelisting
- **Learning:** Persistent correction storage with automatic rule creation
- **Status:** Advanced prototype, **NOT production-ready yet**

### Key Metrics

| Metric | Target | Current Status |
|--------|--------|----------------|
| Test Coverage | >70% | **0%** ❌ Critical |
| API Endpoints | ~20 | 18 ✅ |
| Symbolic Rules | 165+ | 165+ ✅ |
| Documentation | Complete | 90% ✅ |
| Production Readiness | 100% | 60% ⚠️ |

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (Next.js)                      │
│  ┌─────────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │ Preprocessing   │  │   Chatbot    │  │    Metrics     │ │
│  │     Panel       │  │    Panel     │  │   Dashboard    │ │
│  └─────────────────┘  └──────────────┘  └────────────────┘ │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP/REST API
┌───────────────────────────▼─────────────────────────────────┐
│                   Backend (FastAPI)                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              API Layer (server.py)                   │   │
│  │  /preprocess  /correct  /metrics  /explain  ...     │   │
│  └────────────┬──────────────────────┬──────────────────┘   │
│               │                      │                       │
│  ┌────────────▼──────────┐  ┌───────▼────────────────────┐ │
│  │  IntelligentPreprocessor│  │  AdaptiveLearningEngine  │ │
│  │  (preprocessor.py)    │  │  (adaptive_engine.py)    │ │
│  └────────────┬──────────┘  └───────┬────────────────────┘ │
│               │                      │                       │
│  ┌────────────▼──────────────────────▼─────────────────┐   │
│  │           3-Layer Decision Pipeline                  │   │
│  │                                                       │   │
│  │  1. Learned Patterns (from DB)                       │   │
│  │     ↓ (if confidence < 0.9)                          │   │
│  │  2. Symbolic Engine (165+ rules)                     │   │
│  │     ↓ (if confidence < 0.9)                          │   │
│  │  3. Neural Oracle (XGBoost)                          │   │
│  │                                                       │   │
│  └───────────────────────┬───────────────────────────────┘   │
│                          │                                   │
│  ┌───────────────────────▼───────────────────────────────┐  │
│  │              Performance Monitor                       │  │
│  │         (latency, throughput, accuracy)               │  │
│  └───────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                  Database (SQLite/PostgreSQL)                │
│  ┌─────────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │   corrections   │  │ learned_rules│  │ model_versions│  │
│  └─────────────────┘  └──────────────┘  └───────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 Three-Layer Decision Pipeline

**Layer 1: Learned Patterns (Database-backed)**
- Checks for similar patterns from past corrections
- Uses statistical fingerprint matching (k-anonymity)
- Returns decision if similarity > 0.85
- **Fastest:** Immediate DB lookup

**Layer 2: Symbolic Engine (Rule-based)**
- 165+ deterministic rules organized by category
- Pattern detection (dates, emails, IDs, etc.)
- Statistical analysis (skewness, outliers, entropy)
- Returns decision if confidence > 0.9
- **Fast:** <100μs typical latency

**Layer 3: Neural Oracle (XGBoost Meta-learner)**
- Lightweight XGBoost model (~50KB)
- Only activated when symbolic confidence < 0.9
- Trained on synthetic data + real corrections
- 10 minimal features extracted
- **Slow:** <5ms latency

### 2.3 Learning Flow

```
User Correction
      ↓
Extract Statistical Fingerprint
(null%, unique_ratio, skewness, outliers, etc.)
      ↓
Hash Pattern → pattern_hash
      ↓
Store in corrections table
      ↓
Check: >= 5 similar corrections?
      ↓ YES
Create LearnedRule
      ↓
Next request with similar pattern
      ↓
Layer 1 catches it immediately! ✨
```

---

## 3. Current Implementation Status

### 3.1 Completed Features ✅

**Backend Core:**
- ✅ FastAPI server with CORS & JWT authentication
- ✅ Three-layer preprocessing pipeline
- ✅ Symbolic engine (165+ rules)
- ✅ Neural oracle (XGBoost)
- ✅ Feature extraction (10 minimal features)
- ✅ Performance monitoring
- ✅ SQLAlchemy database integration
- ✅ Adaptive learning engine
- ✅ Privacy-preserving correction storage

**API Endpoints (18 total):**
- ✅ `/preprocess` - Single column preprocessing
- ✅ `/batch` - Batch processing
- ✅ `/correct` - Submit corrections
- ✅ `/explain/{id}` - Get decision explanation
- ✅ `/metrics/*` - Performance, learning, neural oracle metrics
- ✅ `/cache/stats` - Cache statistics
- ✅ `/drift/*` - Data drift monitoring
- ✅ `/health` - Health check

**Frontend:**
- ✅ Modern UI with Next.js + TailwindCSS
- ✅ Preprocessing panel with CSV upload
- ✅ Chatbot panel (placeholder)
- ✅ Comprehensive metrics dashboard
- ✅ Real-time updates (2s refresh)
- ✅ Neural oracle status display
- ✅ Learning progress tracking

**Security:**
- ✅ JWT authentication system
- ✅ CORS whitelist (no more `allow_origins=["*"]`)
- ✅ Environment-based configuration
- ✅ Password hashing (bcrypt)

**Learning System:**
- ✅ Persistent correction storage
- ✅ Automatic rule creation (after 5+ similar)
- ✅ Statistical fingerprint extraction
- ✅ Pattern hashing for k-anonymity
- ✅ Training script from corrections

**Documentation:**
- ✅ IMPLEMENTATION_STATUS.md (current status)
- ✅ NEURAL_ORACLE_TRAINING.md (training guide)
- ✅ README.md (quick start)
- ✅ Architecture proposals (V3)
- ✅ Implementation roadmap

### 3.2 Partially Completed ⚠️

**Neural Oracle:**
- ⚠️ Training script exists but model often not trained
- ⚠️ Underutilized (only triggered when symbolic conf < 0.9)
- ⚠️ No A/B testing framework
- ⚠️ No automated retraining

**Monitoring:**
- ⚠️ Basic metrics exist but no alerting
- ⚠️ No Prometheus/Grafana integration
- ⚠️ No distributed tracing
- ⚠️ No error tracking (Sentry, etc.)

**CSV Parser:**
- ⚠️ Basic robust parser exists
- ⚠️ Claims to "handle ANY CSV" - unproven
- ⚠️ No comprehensive edge case tests

### 3.3 Missing/Not Started ❌

**Testing (Critical):**
- ❌ Zero unit tests
- ❌ Zero integration tests
- ❌ Zero E2E tests
- ❌ No benchmark suite
- ❌ No CSV robustness tests

**Production Concerns:**
- ❌ No CI/CD pipeline
- ❌ No containerization (Docker)
- ❌ No database migrations (Alembic unused)
- ❌ No rate limiting enforcement
- ❌ No graceful shutdown
- ❌ No backup/restore procedures
- ❌ No load balancing
- ❌ Still using singletons (can't scale horizontally)

**Security Gaps:**
- ❌ No request size limits
- ❌ No input validation beyond Pydantic
- ❌ No security headers
- ❌ No vulnerability scanning
- ❌ No penetration testing

---

## 4. API Endpoints

### 4.1 Core Preprocessing

#### `POST /preprocess`
Process a single column and get preprocessing recommendation.

**Request:**
```json
{
  "column_data": [1, 2, 3, 100, 200],
  "column_name": "revenue",
  "column_metadata": {
    "dtype": "numeric",
    "source": "csv_upload"
  }
}
```

**Response:**
```json
{
  "action": "log_transform",
  "confidence": 0.92,
  "source": "symbolic",
  "explanation": "Highly skewed positive data (skew: 2.8)",
  "alternatives": [
    {"action": "robust_scale", "confidence": 0.75}
  ],
  "metadata": {
    "latency_ms": 2.3,
    "decision_id": "abc123"
  }
}
```

#### `POST /batch`
Process multiple columns in batch.

#### `POST /correct`
Submit a correction to improve the system.

**Request:**
```json
{
  "column_context": {
    "column_name": "revenue",
    "features": {...}
  },
  "wrong_action": "standard_scale",
  "correct_action": "log_transform",
  "confidence": 0.95
}
```

**Response:**
```json
{
  "status": "recorded",
  "learning_impact": {
    "pattern_learned": true,
    "similar_corrections": 6,
    "rule_created": true,
    "rule_name": "rule_revenue_log_transform"
  }
}
```

### 4.2 Metrics & Monitoring

#### `GET /metrics/dashboard`
**NEW** - Single endpoint for all dashboard metrics.

**Response:**
```json
{
  "timestamp": 1700500000,
  "overview": {
    "total_decisions": 1250,
    "avg_latency_ms": 2.3,
    "system_cpu": 15.2,
    "system_memory": 42.8,
    "uptime_hours": 48.5
  },
  "decision_sources": {
    "counts": {
      "learned": 150,
      "symbolic": 950,
      "neural": 150
    },
    "percentages": {
      "learned": 12.0,
      "symbolic": 76.0,
      "neural": 12.0
    }
  },
  "neural_oracle": {
    "model_loaded": true,
    "model_info": {
      "model_size_kb": 48.3,
      "num_actions": 15,
      "feature_names": [...]
    },
    "training_history": {
      "training_date": "2024-11-20T10:30:00",
      "val_accuracy": 0.85,
      "num_samples": 1250,
      "num_real_corrections": 250
    }
  },
  "learning": {
    "corrections": {
      "total": 250,
      "last_7_days": 85,
      "velocity_per_day": 12.1
    },
    "learned_rules": {
      "total": 8,
      "active": 7,
      "top_rules": [...]
    }
  }
}
```

#### `GET /metrics/neural_oracle`
Neural oracle specific metrics.

#### `GET /metrics/learning`
Learning engine statistics.

#### `GET /metrics/performance`
Component-level performance metrics.

#### `GET /metrics/realtime`
Real-time system metrics (CPU, memory).

### 4.3 Cache & Drift

#### `GET /cache/stats`
Intelligent cache statistics (L1/L2/L3 hit rates).

#### `GET /drift/status`
Data drift monitoring status.

#### `POST /drift/set_reference`
Set reference distribution for drift detection.

#### `POST /drift/check`
Check column for drift.

---

## 5. Database Schema

### 5.1 Tables

**`corrections`** - User correction records
```sql
CREATE TABLE corrections (
    id INTEGER PRIMARY KEY,
    user_id VARCHAR,
    pattern_hash VARCHAR(32) INDEX,
    statistical_fingerprint JSON,  -- Privacy-preserved!
    wrong_action VARCHAR,
    correct_action VARCHAR,
    system_confidence FLOAT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

**`learned_rules`** - Auto-generated rules from corrections
```sql
CREATE TABLE learned_rules (
    id INTEGER PRIMARY KEY,
    rule_name VARCHAR UNIQUE INDEX,
    pattern_template JSON,
    recommended_action VARCHAR,
    base_confidence FLOAT,
    support_count INTEGER,
    validation_successes INTEGER DEFAULT 0,
    validation_failures INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_updated DATETIME
);
```

**`model_versions`** - Neural oracle training history
```sql
CREATE TABLE model_versions (
    id INTEGER PRIMARY KEY,
    version_name VARCHAR UNIQUE,
    training_date DATETIME,
    validation_accuracy FLOAT,
    num_samples INTEGER,
    num_real_corrections INTEGER,
    model_path VARCHAR,
    is_active BOOLEAN DEFAULT FALSE,
    metadata JSON
);
```

### 5.2 Database Files

- **Development:** `aurora.db` (SQLite) in project root
- **Production:** PostgreSQL (set via `DATABASE_URL`)

### 5.3 Migrations

⚠️ **CRITICAL ISSUE:** Alembic is in requirements but NOT CONFIGURED.

**TODO:** Set up proper migrations:
```bash
alembic init alembic
alembic revision --autogenerate -m "Initial schema"
alembic upgrade head
```

---

## 6. Frontend/UI

### 6.1 Technology Stack

- **Framework:** Next.js 14.2.33
- **Language:** TypeScript
- **Styling:** TailwindCSS
- **Charts:** Recharts
- **Icons:** Lucide React
- **HTTP:** Axios
- **Notifications:** React Hot Toast

### 6.2 Components

**`PreprocessingPanel.tsx`**
- CSV file upload
- Manual column data input
- Preprocessing action display
- Correction submission

**`ChatbotPanel.tsx`**
- Placeholder for AI assistant
- Currently non-functional

**`MetricsDashboard.tsx`** (Recently Enhanced)
- Real-time system metrics (CPU, memory, uptime)
- Neural oracle status panel
- Adaptive learning progress panel
- Cache statistics with L1/L2/L3 breakdown
- Data drift monitoring
- Decision source breakdown (pie chart)
- Component latency bars
- Top learned rules display
- Training readiness indicators

**`Header.tsx`**
- Logo and branding
- Metrics toggle button

**`ResultCard.tsx`**
- Display preprocessing results
- Show confidence and alternatives

### 6.3 API Integration

All API calls go through axios with base URL `/api/*`:

```typescript
// Configured in next.config.js
async rewrites() {
  return [
    {
      source: '/api/:path*',
      destination: 'http://localhost:8000/:path*'
    }
  ]
}
```

### 6.4 Build & Deploy

```bash
# Development
cd frontend
npm install
npm run dev  # Runs on http://localhost:3000

# Production build
npm run build
npm start

# Export static
npm run export
```

**Current Build Status:** ✅ Builds successfully (TypeScript errors fixed)

---

## 7. Neural Oracle & Learning System

### 7.1 Neural Oracle Overview

**Model Type:** XGBoost Classifier
**Model Size:** ~50KB compressed
**Features:** 10 minimal features
**Actions:** 15+ preprocessing actions
**Inference:** <5ms average

**Features Used:**
1. null_percentage
2. unique_ratio
3. skewness
4. outlier_percentage
5. entropy
6. pattern_complexity
7. multimodality_score
8. cardinality_bucket
9. detected_dtype
10. column_name_signal

### 7.2 Training Approaches

**Option 1: Synthetic Data (Baseline)**
```bash
python scripts/train_neural_oracle.py
```
- Generates 5000 synthetic edge cases
- Good baseline (~80% accuracy)
- No domain-specific knowledge

**Option 2: From Corrections (Recommended)**
```bash
python scripts/train_from_corrections.py
```
- Uses REAL user corrections from database
- Hybrid: corrections (2x weight) + synthetic data
- Requires 50+ corrections for good results
- Genuinely learns from usage

### 7.3 Training Metadata

Saved as `models/neural_oracle_v1.json`:
```json
{
  "training_date": "2024-11-20T12:30:45",
  "num_samples": 1250,
  "num_real_corrections": 250,
  "num_synthetic": 1000,
  "real_data_weight": 2.0,
  "train_accuracy": 0.89,
  "val_accuracy": 0.85,
  "model_size_kb": 48.3,
  "avg_inference_ms": 3.2
}
```

### 7.4 Adaptive Learning Engine

**Location:** `src/learning/adaptive_engine.py`

**Key Methods:**
- `record_correction()` - Store correction in DB
- `_create_statistical_fingerprint()` - Extract privacy-preserved features
- `_hash_fingerprint()` - Create pattern hash for k-anonymity
- `_try_create_rule()` - Create rule after 5+ similar corrections
- `get_learned_rules()` - Retrieve active rules

**Privacy Guarantees:**
- ✅ NO raw data stored
- ✅ Only statistical fingerprints
- ✅ k-anonymity through pattern hashing
- ✅ Optional differential privacy (Laplace noise)

### 7.5 Current Issues

❌ **Neural Oracle Underutilized**
- With 165+ symbolic rules, confidence is rarely < 0.9
- Oracle sits idle most of the time
- **Solution:** Implement ensemble voting or second-guessing

❌ **No Automated Retraining**
- Requires manual execution
- **Solution:** Set up cron job or GitHub Actions

❌ **No A/B Testing**
- Can't compare old vs new models
- **Solution:** Implement traffic splitting

---

## 8. Current Limitations

### 8.1 Critical Issues 🔴

**1. Zero Tests**
- No unit tests, integration tests, or E2E tests
- Claims about accuracy/performance are unproven
- **Risk:** Unknown bugs in production

**2. No Database Migrations**
- Schema changes will break production
- Alembic installed but not configured
- **Risk:** Data loss on updates

**3. Singleton Pattern**
- Global state prevents horizontal scaling
- Can't run multiple instances
- **Risk:** Scaling bottleneck

**4. No CI/CD Pipeline**
- Manual testing and deployment
- No automated quality checks
- **Risk:** Human error

### 8.2 Important Issues 🟡

**5. Neural Oracle Underutilized**
- Only triggered ~10-20% of the time
- Expensive to maintain for little benefit
- **Impact:** Wasted resources

**6. CSV Parser Unproven**
- Claims "handles ANY CSV" with no proof
- Need 100+ edge case tests
- **Impact:** Unknown failure modes

**7. Monitoring Gaps**
- Basic metrics but no alerting
- No distributed tracing
- No error tracking (Sentry)
- **Impact:** Can't detect issues early

**8. Security Gaps**
- No request size limits (10GB upload = OOM)
- No rate limiting enforcement
- No security headers
- **Impact:** Vulnerable to attacks

### 8.3 Minor Issues 🟢

**9. God Classes**
- `server.py` is 1400+ lines
- Mixing concerns (API + business logic + DB)
- **Impact:** Hard to maintain

**10. Magic Numbers**
- Hardcoded thresholds everywhere (0.85, 0.9, 5, etc.)
- No documentation of why
- **Impact:** Hard to tune

**11. Documentation Overload**
- Too many README files
- Duplicated information
- **Impact:** Confusing for new developers

---

## 9. Future Roadmap

### 9.1 Immediate Priorities (Next 2 Weeks)

**Week 1: Testing Foundation**
- [ ] Write 50+ CSV parser edge case tests
- [ ] Write unit tests for symbolic engine (all 165 rules)
- [ ] Write integration tests for API endpoints
- [ ] Set up pytest with coverage reporting
- [ ] Achieve 70% test coverage

**Week 2: Production Readiness**
- [ ] Set up Alembic database migrations
- [ ] Fix neural oracle utilization (ensemble approach)
- [ ] Refactor server.py into modules
- [ ] Add request size limits
- [ ] Set up GitHub Actions CI/CD

### 9.2 Short Term (Next Month)

**Testing & Quality:**
- [ ] Benchmark suite for performance claims
- [ ] CSV robustness test suite (100+ formats)
- [ ] Load testing (1000 req/s target)
- [ ] Security audit & penetration testing

**Architecture:**
- [ ] Remove singleton pattern (dependency injection)
- [ ] Implement repository pattern for DB
- [ ] Add service layer (separate business logic)
- [ ] Add proper error handling (specific exceptions)

**Monitoring:**
- [ ] Prometheus metrics integration
- [ ] Grafana dashboards
- [ ] Sentry error tracking
- [ ] Alerting rules (accuracy drops, high latency)

### 9.3 Medium Term (Next Quarter)

**Neural Oracle Improvements:**
- [ ] Implement ensemble voting with symbolic engine
- [ ] Add A/B testing framework
- [ ] Automated weekly retraining (cron)
- [ ] Model performance tracking over time

**Scalability:**
- [ ] Containerization (Docker + Docker Compose)
- [ ] Kubernetes deployment manifests
- [ ] Redis for distributed caching
- [ ] Load balancer configuration
- [ ] Horizontal pod autoscaling

**Features:**
- [ ] User management system
- [ ] API key authentication
- [ ] Webhook notifications
- [ ] Batch job processing
- [ ] Data versioning

### 9.4 Long Term (6 Months+)

**Advanced Features:**
- [ ] Multi-table preprocessing
- [ ] Column relationship detection
- [ ] Feature engineering suggestions
- [ ] Automated pipeline generation
- [ ] ML model integration

**Enterprise:**
- [ ] SSO integration (SAML, OAuth)
- [ ] Audit logging
- [ ] Compliance features (GDPR, HIPAA)
- [ ] Multi-tenancy
- [ ] White-labeling

---

## 10. Development Workflow

### 10.1 Environment Setup

```bash
# Clone repository
git clone <repo-url>
cd AURORA-V2

# Backend setup
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env:
#   - Set JWT_SECRET_KEY (use: openssl rand -hex 32)
#   - Set ALLOWED_ORIGINS
#   - Set DATABASE_URL

# Initialize database
# (auto-initializes on first run)

# Train neural oracle (optional)
python scripts/train_neural_oracle.py

# Start backend
uvicorn src.api.server:app --reload --port 8000

# Frontend setup (separate terminal)
cd frontend
npm install
npm run dev
```

### 10.2 Development Tools

**Python:**
- Linting: `ruff check src/`
- Formatting: `black src/`
- Type checking: `mypy src/` (not set up yet)

**Frontend:**
- Linting: `npm run lint`
- Type checking: Built into Next.js build

### 10.3 Git Workflow

**Branch Naming:**
- Feature: `feature/description`
- Bugfix: `bugfix/description`
- Hotfix: `hotfix/description`

**Commit Messages:**
- Format: `type: description`
- Types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`

**Example:**
```bash
git checkout -b feature/add-rate-limiting
# Make changes
git add .
git commit -m "feat: Add rate limiting middleware"
git push origin feature/add-rate-limiting
```

### 10.4 Code Review Checklist

- [ ] Tests written and passing
- [ ] Documentation updated
- [ ] No new linting errors
- [ ] Type hints added
- [ ] Error handling present
- [ ] Performance considered
- [ ] Security implications reviewed

---

## 11. Deployment Guide

### 11.1 Development Deployment

**Backend:**
```bash
uvicorn src.api.server:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm run dev
```

**Access:**
- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs

### 11.2 Production Deployment (Current - Not Recommended)

⚠️ **WARNING:** Not production-ready yet. Follow this at your own risk.

**Backend:**
```bash
# Install production dependencies
pip install -r requirements.txt

# Set production environment variables
export DATABASE_URL="postgresql://user:pass@host:5432/aurora"
export JWT_SECRET_KEY="<strong-secret-key>"
export ALLOWED_ORIGINS="https://yourdomain.com"
export ENVIRONMENT="production"

# Run with Gunicorn
gunicorn src.api.server:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

**Frontend:**
```bash
cd frontend
npm run build
npm start
```

**Reverse Proxy (Nginx):**
```nginx
server {
    listen 80;
    server_name yourdomain.com;

    # Frontend
    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 11.3 Docker Deployment (TODO)

**Not yet implemented.** Need to create:
- `Dockerfile` for backend
- `Dockerfile` for frontend
- `docker-compose.yml` for local development
- Kubernetes manifests for production

---

## 12. Troubleshooting

### 12.1 Common Issues

**Issue: "Module not found: xgboost"**
```bash
pip install xgboost
```

**Issue: "Database connection failed"**
- Check `DATABASE_URL` in `.env`
- Ensure PostgreSQL is running (if not using SQLite)
- Check database permissions

**Issue: "CORS error in frontend"**
- Verify `ALLOWED_ORIGINS` includes frontend URL
- Check Next.js rewrites in `next.config.js`

**Issue: "Neural oracle model not loaded"**
```bash
# Train the model
python scripts/train_neural_oracle.py

# Verify model exists
ls -lh models/neural_oracle_v1.pkl
```

**Issue: "Frontend build fails"**
```bash
cd frontend
rm -rf .next node_modules package-lock.json
npm install
npm run build
```

### 12.2 Debug Mode

**Backend:**
```python
# In server.py, set DEBUG=True
DEBUG = True

# Or via environment
export DEBUG=true
uvicorn src.api.server:app --reload --log-level debug
```

**Frontend:**
```bash
# Check browser console for errors
# Check Network tab for API failures
```

### 12.3 Performance Issues

**Slow API responses:**
```python
# Check monitor metrics
curl http://localhost:8000/metrics/performance | jq

# Check component latencies
# If symbolic_engine > 10ms, investigate rules
# If neural_oracle > 50ms, model too large
```

**High memory usage:**
```python
# Check cache size
curl http://localhost:8000/cache/stats | jq

# Clear cache
curl -X POST http://localhost:8000/cache/clear
```

### 12.4 Database Issues

**SQLite locked:**
```bash
# Check for zombie processes
ps aux | grep python

# If needed, delete lock file
rm aurora.db-shm aurora.db-wal
```

**Migration needed:**
```bash
# When ready, use Alembic
alembic upgrade head
```

---

## 13. Key Files Reference

### 13.1 Backend Core

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `src/api/server.py` | Main API server | 1400+ | ⚠️ Too large |
| `src/core/preprocessor.py` | Decision pipeline | 600+ | ✅ Good |
| `src/symbolic/engine.py` | Rule engine | 800+ | ✅ Good |
| `src/neural/oracle.py` | Neural oracle | 400+ | ✅ Good |
| `src/learning/adaptive_engine.py` | Learning system | 600+ | ✅ Good |
| `src/database/models.py` | DB schema | 80 | ✅ Good |
| `src/auth/__init__.py` | JWT auth | 200 | ✅ Good |

### 13.2 Training Scripts

| File | Purpose | Status |
|------|---------|--------|
| `scripts/train_neural_oracle.py` | Train from synthetic | ✅ Works |
| `scripts/train_from_corrections.py` | Train from corrections | ✅ Works |
| `scripts/generate_synthetic_data.py` | Data generation | ✅ Works |

### 13.3 Frontend Components

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `frontend/src/pages/index.tsx` | Main page | 60 | ✅ Good |
| `frontend/src/components/MetricsDashboard.tsx` | Metrics UI | 490 | ✅ Enhanced |
| `frontend/src/components/PreprocessingPanel.tsx` | Main UI | 180 | ✅ Good |
| `frontend/src/components/ChatbotPanel.tsx` | Chatbot | 200 | ⚠️ Placeholder |

### 13.4 Documentation

| File | Purpose | Status |
|------|---------|--------|
| `README.md` | Quick start | ✅ Updated |
| `IMPLEMENTATION_STATUS.md` | Current status | ✅ Complete |
| `NEURAL_ORACLE_TRAINING.md` | Training guide | ✅ Complete |
| `docs/ARCHITECTURE_V3_PROPOSAL.md` | V3 design | ✅ Reference |
| `docs/HANDOVER.md` | This document | ✅ Current |

---

## 14. Contact & Support

### 14.1 Key Documentation

- **Quick Start:** README.md
- **Current Status:** IMPLEMENTATION_STATUS.md
- **Training:** NEURAL_ORACLE_TRAINING.md
- **This Document:** docs/HANDOVER.md

### 14.2 Getting Help

1. Check documentation (README, STATUS, TRAINING)
2. Check troubleshooting section above
3. Review code comments and docstrings
4. Open issue on GitHub

### 14.3 Handover Checklist

Before handing over this project, ensure:

**Code:**
- [ ] All changes committed and pushed
- [ ] Branch is up to date with main
- [ ] No uncommitted local changes
- [ ] Dependencies listed in requirements.txt

**Environment:**
- [ ] `.env.example` is up to date
- [ ] Database schema documented
- [ ] API endpoints documented
- [ ] Environment variables explained

**Documentation:**
- [ ] README.md updated
- [ ] This handover document complete
- [ ] Architecture diagrams current
- [ ] Known issues documented

**Testing:**
- [ ] Manual testing performed
- [ ] Critical paths verified
- [ ] Known bugs documented
- [ ] Test plan created (even if not executed)

**Deployment:**
- [ ] Deployment steps documented
- [ ] Production environment configured (if applicable)
- [ ] Monitoring set up (if applicable)
- [ ] Backup procedures documented

---

## 15. Final Notes

### What This Project Has

✅ Solid architecture and design
✅ Good separation of concerns
✅ Privacy-preserving learning (novel)
✅ Comprehensive documentation
✅ Working prototype
✅ Modern UI

### What This Project Needs

❌ **Tests** (critical priority #1)
❌ Production-grade error handling
❌ Database migrations
❌ CI/CD pipeline
❌ Monitoring & alerting
❌ Security hardening

### Honest Assessment

**Current State:** Advanced prototype (60% to production)
**Time to Production:** 4-6 weeks with dedicated effort
**Biggest Risk:** Zero tests means unknown bugs
**Biggest Strength:** Privacy-preserving learning loop

### Next Person's Priority

1. **Write tests** (unit, integration, E2E)
2. Set up Alembic migrations
3. Fix neural oracle utilization
4. Add CI/CD pipeline
5. Refactor god classes
6. Add monitoring & alerting

**Don't waste time on:**
- More documentation (enough already)
- New features (fix foundation first)
- Over-engineering (KISS principle)

---

**Document Prepared By:** Claude (AI Assistant)
**Review Status:** Complete
**Last Updated:** November 20, 2024
**Version:** 1.0

---

*This handover document should be kept up to date as the project evolves. Update it whenever major changes are made to architecture, implementation, or deployment procedures.*
