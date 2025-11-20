# AURORA V3: Implementation Roadmap

## Overview

This roadmap transforms AURORA from a **prototype** to a **production-ready, genuinely intelligent system** that learns from corrections and respects privacy.

**Total Estimated Time:** 12 weeks (3 months)
**Required Skills:** Python, PostgreSQL, Redis, Docker, FastAPI, ML fundamentals

---

## Phase 1: Foundation & Infrastructure (Weeks 1-2)

### Week 1: Core Infrastructure Setup

#### Task 1.1: Database Setup ✅ **Priority: CRITICAL**
```bash
# Set up PostgreSQL
docker-compose up -d postgres

# Create database schema
alembic revision --autogenerate -m "initial schema"
alembic upgrade head
```

**Files to create:**
- `alembic/env.py` - Database migration configuration
- `src/database/models.py` - SQLAlchemy models (CorrectionRecord, LearnedRule, etc.)
- `src/database/connection.py` - Database connection handling

**Acceptance Criteria:**
- [ ] PostgreSQL running in Docker
- [ ] Database schema created with migrations
- [ ] Connection pooling configured
- [ ] Health check endpoint returns database status

**Time:** 2 days

---

#### Task 1.2: Redis Cache Setup ✅ **Priority: CRITICAL**
```bash
# Set up Redis
docker-compose up -d redis

# Test connection
redis-cli ping
```

**Files to create:**
- `src/cache/redis_client.py` - Redis connection wrapper
- `src/cache/cache_manager.py` - Cache key management

**Acceptance Criteria:**
- [ ] Redis running in Docker
- [ ] Connection pooling configured
- [ ] Cache hit/miss metrics tracked
- [ ] TTL working correctly

**Time:** 1 day

---

#### Task 1.3: Remove Singleton Pattern ⚠️ **Priority: HIGH**

**Current problem:**
```python
# BAD: Global singleton (can't scale, can't test)
_preprocessor_instance: Optional[IntelligentPreprocessor] = None

def get_preprocessor(**kwargs):
    global _preprocessor_instance
    if _preprocessor_instance is None:
        _preprocessor_instance = IntelligentPreprocessor(**kwargs)
    return _preprocessor_instance
```

**New approach:**
```python
# GOOD: Dependency injection
from fastapi import Depends
from sqlalchemy.orm import Session

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_preprocessor(db: Session = Depends(get_db)):
    return PreprocessingServiceV3(db_session=db, cache=get_cache())
```

**Files to modify:**
- `src/core/preprocessor.py` - Remove singleton, add dependency injection
- `src/api/server.py` - Use Depends() for all endpoints

**Acceptance Criteria:**
- [ ] No global state variables
- [ ] All dependencies injected via FastAPI Depends()
- [ ] Can run multiple instances (test with docker-compose scale)
- [ ] All tests pass

**Time:** 2 days

---

#### Task 1.4: Robust CSV Parser 🆕 **Priority: HIGH**

**Implementation:**
Use the `src/core/robust_parser.py` file already created.

**Integration points:**
```python
# In API endpoint
from src.core.robust_parser import parse_csv_robust

@app.post("/upload")
async def upload_csv(file: UploadFile):
    df = parse_csv_robust(file.file)
    # Process df...
```

**Test cases:**
- [ ] UTF-8 CSV
- [ ] Latin-1 CSV
- [ ] Tab-separated
- [ ] Semicolon-separated (European)
- [ ] Quoted fields with commas
- [ ] Malformed rows (skip gracefully)
- [ ] Large files (>100MB) - stream

**Acceptance Criteria:**
- [ ] Handles all test cases without crashing
- [ ] Logs warnings for problematic rows
- [ ] Returns clean DataFrame
- [ ] Performance: <2s for 10k rows, streaming for >1M rows

**Time:** 2 days

---

### Week 2: Authentication & Security

#### Task 2.1: Implement JWT Authentication 🔒 **Priority: CRITICAL**

**Implementation:**
```python
# src/auth/jwt.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"

security = HTTPBearer()

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """Extract user from JWT token."""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"user_id": user_id}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

**Update endpoints:**
```python
@app.post("/preprocess")
async def preprocess(
    request: PreprocessRequest,
    user = Depends(get_current_user),  # ← Add this
    service = Depends(get_preprocessor)
):
    result = await service.preprocess_column(
        user_id=user["user_id"],  # ← Use authenticated user
        ...
    )
```

**Acceptance Criteria:**
- [ ] All endpoints require authentication (except /docs, /health)
- [ ] Invalid tokens rejected with 401
- [ ] Expired tokens rejected
- [ ] User ID extracted correctly
- [ ] Rate limiting per-user

**Time:** 3 days

---

#### Task 2.2: Fix CORS Properly 🔒 **Priority: CRITICAL**

**Current (INSECURE):**
```python
allow_origins=["*"],  # ← NEVER DO THIS
```

**Fixed:**
```python
# Environment-based configuration
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Only what you need
    allow_headers=["Authorization", "Content-Type"],
)
```

**Acceptance Criteria:**
- [ ] CORS restricted to specific origins
- [ ] Environment variable controls origins
- [ ] Production deployment uses HTTPS only
- [ ] OPTIONS preflight requests work

**Time:** 1 day

---

#### Task 2.3: Add Rate Limiting 🔒 **Priority: HIGH**

**Implementation:**
```python
# src/middleware/rate_limit.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["60/minute"]
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Apply to endpoints
@app.post("/preprocess")
@limiter.limit("30/minute")  # ← Add this
async def preprocess(...):
    ...
```

**Acceptance Criteria:**
- [ ] 429 status code returned when limit exceeded
- [ ] Rate limit headers in response
- [ ] Different limits for different endpoints
- [ ] Redis-backed (distributed rate limiting)

**Time:** 1 day

---

## Phase 2: Adaptive Learning System (Weeks 3-5)

### Week 3: Persistent Correction Storage

#### Task 3.1: Integrate AdaptiveLearningEngine ✨ **Priority: CRITICAL**

**Implementation:**
Use `src/learning/adaptive_engine.py` already created.

**Integration:**
```python
# src/api/server.py
@app.post("/correct")
async def submit_correction(
    request: CorrectionRequest,
    user = Depends(get_current_user),
    service = Depends(get_preprocessor)
):
    result = await service.submit_correction(
        user_id=user["user_id"],
        column_data=request.column_data,
        column_name=request.column_name,
        wrong_action=request.wrong_action,
        correct_action=request.correct_action,
        confidence=request.confidence
    )
    return result
```

**Acceptance Criteria:**
- [ ] Corrections stored in PostgreSQL
- [ ] Statistical fingerprint created (NO raw data stored)
- [ ] Pattern hash computed for similarity
- [ ] Rules created after 5+ similar corrections
- [ ] Rules visible in admin UI

**Time:** 3 days

---

#### Task 3.2: Validation Tracking ✨ **Priority: HIGH**

When a learned rule makes a recommendation, track if it was correct:

```python
@app.post("/validate")
async def validate_prediction(
    request: ValidationRequest,
    user = Depends(get_current_user),
    service = Depends(get_preprocessor)
):
    """User confirms if prediction was correct."""
    service.learning_engine.validate_prediction(
        user_id=user["user_id"],
        rule_name=request.rule_name,
        was_correct=request.was_correct
    )
    return {"status": "validated"}
```

**Acceptance Criteria:**
- [ ] Validation results stored in database
- [ ] Rule confidence adjusted based on validation
- [ ] Poor-performing rules deactivated
- [ ] Metrics tracked (success rate per rule)

**Time:** 2 days

---

### Week 4: Dynamic Confidence Adjustment

#### Task 4.1: Implement Bayesian Confidence Updates ✨ **Priority: HIGH**

**Already implemented in adaptive_engine.py**, but needs testing:

```python
def _compute_dynamic_confidence(self, rule: LearnedRule) -> float:
    """
    Bayesian update: posterior = prior × likelihood

    Starts conservative, increases with evidence.
    """
    total = rule.validation_successes + rule.validation_failures

    if total < 10:
        # Not enough data, blend with prior
        weight = total / 10
        success_rate = rule.validation_successes / total
        confidence = (1 - weight) * 0.5 + weight * success_rate
    else:
        # Enough data, trust the evidence
        confidence = rule.validation_successes / total

    return max(0.3, min(0.95, confidence))
```

**Test scenarios:**
- [ ] Rule with 5/5 successes → confidence ~0.7
- [ ] Rule with 50/50 successes → confidence ~0.98
- [ ] Rule with 3/10 successes → confidence ~0.4 (deactivated)

**Time:** 2 days

---

### Week 5: Nightly Retraining Pipeline

#### Task 5.1: Set Up Celery for Background Tasks 🔧 **Priority: HIGH**

```python
# src/tasks/celery_app.py
from celery import Celery
from celery.schedules import crontab

app = Celery(
    'aurora',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

app.conf.beat_schedule = {
    'retrain-nightly': {
        'task': 'tasks.retrain_models',
        'schedule': crontab(hour=2, minute=0),  # 2 AM daily
    },
}

@app.task
def retrain_models():
    """Retrain ML models on new corrections."""
    from src.learning.training_pipeline import ContinuousLearningPipeline

    pipeline = ContinuousLearningPipeline(
        db_url=os.getenv("DATABASE_URL"),
        mlflow_uri=os.getenv("MLFLOW_URI")
    )

    pipeline.run_nightly_training()
```

**Acceptance Criteria:**
- [ ] Celery worker running
- [ ] Beat scheduler running
- [ ] Retraining runs nightly
- [ ] Logs accessible
- [ ] Failures trigger alerts

**Time:** 3 days

---

#### Task 5.2: Implement Training Pipeline 🧠 **Priority: HIGH**

```python
# src/learning/training_pipeline.py
class ContinuousLearningPipeline:
    def run_nightly_training(self):
        # 1. Fetch new corrections since last training
        corrections = self._fetch_new_corrections()

        if len(corrections) < 100:
            logger.info("Not enough data, skipping")
            return

        # 2. Prepare training data
        X, y = self._prepare_features(corrections)

        # 3. Train ensemble
        models = self._train_ensemble(X, y)

        # 4. Evaluate
        metrics = self._evaluate(models, X_test, y_test)

        # 5. A/B test against production
        if self._should_promote(metrics):
            self._promote_to_staging(models)
```

**Acceptance Criteria:**
- [ ] Fetches corrections from database
- [ ] Trains XGBoost + LightGBM ensemble
- [ ] Evaluates on holdout set
- [ ] Logs to MLflow
- [ ] Only promotes if better than current

**Time:** 4 days

---

## Phase 3: Privacy Guarantees (Weeks 6-7)

### Week 6: Differential Privacy

#### Task 6.1: Audit Data Storage 🔍 **Priority: CRITICAL**

**Goal:** Ensure NO raw data is ever stored.

**Checklist:**
- [ ] CorrectionRecord table: NO column_data field
- [ ] Only statistical_fingerprint stored
- [ ] Fingerprint contains ONLY aggregated stats (mean, std, buckets)
- [ ] No individual values can be reconstructed

**Action:** Code audit + penetration test

**Time:** 2 days

---

#### Task 6.2: Implement Differential Privacy Properly 🔒 **Priority: HIGH**

**Currently implemented** in adaptive_engine.py, but needs validation:

```python
def _add_laplace_noise(self, fingerprint: Dict, epsilon: float = 1.0) -> Dict:
    """Add calibrated Laplace noise for ε-DP."""
    for key in ['skew_bucket', 'kurtosis_bucket', 'entropy_bucket']:
        if key in fingerprint:
            sensitivity = 1.0
            noise = np.random.laplace(0, sensitivity / epsilon)
            fingerprint[key] = int(fingerprint[key] + noise)

    return fingerprint
```

**Validation:**
- [ ] Epsilon budget tracked
- [ ] Noise calibrated correctly
- [ ] Privacy guarantee holds: ε < 1.0
- [ ] Composition theorem applied for multiple queries

**Test:**
```python
def test_differential_privacy():
    # Two datasets differing by one record
    stats1 = compute_stats(dataset1)
    stats2 = compute_stats(dataset2)  # Same except one row

    # Fingerprints should be indistinguishable
    fp1 = create_fingerprint(stats1)
    fp2 = create_fingerprint(stats2)

    # Check ε-indistinguishability
    assert privacy_distance(fp1, fp2) < epsilon
```

**Time:** 3 days

---

### Week 7: Privacy Reporting

#### Task 7.1: Add Privacy Dashboard for Users 📊 **Priority: MEDIUM**

Users should see what data is stored about them:

```python
@app.get("/privacy/report")
async def get_privacy_report(user = Depends(get_current_user)):
    """Show user what data we store."""
    db = get_db()

    # Get correction count
    corrections = db.query(CorrectionRecord).filter(
        CorrectionRecord.user_id == user["user_id"]
    ).count()

    # Get learned rules
    rules = db.query(LearnedRule).filter(
        LearnedRule.user_id == user["user_id"]
    ).count()

    return {
        "data_stored": {
            "corrections": corrections,
            "learned_rules": rules,
            "raw_data_stored": 0,  # ← Always zero!
        },
        "privacy_guarantee": {
            "epsilon": 1.0,
            "mechanism": "Laplace",
            "certification": "Differential Privacy (ε, δ)-DP"
        },
        "data_retention": "90 days",
        "can_delete": True
    }
```

**Acceptance Criteria:**
- [ ] Users can view what's stored
- [ ] Privacy guarantee explained
- [ ] "Delete my data" button works
- [ ] GDPR compliant

**Time:** 2 days

---

## Phase 4: Production Infrastructure (Weeks 8-9)

### Week 8: Monitoring & Observability

#### Task 8.1: Set Up Prometheus + Grafana 📈 **Priority: HIGH**

```yaml
# docker-compose.yml
prometheus:
  image: prom/prometheus:latest
  volumes:
    - ./prometheus.yml:/etc/prometheus/prometheus.yml
  ports:
    - "9090:9090"

grafana:
  image: grafana/grafana:latest
  ports:
    - "3000:3000"
  environment:
    - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
```

**Metrics to track:**
- Request latency (P50, P95, P99)
- Error rate
- Cache hit rate
- Learning rate (corrections per day)
- Rule performance (success rate)

**Dashboards:**
1. System health
2. Learning progress
3. User activity
4. Privacy metrics

**Time:** 3 days

---

#### Task 8.2: Add Structured Logging 📝 **Priority: HIGH**

```python
# Replace print() with proper logging
import structlog

logger = structlog.get_logger()

logger.info(
    "preprocessing_completed",
    user_id=user_id,
    column_name=column_name,
    action=result.action,
    confidence=result.confidence,
    latency_ms=latency
)
```

**Log aggregation:** Use Loki or Elasticsearch

**Time:** 2 days

---

### Week 9: Deployment

#### Task 9.1: Docker Compose for Full Stack 🐳 **Priority: CRITICAL**

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/aurora
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    deploy:
      replicas: 3  # Scale horizontally

  postgres:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine

  celery_worker:
    build: .
    command: celery -A src.tasks worker
    depends_on:
      - redis
      - postgres

  celery_beat:
    build: .
    command: celery -A src.tasks beat
    depends_on:
      - redis

  prometheus:
    image: prom/prometheus:latest

  grafana:
    image: grafana/grafana:latest
```

**Time:** 2 days

---

#### Task 9.2: CI/CD Pipeline ⚙️ **Priority: MEDIUM**

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: |
          pip install -r requirements.txt
          pytest tests/ --cov=src --cov-report=term

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to staging
        run: |
          docker-compose -f docker-compose.staging.yml up -d

      - name: Run smoke tests
        run: |
          curl http://staging.aurora.com/health

      - name: Deploy to production (manual approval)
        if: github.ref == 'refs/heads/main'
        uses: actions/deploy@v1
```

**Time:** 2 days

---

## Phase 5: Intelligence & Optimization (Weeks 10-12)

### Week 10: Ensemble Meta-Learner

#### Task 10.1: Train Ensemble Models 🧠 **Priority: MEDIUM**

**Models:**
1. XGBoost (best for structured data)
2. LightGBM (faster inference)
3. Random Forest (robust baseline)

**Ensemble strategy:**
```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('rf', rf_model)
    ],
    voting='soft',
    weights=[2, 2, 1]  # XGB and LGB weighted higher
)
```

**Time:** 4 days

---

### Week 11: Active Learning

#### Task 11.1: Implement Active Learning for Ambiguous Cases 🎯 **Priority: MEDIUM**

When system is uncertain (confidence < 0.6), ask user:

```python
@app.post("/preprocess")
async def preprocess(...):
    result = await service.preprocess_column(...)

    if result.confidence < 0.6:
        # Trigger active learning
        return {
            "status": "uncertain",
            "message": "We're not sure about this. Can you help?",
            "top_options": [
                {"action": "log_transform", "confidence": 0.55},
                {"action": "robust_scale", "confidence": 0.52},
                {"action": "keep_as_is", "confidence": 0.48}
            ],
            "reasoning": result.explanation
        }

    return result
```

**UI:** Show 3 options, user picks one → system learns

**Time:** 3 days

---

### Week 12: Final Testing & Documentation

#### Task 12.1: Load Testing 🔬 **Priority: HIGH**

```bash
# Use locust.io for load testing
locust -f tests/load/locustfile.py --users 1000 --spawn-rate 10
```

**Targets:**
- [ ] 1000 concurrent users
- [ ] P95 latency < 500ms
- [ ] Error rate < 0.1%
- [ ] No memory leaks

**Time:** 2 days

---

#### Task 12.2: Documentation 📚 **Priority: HIGH**

**Documents to create:**
- [ ] API documentation (Swagger/OpenAPI)
- [ ] User guide
- [ ] Developer guide
- [ ] Deployment guide
- [ ] Privacy policy
- [ ] Security audit report

**Time:** 3 days

---

## Success Metrics

Track these to measure if the system is working:

### Learning Effectiveness
- **Metric:** Recommendation accuracy over time
- **Target:** 70% → 90% after 3 months of use
- **How:** Track validation results

### Adaptation Speed
- **Metric:** Corrections needed to reach 90% accuracy in new domain
- **Target:** < 100 corrections
- **How:** Track per-domain accuracy

### Privacy Guarantee
- **Metric:** Differential privacy budget (ε)
- **Target:** ε < 1.0
- **How:** Automated privacy accounting

### User Satisfaction
- **Metric:** % of recommendations accepted without correction
- **Target:** > 85%
- **How:** Track correction rate

### System Performance
- **Metric:** P95 latency
- **Target:** < 500ms
- **How:** Prometheus metrics

---

## Risk Mitigation

### Risk 1: Learning is too slow
**Mitigation:** Start with larger pre-trained symbolic rule set

### Risk 2: Privacy breach
**Mitigation:** Regular security audits, penetration testing

### Risk 3: Performance degrades
**Mitigation:** Circuit breakers, automatic rollback, A/B testing

### Risk 4: Users don't provide corrections
**Mitigation:** Gamification, show impact of corrections, make it easy

---

## Deliverables Checklist

At the end of 12 weeks:

**Infrastructure:**
- [ ] PostgreSQL + Redis deployed
- [ ] Horizontal scaling working (test with 3+ replicas)
- [ ] CI/CD pipeline deployed
- [ ] Monitoring dashboards live

**Security:**
- [ ] Authentication working (JWT)
- [ ] CORS properly configured
- [ ] Rate limiting active
- [ ] Security audit passed

**Learning:**
- [ ] Corrections stored persistently
- [ ] Rules created automatically
- [ ] Confidence adjusts dynamically
- [ ] Nightly retraining works

**Privacy:**
- [ ] No raw data stored (audit passed)
- [ ] Differential privacy implemented
- [ ] Privacy dashboard for users
- [ ] GDPR compliant

**Performance:**
- [ ] P95 < 500ms
- [ ] Handles 1000 concurrent users
- [ ] Load tests passed

**Documentation:**
- [ ] API docs complete
- [ ] Deployment guide written
- [ ] User guide complete

---

## Next Steps After Completion

Once the 12-week plan is done:

1. **Beta testing** with 10-20 users
2. **Gather feedback** and iterate
3. **Publish research paper** on privacy-preserving adaptive learning
4. **Open source** (if desired) to build community
5. **Scale to production** with paying customers

---

## Estimated Costs (Monthly)

**Development:**
- Developer time: $10k-15k/month (1 senior engineer)

**Infrastructure:**
- AWS/GCP hosting: $200-500/month
- PostgreSQL (managed): $50-100/month
- Redis (managed): $30-50/month
- Monitoring: $50/month

**Total: ~$10,500/month for 3 months = $31,500**

**ROI:** If this becomes a SaaS product at $50/user/month with 100 users, it pays for itself in 6 months.

---

## Conclusion

This roadmap transforms AURORA from a **prototype** into a **production-ready, genuinely novel system**.

**The key innovations:**
1. **Adaptive learning** - Gets smarter with every correction
2. **Privacy-first** - Never stores raw data, formal DP guarantees
3. **Domain-specific** - Learns per-user/organization patterns
4. **Explainable** - Every decision has provable reasoning

**This IS defensible. This IS novel. This IS valuable.**

Now go build it. 🚀
