# AURORA V2 Architecture

## System Overview

AURORA V2 is a three-layer intelligent preprocessing system that combines symbolic rules, machine learning, and adaptive learning.

---

## Architecture Diagram

```
┌───────────────────────────────────────────────────────────────────┐
│                         USER REQUEST                              │
│                  (Upload CSV, Get Recommendations)                │
└────────────────────────────┬──────────────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────────────┐
│                        FASTAPI BACKEND                            │
│                     (src/api/server.py)                           │
└────────────────────────────┬──────────────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────────────┐
│                   INTELLIGENT PREPROCESSOR                        │
│                   (src/core/preprocessor.py)                      │
│                                                                   │
│   ┌───────────────────────────────────────────────────────────┐  │
│   │  LAYER 1: SYMBOLIC ENGINE (Primary Decision Maker)        │  │
│   │  ────────────────────────────────────────────────────     │  │
│   │  • 185+ expert-crafted rules                              │  │
│   │  • Priority-based evaluation (200 → 10)                   │  │
│   │  • Conservative thresholds (80% null, 99.5% unique)       │  │
│   │  • Handles 85% of cases with confidence >0.7              │  │
│   │  • Speed: <1ms per column                                 │  │
│   │                                                            │  │
│   │  Rule Categories:                                         │  │
│   │  ├─ ID Detection (priority 200)                           │  │
│   │  ├─ Safety Net (priority 200)                             │  │
│   │  ├─ Year/Temporal (priority 140)                          │  │
│   │  ├─ Numeric Scaling (priority 100)                        │  │
│   │  ├─ Categorical Encoding (priority 80)                    │  │
│   │  └─ Null Handling (priority 50)                           │  │
│   └───────────────────────────────────────────────────────────┘  │
│                             │                                     │
│                             │ confidence < 0.7                    │
│                             ▼                                     │
│   ┌───────────────────────────────────────────────────────────┐  │
│   │  LAYER 2: NEURAL ORACLE (Edge Case Handler)              │  │
│   │  ────────────────────────────────────────────────────     │  │
│   │  • XGBoost classifier (50 boosters)                       │  │
│   │  • Trained on 149 LLM-validated examples                  │  │
│   │  • 75.9% validation accuracy                              │  │
│   │  • Handles ambiguous cases                                │  │
│   │  • Speed: ~5ms per column                                 │  │
│   │                                                            │  │
│   │  Features (10):                                           │  │
│   │  ├─ null_percentage                                       │  │
│   │  ├─ unique_ratio                                          │  │
│   │  ├─ skewness                                              │  │
│   │  ├─ outlier_percentage                                    │  │
│   │  ├─ entropy                                               │  │
│   │  ├─ pattern_complexity                                    │  │
│   │  ├─ multimodality_score                                   │  │
│   │  ├─ cardinality_bucket                                    │  │
│   │  ├─ detected_dtype                                        │  │
│   │  └─ column_name_signal                                    │  │
│   └───────────────────────────────────────────────────────────┘  │
│                             │                                     │
│                             ▼                                     │
│   ┌───────────────────────────────────────────────────────────┐  │
│   │  LAYER 3: ADAPTIVE LEARNING (Continuous Improvement)     │  │
│   │  ────────────────────────────────────────────────────     │  │
│   │  • Learns from user corrections                           │  │
│   │  • Creates new rules after 10 consistent corrections      │  │
│   │  • LLM validates patterns before deployment               │  │
│   │  • Stores corrections in SQLite database                  │  │
│   │                                                            │  │
│   │  Learning Process:                                        │  │
│   │  1. User corrects decision                                │  │
│   │  2. System stores correction                              │  │
│   │  3. After 10 similar corrections → propose new rule       │  │
│   │  4. LLM validates rule (confidence ≥0.85)                 │  │
│   │  5. Deploy rule to symbolic engine                        │  │
│   └───────────────────────────────────────────────────────────┘  │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING RESULT                           │
│  ────────────────────────────────────────────────────────────     │
│  • action: PreprocessingAction                                    │
│  • confidence: float (0.0-1.0)                                    │
│  • explanation: str                                               │
│  • source: 'symbolic' | 'neural' | 'learned'                      │
│  • alternatives: List[(action, confidence)]                       │
│  • validation: Optional[ValidationResult]                         │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│                      NEXT.JS FRONTEND                             │
│                  (Display & User Interaction)                     │
└───────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Symbolic Engine

**Location:** `src/symbolic/engine.py`

**Responsibilities:**
- Primary decision maker
- Evaluates 185+ rules in priority order
- Returns action + confidence + explanation

**Rule Structure:**
```python
@dataclass
class PreprocessingRule:
    priority: int  # Higher = evaluated first
    
    def matches(self, column, column_name, stats) -> bool:
        # Check if rule applies
        pass
    
    def apply(self, column, column_name, stats) -> RuleResult:
        # Return action + confidence + explanation
        pass
```

**Key Rules:**
- `CONSERVATIVE_FALLBACK` (priority 200) - Safety net
- `KEEP_IF_PRIMARY_KEY` (priority 200) - ID detection
- `KEEP_IF_ORDINAL_OR_YEAR` (priority 140) - Temporal data
- `STANDARD_SCALE_NUMERIC` (priority 100) - Numeric scaling
- `ONEHOT_LOW_CARDINALITY` (priority 80) - Categorical encoding

### 2. Neural Oracle

**Location:** `src/neural/oracle.py`

**Model:** XGBoost Classifier
- 50 boosters
- Multi-class classification (6-8 actions)
- Trained on 149 validated examples
- 75.9% validation accuracy

**Training Data:**
- 18 diverse datasets
- LLM-validated labels (Groq API)
- Strict confidence threshold (0.85)
- Only 7 high-confidence corrections

**Features:**
- Statistical (null%, unique ratio, skewness, outliers)
- Information theory (entropy, pattern complexity)
- Distribution (multimodality)
- Metadata (dtype, column name signals)

### 3. Adaptive Learning

**Location:** `src/learning/adaptive_engine.py`

**Database:** SQLite (`aurora.db`)

**Tables:**
- `user_corrections` - Stores all corrections
- `learned_rules` - Validated patterns
- `rule_performance` - Tracks rule effectiveness

**Learning Pipeline:**
1. User makes correction
2. Store in database
3. Identify patterns (≥10 similar corrections)
4. Validate with LLM (confidence ≥0.85)
5. Generate rule code
6. Deploy to symbolic engine

---

## Data Flow

### Preprocessing Request

```
1. User uploads CSV
2. Frontend sends to /api/preprocess
3. Backend calls IntelligentPreprocessor.preprocess_column()
4. For each column:
   a. Extract features
   b. Symbolic engine evaluates
   c. If confidence ≥ 0.7: Return symbolic decision
   d. Else: Neural oracle predicts
   e. Return final decision
5. Frontend displays results
```

### User Correction

```
1. User corrects decision
2. Frontend sends to /api/correct
3. Backend stores in database
4. Adaptive engine checks for patterns
5. If pattern found (≥10 corrections):
   a. LLM validates pattern
   b. Generate new rule
   c. Add to symbolic engine
6. Return success
```

---

## Performance Characteristics

| Component | Latency | Accuracy | Coverage |
|-----------|---------|----------|----------|
| Symbolic Engine | <1ms | 85-95% | 85% |
| Neural Oracle | ~5ms | 75.9% | 15% |
| Hybrid System | <10ms | ~92% | 100% |

---

## Technology Stack

**Backend:**
- Python 3.10+
- FastAPI (async API)
- XGBoost (neural oracle)
- Pandas, NumPy (data processing)
- SQLite (adaptive learning)

**Frontend:**
- Next.js 14 (React framework)
- TypeScript (type safety)
- TailwindCSS (styling)
- Zustand (state management)

**ML/AI:**
- XGBoost (classification)
- Groq API (LLM validation)
- SHAP (explainability)

---

## Scalability

**Current Limits:**
- Columns: Unlimited
- Rows: 1M+ (memory dependent)
- Concurrent users: 100+ (FastAPI async)

**Bottlenecks:**
- Neural oracle: ~5ms per column
- Database writes: ~10ms per correction

**Optimization:**
- Caching (planned)
- Batch processing (planned)
- GPU acceleration (future)

---

## Security

**API:**
- CORS enabled (configurable origins)
- Rate limiting (planned)
- Input validation (FastAPI Pydantic)

**Data:**
- No data persistence (stateless)
- User corrections stored locally
- No external data transmission

---

**Version:** 2.0  
**Last Updated:** 2024-11-29
