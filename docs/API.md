# API Reference

## Overview

AURORA V2 provides a REST API for data preprocessing recommendations.

**Base URL:** `http://localhost:8000`

---

## Endpoints

### 1. Health Check

**GET** `/`

Check if the API is running.

**Response:**
```json
{
  "message": "AURORA V2 API is running",
  "version": "2.0"
}
```

---

### 2. Preprocess Dataset

**POST** `/api/preprocess`

Get preprocessing recommendations for a dataset.

**Request Body:**
```json
{
  "data": {
    "column1": [1, 2, 3, null, 5],
    "column2": ["a", "b", "c", "d", "e"]
  },
  "options": {
    "target_column": "column2",  // optional
    "confidence_threshold": 0.7   // optional, default: 0.7
  }
}
```

**Response:**
```json
{
  "recommendations": {
    "column1": {
      "action": "standard_scale",
      "confidence": 0.95,
      "explanation": "Numeric column with normal distribution",
      "source": "symbolic",
      "alternatives": [
        {"action": "robust_scale", "confidence": 0.85},
        {"action": "log_transform", "confidence": 0.60}
      ]
    },
    "column2": {
      "action": "onehot_encode",
      "confidence": 0.88,
      "explanation": "Low cardinality categorical (5 unique values)",
      "source": "neural"
    }
  },
  "metadata": {
    "total_columns": 2,
    "processing_time_ms": 45,
    "neural_oracle_used": 1
  }
}
```

---

### 3. Submit Correction

**POST** `/api/correct`

Submit a user correction to improve the system.

**Request Body:**
```json
{
  "column_name": "price",
  "original_action": "standard_scale",
  "corrected_action": "log_transform",
  "column_stats": {
    "dtype": "float64",
    "null_pct": 0.05,
    "unique_ratio": 0.95
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Correction stored successfully",
  "pattern_detected": false,
  "corrections_count": 3
}
```

---

### 4. Get Action Library

**GET** `/api/actions`

Get all available preprocessing actions.

**Response:**
```json
{
  "actions": [
    {
      "name": "keep_as_is",
      "description": "Preserve column unchanged",
      "category": "preservation",
      "use_cases": ["IDs", "timestamps", "ordinal data"]
    },
    {
      "name": "standard_scale",
      "description": "Normalize to mean=0, std=1",
      "category": "scaling",
      "use_cases": ["Normally distributed numeric data"]
    }
    // ... more actions
  ]
}
```

---

### 5. Get System Stats

**GET** `/api/stats`

Get system statistics.

**Response:**
```json
{
  "symbolic_rules": 185,
  "neural_oracle_accuracy": 0.759,
  "total_corrections": 127,
  "learned_rules": 3,
  "uptime_seconds": 3600
}
```

---

## Data Types

### PreprocessingAction

```typescript
enum PreprocessingAction {
  KEEP_AS_IS = "keep_as_is",
  DROP_COLUMN = "drop_column",
  STANDARD_SCALE = "standard_scale",
  ROBUST_SCALE = "robust_scale",
  LOG_TRANSFORM = "log_transform",
  ONEHOT_ENCODE = "onehot_encode",
  LABEL_ENCODE = "label_encode",
  HASH_ENCODE = "hash_encode",
  FILL_NULL_MEAN = "fill_null_mean",
  FILL_NULL_MEDIAN = "fill_null_median",
  FILL_NULL_MODE = "fill_null_mode",
  FILL_NULL_FORWARD = "fill_null_forward"
}
```

### Recommendation

```typescript
interface Recommendation {
  action: PreprocessingAction;
  confidence: number;  // 0.0 - 1.0
  explanation: string;
  source: "symbolic" | "neural" | "learned";
  alternatives?: Array<{
    action: PreprocessingAction;
    confidence: number;
  }>;
  validation?: {
    score: number;
    passed: boolean;
    warnings?: string[];
  };
}
```

---

## Error Responses

### 400 Bad Request

```json
{
  "detail": "Invalid input data format"
}
```

### 500 Internal Server Error

```json
{
  "detail": "Preprocessing failed: <error message>"
}
```

---

## Rate Limiting

**Current:** No rate limiting

**Planned:** 100 requests/minute per IP

---

## CORS

**Allowed Origins:**
- `http://localhost:3000` (development)
- `https://aurora-v2.app` (production)

**Allowed Methods:** GET, POST, OPTIONS

---

## Authentication

**Current:** None (open API)

**Planned:** API key authentication

---

## Examples

### Python

```python
import requests

# Preprocess dataset
response = requests.post(
    "http://localhost:8000/api/preprocess",
    json={
        "data": {
            "price": [10, 20, 30, 40, 50],
            "category": ["A", "B", "A", "C", "B"]
        }
    }
)

recommendations = response.json()["recommendations"]
print(recommendations["price"]["action"])  # "standard_scale"
```

### JavaScript

```javascript
// Preprocess dataset
const response = await fetch('http://localhost:8000/api/preprocess', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    data: {
      price: [10, 20, 30, 40, 50],
      category: ['A', 'B', 'A', 'C', 'B']
    }
  })
});

const { recommendations } = await response.json();
console.log(recommendations.price.action);  // "standard_scale"
```

### cURL

```bash
# Preprocess dataset
curl -X POST http://localhost:8000/api/preprocess \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "price": [10, 20, 30, 40, 50]
    }
  }'
```

---

**Version:** 2.0  
**Last Updated:** 2024-11-29
