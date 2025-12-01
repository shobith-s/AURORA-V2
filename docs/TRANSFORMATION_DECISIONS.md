# Transformation Decisions

This document explains how AURORA makes preprocessing decisions for any CSV column.

---

## Decision Tree

```
Column → Type Detection → Target Check → Validation → Safe Transform
   │          │               │              │             │
   │          ▼               ▼              ▼             ▼
   │    ┌─────────┐     ┌─────────┐    ┌─────────┐   ┌──────────┐
   │    │ empty?  │─Yes─│ DROP    │    │ VALID?  │───│ EXECUTE  │
   │    └────┬────┘     └─────────┘    └────┬────┘   └──────────┘
   │         │No                            │No
   │         ▼                              ▼
   │    ┌─────────┐                   ┌──────────┐
   │    │  URL?   │─Yes─ DROP         │ FALLBACK │
   │    └────┬────┘                   └──────────┘
   │         │No
   │         ▼
   │    ┌─────────┐
   │    │datetime?│─Yes─ EXTRACT
   │    └────┬────┘
   │         │No
   │         ▼
   │    ... (continue for all types)
```

---

## Type-Based Decisions

### Empty Columns
**Detection**: >90% null values
**Decision**: DROP
**Reasoning**: No information content, prevents downstream errors

```python
# Example
col = pd.Series([None, None, None, None])
# → drop_column (confidence: 0.98)
```

### URL Columns
**Detection**: HTTP/HTTPS pattern match, name hints (url, link, photo, image)
**Decision**: DROP
**Reasoning**: URLs are not directly useful for ML, may contain private data

```python
# Example
col = pd.Series(['https://example.com/1.jpg', 'https://example.com/2.jpg'])
# → drop_column (confidence: 0.95)
```

### DateTime Columns
**Detection**: ISO pattern, common date formats, datetime dtype
**Decision**: EXTRACT (year, month, day, dayofweek)
**Reasoning**: Temporal features are more useful than raw timestamps

```python
# Example
col = pd.Series(['2023-01-15', '2023-06-20', '2023-12-31'])
# → datetime_extract (confidence: 0.95)
```

### Email Columns
**Detection**: Email pattern match, name hints (email, mail)
**Decision**: DROP
**Reasoning**: Unique identifiers cause data leakage, privacy concerns

```python
# Example
col = pd.Series(['user@example.com', 'admin@company.org'])
# → drop_column (confidence: 0.96)
```

### Phone Columns
**Detection**: Phone patterns (10+ digits, formatted), name hints (phone, mobile, contact)
**Decision**: DROP
**Reasoning**: Unique identifiers cause data leakage, privacy concerns

```python
# Example
col = pd.Series(['9876543210', '9988776655'])
# → drop_column (confidence: 0.96)
```

### Identifier Columns
**Detection**: High uniqueness (>95%), ID keywords (id, uuid, key, code)
**Decision**: DROP
**Reasoning**: Unique identifiers cause data leakage, no predictive value

```python
# Example
col = pd.Series(['id_001', 'id_002', 'id_003'])
# → drop_column (confidence: 0.97)
```

### Boolean Columns
**Detection**: Binary values (True/False, Yes/No, 0/1), ≤3 unique values
**Decision**: PARSE_BOOLEAN
**Reasoning**: Convert to numeric 0/1 for ML compatibility

```python
# Example
col = pd.Series(['Yes', 'No', 'Yes', 'No'])
# → parse_boolean (confidence: 0.95)
```

### Numeric Columns
**Detection**: Numeric dtype, numeric string patterns
**Decision**: Depends on distribution
- Normal distribution → standard_scale
- High skewness + non-negative → log1p_transform
- Outliers → robust_scale

```python
# Example 1: Normal distribution
col = pd.Series([1, 2, 3, 4, 5])
# → standard_scale or keep_as_is (confidence: 0.90)

# Example 2: Skewed distribution
col = pd.Series([15000, 85000, 200000, 500000])
# → log1p_transform (confidence: 0.92)
```

### Categorical Columns
**Detection**: Low cardinality (<50), moderate unique ratio (<50%)
**Decision**: Depends on cardinality
- ≤10 categories → onehot_encode
- 11-50 categories → frequency_encode or onehot_encode
- >50 categories → frequency_encode

```python
# Example
col = pd.Series(['Active', 'Inactive', 'Pending'] * 10)
# → onehot_encode (confidence: 0.85)
```

### Text Columns
**Detection**: High cardinality, spaces/words, name hints (name, title, description)
**Decision**: text_clean (lowercase, normalize whitespace)
**Reasoning**: Prepare for downstream NLP or drop if not needed

```python
# Example
col = pd.Series(['John Smith', 'Jane Doe', 'Bob Wilson'])
# → text_clean (confidence: 0.88)
```

---

## Target Variable Protection

### Detection Criteria
1. **Keyword Match**: target, label, class, y, price, sales, churn, fraud
2. **Position**: Last column (with binary/target-like properties)
3. **Statistical**: Binary (0/1), low-cardinality multiclass

### Protection Rules
- **Never Drop**: Target columns are never dropped
- **Never Bin**: Binning destroys predictive value
- **Never Clip Outliers**: May contain important edge cases
- **Scaling Allowed**: With warning, may be intentional

```python
# Example: selling_price detected as target
col = pd.Series([450000, 350000, 550000])
# Detection: is_target=True, confidence=0.95
# → keep_as_is (protected)
```

---

## Validation Examples

### Valid Transformations

```python
# Numeric + standard_scale = VALID
col = pd.Series([1, 2, 3, 4, 5])
validate('standard_scale') → is_valid=True

# Categorical + onehot_encode = VALID (if cardinality ≤ 50)
col = pd.Series(['A', 'B', 'C'] * 10)
validate('onehot_encode') → is_valid=True
```

### Invalid Transformations (Blocked)

```python
# Text + standard_scale = INVALID
col = pd.Series(['John', 'Jane', 'Bob'])
validate('standard_scale') → is_valid=False
# Recommended: text_clean

# Target + binning = INVALID
col = pd.Series([100, 200, 300])
validate('binning_equal_width', is_target=True) → is_valid=False
# Recommended: keep_as_is

# High cardinality + onehot = INVALID
col = pd.Series([f'cat_{i}' for i in range(100)])
validate('onehot_encode') → is_valid=False
# Recommended: frequency_encode
```

### Override Transformations

```python
# Log transform with zeros → Override to log1p
col = pd.Series([0, 1, 10, 100])
validate('log_transform') → override to log1p_transform

# Empty column → Override to drop
col = pd.Series([None, None, None])
validate('standard_scale') → override to drop_column
```

---

## Safety Guarantees

### No Crashes
All safe transforms include:
1. Input validation
2. Type checking
3. Error handling
4. Graceful fallbacks

```python
# Example: safe_log_transform
def safe_log_transform(column):
    # 1. Validate numeric
    if not is_numeric(column):
        return fallback_to_keep_as_is()
    
    # 2. Check for negative values
    if column.min() < 0:
        shift_and_warn()
    
    # 3. Use log1p (handles zeros)
    result = np.log1p(column)
    
    # 4. Verify no infinities
    assert not np.isinf(result).any()
    
    return result
```

### No Data Leakage
Identifiers are always dropped:
- Unique IDs (customer_id, transaction_id)
- Email addresses
- Phone numbers
- UUIDs/hashes

### No Target Destruction
Targets are protected from:
- Dropping (would lose target)
- Binning (destroys distribution)
- Clipping (removes important outliers)

---

## Edge Cases

### Empty Series
```python
col = pd.Series([], dtype=float)
# → keep_as_is (status=SKIPPED)
```

### Single Value
```python
col = pd.Series([5])
# Standard scale → keep_as_is (zero variance)
```

### All Unique Text (Names)
```python
col = pd.Series(['John', 'Jane', 'Bob', 'Alice'])
# High uniqueness but text-like name
# → text_clean (not identifier)
```

### Numeric Strings
```python
col = pd.Series(['100', '200', '300'])
# Detected as numeric (95% success rate)
# → numeric transformation
```

---

## Confidence Levels

| Confidence | Meaning | Action |
|------------|---------|--------|
| 0.95+ | Very High | Execute immediately |
| 0.80-0.95 | High | Execute with logging |
| 0.60-0.80 | Medium | Execute with warning |
| 0.40-0.60 | Low | Suggest manual review |
| <0.40 | Very Low | Default to keep_as_is |

---

## Explanation Format

Every decision includes:

```json
{
  "action": "drop_column",
  "confidence": 0.97,
  "source": "symbolic",
  "explanation": "Identifier column 'customer_id' should be dropped to prevent data leakage",
  "alternatives": [
    ["keep_as_is", 0.50],
    ["hash_encode", 0.45]
  ],
  "context": {
    "semantic_type": "identifier",
    "unique_ratio": 1.0,
    "column_name": "customer_id"
  }
}
```
