# Learning System: Training vs Production Phases

**Date:** November 22, 2025
**Status:** ✅ Optimized and Separated

---

## Overview

The AURORA learning system now implements **training/production phase separation** to prevent premature decision-making from insufficient data.

### The Problem (Before)

❌ **Old behavior:**
- User submits 2 corrections → System computes adjustment → **Immediately affects all future decisions**
- Learning system was making decisions without enough training data
- User feedback: *"learning system are taking decisions now itself let them learn for a while"*

✅ **New behavior:**
- User submits 2 corrections → System computes adjustment → **Stores for later use (TRAINING)**
- User submits 10 corrections → System activates adjustment → **Now affects decisions (PRODUCTION)**
- Learning system trains first, then deploys

---

## Architecture

```
User Correction
    ↓
Record to Pattern Bucket
    ↓
┌─────────────────────────────────────────┐
│ TRAINING PHASE (2-9 corrections)        │
│                                         │
│ ✓ Record corrections                   │
│ ✓ Compute adjustments                  │
│ ✗ DON'T affect decisions yet           │
└─────────────────────────────────────────┘
    ↓ (at 10 corrections)
┌─────────────────────────────────────────┐
│ PRODUCTION PHASE (10+ corrections)      │
│                                         │
│ ✓ Apply adjustments to decisions       │
│ ✓ Boost/reduce confidence              │
│ ✓ Influence symbolic rules              │
└─────────────────────────────────────────┘
```

---

## Configuration

### Optimal Settings (Current)

```python
AdaptiveSymbolicRules(
    min_corrections_for_adjustment=2,     # TRAINING: Compute adjustments
    max_confidence_delta=0.20,            # Adjustment strength (20%)
    min_corrections_for_production=10,    # PRODUCTION: Use adjustments
)
```

### What Each Parameter Does

| Parameter | Value | Phase | Purpose |
|-----------|-------|-------|---------|
| `min_corrections_for_adjustment` | 2 | Training | Minimum corrections to compute an adjustment |
| `min_corrections_for_production` | 10 | Production | Minimum corrections to USE adjustment in decisions |
| `max_confidence_delta` | 0.20 | Both | Maximum confidence boost/penalty (20%) |

### Why These Values?

**Training threshold = 2:**
- ✅ Fast feedback: Users see "adjustment computed" after just 2 corrections
- ✅ Requires consistency: Not just a single mistake
- ✅ Establishes pattern: Shows there's a preference forming

**Production threshold = 10:**
- ✅ Sufficient data: 10 corrections provide strong signal
- ✅ Safe from noise: Won't apply adjustments from too few examples
- ✅ Covers diversity: Likely sees different columns within same pattern
- ❌ Not too high: 20+ would take too long to reach

**Confidence delta = 20%:**
- ✅ Strong impact: 20% boost can change decisions
- ✅ Not overwhelming: Won't override obvious cases
- ✅ Reversible: Still leaves room for symbolic rules to matter

---

## User Experience

### Example Flow: Learning to Prefer Log Transform

**Corrections 1-2 (TRAINING):**

```bash
# 1st correction
curl -X POST "http://localhost:8000/correct" \
  -d '{
    "column_data": [1, 5, 100, 500, 1000],
    "wrong_action": "standard_scale",
    "correct_action": "log_transform"
  }'

# Response
{
  "learned": true,
  "production_ready": false,
  "pattern_corrections": 1,
  "corrections_needed_for_training": 1,
  "corrections_needed_for_production": 9,
  "message": "📝 Recording: 1/2 corrections for pattern 'numeric_high_skewness'. 1 more needed to compute adjustment."
}

# 2nd correction
curl -X POST "http://localhost:8000/correct" \
  -d '{
    "column_data": [2, 10, 200, 1000, 5000],
    "wrong_action": "minmax_scale",
    "correct_action": "log_transform"
  }'

# Response
{
  "learned": true,
  "production_ready": false,
  "adjustment_active": true,
  "pattern_corrections": 2,
  "corrections_needed_for_training": 0,
  "corrections_needed_for_production": 8,
  "confidence_boost": "+0.200",
  "preferred_action": "log_transform",
  "message": "⚙ TRAINING: Adjustment computed from 2 corrections. 8 more needed to activate in production decisions."
}
```

**Corrections 3-9 (Still TRAINING):**

Each correction response:
```json
{
  "production_ready": false,
  "pattern_corrections": 5,
  "corrections_needed_for_production": 5,
  "message": "⚙ TRAINING: Adjustment computed from 5 corrections. 5 more needed to activate in production decisions."
}
```

**During this phase:**
- ❌ System does NOT use learned adjustments in decisions
- ✓ System IS recording and updating adjustments
- ✓ Users get clear progress feedback

**Correction 10 (PRODUCTION ACTIVATED):**

```bash
curl -X POST "http://localhost:8000/correct" \
  -d '{...}'  # 10th correction

# Response
{
  "learned": true,
  "production_ready": true,  # ✓ NOW PRODUCTION!
  "adjustment_active": true,
  "pattern_corrections": 10,
  "corrections_needed_for_production": 0,
  "confidence_boost": "+0.200",
  "preferred_action": "log_transform",
  "message": "✓ PRODUCTION: Adjustments active! Pattern 'numeric_high_skewness' learned from 10 corrections. Now boosting 'log_transform' by 20% for similar columns."
}
```

**From now on:**
- ✅ All columns matching `numeric_high_skewness` pattern:
  - If symbolic suggests `log_transform` → confidence boosted +20%
  - If symbolic suggests other actions → confidence reduced -10%
- ✅ Learning actively influences decisions
- ✅ Users see adapted decisions in real-time

---

## Implementation Details

### 1. Check Production Readiness

```python
def is_production_ready(self, column_stats: Dict[str, Any]) -> bool:
    """Check if pattern has enough corrections for production use."""
    pattern_key = self._identify_pattern(column_stats)
    corrections = self.correction_patterns.get(pattern_key, [])
    return len(corrections) >= self.min_corrections_for_production
```

### 2. Apply Adjustments Only When Ready

```python
def adjust_confidence(
    self,
    action: PreprocessingAction,
    original_confidence: float,
    column_stats: Dict[str, Any]
) -> float:
    """Adjust confidence only if pattern is production-ready."""

    # CRITICAL: Check production readiness first
    if not self.is_production_ready(column_stats):
        # TRAINING PHASE: Don't affect decisions yet
        return original_confidence

    # PRODUCTION PHASE: Apply learned adjustments
    adjustment = self.get_adjustment(column_stats)
    if not adjustment:
        return original_confidence

    if action == adjustment.action:
        return min(0.98, original_confidence + adjustment.confidence_delta)
    else:
        return max(0.3, original_confidence - adjustment.confidence_delta * 0.5)
```

### 3. Clear User Feedback

```python
# After correction is recorded:
is_production_ready = self.adaptive_rules.is_production_ready(stats_dict)

if adjustment:
    if is_production_ready:
        message = "✓ PRODUCTION: Adjustments active!"
    else:
        message = f"⚙ TRAINING: {corrections_left} more needed for production"
```

---

## API Response Schema

### Correction Response Fields

```python
{
    "learned": bool,                        # Always true if correction accepted
    "production_ready": bool,               # Is this pattern ready for production?
    "adjustment_active": bool,              # Is adjustment computed? (>= 2 corrections)
    "pattern_category": str,                # e.g., "numeric_high_skewness"
    "pattern_corrections": int,             # Total corrections for this pattern
    "corrections_needed_for_training": int, # How many more to compute adjustment (0 if done)
    "corrections_needed_for_production": int, # How many more to activate adjustment (0 if done)
    "confidence_boost": str,                # e.g., "+0.200" (if adjustment active)
    "preferred_action": str,                # e.g., "log_transform" (if adjustment active)
    "message": str                          # Human-readable status
}
```

---

## Benefits of This Architecture

### ✅ Advantages

1. **Safe Learning:**
   - System won't affect production decisions with insufficient data
   - Requires 10 diverse corrections before deployment

2. **Clear Feedback:**
   - Users always know which phase they're in (📝 Recording, ⚙ Training, ✓ Production)
   - Progress bars show exactly how many corrections needed

3. **Gradual Deployment:**
   - Training happens first (collect data, compute adjustments)
   - Production happens later (use adjustments confidently)

4. **Reversible:**
   - Can adjust thresholds without code changes
   - Can delete `data/adaptive_rules.json` to reset

5. **Per-Pattern Activation:**
   - Some patterns may be in production while others still training
   - Prevents global all-or-nothing approach

### ⚠️ Trade-offs

1. **Slower Initial Learning:**
   - Was: 2 corrections → affects decisions
   - Now: 10 corrections → affects decisions
   - But this is intentional and safer

2. **More Corrections Needed:**
   - To activate all 13 patterns: ~130 total corrections (13 × 10)
   - But each pattern activates independently

---

## Monitoring & Debugging

### Check Pattern Status

```bash
# Get learning statistics
curl http://localhost:8000/stats

# Response shows which patterns are production-ready
{
  "total_corrections": 25,
  "patterns_tracked": 5,
  "active_adjustments": 3,
  "adjustments": {
    "numeric_high_skewness": {
      "action": "log_transform",
      "confidence_delta": "+0.200",
      "corrections": 12,  # ✓ Production-ready
      "production_ready": true
    },
    "categorical_high_cardinality": {
      "action": "drop_column",
      "confidence_delta": "+0.180",
      "corrections": 5,   # ⚙ Still training
      "production_ready": false
    }
  }
}
```

### View Persistence File

```bash
cat data/adaptive_rules.json | python -m json.tool

# Shows:
# - correction_patterns: All recorded corrections
# - rule_adjustments: Computed adjustments (may not be active yet)
```

### Decision Pipeline Logging

When processing a column:
```
[2025-11-22 10:30:00] INFO: Processing column 'revenue' (numeric, skewness=3.5)
[2025-11-22 10:30:00] INFO: Pattern identified: numeric_high_skewness
[2025-11-22 10:30:00] INFO: Symbolic decision: standard_scale (confidence=0.75)
[2025-11-22 10:30:00] INFO: Adaptive learning check: Production-ready = True
[2025-11-22 10:30:00] INFO: Applying adjustment: log_transform (+0.20 boost)
[2025-11-22 10:30:00] INFO: Adjusted confidence: 0.75 → 0.95
[2025-11-22 10:30:00] INFO: Final decision: log_transform (confidence=0.95)
```

If not production-ready:
```
[2025-11-22 10:30:00] INFO: Adaptive learning check: Production-ready = False (5/10 corrections)
[2025-11-22 10:30:00] INFO: Skipping adjustment (training phase)
[2025-11-22 10:30:00] INFO: Final decision: standard_scale (confidence=0.75, unchanged)
```

---

## Tuning for Different Use Cases

### More Aggressive (Not Recommended)

```python
AdaptiveSymbolicRules(
    min_corrections_for_adjustment=1,   # Compute after 1 correction
    min_corrections_for_production=5,   # Activate after 5 corrections
    max_confidence_delta=0.25,          # Very strong adjustments
)
```

**Use when:**
- Single expert user (low mistake risk)
- Highly consistent domain
- Rapid iteration needed

**Risks:**
- May deploy bad adjustments from noise
- Overconfident early decisions

### More Conservative

```python
AdaptiveSymbolicRules(
    min_corrections_for_adjustment=3,   # More data to compute
    min_corrections_for_production=20,  # Much more data to activate
    max_confidence_delta=0.15,          # Weaker adjustments
)
```

**Use when:**
- Multiple users with different preferences
- Safety-critical domain
- Symbolic rules should remain dominant

**Risks:**
- Very slow to see learning effects
- Users may get frustrated

### Recommended (Current)

```python
AdaptiveSymbolicRules(
    min_corrections_for_adjustment=2,   # Fast feedback
    min_corrections_for_production=10,  # Safe deployment
    max_confidence_delta=0.20,          # Strong but not overwhelming
)
```

**Sweet spot for:**
- Single-team usage
- Domain-specific data
- Balance of speed and safety

---

## Testing the System

### Manual Test: Full Training → Production Cycle

```bash
# Start server
uvicorn src.api.server:app --reload

# 1. Get initial recommendation (no learning yet)
curl -X POST "http://localhost:8000/preprocess" \
  -H "Content-Type: application/json" \
  -d '{"column_data": [1, 5, 100, 500, 1000], "column_name": "test1"}'

# Response: {"action": "standard_scale", "confidence": 0.75}

# 2. Submit 10 corrections for same pattern
for i in {1..10}; do
  curl -X POST "http://localhost:8000/correct" \
    -H "Content-Type: application/json" \
    -d "{
      \"column_data\": [1, 5, 100, 500, 1000],
      \"column_name\": \"test$i\",
      \"wrong_action\": \"standard_scale\",
      \"correct_action\": \"log_transform\"
    }"

  echo "\n--- Correction $i submitted ---\n"
  sleep 1
done

# Watch messages change:
# Corrections 1: "📝 Recording: 1/2 corrections..."
# Corrections 2: "⚙ TRAINING: Adjustment computed... 8 more needed"
# Corrections 3-9: "⚙ TRAINING: ... X more needed"
# Corrections 10: "✓ PRODUCTION: Adjustments active!"

# 3. Get recommendation again (should be adapted now)
curl -X POST "http://localhost:8000/preprocess" \
  -H "Content-Type: application/json" \
  -d '{"column_data": [1, 5, 100, 500, 1000], "column_name": "new_column"}'

# Response: {"action": "log_transform", "confidence": 0.95}
# ✓ Confidence boosted from 0.75 → 0.95 due to learning!
```

---

## Migration from Old System

### If You Have Existing Corrections

The system will automatically:
1. Load existing corrections from `data/adaptive_rules.json`
2. Count corrections per pattern
3. Activate production mode for patterns with >= 10 corrections
4. Keep training mode for patterns with < 10 corrections

No manual migration needed!

### Resetting Learning

```bash
# Backup existing learning
cp data/adaptive_rules.json data/adaptive_rules.backup.json

# Reset to fresh state
rm data/adaptive_rules.json

# Restart server
# System starts in training mode for all patterns
```

---

## Summary

### Key Changes

| Aspect | Old System | New System |
|--------|-----------|------------|
| **Activation** | 2 corrections → immediate effect | 2 corrections → training, 10 → production |
| **User Feedback** | "Learned!" (unclear) | "📝 Recording", "⚙ Training", "✓ Production" |
| **Safety** | Risky (affects decisions too early) | Safe (requires sufficient data) |
| **Perception** | "Making decisions without learning" | "Learning first, then deploying" |

### Configuration

```python
min_corrections_for_adjustment=2   # Compute adjustments (TRAINING)
min_corrections_for_production=10  # Use adjustments (PRODUCTION)
max_confidence_delta=0.20          # Adjustment strength
```

### User Experience

- **Corrections 1-2:** Recording phase, system acknowledges learning
- **Corrections 2-9:** Training phase, adjustments computed but NOT used
- **Corrections 10+:** Production phase, adjustments actively influence decisions

**The learning system now learns responsibly.** ✅
