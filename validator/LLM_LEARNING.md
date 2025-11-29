# LLM-Validated Learning System

**Two validation systems working together:**

## 1. Neural Oracle Training (DONE ✅)

**What:** Validate symbolic labels with LLM before training XGBoost

**Process:**
```
Datasets → Symbolic labels → LLM validates → Train neural oracle
```

**Status:** Working, 71.3% accuracy

---

## 2. Symbolic Rule Learning (NEW 🆕)

**What:** Validate user correction patterns with LLM before creating symbolic rules

**Process:**
```
User corrections → Identify patterns → LLM validates → Create symbolic rules
```

**How to use:**
```bash
# After collecting 50+ user corrections
python validator/scripts/validate_corrections.py \
    --mode huggingface \
    --api-key $HF_TOKEN \
    --db-path aurora.db
```

**What it does:**
1. Loads user corrections from database
2. Groups into patterns (e.g., "Year columns → keep_as_is")
3. Asks LLM: "Should this be a rule?"
4. Generates Python code for validated rules
5. Saves to `src/symbolic/learned_rules/`

**Example output:**
```python
# AUTO-GENERATED RULE
@dataclass
class LEARNED_KEEP_AS_IS(PreprocessingRule):
    """
    Year columns should be kept as-is for temporal analysis.
    Learned from 15 user corrections.
    """
    priority: int = 150
    
    def matches(self, column, column_name, stats):
        # Check if column name contains 'year'
        # and values in range 1900-2100
        return 'year' in column_name.lower()
    
    def apply(self, column, column_name, stats):
        return RuleResult(
            action=PreprocessingAction.KEEP_AS_IS,
            confidence=0.85,
            explanation="Learned: Year columns for temporal analysis"
        )
```

---

## Benefits

**Quality Control:**
- LLM validates patterns before creating rules
- Prevents bad rules from user mistakes
- Ensures rules are generalizable

**Scalability:**
- Process 100s of corrections at once
- Batch validation is efficient
- Continuous improvement

**Better Learning:**
- LLM suggests optimal conditions
- Better priority assignment
- More robust rules

---

## Workflow

### **Phase 1: Collect Corrections**
```
User uses AURORA → Makes corrections → Stored in database
```

### **Phase 2: Validate Patterns (Weekly/Monthly)**
```bash
# Run validation script
python validator/scripts/validate_corrections.py \
    --mode huggingface \
    --api-key $HF_TOKEN
```

### **Phase 3: Review & Deploy**
```
1. Review generated rules in src/symbolic/learned_rules/
2. Implement matching logic (currently placeholder)
3. Test on validation data
4. Add to symbolic engine
5. Deploy
```

---

## Example

**User corrections (15 times):**
```
"Year-Of-Publication" → keep_as_is (was: scale)
"Birth_Year" → keep_as_is (was: scale)
"Year_Built" → keep_as_is (was: scale)
```

**LLM validation:**
```json
{
  "should_create_rule": true,
  "rule_name": "KEEP_YEAR_COLUMNS",
  "confidence": 0.95,
  "reasoning": "Year columns should be kept as-is for temporal analysis"
}
```

**Generated rule:** `src/symbolic/learned_rules/learned_rule_1_keep_year_columns.py`

---

## Status

- ✅ Script created
- ✅ Pattern identification
- ✅ LLM validation
- ✅ Rule generation
- ⚠️  Needs manual review & implementation
- ⚠️  Needs testing

---

**Ready to use after collecting user corrections!**
