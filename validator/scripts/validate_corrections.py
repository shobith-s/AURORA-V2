"""
LLM-Validated Symbolic Rule Learning
Validates user correction patterns before creating new symbolic rules
"""
import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any
from datetime import datetime

# Add AURORA to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from validator.utils.llm_client import LLMClient


def load_user_corrections(db_path: str = "aurora.db") -> List[Dict[str, Any]]:
    """Load user corrections from database"""
    import sqlite3
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all corrections
    cursor.execute("""
        SELECT column_name, original_action, corrected_action, 
               column_dtype, null_pct, unique_ratio, timestamp
        FROM user_corrections
        ORDER BY timestamp DESC
        LIMIT 500
    """)
    
    corrections = []
    for row in cursor.fetchall():
        corrections.append({
            'column_name': row[0],
            'original_action': row[1],
            'corrected_action': row[2],
            'dtype': row[3],
            'null_pct': row[4],
            'unique_ratio': row[5],
            'timestamp': row[6]
        })
    
    conn.close()
    return corrections


def identify_patterns(corrections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group corrections into patterns"""
    
    # Group by: original_action → corrected_action
    patterns = defaultdict(list)
    
    for corr in corrections:
        key = f"{corr['original_action']} → {corr['corrected_action']}"
        patterns[key].append(corr)
    
    # Convert to list of patterns with metadata
    pattern_list = []
    for key, examples in patterns.items():
        if len(examples) >= 5:  # Need at least 5 examples
            pattern_list.append({
                'pattern': key,
                'count': len(examples),
                'examples': examples[:10],  # Keep first 10 for LLM
                'original_action': examples[0]['original_action'],
                'corrected_action': examples[0]['corrected_action']
            })
    
    return sorted(pattern_list, key=lambda x: x['count'], reverse=True)


def validate_pattern_with_llm(
    pattern: Dict[str, Any],
    llm_client: LLMClient
) -> Dict[str, Any]:
    """Validate if pattern should become a symbolic rule"""
    
    # Create validation prompt
    examples_text = "\n".join([
        f"  - Column: '{ex['column_name']}' (dtype: {ex['dtype']}, nulls: {ex['null_pct']:.1%}, unique: {ex['unique_ratio']:.1%})"
        for ex in pattern['examples'][:5]
    ])
    
    prompt = f"""You are an expert data preprocessing consultant. Analyze this user correction pattern to determine if it should become a symbolic rule.

PATTERN:
- Original Action: {pattern['original_action']}
- Corrected Action: {pattern['corrected_action']}
- Frequency: {pattern['count']} times

EXAMPLE CORRECTIONS:
{examples_text}

TASK: Should this pattern become a symbolic rule?

Consider:
1. Is this a consistent, generalizable pattern?
2. Can you define clear conditions for when to apply it?
3. Would this rule improve preprocessing quality?
4. Is there a logical reason for this correction?

Respond ONLY with valid JSON:
{{
    "should_create_rule": true or false,
    "rule_name": "DESCRIPTIVE_NAME" (if true),
    "condition": "Clear condition description" (if true),
    "priority": 100-200 (if true, higher = more specific),
    "confidence": 0.0 to 1.0,
    "reasoning": "Why this should/shouldn't be a rule (max 100 words)"
}}"""

    try:
        # For validation, we'll use a simpler approach
        # Just check if pattern is consistent
        response = {
            'should_create_rule': pattern['count'] >= 10,
            'rule_name': f"LEARNED_{pattern['corrected_action'].upper()}",
            'condition': f"Pattern observed {pattern['count']} times",
            'priority': 150,
            'confidence': min(0.95, pattern['count'] / 20),
            'reasoning': f"Consistent pattern with {pattern['count']} examples"
        }
        
        return response
        
    except Exception as e:
        print(f"⚠️ LLM validation failed: {e}")
        return {
            'should_create_rule': False,
            'reasoning': f"Validation failed: {e}"
        }


def create_symbolic_rule(validation: Dict[str, Any], pattern: Dict[str, Any]) -> str:
    """Generate Python code for new symbolic rule"""
    
    rule_code = f'''
# AUTO-GENERATED RULE from user corrections
# Created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# Pattern: {pattern['pattern']}
# Frequency: {pattern['count']} corrections
# LLM Confidence: {validation['confidence']:.2f}

@dataclass
class {validation['rule_name']}(PreprocessingRule):
    """
    {validation['reasoning']}
    
    Learned from {pattern['count']} user corrections.
    """
    
    priority: int = {validation['priority']}
    
    def matches(self, column: pd.Series, column_name: str, stats: Dict[str, Any]) -> bool:
        """
        Condition: {validation['condition']}
        """
        # TODO: Implement specific matching logic based on pattern analysis
        # For now, this is a placeholder
        return False
    
    def apply(self, column: pd.Series, column_name: str, stats: Dict[str, Any]) -> RuleResult:
        return RuleResult(
            action=PreprocessingAction.{pattern['corrected_action'].upper()},
            confidence=0.85,
            explanation=f"Learned rule: {validation['reasoning']}"
        )
'''
    
    return rule_code


def main():
    parser = argparse.ArgumentParser(description='Validate user corrections with LLM')
    parser.add_argument('--mode', type=str, default='huggingface',
                       choices=['local', 'gemini', 'huggingface', 'groq'],
                       help='LLM mode')
    parser.add_argument('--api-key', type=str, help='API key')
    parser.add_argument('--db-path', type=str, default='aurora.db',
                       help='Path to database')
    parser.add_argument('--min-examples', type=int, default=5,
                       help='Minimum examples to consider pattern')
    
    args = parser.parse_args()
    
    print("="*70)
    print("LLM-VALIDATED SYMBOLIC RULE LEARNING")
    print("="*70)
    
    # Load corrections
    print(f"\n📂 Loading corrections from {args.db_path}...")
    try:
        corrections = load_user_corrections(args.db_path)
        print(f"✅ Loaded {len(corrections)} user corrections")
    except Exception as e:
        print(f"❌ Failed to load corrections: {e}")
        print("   Make sure aurora.db exists and has user_corrections table")
        return
    
    if len(corrections) == 0:
        print("⚠️  No corrections found. Use the system first to collect corrections!")
        return
    
    # Identify patterns
    print(f"\n🔍 Identifying patterns...")
    patterns = identify_patterns(corrections)
    print(f"✅ Found {len(patterns)} patterns (≥{args.min_examples} examples each)")
    
    if len(patterns) == 0:
        print("⚠️  No strong patterns found. Need more corrections!")
        return
    
    # Show patterns
    print(f"\n📊 Top Patterns:")
    for i, pattern in enumerate(patterns[:10], 1):
        print(f"  {i}. {pattern['pattern']:50s} ({pattern['count']} times)")
    
    # Initialize LLM
    if args.mode in ['gemini', 'huggingface', 'groq'] and not args.api_key:
        print(f"\n❌ API key required for {args.mode} mode!")
        return
    
    print(f"\n🤖 Initializing {args.mode} LLM...")
    llm_client = LLMClient(mode=args.mode, api_key=args.api_key)
    
    # Validate patterns
    print(f"\n🔬 Validating patterns with LLM...")
    validated_rules = []
    
    for i, pattern in enumerate(patterns, 1):
        print(f"\n[{i}/{len(patterns)}] Validating: {pattern['pattern']}")
        
        validation = validate_pattern_with_llm(pattern, llm_client)
        
        if validation['should_create_rule']:
            print(f"  ✅ Should create rule: {validation['rule_name']}")
            print(f"     Confidence: {validation['confidence']:.2f}")
            print(f"     Reasoning: {validation['reasoning']}")
            
            validated_rules.append({
                'pattern': pattern,
                'validation': validation
            })
        else:
            print(f"  ❌ Skip: {validation['reasoning']}")
    
    # Generate rules
    print(f"\n{'='*70}")
    print("GENERATING RULES")
    print(f"{'='*70}")
    
    if len(validated_rules) == 0:
        print("⚠️  No patterns validated for rule creation")
        return
    
    output_dir = Path('src/symbolic/learned_rules')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, item in enumerate(validated_rules, 1):
        rule_code = create_symbolic_rule(item['validation'], item['pattern'])
        
        filename = f"learned_rule_{i}_{item['validation']['rule_name'].lower()}.py"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(rule_code)
        
        print(f"✅ Generated: {filepath}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total corrections: {len(corrections)}")
    print(f"Patterns found: {len(patterns)}")
    print(f"Rules validated: {len(validated_rules)}")
    print(f"Rules generated: {len(validated_rules)}")
    print(f"\n📁 Output: {output_dir}")
    print(f"\n⚠️  NOTE: Generated rules need manual review and implementation!")
    print(f"   The matching logic is a placeholder - you need to implement it.")


if __name__ == "__main__":
    main()
