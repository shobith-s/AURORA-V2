"""
Generate Simulated User Corrections using FREE Groq API
Feeds corrections to EXISTING adaptive learning system
"""
import os
import sys
import json
import time
from pathlib import Path
from tqdm import tqdm

# Add AURORA to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from validator.utils.llm_client import LLMClient
from src.core.preprocessor import IntelligentPreprocessor


def main():
    print("="*70)
    print("GENERATING SIMULATED USER CORRECTIONS (FREE)")
    print("="*70)
    
    # Load symbolic labels (already generated)
    labels_dir = Path('validator/labels')
    
    print(f"\n📂 Loading symbolic labels...")
    with open(labels_dir / 'all_labels.json', 'r') as f:
        all_labels = json.load(f)
    
    print(f"✅ Loaded {len(all_labels)} symbolic decisions")
    
    # Initialize Groq LLM (FREE)
    groq_key = input("\nEnter Groq API key (free from console.groq.com): ").strip()
    llm = LLMClient(mode='groq', api_key=groq_key)
    
    print(f"\n🔌 Testing Groq connection...")
    if not llm.test_connection():
        print("❌ Groq not available!")
        return
    print(f"✅ Groq is ready!")
    
    # Initialize preprocessor (has adaptive learning built-in)
    print(f"\n🧠 Initializing AURORA preprocessor...")
    preprocessor = IntelligentPreprocessor()
    print(f"✅ Preprocessor ready!")
    
    # Generate corrections and feed to adaptive learning
    print(f"\n🔬 Generating corrections (LLM validates symbolic decisions)...")
    corrections_count = 0
    rules_created = 0
    
    for i, label in enumerate(tqdm(all_labels, desc="Validating")):
        try:
            # Prepare column info
            column_info = {
                'name': label['column'],
                'dtype': label['dtype'],
                'row_count': 1000,
                'null_pct': label['null_pct'],
                'unique_count': label['unique_count'],
                'unique_ratio': label['unique_count'] / 1000,
                'sample_values': [],
                'skewness': label['features'].get('skewness', 0)
            }
            
            symbolic_decision = {
                'action': label['action'],
                'confidence': label['confidence'],
                'explanation': label['explanation']
            }
            
            # LLM validates
            validation = llm.validate_decision(column_info, symbolic_decision)
            
            # If LLM disagrees with high confidence → correction!
            if not validation['is_correct'] and validation.get('llm_confidence', 0) >= 0.85:
                # Feed to EXISTING adaptive learning system
                # Create dummy column data for stats extraction
                dummy_column = pd.Series([0] * 100, name=label['column'])
                
                result = preprocessor.process_correction(
                    column=dummy_column,
                    column_name=label['column'],
                    wrong_action=label['action'],
                    correct_action=validation['correct_action'],
                    confidence=label['confidence']
                )
                
                corrections_count += 1
                
                # Check if new rule was created
                if result.get('new_rule_created', False):
                    rules_created += 1
                    print(f"\n  ✅ NEW RULE CREATED!")
                    print(f"     Pattern: {label['action']} → {validation['correct_action']}")
                    print(f"     Support: {result.get('rule_support', 0)} corrections")
                
                elif result.get('correction_support', 0) > 0:
                    needed = result.get('corrections_needed_for_production', 0)
                    print(f"\n  [{i+1}/{len(all_labels)}] Correction recorded:")
                    print(f"     {label['action']} → {validation['correct_action']}")
                    print(f"     Progress: {result.get('correction_support', 0)}/10 (need {needed} more)")
            
        except Exception as e:
            print(f"\n  ⚠️ Error on {label['column']}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Total corrections: {corrections_count}")
    print(f"New rules created: {rules_created}")
    
    # Get adaptive learning stats
    stats = preprocessor.get_statistics()
    learning_stats = stats.get('learning', {})
    
    print(f"\nAdaptive Learning Stats:")
    print(f"  Total corrections in DB: {learning_stats.get('total_corrections', 0)}")
    print(f"  Learned rules: {learning_stats.get('learned_rules', 0)}")
    print(f"  Pattern clusters: {learning_stats.get('pattern_clusters', 0)}")
    
    print(f"\n✅ All corrections fed to adaptive learning!")
    print(f"   Rules are automatically created when ≥10 similar corrections")
    print(f"   Check database for learned rules")


if __name__ == "__main__":
    main()

