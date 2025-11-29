"""
Generate Simulated User Corrections using FREE Groq API
Same approach as neural oracle training, but for adaptive learning
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


def generate_corrections_from_datasets():
    """
    Generate simulated user corrections using Groq (FREE)
    Same pipeline as neural oracle training
    """
    
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
    
    # Generate corrections
    print(f"\n🔬 Generating corrections (LLM validates symbolic decisions)...")
    corrections = []
    
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
                correction = {
                    'column_name': label['column'],
                    'dataset': label['dataset'],
                    'original_action': label['action'],
                    'corrected_action': validation['correct_action'],
                    'reasoning': validation['reasoning'],
                    'llm_confidence': validation['llm_confidence'],
                    'column_dtype': label['dtype'],
                    'null_pct': label['null_pct'],
                    'unique_ratio': label['unique_count'] / 1000
                }
                corrections.append(correction)
                
                print(f"\n  [{i+1}/{len(all_labels)}] Correction found:")
                print(f"    Column: {label['column']}")
                print(f"    {label['action']} → {validation['correct_action']}")
                print(f"    Reason: {validation['reasoning'][:80]}...")
            
            # Rate limiting (Groq: 14,400/day, very generous)
            # No pause needed unless you hit limit
            
        except Exception as e:
            print(f"\n  ⚠️ Error on {label['column']}: {e}")
            continue
    
    # Save corrections
    print(f"\n{'='*70}")
    print("SAVING CORRECTIONS")
    print(f"{'='*70}")
    
    corrections_dir = Path('validator/corrections')
    corrections_dir.mkdir(parents=True, exist_ok=True)
    
    corrections_file = corrections_dir / 'simulated_corrections.json'
    with open(corrections_file, 'w') as f:
        json.dump(corrections, f, indent=2)
    
    print(f"✅ Saved {len(corrections)} corrections")
    print(f"📁 Output: {corrections_file}")
    
    # Analyze corrections
    from collections import Counter
    patterns = Counter(f"{c['original_action']} → {c['corrected_action']}" for c in corrections)
    
    print(f"\n📊 Top Correction Patterns:")
    for pattern, count in patterns.most_common(10):
        print(f"  {pattern:50s}: {count} times")
    
    # Identify rule candidates (≥10 similar corrections)
    rule_candidates = [p for p, c in patterns.items() if c >= 10]
    print(f"\n✅ Rule candidates (≥10 corrections): {len(rule_candidates)}")
    
    return corrections


def feed_corrections_to_adaptive_learning(corrections):
    """
    Feed corrections to AURORA's adaptive learning system
    This will create new symbolic rules automatically
    """
    
    print(f"\n{'='*70}")
    print("FEEDING CORRECTIONS TO ADAPTIVE LEARNING")
    print(f"{'='*70}")
    
    from src.core.preprocessor import IntelligentPreprocessor
    
    preprocessor = IntelligentPreprocessor()
    
    print(f"\nProcessing {len(corrections)} corrections...")
    
    for i, correction in enumerate(corrections, 1):
        result = preprocessor.process_correction(
            column_name=correction['column_name'],
            wrong_action=correction['original_action'],
            correct_action=correction['corrected_action'],
            column_stats={
                'dtype': correction['column_dtype'],
                'null_pct': correction['null_pct'],
                'unique_ratio': correction['unique_ratio']
            }
        )
        
        if result.get('learned', False):
            print(f"  [{i}] ✅ New rule created!")
            print(f"      Pattern: {correction['original_action']} → {correction['corrected_action']}")
    
    print(f"\n✅ All corrections processed!")
    print(f"   Check adaptive learning stats for new rules")


if __name__ == "__main__":
    # Step 1: Generate corrections using FREE Groq
    corrections = generate_corrections_from_datasets()
    
    if corrections and len(corrections) > 0:
        # Step 2: Feed to adaptive learning
        feed_corrections_to_adaptive_learning(corrections)
        
        print(f"\n{'='*70}")
        print("NEXT STEPS")
        print(f"{'='*70}")
        print(f"1. New symbolic rules have been created")
        print(f"2. Run benchmarks to show improvement")
        print(f"3. Compare: Before corrections vs After corrections")
        print(f"4. Expected: 5-10% accuracy improvement")
    else:
        print(f"\n⚠️  No corrections generated")
