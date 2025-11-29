"""
LLM Validator - Validates symbolic decisions using Gemini API
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from tqdm import tqdm

# Add AURORA to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from validator.utils.llm_client import LLMClient

def main():
    parser = argparse.ArgumentParser(description='Validate symbolic labels with LLM')
    parser.add_argument('--mode', type=str, default='groq', 
                       choices=['local', 'gemini', 'huggingface', 'groq'],
                       help='LLM mode: local (Ollama), gemini (Google), huggingface (HF), or groq (FAST)')
    parser.add_argument('--api-key', type=str, help='API key (required for gemini/huggingface/groq mode)')
    parser.add_argument('--validate-medium', action='store_true', default=True,
                       help='Validate medium confidence cases')
    parser.add_argument('--validate-low', action='store_true', default=True,
                       help='Validate low confidence cases')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Batch size for processing')
    
    args = parser.parse_args()
    
    print("="*70)
    print("LLM VALIDATION")
    print("="*70)
    
    # Initialize LLM client
    if args.mode in ['gemini', 'huggingface', 'groq'] and not args.api_key:
        print(f"❌ {args.mode.title()} API key required!")
        if args.mode == 'gemini':
            print("   Get one at: https://aistudio.google.com/app/apikey")
        elif args.mode == 'groq':
            print("   Get one at: https://console.groq.com")
        else:
            print("   Get one at: https://huggingface.co/settings/tokens")
        return
    
    client = LLMClient(mode=args.mode, api_key=args.api_key)
    
    # Test connection
    print(f"\n🔌 Testing {args.mode} connection...")
    if not client.test_connection():
        print(f"❌ {args.mode} not available!")
        return
    print(f"✅ {args.mode} is ready!")
    
    # Load labels
    labels_dir = Path('validator/labels')
    
    # Load high confidence (trust these)
    with open(labels_dir / 'high_confidence.json', 'r') as f:
        high_conf = json.load(f)
    print(f"\n✅ High confidence: {len(high_conf)} (trusted)")
    
    # Load medium confidence (validate these)
    medium_conf = []
    if args.validate_medium:
        with open(labels_dir / 'medium_confidence.json', 'r') as f:
            medium_conf = json.load(f)
        print(f"⚠️  Medium confidence: {len(medium_conf)} (will validate)")
    
    # Load low confidence (LLM decides)
    low_conf = []
    if args.validate_low:
        with open(labels_dir / 'low_confidence.json', 'r') as f:
            low_conf = json.load(f)
        print(f"⚠️  Low confidence: {len(low_conf)} (LLM will decide)")
    
    # Validate medium confidence
    validated_labels = high_conf.copy()  # Start with trusted high-conf labels
    corrections = 0
    
    if medium_conf:
        print(f"\n{'='*70}")
        print("VALIDATING MEDIUM CONFIDENCE CASES")
        print(f"{'='*70}")
        
        for i, label in enumerate(tqdm(medium_conf, desc="Validating")):
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
                
                # Validate with LLM
                validation = client.validate_decision(column_info, symbolic_decision)
                
                if validation['is_correct']:
                    validated_labels.append(label)
                else:
                    # LLM corrected it
                    corrected_label = label.copy()
                    corrected_label['action'] = validation['correct_action']
                    corrected_label['llm_corrected'] = True
                    corrected_label['llm_reasoning'] = validation['reasoning']
                    validated_labels.append(corrected_label)
                    corrections += 1
                    print(f"\n  [{i+1}/{len(medium_conf)}] Corrected: {label['column']}")
                    print(f"    {label['action']} → {validation['correct_action']}")
                    print(f"    Reason: {validation['reasoning']}")
                
                # Rate limiting
                # Groq: 14,400/day (no pause needed)
                # Gemini/HF: 15 req/min (pause every 15)
                if args.mode in ['gemini', 'huggingface'] and (i + 1) % 15 == 0:
                    print(f"\n  ⏸️  Rate limit pause (15 requests)...")
                    time.sleep(60)
                
            except Exception as e:
                print(f"\n  ⚠️ Error validating {label['column']}: {e}")
                validated_labels.append(label)  # Keep original if validation fails
                continue
        
        print(f"\n✅ Medium confidence validated: {corrections} corrections")
    
    # Validate low confidence
    if low_conf:
        print(f"\n{'='*70}")
        print("VALIDATING LOW CONFIDENCE CASES")
        print(f"{'='*70}")
        
        for i, label in enumerate(tqdm(low_conf, desc="Validating")):
            try:
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
                
                validation = client.validate_decision(column_info, symbolic_decision)
                
                # Use LLM's decision
                corrected_label = label.copy()
                corrected_label['action'] = validation['correct_action']
                corrected_label['llm_decided'] = True
                corrected_label['llm_reasoning'] = validation['reasoning']
                validated_labels.append(corrected_label)
                
                # Rate limiting
                if args.mode in ['gemini', 'huggingface'] and (i + 1) % 15 == 0:
                    print(f"\n  ⏸️  Rate limit pause...")
                    time.sleep(60)
                
            except Exception as e:
                print(f"\n  ⚠️ Error: {e}")
                validated_labels.append(label)
                continue
        
        print(f"\n✅ Low confidence validated")
    
    # Save validated labels
    print(f"\n{'='*70}")
    print("SAVING VALIDATED LABELS")
    print(f"{'='*70}")
    
    validated_dir = Path('validator/validated')
    validated_dir.mkdir(parents=True, exist_ok=True)
    
    with open(validated_dir / 'validated_labels.json', 'w') as f:
        json.dump(validated_labels, f, indent=2)
    
    print(f"✅ Saved {len(validated_labels)} validated labels")
    print(f"   High confidence (trusted): {len(high_conf)}")
    print(f"   Medium (validated): {len(medium_conf)} ({corrections} corrected)")
    print(f"   Low (LLM decided): {len(low_conf)}")
    print(f"\n📁 Output: {validated_dir / 'validated_labels.json'}")

if __name__ == "__main__":
    main()
