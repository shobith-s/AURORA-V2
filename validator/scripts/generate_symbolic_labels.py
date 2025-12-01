"""
Generate symbolic labels for all datasets
Runs symbolic engine on each column and categorizes by confidence
"""
import os
import sys
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add AURORA to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.symbolic.engine import SymbolicEngine
from src.features.minimal_extractor import MinimalFeatureExtractor

def main():
    print("="*70)
    print("GENERATING SYMBOLIC LABELS")
    print("="*70)
    
    # Initialize engines
    symbolic_engine = SymbolicEngine()
    feature_extractor = MinimalFeatureExtractor()
    
    # Load datasets
    data_dir = Path('validator/data')
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("   Run download_datasets.py first!")
        return
    
    # Results storage
    all_labels = []
    high_confidence = []
    medium_confidence = []
    low_confidence = []
    
    # Process each dataset
    csv_files = list(data_dir.glob('*.csv'))
    print(f"\n📂 Found {len(csv_files)} datasets\n")
    
    for csv_file in csv_files:
        dataset_name = csv_file.stem
        print(f"\n{'='*70}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*70}")
        
        try:
            # Load dataset
            df = pd.read_csv(csv_file, low_memory=False, nrows=1000)
            print(f"  Shape: {df.shape}")
            
            # Process each column
            for col_name in tqdm(df.columns, desc=f"  {dataset_name}"):
                try:
                    column = df[col_name]
                    
                    # Skip if too few values
                    if len(column.dropna()) < 10:
                        continue
                    
                    # Get symbolic decision
                    result = symbolic_engine.evaluate(column, col_name)
                    
                    # Extract features
                    features = feature_extractor.extract(column, col_name)
                    
                    # Store label
                    label = {
                        'dataset': dataset_name,
                        'column': col_name,
                        'action': result.action.value,
                        'confidence': result.confidence,
                        'explanation': result.explanation,
                        'features': features.to_dict(),
                        'dtype': str(column.dtype),
                        'null_pct': column.isnull().mean(),
                        'unique_count': column.nunique()
                    }
                    
                    all_labels.append(label)
                    
                    # Categorize by confidence
                    if result.confidence >= 0.90:
                        high_confidence.append(label)
                    elif result.confidence >= 0.70:
                        medium_confidence.append(label)
                    else:
                        low_confidence.append(label)
                        
                except Exception as e:
                    print(f"    ⚠️ Error on {col_name}: {e}")
                    continue
            
            print(f"  ✅ Processed {len([l for l in all_labels if l['dataset'] == dataset_name])} columns")
            
        except Exception as e:
            print(f"  ❌ Failed to process {dataset_name}: {e}")
            continue
    
    # Save results
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    labels_dir = Path('validator/labels')
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all labels
    with open(labels_dir / 'all_labels.json', 'w') as f:
        json.dump(all_labels, f, indent=2)
    print(f"✅ Saved all labels: {len(all_labels)} examples")
    
    # Save by confidence
    with open(labels_dir / 'high_confidence.json', 'w') as f:
        json.dump(high_confidence, f, indent=2)
    print(f"✅ High confidence (≥0.90): {len(high_confidence)} examples")
    
    with open(labels_dir / 'medium_confidence.json', 'w') as f:
        json.dump(medium_confidence, f, indent=2)
    print(f"✅ Medium confidence (0.70-0.90): {len(medium_confidence)} examples")
    
    with open(labels_dir / 'low_confidence.json', 'w') as f:
        json.dump(low_confidence, f, indent=2)
    print(f"✅ Low confidence (<0.70): {len(low_confidence)} examples")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total examples: {len(all_labels)}")
    print(f"High confidence: {len(high_confidence)} ({len(high_confidence)/len(all_labels)*100:.1f}%)")
    print(f"Medium confidence: {len(medium_confidence)} ({len(medium_confidence)/len(all_labels)*100:.1f}%)")
    print(f"Low confidence: {len(low_confidence)} ({len(low_confidence)/len(all_labels)*100:.1f}%)")
    print(f"{'='*70}")
    
    # Action distribution
    from collections import Counter
    actions = Counter(l['action'] for l in all_labels)
    print(f"\nTop 10 Actions:")
    for action, count in actions.most_common(10):
        print(f"  {action:30s}: {count:4d} ({count/len(all_labels)*100:.1f}%)")
    
    print(f"\n✅ Labels saved to: {labels_dir}")

if __name__ == "__main__":
    main()
