"""
Simple Benchmark for AURORA V2
Measures: Accuracy, Speed, Confidence
No over-engineering - just concrete numbers
"""
import pandas as pd
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.preprocessor import IntelligentPreprocessor


def benchmark_on_dataset(csv_path: str):
    """
    Simple benchmark on a single dataset
    Returns concrete metrics
    """
    
    print(f"\n{'='*70}")
    print(f"BENCHMARKING: {Path(csv_path).name}")
    print(f"{'='*70}")
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"\nDataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Initialize preprocessor
    preprocessor = IntelligentPreprocessor()
    
    # Benchmark each column
    results = []
    total_time = 0
    
    for col in df.columns:
        start = time.time()
        
        result = preprocessor.preprocess_column(
            column=df[col],
            column_name=col
        )
        
        elapsed = (time.time() - start) * 1000  # ms
        total_time += elapsed
        
        results.append({
            'column': col,
            'action': result.action.value,
            'confidence': result.confidence,
            'source': result.source,
            'time_ms': elapsed
        })
    
    # Calculate metrics
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    avg_time = total_time / len(results)
    
    # Source breakdown
    sources = {}
    for r in results:
        sources[r['source']] = sources.get(r['source'], 0) + 1
    
    # Print results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Total columns: {len(results)}")
    print(f"Avg confidence: {avg_confidence:.1%}")
    print(f"Avg time: {avg_time:.1f}ms")
    print(f"Total time: {total_time:.1f}ms")
    
    print(f"\nDecision Sources:")
    for source, count in sources.items():
        pct = (count / len(results)) * 100
        print(f"  {source}: {count} ({pct:.1f}%)")
    
    print(f"\nPer-Column Results:")
    for r in results:
        print(f"  {r['column']:30s} → {r['action']:20s} ({r['confidence']:.1%}, {r['time_ms']:.1f}ms)")
    
    return {
        'total_columns': len(results),
        'avg_confidence': avg_confidence,
        'avg_time_ms': avg_time,
        'total_time_ms': total_time,
        'sources': sources,
        'results': results
    }


def main():
    """Run benchmarks on available datasets"""
    
    print("="*70)
    print("AURORA V2 BENCHMARK")
    print("="*70)
    
    # Test on Books.csv (if available)
    books_path = Path('datas/Books.csv')
    
    if books_path.exists():
        metrics = benchmark_on_dataset(str(books_path))
        
        # Save results
        import json
        output_path = Path('benchmark_results.json')
        with open(output_path, 'w') as f:
            # Convert to JSON-safe format
            safe_metrics = {
                'total_columns': metrics['total_columns'],
                'avg_confidence': float(metrics['avg_confidence']),
                'avg_time_ms': float(metrics['avg_time_ms']),
                'total_time_ms': float(metrics['total_time_ms']),
                'sources': metrics['sources']
            }
            json.dump(safe_metrics, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_path}")
    else:
        print(f"\n❌ Books.csv not found at: {books_path}")
        print(f"   Place a CSV file there to benchmark")
    
    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
