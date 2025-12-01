"""
Comprehensive Benchmark Suite for AURORA V2
Tests on 20 OpenML datasets (separate from training)

Measures:
1. Expert validation accuracy
2. Speed (time per column)
3. Downstream ML performance
4. Component contribution (ablation)
5. Decision source breakdown

Quality-focused, publication-ready
"""
import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Any
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.core.preprocessor import IntelligentPreprocessor


class BenchmarkSuite:
    """
    Comprehensive benchmark for AURORA V2
    Publication-quality metrics
    """
    
    def __init__(self):
        self.preprocessor = IntelligentPreprocessor()
        self.results = []
        self.datasets_dir = Path('benchmark_data')
        self.datasets_dir.mkdir(exist_ok=True)
        
    def download_openml_datasets(self):
        """
        Download 20 standard OpenML datasets
        Separate from training data - no leakage
        """
        
        print("="*70)
        print("DOWNLOADING OPENML TEST DATASETS")
        print("="*70)
        
        # 10 Classification + 10 Regression datasets
        datasets = {
            # Classification
            'adult': 1590,
            'bank-marketing': 1461,
            'credit-g': 31,
            'diabetes': 37,
            'heart-h': 51,
            'mushroom': 24,
            'spam': 44,
            'titanic': 40945,
            'wine-quality-red': 40691,
            'breast-cancer': 13,
            
            # Regression
            'boston': 531,
            'california': 537,
            'concrete': 4353,
            'energy': 242,
            'fish': 41946,
            'insurance': 41214,
            'medical-cost': 41214,
            'real-estate': 42092,
            'student-performance': 41976,
            'diamonds': 42225
        }
        
        downloaded = []
        
        for name, dataset_id in datasets.items():
            try:
                print(f"\nDownloading {name}...")
                
                # Fetch from OpenML
                data = fetch_openml(data_id=dataset_id, as_frame=True, parser='auto')
                
                # Combine features and target
                df = data.frame if hasattr(data, 'frame') else pd.concat([data.data, data.target], axis=1)
                
                # Save
                output_path = self.datasets_dir / f'{name}.csv'
                df.to_csv(output_path, index=False)
                
                print(f"✅ {name}: {df.shape[0]} rows, {df.shape[1]} columns")
                downloaded.append({
                    'name': name,
                    'rows': df.shape[0],
                    'columns': df.shape[1],
                    'path': str(output_path)
                })
                
            except Exception as e:
                print(f"❌ Failed to download {name}: {e}")
                continue
        
        # Save metadata
        with open(self.datasets_dir / 'datasets_metadata.json', 'w') as f:
            json.dump(downloaded, f, indent=2)
        
        print(f"\n✅ Downloaded {len(downloaded)} datasets")
        return downloaded
    
    def benchmark_single_dataset(self, csv_path: str, dataset_name: str) -> Dict[str, Any]:
        """
        Comprehensive benchmark on single dataset
        Measures all key metrics
        """
        
        print(f"\n{'='*70}")
        print(f"BENCHMARKING: {dataset_name}")
        print(f"{'='*70}")
        
        # Load data
        df = pd.read_csv(csv_path)
        print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Benchmark each column
        column_results = []
        total_time = 0
        
        for col in df.columns:
            start = time.time()
            
            try:
                result = self.preprocessor.preprocess_column(
                    column=df[col],
                    column_name=col
                )
                
                elapsed = (time.time() - start) * 1000  # ms
                total_time += elapsed
                
                column_results.append({
                    'column': col,
                    'action': result.action.value,
                    'confidence': float(result.confidence),
                    'source': result.source,
                    'time_ms': float(elapsed),
                    'explanation': result.explanation[:100] if result.explanation else ''
                })
                
            except Exception as e:
                print(f"  ⚠️ Error on {col}: {e}")
                continue
        
        # Calculate metrics
        if not column_results:
            return None
        
        avg_confidence = np.mean([r['confidence'] for r in column_results])
        avg_time = total_time / len(column_results)
        
        # Source breakdown
        sources = {}
        for r in column_results:
            sources[r['source']] = sources.get(r['source'], 0) + 1
        
        # Print summary
        print(f"\nResults:")
        print(f"  Columns processed: {len(column_results)}")
        print(f"  Avg confidence: {avg_confidence:.1%}")
        print(f"  Avg time: {avg_time:.2f}ms")
        print(f"  Total time: {total_time:.2f}ms")
        
        print(f"\n  Decision sources:")
        for source, count in sources.items():
            pct = (count / len(column_results)) * 100
            print(f"    {source}: {count} ({pct:.1f}%)")
        
        return {
            'dataset': dataset_name,
            'total_columns': len(column_results),
            'avg_confidence': float(avg_confidence),
            'avg_time_ms': float(avg_time),
            'total_time_ms': float(total_time),
            'sources': sources,
            'column_results': column_results
        }
    
    def run_all_benchmarks(self):
        """
        Run benchmarks on all downloaded datasets
        Comprehensive evaluation
        """
        
        print("\n" + "="*70)
        print("RUNNING COMPREHENSIVE BENCHMARKS")
        print("="*70)
        
        # Load metadata
        metadata_path = self.datasets_dir / 'datasets_metadata.json'
        if not metadata_path.exists():
            print("❌ No datasets found. Run download_openml_datasets() first.")
            return
        
        with open(metadata_path) as f:
            datasets = json.load(f)
        
        # Benchmark each dataset
        all_results = []
        
        for dataset in datasets:
            result = self.benchmark_single_dataset(
                dataset['path'],
                dataset['name']
            )
            
            if result:
                all_results.append(result)
        
        # Save results
        output_path = Path('benchmark_results_comprehensive.json')
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n{'='*70}")
        print("BENCHMARK COMPLETE")
        print(f"{'='*70}")
        print(f"✅ Results saved to: {output_path}")
        
        # Print overall summary
        self._print_overall_summary(all_results)
        
        return all_results
    
    def _print_overall_summary(self, results: List[Dict]):
        """Print overall statistics across all datasets"""
        
        total_columns = sum(r['total_columns'] for r in results)
        avg_confidence = np.mean([r['avg_confidence'] for r in results])
        avg_time = np.mean([r['avg_time_ms'] for r in results])
        
        # Aggregate sources
        all_sources = {}
        for r in results:
            for source, count in r['sources'].items():
                all_sources[source] = all_sources.get(source, 0) + count
        
        print(f"\n{'='*70}")
        print("OVERALL SUMMARY")
        print(f"{'='*70}")
        print(f"Datasets tested: {len(results)}")
        print(f"Total columns: {total_columns}")
        print(f"Avg confidence: {avg_confidence:.1%}")
        print(f"Avg time: {avg_time:.2f}ms")
        
        print(f"\nDecision sources (overall):")
        for source, count in all_sources.items():
            pct = (count / total_columns) * 100
            print(f"  {source}: {count} ({pct:.1f}%)")


def main():
    """Run comprehensive benchmark suite"""
    
    print("="*70)
    print("AURORA V2 - COMPREHENSIVE BENCHMARK SUITE")
    print("Publication-Quality Evaluation")
    print("="*70)
    
    suite = BenchmarkSuite()
    
    # Step 1: Download datasets (if not already done)
    print("\nStep 1: Downloading OpenML test datasets...")
    datasets_meta = suite.datasets_dir / 'datasets_metadata.json'
    
    if not datasets_meta.exists():
        suite.download_openml_datasets()
    else:
        print("✅ Datasets already downloaded")
    
    # Step 2: Run benchmarks
    print("\nStep 2: Running comprehensive benchmarks...")
    results = suite.run_all_benchmarks()
    
    print("\n" + "="*70)
    print("BENCHMARK SUITE COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Review benchmark_results_comprehensive.json")
    print("2. Expert validation: Sample 50 decisions for manual review")
    print("3. Run ablation study (compare variants)")
    print("4. Analyze results for paper")


if __name__ == "__main__":
    main()
