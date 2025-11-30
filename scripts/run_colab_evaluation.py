#!/usr/bin/env python3
"""
Colab Full Evaluation Pipeline for AURORA V2

Comprehensive automated evaluation system for Google Colab (2-3 hours).
Generates publication-ready results with zero manual intervention.

Usage:
    python scripts/run_colab_evaluation.py [--datasets 5] [--verbose]

Output:
    results/
    ├── ablation_results.json
    ├── benchmark_results.json
    ├── statistical_tests.json
    ├── paper_tables.md
    ├── EVALUATION_REPORT.pdf
    ├── figures/
    │   ├── accuracy_comparison.png
    │   ├── decision_sources.png
    │   ├── latency_distribution.png
    │   └── confusion_matrix.png
    └── checkpoints/
"""

import gc
import json
import sys
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    
    def _tqdm_fallback(iterable, desc=None, **kwargs):
        """Fallback progress indicator when tqdm is not available."""
        if desc:
            print(f"Processing: {desc}...")
        return iterable
    
    tqdm = _tqdm_fallback

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for Colab
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def is_colab() -> bool:
    """Detect if running in Google Colab."""
    try:
        from google.colab import files
        return True
    except ImportError:
        return False


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load evaluation configuration."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'configs' / 'evaluation_config.yaml'
    else:
        config_path = Path(config_path)
    
    default_config = {
        'evaluation': {
            'num_datasets': 5,
            'datasets': ['titanic', 'wine', 'diabetes', 'breast_cancer', 'california_housing'],
            'dataset_ids': {
                'titanic': 40945,
                'wine': 187,
                'diabetes': 37,
                'breast_cancer': 13,
                'california_housing': 537,
                'adult': 1590,
            },
            'variants': ['random', 'symbolic_only', 'neural_only', 'aurora_hybrid'],
            'cv_folds': 3,
            'random_forest_estimators': 50,
            'random_forest_max_depth': 10,
            'results_dir': 'results',
            'figures_dpi': 300,
            'checkpoint_frequency': 1,
            'verbose': True
        },
        'statistics': {
            'confidence_level': 0.95,
            'min_samples_for_ttest': 5
        }
    }
    
    if config_path.exists() and YAML_AVAILABLE:
        with open(config_path, 'r') as f:
            loaded = yaml.safe_load(f)
            # Merge with defaults
            for key in default_config:
                if key in loaded:
                    default_config[key].update(loaded[key])
    
    return default_config


class CheckpointManager:
    """Manages checkpoints for resumable evaluation."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, stage: str, data: Dict[str, Any]) -> None:
        """Save checkpoint for a stage."""
        checkpoint_path = self.checkpoint_dir / f"{stage}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, stage: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint for a stage if exists."""
        checkpoint_path = self.checkpoint_dir / f"{stage}.json"
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        return None
    
    def clear_checkpoint(self, stage: str) -> None:
        """Clear a specific checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{stage}.json"
        if checkpoint_path.exists():
            checkpoint_path.unlink()


class AblationVariant:
    """Base class for ablation variants."""
    
    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class RandomBaseline(AblationVariant):
    """Random preprocessing baseline - just standard scaling."""
    
    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        for col in X_copy.columns:
            if X_copy[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                # Fill nulls with mean
                X_copy[col] = X_copy[col].fillna(X_copy[col].mean())
                # Standard scale
                scaler = StandardScaler()
                X_copy[col] = scaler.fit_transform(X_copy[[col]])
            else:
                # Label encode categorical
                X_copy[col] = X_copy[col].fillna('missing')
                le = LabelEncoder()
                X_copy[col] = le.fit_transform(X_copy[col].astype(str))
        return X_copy


class SymbolicOnlyVariant(AblationVariant):
    """AURORA with only symbolic rules (no neural oracle)."""
    
    def __init__(self):
        # Lazy import to avoid import errors if preprocessor not available
        from src.core.preprocessor import IntelligentPreprocessor
        self.preprocessor = IntelligentPreprocessor(use_neural_oracle=False)
    
    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._preprocess_with_aurora(X)
    
    def _preprocess_with_aurora(self, X: pd.DataFrame) -> pd.DataFrame:
        from src.core.executor import get_executor
        executor = get_executor()
        
        X_copy = X.copy()
        for col in X_copy.columns:
            try:
                result = self.preprocessor.preprocess_column(X_copy[col], col)
                action = result.action.value
                
                # Apply simple transformations
                if action == 'keep_as_is':
                    # Handle nulls
                    if X_copy[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                        X_copy[col] = X_copy[col].fillna(X_copy[col].mean())
                    else:
                        X_copy[col] = X_copy[col].fillna('missing')
                        le = LabelEncoder()
                        X_copy[col] = le.fit_transform(X_copy[col].astype(str))
                elif action in ['fill_null_mean', 'fill_null_median']:
                    X_copy[col] = X_copy[col].fillna(X_copy[col].median())
                elif action == 'standard_scale':
                    X_copy[col] = X_copy[col].fillna(X_copy[col].mean())
                    scaler = StandardScaler()
                    X_copy[col] = scaler.fit_transform(X_copy[[col]])
                elif action in ['label_encode', 'onehot_encode']:
                    X_copy[col] = X_copy[col].fillna('missing')
                    le = LabelEncoder()
                    X_copy[col] = le.fit_transform(X_copy[col].astype(str))
                elif action == 'drop_column':
                    X_copy = X_copy.drop(columns=[col])
                else:
                    # Default handling
                    if X_copy[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                        X_copy[col] = X_copy[col].fillna(X_copy[col].mean())
                    else:
                        X_copy[col] = X_copy[col].fillna('missing')
                        le = LabelEncoder()
                        X_copy[col] = le.fit_transform(X_copy[col].astype(str))
            except Exception as e:
                logger.warning(f"Error processing {col}: {e}")
                # Safe fallback
                if X_copy[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    X_copy[col] = X_copy[col].fillna(X_copy[col].mean())
                else:
                    X_copy[col] = X_copy[col].fillna('missing')
                    le = LabelEncoder()
                    X_copy[col] = le.fit_transform(X_copy[col].astype(str))
        
        return X_copy


class NeuralOnlyVariant(AblationVariant):
    """AURORA with only neural oracle (symbolic rules disabled)."""
    
    def __init__(self):
        from src.core.preprocessor import IntelligentPreprocessor
        # Use very low threshold to force neural oracle usage
        self.preprocessor = IntelligentPreprocessor(
            confidence_threshold=0.99,  # Force neural oracle
            use_neural_oracle=True
        )
    
    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        from src.core.executor import get_executor
        
        X_copy = X.copy()
        for col in X_copy.columns:
            try:
                result = self.preprocessor.preprocess_column(X_copy[col], col)
                action = result.action.value
                
                # Apply based on action
                if action == 'keep_as_is':
                    if X_copy[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                        X_copy[col] = X_copy[col].fillna(X_copy[col].mean())
                    else:
                        X_copy[col] = X_copy[col].fillna('missing')
                        le = LabelEncoder()
                        X_copy[col] = le.fit_transform(X_copy[col].astype(str))
                elif action in ['fill_null_mean', 'fill_null_median']:
                    X_copy[col] = X_copy[col].fillna(X_copy[col].median())
                elif action == 'standard_scale':
                    X_copy[col] = X_copy[col].fillna(X_copy[col].mean())
                    scaler = StandardScaler()
                    X_copy[col] = scaler.fit_transform(X_copy[[col]])
                elif action in ['label_encode', 'onehot_encode']:
                    X_copy[col] = X_copy[col].fillna('missing')
                    le = LabelEncoder()
                    X_copy[col] = le.fit_transform(X_copy[col].astype(str))
                elif action == 'drop_column':
                    X_copy = X_copy.drop(columns=[col])
                else:
                    if X_copy[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                        X_copy[col] = X_copy[col].fillna(X_copy[col].mean())
                    else:
                        X_copy[col] = X_copy[col].fillna('missing')
                        le = LabelEncoder()
                        X_copy[col] = le.fit_transform(X_copy[col].astype(str))
            except Exception as e:
                logger.warning(f"Error processing {col}: {e}")
                if X_copy[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    X_copy[col] = X_copy[col].fillna(X_copy[col].mean())
                else:
                    X_copy[col] = X_copy[col].fillna('missing')
                    le = LabelEncoder()
                    X_copy[col] = le.fit_transform(X_copy[col].astype(str))
        
        return X_copy


class AuroraHybridVariant(AblationVariant):
    """Full AURORA hybrid (symbolic + neural)."""
    
    def __init__(self):
        from src.core.preprocessor import IntelligentPreprocessor
        self.preprocessor = IntelligentPreprocessor(
            use_neural_oracle=True,
            confidence_threshold=0.65
        )
    
    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        for col in X_copy.columns:
            try:
                result = self.preprocessor.preprocess_column(X_copy[col], col)
                action = result.action.value
                
                # Apply based on action
                if action == 'keep_as_is':
                    if X_copy[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                        X_copy[col] = X_copy[col].fillna(X_copy[col].mean())
                    else:
                        X_copy[col] = X_copy[col].fillna('missing')
                        le = LabelEncoder()
                        X_copy[col] = le.fit_transform(X_copy[col].astype(str))
                elif action in ['fill_null_mean', 'fill_null_median']:
                    X_copy[col] = X_copy[col].fillna(X_copy[col].median())
                elif action == 'standard_scale':
                    X_copy[col] = X_copy[col].fillna(X_copy[col].mean())
                    scaler = StandardScaler()
                    X_copy[col] = scaler.fit_transform(X_copy[[col]])
                elif action in ['label_encode', 'onehot_encode']:
                    X_copy[col] = X_copy[col].fillna('missing')
                    le = LabelEncoder()
                    X_copy[col] = le.fit_transform(X_copy[col].astype(str))
                elif action == 'drop_column':
                    X_copy = X_copy.drop(columns=[col])
                else:
                    if X_copy[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                        X_copy[col] = X_copy[col].fillna(X_copy[col].mean())
                    else:
                        X_copy[col] = X_copy[col].fillna('missing')
                        le = LabelEncoder()
                        X_copy[col] = le.fit_transform(X_copy[col].astype(str))
            except Exception as e:
                logger.warning(f"Error processing {col}: {e}")
                if X_copy[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    X_copy[col] = X_copy[col].fillna(X_copy[col].mean())
                else:
                    X_copy[col] = X_copy[col].fillna('missing')
                    le = LabelEncoder()
                    X_copy[col] = le.fit_transform(X_copy[col].astype(str))
        
        return X_copy


class ColabEvaluation:
    """Main evaluation pipeline for Colab."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.eval_config = config['evaluation']
        self.results_dir = Path(self.eval_config['results_dir'])
        self.figures_dir = self.results_dir / 'figures'
        self.checkpoint_dir = self.results_dir / 'checkpoints'
        
        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize checkpoint manager
        self.checkpoint_mgr = CheckpointManager(self.checkpoint_dir)
        
        # Results storage
        self.ablation_results = {}
        self.benchmark_results = {}
        self.statistical_results = {}
    
    def setup(self) -> bool:
        """Stage 1: Setup and verify environment."""
        print("\n" + "=" * 70)
        print("STAGE 1: Setup")
        print("=" * 70)
        
        # Check environment
        if is_colab():
            print("✅ Running in Google Colab")
        else:
            print("ℹ️ Running locally (not in Colab)")
        
        # Verify AURORA components
        try:
            from src.core.preprocessor import IntelligentPreprocessor
            preprocessor = IntelligentPreprocessor()
            print("✅ AURORA preprocessor loaded")
        except Exception as e:
            print(f"❌ Failed to load preprocessor: {e}")
            return False
        
        # Check for neural oracle
        try:
            from src.neural.oracle import get_neural_oracle
            print("✅ Neural oracle available")
        except Exception as e:
            print(f"⚠️ Neural oracle not available: {e}")
        
        # Check plotting
        if PLOTTING_AVAILABLE:
            print("✅ Matplotlib/Seaborn available")
        else:
            print("⚠️ Matplotlib/Seaborn not available (no figures)")
        
        # Check PDF generation
        if PDF_AVAILABLE:
            print("✅ ReportLab available")
        else:
            print("⚠️ ReportLab not available (no PDF report)")
        
        print(f"\n📁 Results directory: {self.results_dir}")
        return True
    
    def download_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Stage 2: Download OpenML datasets."""
        print("\n" + "=" * 70)
        print("STAGE 2: Download Datasets")
        print("=" * 70)
        
        # Check for checkpoint
        checkpoint = self.checkpoint_mgr.load_checkpoint('datasets')
        if checkpoint:
            print("✅ Resuming from checkpoint")
            return checkpoint
        
        datasets = {}
        dataset_names = self.eval_config['datasets'][:self.eval_config['num_datasets']]
        dataset_ids = self.eval_config['dataset_ids']
        
        for name in tqdm(dataset_names, desc="Downloading"):
            try:
                dataset_id = dataset_ids.get(name)
                if not dataset_id:
                    print(f"⚠️ No ID for {name}, skipping")
                    continue
                
                print(f"\n  Downloading {name} (ID: {dataset_id})...")
                data = fetch_openml(data_id=dataset_id, as_frame=True, parser='auto')
                
                # Get dataframe
                if hasattr(data, 'frame') and data.frame is not None:
                    df = data.frame
                else:
                    df = pd.concat([data.data, pd.Series(data.target, name='target')], axis=1)
                
                # Determine target column
                target_col = data.target_names[0] if hasattr(data, 'target_names') and data.target_names else df.columns[-1]
                
                datasets[name] = {
                    'df': df,
                    'target_col': target_col,
                    'n_rows': len(df),
                    'n_cols': len(df.columns),
                    'task': 'classification' if df[target_col].nunique() < 20 else 'regression'
                }
                
                print(f"  ✅ {name}: {len(df)} rows, {len(df.columns)} columns")
                
                # Save checkpoint after each download
                self.checkpoint_mgr.save_checkpoint('datasets_progress', {
                    'completed': list(datasets.keys())
                })
                
            except Exception as e:
                print(f"  ❌ Failed to download {name}: {e}")
                continue
        
        # Save final checkpoint (without DataFrames for JSON serialization)
        checkpoint_data = {
            name: {k: v for k, v in info.items() if k != 'df'}
            for name, info in datasets.items()
        }
        self.checkpoint_mgr.save_checkpoint('datasets', checkpoint_data)
        
        print(f"\n✅ Downloaded {len(datasets)} datasets")
        return datasets
    
    def run_ablation_study(self, datasets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Stage 3: Run ablation study."""
        print("\n" + "=" * 70)
        print("STAGE 3: Ablation Study")
        print("=" * 70)
        
        # Check for checkpoint
        checkpoint = self.checkpoint_mgr.load_checkpoint('ablation')
        if checkpoint:
            print("✅ Resuming from checkpoint")
            self.ablation_results = checkpoint
            return checkpoint
        
        # Create variants
        variants = {
            'random': RandomBaseline(),
            'symbolic_only': SymbolicOnlyVariant(),
            'neural_only': NeuralOnlyVariant(),
            'aurora_hybrid': AuroraHybridVariant()
        }
        
        results = {}
        cv_folds = self.eval_config['cv_folds']
        n_estimators = self.eval_config['random_forest_estimators']
        max_depth = self.eval_config['random_forest_max_depth']
        
        for dataset_name, dataset_info in tqdm(datasets.items(), desc="Datasets"):
            if 'df' not in dataset_info:
                # Need to reload dataset
                try:
                    dataset_id = self.eval_config['dataset_ids'].get(dataset_name)
                    data = fetch_openml(data_id=dataset_id, as_frame=True, parser='auto')
                    if hasattr(data, 'frame') and data.frame is not None:
                        df = data.frame
                    else:
                        df = pd.concat([data.data, pd.Series(data.target, name='target')], axis=1)
                    dataset_info['df'] = df
                except Exception as e:
                    print(f"  ❌ Failed to reload {dataset_name}: {e}")
                    continue
            
            df = dataset_info['df']
            target_col = dataset_info['target_col']
            task = dataset_info.get('task', 'classification')
            
            results[dataset_name] = {}
            
            print(f"\n  {dataset_name}:")
            
            for variant_name, variant in tqdm(variants.items(), desc="Variants", leave=False):
                try:
                    start_time = time.time()
                    
                    # Prepare data
                    X = df.drop(columns=[target_col])
                    y = df[target_col]
                    
                    # Encode target if needed (only for categorical/object types)
                    if y.dtype == 'object' or y.dtype.name == 'category':
                        le = LabelEncoder()
                        y = le.fit_transform(y.astype(str))
                    
                    # Preprocess with variant
                    X_processed = variant.preprocess(X)
                    
                    # Handle remaining NaN values with column-appropriate defaults
                    for col in X_processed.columns:
                        if X_processed[col].isna().any():
                            if X_processed[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                                X_processed[col] = X_processed[col].fillna(X_processed[col].median())
                            else:
                                X_processed[col] = X_processed[col].fillna(0)
                    
                    # Select model
                    if task == 'classification':
                        model = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=42
                        )
                        scoring = 'accuracy'
                    else:
                        model = RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=42
                        )
                        scoring = 'r2'
                    
                    # Cross-validation
                    scores = cross_val_score(model, X_processed, y, cv=cv_folds, scoring=scoring)
                    
                    elapsed_ms = (time.time() - start_time) * 1000
                    
                    results[dataset_name][variant_name] = {
                        'accuracy': float(np.mean(scores)),
                        'std': float(np.std(scores)),
                        'latency_ms': float(elapsed_ms / len(X.columns)),
                        'scores': [float(s) for s in scores]
                    }
                    
                    print(f"    {variant_name}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
                    
                except Exception as e:
                    print(f"    ❌ {variant_name} failed: {e}")
                    results[dataset_name][variant_name] = {
                        'accuracy': 0.0,
                        'std': 0.0,
                        'latency_ms': 0.0,
                        'error': str(e)
                    }
            
            # Clean up memory
            del df
            gc.collect()
            
            # Save checkpoint
            self.checkpoint_mgr.save_checkpoint('ablation', results)
        
        self.ablation_results = results
        
        # Save final results
        with open(self.results_dir / 'ablation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Ablation study complete")
        return results
    
    def run_benchmarks(self, datasets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Stage 4: Comprehensive benchmarks."""
        print("\n" + "=" * 70)
        print("STAGE 4: Comprehensive Benchmarks")
        print("=" * 70)
        
        # Check for checkpoint
        checkpoint = self.checkpoint_mgr.load_checkpoint('benchmarks')
        if checkpoint:
            print("✅ Resuming from checkpoint")
            self.benchmark_results = checkpoint
            return checkpoint
        
        from src.core.preprocessor import IntelligentPreprocessor
        preprocessor = IntelligentPreprocessor()
        
        results = {}
        
        for dataset_name, dataset_info in tqdm(datasets.items(), desc="Benchmarking"):
            if 'df' not in dataset_info:
                try:
                    dataset_id = self.eval_config['dataset_ids'].get(dataset_name)
                    data = fetch_openml(data_id=dataset_id, as_frame=True, parser='auto')
                    if hasattr(data, 'frame') and data.frame is not None:
                        df = data.frame
                    else:
                        df = pd.concat([data.data, pd.Series(data.target, name='target')], axis=1)
                    dataset_info['df'] = df
                except Exception as e:
                    print(f"  ❌ Failed to reload {dataset_name}: {e}")
                    continue
            
            df = dataset_info['df']
            target_col = dataset_info['target_col']
            
            column_results = []
            sources = defaultdict(int)
            actions = defaultdict(int)
            
            print(f"\n  {dataset_name}:")
            
            for col in tqdm(df.columns, desc=f"  Columns", leave=False):
                if col == target_col:
                    continue
                
                try:
                    start_time = time.time()
                    result = preprocessor.preprocess_column(df[col], col)
                    elapsed_ms = (time.time() - start_time) * 1000
                    
                    column_results.append({
                        'column': col,
                        'action': result.action.value,
                        'confidence': float(result.confidence),
                        'source': result.source,
                        'latency_ms': float(elapsed_ms),
                        'explanation': result.explanation[:100] if result.explanation else ''
                    })
                    
                    sources[result.source] += 1
                    actions[result.action.value] += 1
                    
                except Exception as e:
                    logger.warning(f"Error on {col}: {e}")
            
            # Calculate metrics
            if column_results:
                avg_confidence = np.mean([r['confidence'] for r in column_results])
                avg_latency = np.mean([r['latency_ms'] for r in column_results])
            else:
                avg_confidence = 0.0
                avg_latency = 0.0
            
            results[dataset_name] = {
                'columns_processed': len(column_results),
                'avg_confidence': float(avg_confidence),
                'avg_latency_ms': float(avg_latency),
                'source_breakdown': dict(sources),
                'action_distribution': dict(actions),
                'column_results': column_results
            }
            
            print(f"    Columns: {len(column_results)}, Avg confidence: {avg_confidence:.1%}")
            
            # Clean up
            del df
            gc.collect()
            
            # Save checkpoint
            self.checkpoint_mgr.save_checkpoint('benchmarks', results)
        
        self.benchmark_results = results
        
        # Save results
        with open(self.results_dir / 'benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Benchmarks complete")
        return results
    
    def run_statistical_analysis(self) -> Dict[str, Any]:
        """Stage 5: Statistical analysis."""
        print("\n" + "=" * 70)
        print("STAGE 5: Statistical Analysis")
        print("=" * 70)
        
        if not self.ablation_results:
            print("⚠️ No ablation results to analyze")
            return {}
        
        results = {
            'comparisons': {},
            'summary': {}
        }
        
        # Collect scores by variant
        variant_scores = defaultdict(list)
        for dataset_name, dataset_results in self.ablation_results.items():
            for variant_name, variant_results in dataset_results.items():
                if 'accuracy' in variant_results:
                    variant_scores[variant_name].append(variant_results['accuracy'])
        
        # Calculate summary statistics
        for variant, scores in variant_scores.items():
            results['summary'][variant] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'n': len(scores)
            }
        
        # Pairwise t-tests
        variants = list(variant_scores.keys())
        min_samples = self.config['statistics']['min_samples_for_ttest']
        
        for i, v1 in enumerate(variants):
            for v2 in variants[i+1:]:
                s1 = variant_scores[v1]
                s2 = variant_scores[v2]
                
                if len(s1) >= min_samples and len(s2) >= min_samples:
                    try:
                        t_stat, p_value = stats.ttest_ind(s1, s2)
                        
                        # Effect size (Cohen's d)
                        pooled_std = np.sqrt((np.var(s1) + np.var(s2)) / 2)
                        effect_size = (np.mean(s1) - np.mean(s2)) / pooled_std if pooled_std > 0 else 0
                        
                        results['comparisons'][f"{v1}_vs_{v2}"] = {
                            't_statistic': float(t_stat),
                            'p_value': float(p_value),
                            'effect_size': float(effect_size),
                            'significant': p_value < 0.05,
                            'mean_diff': float(np.mean(s1) - np.mean(s2))
                        }
                        
                        print(f"  {v1} vs {v2}: p={p_value:.4f}, d={effect_size:.3f}")
                        
                    except Exception as e:
                        logger.warning(f"T-test failed for {v1} vs {v2}: {e}")
        
        self.statistical_results = results
        
        # Save results
        with open(self.results_dir / 'statistical_tests.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Statistical analysis complete")
        return results
    
    def generate_tables(self) -> str:
        """Stage 6: Generate paper tables in Markdown."""
        print("\n" + "=" * 70)
        print("STAGE 6: Generate Tables")
        print("=" * 70)
        
        tables = []
        
        # Table 1: Ablation Study Results
        tables.append("# AURORA V2 Evaluation Results\n")
        tables.append("## Table 1: Ablation Study Results\n")
        tables.append("| Dataset | Random | Symbolic-Only | Neural-Only | AURORA Hybrid |")
        tables.append("|---------|--------|---------------|-------------|---------------|")
        
        for dataset, results in self.ablation_results.items():
            row = f"| {dataset} "
            for variant in ['random', 'symbolic_only', 'neural_only', 'aurora_hybrid']:
                if variant in results and 'accuracy' in results[variant]:
                    acc = results[variant]['accuracy']
                    std = results[variant].get('std', 0)
                    row += f"| {acc:.3f} ± {std:.3f} "
                else:
                    row += "| N/A "
            row += "|"
            tables.append(row)
        
        tables.append("")
        
        # Table 2: Average Performance
        tables.append("## Table 2: Average Performance Across Datasets\n")
        tables.append("| Variant | Mean Accuracy | Std Dev |")
        tables.append("|---------|---------------|---------|")
        
        if self.statistical_results and 'summary' in self.statistical_results:
            for variant, stats in self.statistical_results['summary'].items():
                tables.append(f"| {variant} | {stats['mean']:.3f} | {stats['std']:.3f} |")
        
        tables.append("")
        
        # Table 3: Statistical Comparisons
        tables.append("## Table 3: Statistical Significance Tests\n")
        tables.append("| Comparison | p-value | Effect Size | Significant |")
        tables.append("|------------|---------|-------------|-------------|")
        
        if self.statistical_results and 'comparisons' in self.statistical_results:
            for comparison, stats in self.statistical_results['comparisons'].items():
                sig = "Yes" if stats['significant'] else "No"
                tables.append(f"| {comparison} | {stats['p_value']:.4f} | {stats['effect_size']:.3f} | {sig} |")
        
        tables.append("")
        
        # Table 4: Decision Source Breakdown
        tables.append("## Table 4: Decision Source Breakdown\n")
        tables.append("| Dataset | Symbolic | Neural | Fallback |")
        tables.append("|---------|----------|--------|----------|")
        
        for dataset, results in self.benchmark_results.items():
            sources = results.get('source_breakdown', {})
            total = sum(sources.values()) or 1
            sym_pct = sources.get('symbolic', 0) / total * 100
            neu_pct = sources.get('neural', 0) / total * 100
            fall_pct = sources.get('conservative_fallback', 0) / total * 100
            tables.append(f"| {dataset} | {sym_pct:.1f}% | {neu_pct:.1f}% | {fall_pct:.1f}% |")
        
        tables.append("")
        
        # Table 5: Performance Metrics
        tables.append("## Table 5: Performance Metrics\n")
        tables.append("| Dataset | Columns | Avg Confidence | Avg Latency (ms) |")
        tables.append("|---------|---------|----------------|------------------|")
        
        for dataset, results in self.benchmark_results.items():
            tables.append(f"| {dataset} | {results['columns_processed']} | "
                         f"{results['avg_confidence']:.1%} | {results['avg_latency_ms']:.2f} |")
        
        tables.append("")
        tables.append(f"\n*Generated on {datetime.now(timezone.utc).isoformat()}*")
        
        markdown = "\n".join(tables)
        
        # Save tables
        with open(self.results_dir / 'paper_tables.md', 'w') as f:
            f.write(markdown)
        
        print(f"✅ Tables saved to {self.results_dir / 'paper_tables.md'}")
        return markdown
    
    def generate_figures(self) -> None:
        """Stage 7: Generate figures."""
        print("\n" + "=" * 70)
        print("STAGE 7: Generate Figures")
        print("=" * 70)
        
        if not PLOTTING_AVAILABLE:
            print("⚠️ Matplotlib/Seaborn not available. Skipping figures.")
            return
        
        dpi = self.eval_config['figures_dpi']
        
        # Figure 1: Accuracy Comparison Bar Chart
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            datasets = list(self.ablation_results.keys())
            variants = ['random', 'symbolic_only', 'neural_only', 'aurora_hybrid']
            x = np.arange(len(datasets))
            width = 0.2
            
            for i, variant in enumerate(variants):
                accuracies = []
                errors = []
                for dataset in datasets:
                    if variant in self.ablation_results[dataset]:
                        accuracies.append(self.ablation_results[dataset][variant].get('accuracy', 0))
                        errors.append(self.ablation_results[dataset][variant].get('std', 0))
                    else:
                        accuracies.append(0)
                        errors.append(0)
                
                bars = ax.bar(x + i * width, accuracies, width, label=variant.replace('_', ' ').title(),
                             yerr=errors, capsize=3)
            
            ax.set_xlabel('Dataset')
            ax.set_ylabel('Accuracy')
            ax.set_title('AURORA V2 Ablation Study: Accuracy Comparison')
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(datasets, rotation=45, ha='right')
            ax.legend()
            ax.set_ylim(0, 1.1)
            
            plt.tight_layout()
            plt.savefig(self.figures_dir / 'accuracy_comparison.png', dpi=dpi)
            plt.close()
            print("  ✅ accuracy_comparison.png")
        except Exception as e:
            print(f"  ❌ accuracy_comparison.png failed: {e}")
        
        # Figure 2: Decision Sources Pie Chart
        try:
            # Aggregate sources across all datasets
            total_sources = defaultdict(int)
            for dataset, results in self.benchmark_results.items():
                for source, count in results.get('source_breakdown', {}).items():
                    total_sources[source] += count
            
            if total_sources:
                fig, ax = plt.subplots(figsize=(8, 8))
                labels = list(total_sources.keys())
                sizes = list(total_sources.values())
                colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
                
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
                ax.set_title('AURORA V2: Decision Source Distribution')
                
                plt.tight_layout()
                plt.savefig(self.figures_dir / 'decision_sources.png', dpi=dpi)
                plt.close()
                print("  ✅ decision_sources.png")
        except Exception as e:
            print(f"  ❌ decision_sources.png failed: {e}")
        
        # Figure 3: Latency Distribution
        try:
            all_latencies = []
            for dataset, results in self.benchmark_results.items():
                for col_result in results.get('column_results', []):
                    all_latencies.append(col_result.get('latency_ms', 0))
            
            if all_latencies:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(all_latencies, bins=50, edgecolor='black', alpha=0.7)
                ax.set_xlabel('Latency (ms)')
                ax.set_ylabel('Frequency')
                ax.set_title('AURORA V2: Latency Distribution')
                ax.axvline(np.mean(all_latencies), color='red', linestyle='--',
                          label=f'Mean: {np.mean(all_latencies):.2f}ms')
                ax.legend()
                
                plt.tight_layout()
                plt.savefig(self.figures_dir / 'latency_distribution.png', dpi=dpi)
                plt.close()
                print("  ✅ latency_distribution.png")
        except Exception as e:
            print(f"  ❌ latency_distribution.png failed: {e}")
        
        # Figure 4: Confusion Matrix (if ground truth validation exists)
        try:
            gt_path = self.results_dir / 'ground_truth_validation.json'
            if gt_path.exists():
                with open(gt_path, 'r') as f:
                    gt_results = json.load(f)
                
                if 'confusion_matrix' in gt_results and 'action_labels' in gt_results:
                    cm = np.array(gt_results['confusion_matrix'])
                    labels = gt_results['action_labels']
                    
                    fig, ax = plt.subplots(figsize=(12, 10))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                               xticklabels=labels, yticklabels=labels, ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title('AURORA V2: Confusion Matrix')
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0)
                    
                    plt.tight_layout()
                    plt.savefig(self.figures_dir / 'confusion_matrix.png', dpi=dpi)
                    plt.close()
                    print("  ✅ confusion_matrix.png")
        except Exception as e:
            print(f"  ❌ confusion_matrix.png failed: {e}")
        
        print(f"\n✅ Figures saved to {self.figures_dir}")
    
    def generate_pdf_report(self) -> None:
        """Stage 8: Generate PDF report."""
        print("\n" + "=" * 70)
        print("STAGE 8: Generate PDF Report")
        print("=" * 70)
        
        if not PDF_AVAILABLE:
            print("⚠️ ReportLab not available. Skipping PDF generation.")
            return
        
        try:
            pdf_path = self.results_dir / 'EVALUATION_REPORT.pdf'
            doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30
            )
            story.append(Paragraph("AURORA V2 Evaluation Report", title_style))
            story.append(Spacer(1, 20))
            
            # Executive Summary
            story.append(Paragraph("Executive Summary", styles['Heading2']))
            
            summary_text = f"""
            This report presents the comprehensive evaluation of AURORA V2, an intelligent 
            data preprocessing system. The evaluation covers {len(self.ablation_results)} datasets 
            with four preprocessing variants: Random baseline, Symbolic-only, Neural-only, 
            and the full AURORA Hybrid approach.
            """
            story.append(Paragraph(summary_text, styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Key Findings
            story.append(Paragraph("Key Findings", styles['Heading2']))
            
            if self.statistical_results and 'summary' in self.statistical_results:
                for variant, stats in self.statistical_results['summary'].items():
                    finding = f"• {variant.replace('_', ' ').title()}: Mean accuracy = {stats['mean']:.3f}"
                    story.append(Paragraph(finding, styles['Normal']))
            
            story.append(Spacer(1, 20))
            
            # Statistical Significance
            story.append(Paragraph("Statistical Significance", styles['Heading2']))
            
            if self.statistical_results and 'comparisons' in self.statistical_results:
                for comparison, stats in self.statistical_results['comparisons'].items():
                    sig_text = "significant" if stats['significant'] else "not significant"
                    finding = f"• {comparison}: p-value = {stats['p_value']:.4f} ({sig_text})"
                    story.append(Paragraph(finding, styles['Normal']))
            
            story.append(Spacer(1, 20))
            
            # Timestamp
            story.append(Paragraph(f"Generated: {datetime.now(timezone.utc).isoformat()}", 
                                  styles['Normal']))
            
            # Add figures if they exist
            figures = ['accuracy_comparison.png', 'decision_sources.png', 
                      'latency_distribution.png', 'confusion_matrix.png']
            
            for fig_name in figures:
                fig_path = self.figures_dir / fig_name
                if fig_path.exists():
                    story.append(Spacer(1, 20))
                    story.append(Paragraph(fig_name.replace('.png', '').replace('_', ' ').title(),
                                          styles['Heading3']))
                    # Add image (scaled to fit page)
                    img = Image(str(fig_path), width=400, height=300)
                    story.append(img)
            
            doc.build(story)
            print(f"✅ PDF report saved to {pdf_path}")
            
        except Exception as e:
            print(f"❌ PDF generation failed: {e}")
    
    def run_all(self) -> Dict[str, Any]:
        """Run complete evaluation pipeline."""
        start_time = time.time()
        
        print("\n" + "=" * 70)
        print("AURORA V2 - Comprehensive Colab Evaluation")
        print("=" * 70)
        print(f"Started at: {datetime.now(timezone.utc).isoformat()}")
        
        # Stage 1: Setup
        if not self.setup():
            print("❌ Setup failed. Aborting.")
            return {}
        
        # Stage 2: Download datasets
        datasets = self.download_datasets()
        if not datasets:
            print("❌ No datasets available. Aborting.")
            return {}
        
        # Stage 3: Ablation study
        self.run_ablation_study(datasets)
        
        # Stage 4: Benchmarks
        self.run_benchmarks(datasets)
        
        # Stage 5: Statistical analysis
        self.run_statistical_analysis()
        
        # Stage 6: Generate tables
        self.generate_tables()
        
        # Stage 7: Generate figures
        self.generate_figures()
        
        # Stage 8: Generate PDF
        self.generate_pdf_report()
        
        # Final summary
        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print("EVALUATION COMPLETE")
        print("=" * 70)
        print(f"Total time: {elapsed / 60:.1f} minutes")
        print(f"\n📁 Results saved to: {self.results_dir}")
        print("\nOutput files:")
        for f in self.results_dir.iterdir():
            if f.is_file():
                print(f"  • {f.name}")
        
        return {
            'ablation': self.ablation_results,
            'benchmarks': self.benchmark_results,
            'statistics': self.statistical_results,
            'elapsed_minutes': elapsed / 60
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run AURORA V2 comprehensive evaluation'
    )
    parser.add_argument(
        '--datasets',
        type=int,
        default=5,
        help='Number of datasets to evaluate (default: 5)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to evaluation config YAML file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with command line args
    config['evaluation']['num_datasets'] = args.datasets
    config['evaluation']['verbose'] = args.verbose
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run evaluation
    evaluator = ColabEvaluation(config)
    results = evaluator.run_all()
    
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
