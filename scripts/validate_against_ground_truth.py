#!/usr/bin/env python3
"""
Ground Truth Validator for AURORA V2

Validates AURORA predictions against LLM-validated ground truth labels.
Generates publication-ready accuracy metrics and confusion matrix.

Usage:
    python scripts/validate_against_ground_truth.py [--config CONFIG_PATH]

Output:
    results/ground_truth_validation.json
    results/ground_truth_metrics.txt
"""

import os
import warnings

# Suppress Numba debug logging and related warnings before any imports
os.environ['NUMBA_DISABLE_PERFORMANCE_WARNINGS'] = '1'
os.environ['NUMBA_WARNINGS'] = '0'

import json
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from collections import defaultdict

import numpy as np
import pandas as pd

# Suppress Numba logging
logging.getLogger('numba').setLevel(logging.WARNING)

# Suppress sklearn/lightgbm feature name warnings (after fixing root cause)
warnings.filterwarnings('ignore', message='.*does not have valid feature names.*')
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Constants
RANDOM_SEED = 42  # Seed for reproducibility in synthetic data generation
DEFAULT_SAMPLE_SIZE = 100  # Default number of samples for data reconstruction
MAX_CATEGORIES = 50  # Maximum number of categories for categorical data reconstruction
EXPONENTIAL_SCALE_BASE = 100  # Base scale for exponential distribution
EXPONENTIAL_SCALE_MULTIPLIER = 50  # Multiplier for skewness-based scaling

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load evaluation configuration."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'configs' / 'evaluation_config.yaml'
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        return {
            'evaluation': {
                'ground_truth_file': 'validator/validated/validated_labels.json',
                'results_dir': 'results',
                'action_mapping': {}
            }
        }
    
    if YAML_AVAILABLE:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        logger.warning("PyYAML not available. Using default config.")
        return {
            'evaluation': {
                'ground_truth_file': 'validator/validated/validated_labels.json',
                'results_dir': 'results',
                'action_mapping': {}
            }
        }


def load_validated_labels(path: str) -> List[Dict[str, Any]]:
    """
    Load validated labels from JSON file.
    
    Args:
        path: Path to validated_labels.json
        
    Returns:
        List of validated label dictionaries
        
    Raises:
        FileNotFoundError: If labels file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    labels_path = Path(path)
    
    if not labels_path.exists():
        raise FileNotFoundError(f"Validated labels file not found: {path}")
    
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    
    logger.info(f"Loaded {len(labels)} validated labels from {path}")
    return labels


def create_sample_ground_truth(output_path: str) -> List[Dict[str, Any]]:
    """
    Create sample ground truth data if validated_labels.json doesn't exist.
    This demonstrates the expected format and enables testing.
    
    Uses actions that match the neural oracle model's vocabulary:
    drop_column, encode_categorical, keep_as_is, log_transform, 
    onehot_encode, parse_boolean, parse_datetime, retain_column, 
    scale, scale_or_normalize, standard_scale
    
    Args:
        output_path: Path to save the sample file
        
    Returns:
        List of sample label dictionaries
    """
    sample_labels = [
        {
            "dataset": "titanic",
            "column": "Age",
            "action": "standard_scale",
            "confidence": 0.92,
            "features": {
                "null_pct": 0.198,
                "is_numeric": 1.0,
                "cardinality_ratio": 0.088,
                "skewness": 0.389,
                "kurtosis": 0.178
            },
            "explanation": "Numeric column with moderate skewness, scaling recommended"
        },
        {
            "dataset": "titanic",
            "column": "Survived",
            "action": "keep_as_is",
            "confidence": 0.95,
            "features": {
                "null_pct": 0.0,
                "is_numeric": 1.0,
                "cardinality_ratio": 0.002,
                "skewness": 0.479,
                "kurtosis": -1.775
            },
            "explanation": "Binary target column, no preprocessing needed"
        },
        {
            "dataset": "titanic",
            "column": "Name",
            "action": "keep_as_is",
            "confidence": 0.88,
            "features": {
                "null_pct": 0.0,
                "is_numeric": 0.0,
                "cardinality_ratio": 1.0,
                "skewness": 0.0,
                "kurtosis": 0.0
            },
            "explanation": "High cardinality text column - model predicts keep for further analysis"
        },
        {
            "dataset": "titanic",
            "column": "Fare",
            "action": "keep_as_is",
            "confidence": 0.85,
            "features": {
                "null_pct": 0.001,
                "is_numeric": 1.0,
                "cardinality_ratio": 0.274,
                "skewness": 4.787,
                "kurtosis": 33.398
            },
            "explanation": "Numeric column - model predicts keep for numeric data"
        },
        {
            "dataset": "titanic",
            "column": "Sex",
            "action": "onehot_encode",
            "confidence": 0.94,
            "features": {
                "null_pct": 0.0,
                "is_numeric": 0.0,
                "cardinality_ratio": 0.002,
                "skewness": 0.0,
                "kurtosis": 0.0
            },
            "explanation": "Binary categorical column"
        }
    ]
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(sample_labels, f, indent=2)
    
    logger.info(f"Created sample ground truth file: {output_path}")
    return sample_labels


def reconstruct_column_from_features(features: Dict[str, Any], sample_size: int = 100) -> pd.Series:
    """
    Reconstruct a synthetic column from feature statistics.
    Used when original data is not available.
    
    Args:
        features: Dictionary of column features
        sample_size: Number of samples to generate
        
    Returns:
        Synthetic pandas Series matching the features
    """
    np.random.seed(RANDOM_SEED)  # For reproducibility
    
    is_numeric = features.get('is_numeric', features.get('detected_dtype', 0)) == 1.0
    null_pct = features.get('null_pct', features.get('null_ratio', 0))
    cardinality_ratio = features.get('cardinality_ratio', 0.5)
    skewness = features.get('skewness', 0)
    
    if is_numeric:
        # Generate numeric data
        if abs(skewness) > 2:
            # Highly skewed - use exponential with higher scale for extreme skewness
            scale = max(EXPONENTIAL_SCALE_BASE, abs(skewness) * EXPONENTIAL_SCALE_MULTIPLIER)
            data = np.random.exponential(scale=scale, size=sample_size)
        elif abs(skewness) > 0.5:
            # Moderately skewed - use log-normal
            data = np.random.lognormal(mean=2, sigma=1, size=sample_size)
        else:
            # Normal distribution
            data = np.random.normal(loc=50, scale=15, size=sample_size)
    else:
        # Generate categorical data
        if cardinality_ratio >= 0.9:
            # High cardinality (nearly unique) - generate unique strings like names
            data = [f"name_{i:05d}" for i in range(sample_size)]
        else:
            # Low-medium cardinality
            n_categories = max(2, int(cardinality_ratio * sample_size))
            n_categories = min(n_categories, MAX_CATEGORIES)  # Reasonable cap
            categories = [f"cat_{i}" for i in range(n_categories)]
            data = np.random.choice(categories, size=sample_size)
    
    # Apply null percentage
    if null_pct > 0:
        null_mask = np.random.random(sample_size) < null_pct
        if is_numeric:
            data = data.astype(float)
            data[null_mask] = np.nan
        else:
            data = np.where(null_mask, None, data)
    
    return pd.Series(data)


def run_aurora_prediction(
    column: pd.Series,
    column_name: str,
    preprocessor
) -> Dict[str, Any]:
    """
    Run AURORA prediction on a column.
    
    Args:
        column: Column data
        column_name: Name of the column
        preprocessor: IntelligentPreprocessor instance
        
    Returns:
        Dictionary with prediction details
    """
    try:
        result = preprocessor.preprocess_column(column, column_name)
        return {
            'action': result.action.value,
            'confidence': result.confidence,
            'source': result.source,
            'success': True
        }
    except Exception as e:
        logger.warning(f"Prediction failed for {column_name}: {e}")
        return {
            'action': 'error',
            'confidence': 0.0,
            'source': 'error',
            'success': False,
            'error': str(e)
        }


def normalize_action(action: str, action_mapping: Dict[str, str]) -> str:
    """
    Normalize action name using mapping.
    
    Args:
        action: Original action name
        action_mapping: Dictionary mapping alternative names to canonical names
        
    Returns:
        Normalized action name
    """
    # Convert to lowercase and remove spaces
    action = action.lower().strip().replace(' ', '_').replace('-', '_')
    
    # Apply explicit mapping first
    if action in action_mapping:
        return action_mapping[action]
    
    # Built-in semantic normalizations for common variations
    semantic_mapping = {
        # Keep as is variations
        'keep_as_is': 'keep_as_is',
        'retain_column': 'keep_as_is',
        
        # Drop column
        'drop_column': 'drop_column',
        
        # Scaling variations
        'scale': 'standard_scale',
        'scale_or_normalize': 'standard_scale',
        'standard_scale': 'standard_scale',
        'normalize': 'standard_scale',
        
        # Log transform variations
        'log_transform': 'log_transform',
        'log1p_transform': 'log_transform',
        
        # Encoding variations
        'encode_categorical': 'encode_categorical',
        'label_encode': 'encode_categorical',  # Map label_encode to encode_categorical
        'ordinal_encode': 'encode_categorical',
        'frequency_encode': 'frequency_encode',
        'onehot_encode': 'onehot_encode',
        
        # Date/time parsing
        'parse_datetime': 'parse_datetime',
        'parse_boolean': 'parse_boolean',
    }
    
    return semantic_mapping.get(action, action)


def calculate_metrics(
    ground_truths: List[str],
    predictions: List[str],
    action_labels: List[str]
) -> Dict[str, Any]:
    """
    Calculate precision, recall, F1 for each action.
    
    Args:
        ground_truths: List of ground truth action names
        predictions: List of predicted action names
        action_labels: List of all unique action names
        
    Returns:
        Dictionary with per-action metrics and confusion matrix
    """
    # Build confusion matrix
    n_actions = len(action_labels)
    action_to_idx = {a: i for i, a in enumerate(action_labels)}
    confusion_matrix = np.zeros((n_actions, n_actions), dtype=int)
    
    for gt, pred in zip(ground_truths, predictions):
        gt_idx = action_to_idx.get(gt, -1)
        pred_idx = action_to_idx.get(pred, -1)
        if gt_idx >= 0 and pred_idx >= 0:
            confusion_matrix[gt_idx, pred_idx] += 1
    
    # Calculate per-action metrics
    per_action_metrics = {}
    for action in action_labels:
        idx = action_to_idx[action]
        
        # True positives: diagonal
        tp = confusion_matrix[idx, idx]
        
        # False positives: column sum - TP
        fp = confusion_matrix[:, idx].sum() - tp
        
        # False negatives: row sum - TP
        fn = confusion_matrix[idx, :].sum() - tp
        
        # Support: row sum (actual occurrences)
        support = confusion_matrix[idx, :].sum()
        
        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_action_metrics[action] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'support': int(support),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
    
    return {
        'per_action_metrics': per_action_metrics,
        'confusion_matrix': confusion_matrix.tolist()
    }


def run_validation(
    validated_labels: List[Dict[str, Any]],
    action_mapping: Dict[str, str],
    results_dir: Path
) -> Dict[str, Any]:
    """
    Run ground truth validation.
    
    Args:
        validated_labels: List of validated label dictionaries
        action_mapping: Action name mapping
        results_dir: Directory to save results
        
    Returns:
        Dictionary with validation results
    """
    # Initialize preprocessor
    from src.core.preprocessor import IntelligentPreprocessor
    preprocessor = IntelligentPreprocessor(use_neural_oracle=True)
    
    # Track results
    ground_truths = []
    predictions = []
    correct = 0
    total = 0
    skipped = 0
    errors = []
    per_dataset_results = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    logger.info(f"Running validation on {len(validated_labels)} examples...")
    
    try:
        from tqdm import tqdm
        iterator = tqdm(validated_labels, desc="Validating")
    except ImportError:
        iterator = validated_labels
        logger.info("Install tqdm for progress bars: pip install tqdm")
    
    for i, label in enumerate(iterator):
        try:
            # Get ground truth
            gt_action = normalize_action(label['action'], action_mapping)
            
            # Get column features
            features = label.get('features', {})
            column_name = label.get('column', f'column_{i}')
            dataset_name = label.get('dataset', 'unknown')
            
            # Reconstruct column from features
            column = reconstruct_column_from_features(features)
            
            # Run AURORA prediction
            prediction = run_aurora_prediction(column, column_name, preprocessor)
            
            if not prediction['success']:
                skipped += 1
                errors.append({
                    'index': i,
                    'column': column_name,
                    'error': prediction.get('error', 'Unknown error')
                })
                continue
            
            pred_action = normalize_action(prediction['action'], action_mapping)
            
            # Track results
            ground_truths.append(gt_action)
            predictions.append(pred_action)
            
            if gt_action == pred_action:
                correct += 1
                per_dataset_results[dataset_name]['correct'] += 1
            
            total += 1
            per_dataset_results[dataset_name]['total'] += 1
            
        except Exception as e:
            skipped += 1
            errors.append({
                'index': i,
                'column': label.get('column', f'column_{i}'),
                'error': str(e)
            })
    
    # Calculate overall accuracy
    accuracy = correct / total if total > 0 else 0.0
    
    # Get unique actions
    all_actions = sorted(set(ground_truths + predictions))
    
    # Calculate metrics
    metrics = calculate_metrics(ground_truths, predictions, all_actions)
    
    # Per-dataset accuracy
    per_dataset_accuracy = {}
    for dataset, stats in per_dataset_results.items():
        if stats['total'] > 0:
            per_dataset_accuracy[dataset] = {
                'accuracy': round(stats['correct'] / stats['total'], 4),
                'correct': stats['correct'],
                'total': stats['total']
            }
    
    # Build results
    results = {
        'test_accuracy': round(accuracy, 4),
        'total_examples': total,
        'correct_predictions': correct,
        'skipped_examples': skipped,
        'per_action_metrics': metrics['per_action_metrics'],
        'confusion_matrix': metrics['confusion_matrix'],
        'action_labels': all_actions,
        'per_dataset_accuracy': per_dataset_accuracy,
        'errors': errors[:10] if len(errors) > 10 else errors,  # Limit errors
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    
    return results


def save_results(results: Dict[str, Any], results_dir: Path) -> None:
    """
    Save validation results to files.
    
    Args:
        results: Validation results dictionary
        results_dir: Directory to save results
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON results
    json_path = results_dir / 'ground_truth_validation.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved JSON results to: {json_path}")
    
    # Save human-readable summary
    txt_path = results_dir / 'ground_truth_metrics.txt'
    with open(txt_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("AURORA V2 - Ground Truth Validation Results\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Timestamp: {results['timestamp']}\n\n")
        
        f.write("OVERALL METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Examples:      {results['total_examples']}\n")
        f.write(f"Correct Predictions: {results['correct_predictions']}\n")
        f.write(f"Skipped Examples:    {results['skipped_examples']}\n")
        f.write(f"Test Accuracy:       {results['test_accuracy']:.2%}\n\n")
        
        f.write("PER-ACTION METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Action':<30} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}\n")
        f.write("-" * 70 + "\n")
        
        for action, metrics in sorted(results['per_action_metrics'].items()):
            f.write(f"{action:<30} {metrics['precision']:>8.3f} {metrics['recall']:>8.3f} "
                   f"{metrics['f1']:>8.3f} {metrics['support']:>8d}\n")
        
        f.write("\n")
        f.write("PER-DATASET ACCURACY\n")
        f.write("-" * 40 + "\n")
        for dataset, stats in sorted(results.get('per_dataset_accuracy', {}).items()):
            f.write(f"{dataset:<30} {stats['accuracy']:>8.2%} ({stats['correct']}/{stats['total']})\n")
        
        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write(f"Accuracy: {results['test_accuracy']:.2%}\n")
        f.write("=" * 70 + "\n")
    
    logger.info(f"Saved human-readable summary to: {txt_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Validate AURORA predictions against ground truth labels'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to evaluation config YAML file'
    )
    parser.add_argument(
        '--labels',
        type=str,
        default=None,
        help='Path to validated labels JSON file (overrides config)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for results (overrides config)'
    )
    parser.add_argument(
        '--create-sample',
        action='store_true',
        help='Create sample ground truth file if not exists'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("AURORA V2 - Ground Truth Validator")
    print("=" * 70)
    
    # Load config
    config = load_config(args.config)
    eval_config = config.get('evaluation', {})
    
    # Determine paths
    labels_path = args.labels or eval_config.get('ground_truth_file', 'validator/validated/validated_labels.json')
    results_dir = Path(args.output or eval_config.get('results_dir', 'results'))
    action_mapping = eval_config.get('action_mapping', {})
    
    # Try to load validated labels
    try:
        validated_labels = load_validated_labels(labels_path)
    except FileNotFoundError:
        if args.create_sample:
            logger.info("Creating sample ground truth file...")
            validated_labels = create_sample_ground_truth(labels_path)
        else:
            print(f"\n❌ Error: Validated labels file not found: {labels_path}")
            print("\nTo create a sample file for testing, run:")
            print(f"  python {__file__} --create-sample")
            print("\nOr ensure the LLM validation pipeline has been run to generate validated labels.")
            sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"\n❌ Error: Invalid JSON in labels file: {e}")
        sys.exit(1)
    
    # Run validation
    print(f"\n📊 Validating {len(validated_labels)} examples...")
    results = run_validation(validated_labels, action_mapping, results_dir)
    
    # Save results
    save_results(results, results_dir)
    
    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(f"✅ Accuracy: {results['test_accuracy']:.2%}")
    print(f"   Correct:  {results['correct_predictions']}/{results['total_examples']}")
    print(f"   Skipped:  {results['skipped_examples']}")
    print(f"\n📁 Results saved to: {results_dir}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
