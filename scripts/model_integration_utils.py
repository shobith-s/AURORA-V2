"""
Model Integration Utilities for Neural Oracle
===============================================

This module provides utilities for integrating trained neural oracle models
with the AURORA preprocessing system.

Functions:
- validate_model_compatibility(): Ensures model works with existing NeuralOracle
- export_for_aurora(): Packages model in correct format for AURORA
- test_on_bestsellers(): Validates model against known test case
- compare_with_current(): Benchmarks new model vs existing model

Usage:
    from scripts.model_integration_utils import (
        validate_model_compatibility,
        export_for_aurora,
        test_on_bestsellers,
        compare_with_current
    )
    
    # After training
    if validate_model_compatibility(model, feature_names):
        export_for_aurora(model, 'models/neural_oracle_meta_v3.pkl')
        accuracy = test_on_bestsellers('models/neural_oracle_meta_v3.pkl')
        print(f"Bestsellers accuracy: {accuracy:.1%}")
"""

import os
import sys
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Module-level constant for action mappings (performance optimization)
ACTION_MAPPINGS = {
    'scale': 'standard_scale',
    'scale_or_normalize': 'standard_scale',
    'encode_categorical': 'label_encode',
    'retain_column': 'keep_as_is',
}


def validate_model_compatibility(
    model: Any,
    feature_names: Optional[List[str]] = None,
    expected_classes: Optional[List[str]] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate that a trained model is compatible with NeuralOracle.
    
    Checks:
    1. Model has predict_proba method (required for confidence scores)
    2. Model has classes_ attribute (required for action mapping)
    3. Feature count matches expected (62 for MetaLearning, 20 for Minimal)
    4. Output classes are valid preprocessing actions
    
    Args:
        model: Trained model (VotingClassifier, XGBClassifier, etc.)
        feature_names: Expected feature names
        expected_classes: Expected output classes (action names)
    
    Returns:
        Tuple of (is_compatible, validation_details)
    """
    validation = {
        'has_predict_proba': False,
        'has_classes': False,
        'classes_valid': False,
        'feature_count_valid': False,
        'inference_test_passed': False,
        'errors': [],
    }
    
    # Check 1: predict_proba method
    if hasattr(model, 'predict_proba'):
        validation['has_predict_proba'] = True
    else:
        validation['errors'].append("Model missing predict_proba method")
    
    # Check 2: classes_ attribute
    if hasattr(model, 'classes_'):
        validation['has_classes'] = True
        validation['model_classes'] = list(model.classes_)
    else:
        validation['errors'].append("Model missing classes_ attribute")
    
    # Check 3: Validate classes are valid actions
    if validation['has_classes']:
        from src.core.actions import PreprocessingAction
        
        valid_actions = {a.value for a in PreprocessingAction}
        model_classes = set(model.classes_)
        
        # Check if all model classes map to valid actions
        invalid_classes = []
        for cls in model_classes:
            if cls not in valid_actions:
                # Use module-level constant for common mappings
                if cls not in ACTION_MAPPINGS and cls not in valid_actions:
                    invalid_classes.append(cls)
        
        if not invalid_classes:
            validation['classes_valid'] = True
        else:
            validation['errors'].append(f"Invalid classes: {invalid_classes}")
    
    # Check 4: Feature count
    if feature_names:
        validation['expected_features'] = len(feature_names)
        # Model should accept this many features
        try:
            # Try inference with dummy data
            n_features = len(feature_names)
            X_test = np.random.randn(1, n_features)
            _ = model.predict_proba(X_test)
            validation['feature_count_valid'] = True
        except Exception as e:
            validation['errors'].append(f"Feature count mismatch: {e}")
    else:
        validation['feature_count_valid'] = True  # Skip if not provided
    
    # Check 5: Inference test
    try:
        n_features = len(feature_names) if feature_names else 62
        X_test = np.random.randn(5, n_features)
        probs = model.predict_proba(X_test)
        preds = model.predict(X_test)
        
        if probs.shape[0] == 5 and len(preds) == 5:
            validation['inference_test_passed'] = True
        else:
            validation['errors'].append("Inference output shape mismatch")
    except Exception as e:
        validation['errors'].append(f"Inference test failed: {e}")
    
    # Overall compatibility
    is_compatible = (
        validation['has_predict_proba'] and
        validation['has_classes'] and
        validation['classes_valid'] and
        validation['inference_test_passed']
    )
    
    return is_compatible, validation


def export_for_aurora(
    model: Any,
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    include_wrapper: bool = False
) -> Path:
    """
    Export a trained model in the format expected by NeuralOracle.
    
    The NeuralOracle class supports two formats:
    1. Direct model: Just the sklearn model (VotingClassifier, etc.)
    2. Dictionary with model and metadata
    
    Args:
        model: Trained model
        output_path: Path to save the model
        metadata: Optional metadata (accuracy, training info, etc.)
        include_wrapper: If True, wraps in dictionary with metadata
    
    Returns:
        Path to saved model
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if include_wrapper and metadata:
        # Dictionary format with metadata
        export_data = {
            'model': model,
            'metadata': {
                'version': metadata.get('version', 'meta_v3'),
                'accuracy': metadata.get('accuracy', 0.0),
                'training_samples': metadata.get('training_samples', 0),
                'datasets': metadata.get('datasets', 0),
                'timestamp': metadata.get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S')),
                'feature_count': metadata.get('feature_count', 62),
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(export_data, f)
    else:
        # Direct model format (preferred for compatibility)
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
    
    # Verify the saved model loads correctly
    try:
        with open(output_path, 'rb') as f:
            loaded = pickle.load(f)
        
        if include_wrapper and metadata:
            assert 'model' in loaded, "Model not found in wrapper"
        else:
            assert hasattr(loaded, 'predict_proba'), "Loaded model missing predict_proba"
        
        logger.info(f"✅ Model exported successfully to {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Model verification failed: {e}")
        raise
    
    return output_path


def test_on_bestsellers(
    model_path: str,
    bestsellers_path: Optional[str] = None,
    expected_accuracy: float = 0.9
) -> Dict[str, Any]:
    """
    Test the model on bestsellers.csv dataset.
    
    This is the primary validation test case for neural oracle models.
    The model should achieve 90%+ accuracy on this dataset.
    
    Args:
        model_path: Path to the trained model
        bestsellers_path: Path to bestsellers.csv (optional, uses default)
        expected_accuracy: Minimum expected accuracy (default: 0.9)
    
    Returns:
        Dictionary with test results
    """
    from src.neural.oracle import NeuralOracle
    from src.core.preprocessor import IntelligentPreprocessor
    
    results = {
        'model_loaded': False,
        'data_loaded': False,
        'predictions_made': False,
        'accuracy': 0.0,
        'meets_threshold': False,
        'per_column_results': {},
        'errors': [],
    }
    
    # Load model
    try:
        oracle = NeuralOracle(Path(model_path))
        results['model_loaded'] = True
    except Exception as e:
        results['errors'].append(f"Model load failed: {e}")
        return results
    
    # Load bestsellers data
    if bestsellers_path is None:
        # Try common locations
        candidates = [
            project_root / 'data' / 'bestsellers.csv',
            project_root / 'data' / 'raw' / 'bestsellers.csv',
            Path.home() / 'bestsellers.csv',
        ]
        for candidate in candidates:
            if candidate.exists():
                bestsellers_path = str(candidate)
                break
    
    if bestsellers_path is None or not Path(bestsellers_path).exists():
        # Create sample bestsellers data for testing
        logger.warning("Bestsellers.csv not found, using synthetic data")
        df = pd.DataFrame({
            'Name': ['Book A', 'Book B', 'Book C'] * 100,
            'Author': ['Author 1', 'Author 2', 'Author 3'] * 100,
            'User Rating': np.random.uniform(3.5, 5.0, 300),
            'Reviews': np.random.randint(100, 50000, 300),
            'Price': np.random.randint(5, 100, 300),
            'Year': np.random.randint(2009, 2020, 300),
            'Genre': np.random.choice(['Fiction', 'Non Fiction'], 300),
        })
        results['using_synthetic'] = True
    else:
        df = pd.read_csv(bestsellers_path)
        results['data_loaded'] = True
    
    # Ground truth labels for bestsellers columns
    ground_truth = {
        'Name': 'drop_column',  # High cardinality text
        'Author': 'label_encode',  # Categorical with many values
        'User Rating': 'keep_as_is',  # Already good numeric
        'Reviews': 'log_transform',  # Skewed numeric
        'Price': 'keep_as_is',  # Numeric
        'Year': 'keep_as_is',  # Numeric (year)
        'Genre': 'onehot_encode',  # Low cardinality categorical
    }
    
    # Make predictions
    try:
        preprocessor = IntelligentPreprocessor(
            confidence_threshold=0.75,
            use_neural_oracle=True,
            enable_learning=False
        )
        
        correct = 0
        total = 0
        
        for col_name, expected_action in ground_truth.items():
            if col_name not in df.columns:
                continue
            
            column = df[col_name]
            result = preprocessor.preprocess_column(
                column=column,
                column_name=col_name,
                target_available=True,
                context='classification'
            )
            
            predicted = result.action.value
            is_correct = predicted == expected_action
            
            results['per_column_results'][col_name] = {
                'predicted': predicted,
                'expected': expected_action,
                'correct': is_correct,
                'confidence': result.confidence,
                'source': result.source,
            }
            
            if is_correct:
                correct += 1
            total += 1
        
        results['predictions_made'] = True
        results['accuracy'] = correct / total if total > 0 else 0.0
        results['meets_threshold'] = results['accuracy'] >= expected_accuracy
        results['correct'] = correct
        results['total'] = total
        
    except Exception as e:
        results['errors'].append(f"Prediction failed: {e}")
    
    return results


def compare_with_current(
    new_model_path: str,
    current_model_path: Optional[str] = None,
    test_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Compare a new model with the current production model.
    
    Args:
        new_model_path: Path to the new model
        current_model_path: Path to current model (uses default if None)
        test_df: Test DataFrame (uses synthetic if None)
    
    Returns:
        Comparison results including accuracy difference
    """
    from src.features.minimal_extractor import get_feature_extractor
    
    comparison = {
        'new_model': {'loaded': False, 'accuracy': 0.0},
        'current_model': {'loaded': False, 'accuracy': 0.0},
        'improvement': 0.0,
        'is_better': False,
    }
    
    # Default current model path - find latest model in models directory
    if current_model_path is None:
        models_dir = project_root / 'models'
        if models_dir.exists():
            # Find the latest neural_oracle model file
            model_files = sorted(
                models_dir.glob('neural_oracle*.pkl'),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            if model_files:
                current_model_path = str(model_files[0])
            else:
                comparison['current_model']['error'] = 'No model files found in models/'
                return comparison
        else:
            comparison['current_model']['error'] = 'Models directory not found'
            return comparison
    
    # Create test data if not provided
    if test_df is None:
        np.random.seed(42)
        n = 500
        test_df = pd.DataFrame({
            'numeric_normal': np.random.randn(n),
            'numeric_skewed': np.random.lognormal(0, 1, n),
            'categorical_low': np.random.choice(['A', 'B', 'C'], n),
            'categorical_high': np.random.choice([f'Cat_{i}' for i in range(50)], n),
            'id_column': range(n),
            'constant': ['same'] * n,
            'target': np.random.choice([0, 1], n),
        })
    
    extractor = get_feature_extractor()
    
    def evaluate_model(model_path: str, model_key: str):
        try:
            with open(model_path, 'rb') as f:
                loaded = pickle.load(f)
            
            # Handle wrapped vs direct model
            if isinstance(loaded, dict) and 'model' in loaded:
                model = loaded['model']
            else:
                model = loaded
            
            comparison[model_key]['loaded'] = True
            
            # Extract features for all columns
            correct = 0
            total = 0
            
            for col_name in test_df.columns:
                if col_name == 'target':
                    continue
                
                column = test_df[col_name]
                features = extractor.extract(column, col_name)
                X = features.to_array().reshape(1, -1)
                
                # Get prediction
                pred_probs = model.predict_proba(X)[0]
                pred_class = model.classes_[np.argmax(pred_probs)]
                
                # Simple heuristic for "ground truth"
                if 'id' in col_name.lower() or 'constant' in col_name.lower():
                    expected = 'drop_column'
                elif 'categorical' in col_name.lower():
                    expected = 'label_encode'
                elif 'numeric_skewed' in col_name.lower():
                    expected = 'log_transform'
                else:
                    expected = 'keep_as_is'
                
                if pred_class == expected:
                    correct += 1
                total += 1
            
            comparison[model_key]['accuracy'] = correct / total if total > 0 else 0.0
            
        except Exception as e:
            comparison[model_key]['error'] = str(e)
    
    # Evaluate both models
    evaluate_model(new_model_path, 'new_model')
    evaluate_model(current_model_path, 'current_model')
    
    # Calculate improvement
    if comparison['new_model']['loaded'] and comparison['current_model']['loaded']:
        comparison['improvement'] = (
            comparison['new_model']['accuracy'] - 
            comparison['current_model']['accuracy']
        )
        comparison['is_better'] = comparison['improvement'] > 0
    
    return comparison


def create_deployment_package(
    model_path: str,
    output_dir: str,
    include_tests: bool = True
) -> Dict[str, str]:
    """
    Create a deployment package for the neural oracle model.
    
    Creates:
    - Model file (pickle)
    - Metadata JSON
    - Test script
    - README with deployment instructions
    
    Args:
        model_path: Path to the trained model
        output_dir: Directory to create package in
        include_tests: Whether to include test script
    
    Returns:
        Dictionary of created file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    created_files = {}
    
    # Copy model
    import shutil
    model_dest = output_dir / 'neural_oracle_meta_v3.pkl'
    shutil.copy(model_path, model_dest)
    created_files['model'] = str(model_dest)
    
    # Create metadata
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    if isinstance(model, dict) and 'metadata' in model:
        metadata = model['metadata']
    else:
        metadata = {
            'version': 'meta_v3',
            'created': datetime.now().isoformat(),
        }
    
    metadata_path = output_dir / 'model_metadata.json'
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    created_files['metadata'] = str(metadata_path)
    
    # Create README
    readme_content = f"""# Neural Oracle Model Deployment

## Model Information
- Version: {metadata.get('version', 'meta_v3')}
- Created: {metadata.get('timestamp', 'unknown')}
- Accuracy: {metadata.get('accuracy', 'N/A')}

## Deployment Instructions

1. Copy `neural_oracle_meta_v3.pkl` to `models/` directory:
   ```bash
   cp neural_oracle_meta_v3.pkl /path/to/AURORA-V2/models/
   ```

2. Update NeuralOracle to use new model (optional, auto-detected):
   - The NeuralOracle class will automatically load the latest model
   - Or specify path: `NeuralOracle(model_path='models/neural_oracle_meta_v3.pkl')`

3. Verify deployment:
   ```python
   from src.neural.oracle import NeuralOracle
   oracle = NeuralOracle()
   # Should load without errors
   ```

## Testing

Run validation tests:
```bash
python -c "from scripts.model_integration_utils import test_on_bestsellers; print(test_on_bestsellers('models/neural_oracle_meta_v3.pkl'))"
```

## Rollback

If issues occur, restore previous model:
```bash
cp models/neural_oracle_v2_improved_20251129_150244.pkl models/neural_oracle_meta_v3.pkl
```
"""
    
    readme_path = output_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    created_files['readme'] = str(readme_path)
    
    logger.info(f"✅ Deployment package created in {output_dir}")
    return created_files


if __name__ == '__main__':
    # Demo
    print("Model Integration Utilities")
    print("=" * 50)
    
    # Check if there's a current model
    model_path = project_root / 'models' / 'neural_oracle_v2_improved_20251129_150244.pkl'
    if model_path.exists():
        print(f"\nFound existing model: {model_path}")
        
        # Test on bestsellers
        results = test_on_bestsellers(str(model_path))
        print(f"\nBestsellers test:")
        print(f"  Model loaded: {results['model_loaded']}")
        print(f"  Accuracy: {results['accuracy']:.1%}")
        print(f"  Meets threshold: {results['meets_threshold']}")
    else:
        print(f"\nNo model found at {model_path}")
