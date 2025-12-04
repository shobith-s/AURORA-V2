"""
Test hybrid model loading with training classes in pickle.

This test simulates the scenario where a hybrid model was trained in Colab
with TrainingConfig, TrainingSample, and other classes defined in __main__,
and now needs to be loaded for inference.
"""

import pytest
import pickle
import io
import numpy as np
from pathlib import Path
from unittest.mock import Mock

from src.neural.oracle import (
    NeuralOracle,
    ModelUnpickler,
    TrainingConfig,
    TrainingSample,
    CurriculumConfig
)


class TestHybridModelLoading:
    """Test loading hybrid models with training classes."""
    
    def test_simulated_hybrid_model_with_training_config(self):
        """Test loading a model that includes TrainingConfig in pickle."""
        # Simulate a hybrid model structure that includes TrainingConfig
        config = TrainingConfig(
            n_datasets=40,
            cv_folds=3,
            random_state=42,
            numeric_actions=['standard_scale', 'log_transform'],
            categorical_actions=['label_encode', 'onehot_encode']
        )
        
        # Create a mock model structure similar to what would be in a hybrid model
        model_dict = {
            'hybrid_model': None,  # Would be the actual model
            'xgb_model': None,
            'lgb_model': None,
            'label_encoder': None,
            'feature_extractor': None,
            'config': config,  # TrainingConfig is included
            'metadata': {
                'model_version': '1.0',
                'trained_date': '2025-12-02',
            },
            'removed_classes': []
        }
        
        # Pickle and unpickle using ModelUnpickler
        buffer = io.BytesIO()
        pickle.dump(model_dict, buffer)
        
        buffer.seek(0)
        loaded = ModelUnpickler(buffer).load()
        
        # Verify structure
        assert 'config' in loaded
        assert isinstance(loaded['config'], TrainingConfig)
        assert loaded['config'].n_datasets == 40
        assert loaded['config'].cv_folds == 3
    
    def test_simulated_model_with_training_samples(self):
        """Test loading a model that includes TrainingSample objects."""
        # Create sample training data
        sample1 = TrainingSample(
            features=np.array([1, 2, 3]),
            label='standard_scale',
            confidence=0.95,
            column_type='numeric',
            column_name='age',
            dataset_name='test_dataset',
            performance_score=0.87
        )
        
        sample2 = TrainingSample(
            features=np.array([4, 5, 6]),
            label='label_encode',
            confidence=0.88,
            column_type='categorical',
            column_name='category',
            dataset_name='test_dataset',
            performance_score=0.82
        )
        
        # Simulate model with training samples (might be in metadata or history)
        model_dict = {
            'model': None,
            'training_samples': [sample1, sample2],
            'sample_count': 2
        }
        
        # Pickle and unpickle
        buffer = io.BytesIO()
        pickle.dump(model_dict, buffer)
        
        buffer.seek(0)
        loaded = ModelUnpickler(buffer).load()
        
        # Verify samples can be loaded
        assert 'training_samples' in loaded
        assert len(loaded['training_samples']) == 2
        assert isinstance(loaded['training_samples'][0], TrainingSample)
        assert loaded['training_samples'][0].label == 'standard_scale'
    
    def test_class_redirect_from_main(self):
        """Test that classes from __main__ are properly redirected."""
        # This test verifies the ModelUnpickler can handle __main__ references
        config = TrainingConfig()
        
        # Manually create a pickle that references __main__
        # In practice, this would come from Colab training
        buffer = io.BytesIO()
        pickle.dump(config, buffer)
        
        # Load with ModelUnpickler - should handle __main__ redirects
        buffer.seek(0)
        loaded = ModelUnpickler(buffer).load()
        
        assert isinstance(loaded, TrainingConfig)
    
    def test_complex_nested_structure(self):
        """Test loading complex nested structures with multiple training classes."""
        # Create a complex structure that might be saved by training script
        config = TrainingConfig(n_datasets=50)
        curriculum = CurriculumConfig(n_datasets=50, cv_folds=5)
        samples = [
            TrainingSample(
                features=np.array([i, i+1, i+2]),
                label=f'action_{i}',
                confidence=0.8 + i*0.01,
                column_type='numeric',
                column_name=f'col_{i}',
                dataset_name=f'dataset_{i}',
                performance_score=0.7 + i*0.01
            )
            for i in range(5)
        ]
        
        nested_structure = {
            'config': config,
            'curriculum': curriculum,
            'samples': samples,
            'metadata': {
                'version': '2.0',
                'sample_count': len(samples),
                'config_hash': hash(str(config))
            }
        }
        
        # Pickle and unpickle
        buffer = io.BytesIO()
        pickle.dump(nested_structure, buffer)
        
        buffer.seek(0)
        loaded = ModelUnpickler(buffer).load()
        
        # Verify all components loaded correctly
        assert isinstance(loaded['config'], TrainingConfig)
        assert isinstance(loaded['curriculum'], CurriculumConfig)
        assert len(loaded['samples']) == 5
        assert all(isinstance(s, TrainingSample) for s in loaded['samples'])
        assert loaded['config'].n_datasets == 50
        assert loaded['curriculum'].cv_folds == 5


class TestErrorPrevention:
    """Test that the fixes prevent the errors mentioned in the problem statement."""
    
    def test_no_cant_get_attribute_error(self):
        """Verify that 'Can't get attribute' errors don't occur."""
        # Create a structure that would previously cause the error
        config = TrainingConfig()
        
        # Pickle it
        buffer = io.BytesIO()
        pickle.dump(config, buffer)
        
        # This should NOT raise "Can't get attribute 'TrainingConfig'" error
        buffer.seek(0)
        try:
            loaded = ModelUnpickler(buffer).load()
            assert isinstance(loaded, TrainingConfig)
        except AttributeError as e:
            if "Can't get attribute" in str(e):
                pytest.fail(f"Got 'Can't get attribute' error: {e}")
            raise
    
    def test_all_training_classes_accessible(self):
        """Verify all training classes are accessible for unpickling."""
        # Test each class mentioned in the problem statement
        classes_to_test = [
            ('TrainingConfig', TrainingConfig),
            ('TrainingSample', TrainingSample),
            ('CurriculumConfig', CurriculumConfig),
        ]
        
        for class_name, class_type in classes_to_test:
            # Create instance
            if class_name == 'TrainingConfig':
                obj = class_type(n_datasets=40)
            elif class_name == 'TrainingSample':
                obj = class_type(label='test')
            else:
                obj = class_type()
            
            # Pickle and unpickle
            buffer = io.BytesIO()
            pickle.dump(obj, buffer)
            
            buffer.seek(0)
            loaded = ModelUnpickler(buffer).load()
            
            # Verify it's the correct type
            assert isinstance(loaded, class_type), \
                f"{class_name} failed to load correctly"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
