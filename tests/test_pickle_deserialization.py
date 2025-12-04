"""
Tests for pickle deserialization with stub classes.

This test suite validates that models trained in Colab notebooks with
classes defined in __main__ can be loaded successfully using stub classes.
"""

import pytest
import pickle
import io
from pathlib import Path

from src.neural.oracle import (
    TrainingConfig,
    CurriculumConfig,
    TrainingSample,
    ModelUnpickler
)


class TestStubClasses:
    """Test that stub classes can be instantiated and pickled."""
    
    def test_training_config_instantiation(self):
        """Test that TrainingConfig stub can be instantiated."""
        config = TrainingConfig()
        assert config is not None
        assert config.n_datasets == 40
        assert config.cv_folds == 3
        assert config.random_state == 42
    
    def test_curriculum_config_instantiation(self):
        """Test that CurriculumConfig stub can be instantiated."""
        config = CurriculumConfig()
        assert config is not None
        assert config.n_datasets == 40
        assert config.cv_folds == 3
    
    def test_training_sample_instantiation(self):
        """Test that TrainingSample stub can be instantiated."""
        sample = TrainingSample()
        assert sample is not None
        assert sample.label == ""
        assert sample.confidence == 0.0
    
    def test_stub_classes_pickle_roundtrip(self):
        """Test that stub classes can be pickled and unpickled."""
        config = TrainingConfig(n_datasets=50, cv_folds=5)
        sample = TrainingSample(label="test_action", confidence=0.8)
        
        # Pickle
        buffer = io.BytesIO()
        pickle.dump({'config': config, 'sample': sample}, buffer)
        
        # Unpickle
        buffer.seek(0)
        loaded = pickle.load(buffer)
        
        assert loaded['config'].n_datasets == 50
        assert loaded['config'].cv_folds == 5
        assert loaded['sample'].label == "test_action"
        assert loaded['sample'].confidence == 0.8


class TestModelUnpickler:
    """Test ModelUnpickler class redirection."""
    
    def test_class_redirects_defined(self):
        """Test that CLASS_REDIRECTS includes all necessary classes."""
        assert 'TrainingConfig' in ModelUnpickler.CLASS_REDIRECTS
        assert 'CurriculumConfig' in ModelUnpickler.CLASS_REDIRECTS
        assert 'TrainingSample' in ModelUnpickler.CLASS_REDIRECTS
        assert 'HybridPreprocessingOracle' in ModelUnpickler.CLASS_REDIRECTS
        assert 'MetaFeatureExtractor' in ModelUnpickler.CLASS_REDIRECTS
    
    def test_unpickler_redirects_training_classes(self):
        """Test that ModelUnpickler correctly redirects training classes."""
        # Create a pickle that references __main__.TrainingConfig
        config = TrainingConfig(n_datasets=30)
        
        # Pickle it normally
        buffer = io.BytesIO()
        pickle.dump(config, buffer)
        
        # Unpickle using ModelUnpickler
        buffer.seek(0)
        loaded = ModelUnpickler(buffer).load()
        
        assert loaded.n_datasets == 30
        assert isinstance(loaded, TrainingConfig)
    
    def test_model_loading_with_existing_model(self):
        """Test that existing model can be loaded successfully."""
        from src.neural.oracle import NeuralOracle
        
        model_path = Path('models/neural_oracle_v2_improved_20251129_150244.pkl')
        if not model_path.exists():
            pytest.skip("Model file not found")
        
        # This should not raise any errors
        oracle = NeuralOracle(model_path)
        assert oracle is not None


class TestBackwardCompatibility:
    """Test backward compatibility with existing models."""
    
    def test_existing_model_loads_without_errors(self):
        """Test that existing model loads without AttributeError."""
        from src.neural.oracle import NeuralOracle
        
        model_path = Path('models/neural_oracle_v2_improved_20251129_150244.pkl')
        if not model_path.exists():
            pytest.skip("Model file not found")
        
        # Should load successfully without "Can't get attribute" errors
        try:
            oracle = NeuralOracle(model_path)
            assert oracle is not None
        except AttributeError as e:
            if "Can't get attribute" in str(e):
                pytest.fail(f"Model loading failed with attribute error: {e}")
            raise


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
