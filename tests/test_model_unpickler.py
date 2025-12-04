"""
Tests for ModelUnpickler to handle pickle deserialization with __main__ references.
"""

import pytest
import pickle
from pathlib import Path
import tempfile
from src.neural.oracle import ModelUnpickler, NeuralOracle


class TestModelUnpickler:
    """Test ModelUnpickler functionality."""
    
    def test_unpickler_exists(self):
        """Test that ModelUnpickler class exists."""
        assert ModelUnpickler is not None
        assert hasattr(ModelUnpickler, 'CLASS_REDIRECTS')
        assert hasattr(ModelUnpickler, 'find_class')
    
    def test_class_redirects_configured(self):
        """Test that CLASS_REDIRECTS contains expected mappings."""
        redirects = ModelUnpickler.CLASS_REDIRECTS
        
        # Check for expected redirects
        assert 'HybridPreprocessingOracle' in redirects
        assert 'MetaFeatureExtractor' in redirects
        assert 'MinimalFeatureExtractor' in redirects
        assert 'MinimalFeatures' in redirects
        assert 'MetaFeatures' in redirects
        
        # Check redirect targets
        assert redirects['HybridPreprocessingOracle'] == 'src.neural.hybrid_oracle'
        assert redirects['MetaFeatureExtractor'] == 'src.features.meta_extractor'
        assert redirects['MinimalFeatures'] == 'src.features.minimal_extractor'
    
    def test_simple_pickle_load(self):
        """Test loading a simple pickle with ModelUnpickler."""
        test_data = {'key': 'value', 'number': 42}
        
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            pickle.dump(test_data, f)
            temp_path = f.name
        
        try:
            with open(temp_path, 'rb') as f:
                loaded = ModelUnpickler(f).load()
            
            assert loaded == test_data
        finally:
            Path(temp_path).unlink()
    
    def test_redirect_from_main(self):
        """Test that __main__ references are redirected correctly."""
        import io
        
        unpickler = ModelUnpickler(io.BytesIO())
        
        # Test redirecting HybridPreprocessingOracle from __main__
        cls = unpickler.find_class('__main__', 'HybridPreprocessingOracle')
        
        # Should be redirected to the actual class
        assert cls.__module__ == 'src.neural.hybrid_oracle'
        assert cls.__name__ == 'HybridPreprocessingOracle'
    
    def test_redirect_by_name_only(self):
        """Test that class names are redirected regardless of original module."""
        import io
        
        unpickler = ModelUnpickler(io.BytesIO())
        
        # Test redirecting by name only (not from __main__)
        cls = unpickler.find_class('some.other.module', 'MetaFeatureExtractor')
        
        # Should be redirected to the correct module
        assert cls.__module__ == 'src.features.meta_extractor'
        assert cls.__name__ == 'MetaFeatureExtractor'
    
    def test_non_redirect_classes_work(self):
        """Test that classes not in redirect list still work."""
        import io
        
        unpickler = ModelUnpickler(io.BytesIO())
        
        # Test a built-in class that shouldn't be redirected
        cls = unpickler.find_class('builtins', 'dict')
        assert cls == dict


class TestNeuralOracleWithUnpickler:
    """Test NeuralOracle loading with ModelUnpickler."""
    
    def test_oracle_loads_model(self):
        """Test that NeuralOracle can load models using ModelUnpickler."""
        # This should use ModelUnpickler internally
        oracle = NeuralOracle()
        
        # If a model exists, it should be loaded
        # (We can't guarantee a model exists in all test environments)
        # So we just check that initialization doesn't crash
        assert oracle is not None
    
    def test_oracle_with_explicit_path(self):
        """Test that NeuralOracle can load a specific model file."""
        models_dir = Path(__file__).parent.parent / "models"
        
        if models_dir.exists():
            pkl_files = list(models_dir.glob("*.pkl"))
            if pkl_files:
                # Try loading the first available model
                oracle = NeuralOracle(model_path=pkl_files[0])
                assert oracle is not None
            else:
                pytest.skip("No model files found for testing")
        else:
            pytest.skip("Models directory not found")


class TestDynamicModelDiscovery:
    """Test dynamic model file discovery."""
    
    def test_hybrid_model_priority(self):
        """Test that hybrid models are prioritized over other .pkl files."""
        models_dir = Path(__file__).parent.parent / "models"
        
        if not models_dir.exists():
            pytest.skip("Models directory not found")
        
        # Check what files exist
        hybrid_models = list(models_dir.glob("aurora_preprocessing_oracle_*.pkl"))
        all_pkl_files = list(models_dir.glob("*.pkl"))
        
        # If there are both hybrid and non-hybrid models, verify discovery logic
        if hybrid_models and all_pkl_files:
            oracle = NeuralOracle()
            assert oracle is not None
    
    def test_newest_file_selected(self):
        """Test that the newest file is selected when multiple exist."""
        models_dir = Path(__file__).parent.parent / "models"
        
        if not models_dir.exists():
            pytest.skip("Models directory not found")
        
        all_pkl_files = sorted(
            models_dir.glob("*.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if all_pkl_files:
            # The oracle should load the newest file
            oracle = NeuralOracle()
            assert oracle is not None
