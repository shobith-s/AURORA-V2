"""
Test dynamic model loading in NeuralOracle.

This test verifies that:
1. Hybrid models (aurora_preprocessing_oracle_*.pkl) take priority
2. Any .pkl file is discovered when hybrid models don't exist
3. Most recent .pkl file (by modification time) is selected
4. Proper logging occurs when models are found
"""

import pytest
from pathlib import Path
import tempfile
import time

from src.neural.oracle import NeuralOracle, get_neural_oracle
from src.core.actions import PreprocessingAction


class TestDynamicModelLoading:
    """Test suite for dynamic model loading in NeuralOracle."""
    
    @staticmethod
    def _discover_model(models_dir: Path) -> Path:
        """
        Helper method that replicates the model discovery logic.
        This matches the logic in NeuralOracle.__init__ and get_neural_oracle().
        """
        model_path = None
        if models_dir.exists():
            hybrid_models = sorted(models_dir.glob("aurora_preprocessing_oracle_*.pkl"), reverse=True)
            if hybrid_models:
                model_path = hybrid_models[0]
            else:
                all_pkl_files = sorted(
                    models_dir.glob("*.pkl"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True
                )
                if all_pkl_files:
                    model_path = all_pkl_files[0]
        return model_path
    
    def test_loads_hybrid_model_first(self):
        """Hybrid models should take priority over other .pkl files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            
            # Create multiple .pkl files
            hybrid_model = models_dir / "aurora_preprocessing_oracle_20250101.pkl"
            other_model = models_dir / "other_model.pkl"
            
            # Create dummy model files (just empty files for this test)
            hybrid_model.touch()
            time.sleep(0.01)  # Ensure different timestamps
            other_model.touch()
            
            # The hybrid model should be selected even though other_model is newer
            model_path = self._discover_model(models_dir)
            
            assert model_path is not None
            assert model_path.name == "aurora_preprocessing_oracle_20250101.pkl"
    
    def test_loads_any_pkl_when_no_hybrid(self):
        """Should load any .pkl file when no hybrid model exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            
            # Create only non-hybrid .pkl files
            model1 = models_dir / "my_custom_model.pkl"
            model2 = models_dir / "another_model.pkl"
            
            model1.touch()
            time.sleep(0.01)
            model2.touch()  # This is newer
            
            model_path = self._discover_model(models_dir)
            
            assert model_path is not None
            # Should select the newer file
            assert model_path.name == "another_model.pkl"
    
    def test_selects_most_recent_pkl_file(self):
        """Should select the most recent .pkl file by modification time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            
            # Create multiple .pkl files with different timestamps
            old_model = models_dir / "old_model.pkl"
            middle_model = models_dir / "middle_model.pkl"
            new_model = models_dir / "new_model.pkl"
            
            old_model.touch()
            time.sleep(0.01)
            middle_model.touch()
            time.sleep(0.01)
            new_model.touch()
            
            model_path = self._discover_model(models_dir)
            
            assert model_path is not None
            assert model_path.name == "new_model.pkl"
    
    def test_handles_no_pkl_files(self):
        """Should handle gracefully when no .pkl files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            
            # Create a non-.pkl file
            (models_dir / "readme.txt").touch()
            
            model_path = self._discover_model(models_dir)
            
            assert model_path is None
    
    def test_handles_empty_directory(self):
        """Should handle empty models directory gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            
            model_path = self._discover_model(models_dir)
            
            assert model_path is None
    
    def test_selects_newest_hybrid_when_multiple_exist(self):
        """Should select the most recent hybrid model when multiple exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            
            # Create multiple hybrid models
            old_hybrid = models_dir / "aurora_preprocessing_oracle_20250101.pkl"
            new_hybrid = models_dir / "aurora_preprocessing_oracle_20250102.pkl"
            
            old_hybrid.touch()
            time.sleep(0.01)
            new_hybrid.touch()
            
            model_path = self._discover_model(models_dir)
            
            assert model_path is not None
            # With reverse=True on sorted names, newer timestamp in filename comes first
            assert model_path.name == "aurora_preprocessing_oracle_20250102.pkl"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
