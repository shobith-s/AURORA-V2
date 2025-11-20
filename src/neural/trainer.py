"""
Neural Oracle Trainer - Train the neural oracle model.

Note: Main training logic is in scripts/train_neural_oracle.py
This module provides utilities for training workflow.
"""

from typing import Dict, Any, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class NeuralOracleTrainer:
    """
    Trainer for the neural oracle model.

    For actual training, use: scripts/train_neural_oracle.py
    This class provides utilities and validation.
    """

    def __init__(self, output_dir: str = "./models"):
        """
        Initialize trainer.

        Args:
            output_dir: Directory to save trained models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate_training_data(
        self,
        features: List,
        labels: List
    ) -> Dict[str, Any]:
        """
        Validate training data before training.

        Args:
            features: List of feature arrays
            labels: List of labels

        Returns:
            Validation results
        """
        if len(features) != len(labels):
            raise ValueError(
                f"Feature count ({len(features)}) must match label count ({len(labels)})"
            )

        if len(features) < 100:
            logger.warning(
                f"Only {len(features)} training samples. Recommend at least 100."
            )

        return {
            'valid': True,
            'num_samples': len(features),
            'num_features': len(features[0]) if features else 0
        }


# For backwards compatibility
def train_neural_oracle(*args, **kwargs):
    """
    Legacy training function.

    Use scripts/train_neural_oracle.py for actual training.
    """
    logger.warning(
        "Direct training from this module is deprecated. "
        "Use: python scripts/train_neural_oracle.py"
    )
    raise NotImplementedError("Use scripts/train_neural_oracle.py")
