#!/usr/bin/env python3
"""
Test script to verify all imports work after simplification.
Run this to ensure the server will start without import errors.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing imports...")
print("-" * 50)

errors = []

# Test core imports
try:
    from src.core.preprocessor import IntelligentPreprocessor, get_preprocessor
    print("✓ Core preprocessor imports OK")
except Exception as e:
    errors.append(f"✗ Core preprocessor: {e}")
    print(f"✗ Core preprocessor: {e}")

# Test API imports
try:
    from src.api.schemas import (
        PreprocessRequest, PreprocessResponse,
        CorrectionRequest, CorrectionResponse
    )
    print("✓ API schemas imports OK")
except Exception as e:
    errors.append(f"✗ API schemas: {e}")
    print(f"✗ API schemas: {e}")

# Test validation imports
try:
    from src.validation.metrics_tracker import MetricsTracker, get_metrics_tracker
    print("✓ Validation metrics_tracker imports OK")
except Exception as e:
    errors.append(f"✗ Validation metrics_tracker: {e}")
    print(f"✗ Validation metrics_tracker: {e}")

# Test symbolic imports
try:
    from src.symbolic.engine import SymbolicEngine
    from src.symbolic.rules import get_all_rules
    print("✓ Symbolic engine imports OK")
except Exception as e:
    errors.append(f"✗ Symbolic engine: {e}")
    print(f"✗ Symbolic engine: {e}")

# Test neural imports
try:
    from src.neural.oracle import NeuralOracle, get_neural_oracle
    print("✓ Neural oracle imports OK")
except Exception as e:
    errors.append(f"✗ Neural oracle: {e}")
    print(f"✗ Neural oracle: {e}")

# Test learning imports
try:
    from src.learning.adaptive_engine import AdaptiveLearningEngine
    from src.learning.adaptive_rules import AdaptiveSymbolicRules
    print("✓ Learning system imports OK")
except Exception as e:
    errors.append(f"✗ Learning system: {e}")
    print(f"✗ Learning system: {e}")

# Test features imports
try:
    from src.features.minimal_extractor import MinimalFeatureExtractor, get_feature_extractor
    from src.features.intelligent_cache import get_cache
    print("✓ Features imports OK")
except Exception as e:
    errors.append(f"✗ Features: {e}")
    print(f"✗ Features: {e}")

# Test database imports
try:
    from src.database.models import CorrectionRecord, LearnedRule
    from src.database.connection import init_db, get_db
    print("✓ Database imports OK")
except Exception as e:
    errors.append(f"✗ Database: {e}")
    print(f"✗ Database: {e}")

# Test server import (main test)
try:
    from src.api.server import app
    print("✓ API server imports OK")
except Exception as e:
    errors.append(f"✗ API server: {e}")
    print(f"✗ API server: {e}")

print("-" * 50)

if errors:
    print(f"\n❌ {len(errors)} IMPORT ERRORS FOUND:")
    for error in errors:
        print(f"  {error}")
    sys.exit(1)
else:
    print("\n✅ ALL IMPORTS SUCCESSFUL!")
    print("Server should start without errors.")
    print("\nRun: uvicorn src.api.server:app --reload")
    sys.exit(0)
