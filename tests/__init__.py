"""Test suite for WakeyWakey wake word detection package."""

import sys
import os
from pathlib import Path

# Add the project root to Python path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
TEST_DATA_DIR = project_root / "tests" / "data"
TEST_MODELS_DIR = project_root / "tests" / "models"
TEST_AUDIO_DIR = project_root / "tests" / "audio"

# Create test directories if they don't exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_MODELS_DIR.mkdir(exist_ok=True)
TEST_AUDIO_DIR.mkdir(exist_ok=True) 