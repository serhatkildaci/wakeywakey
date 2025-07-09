"""
WakeyWakey - Lightweight Wake Word Detection

A comprehensive wake word detection package optimized for deployment from
microcontrollers to Raspberry Pi. Features neural network architectures
with real-time audio processing and terminal-style CLI.

Key Features:
- Multiple neural network architectures (CNN, RNN, Hybrid)
- Real-time audio processing with MFCC feature extraction
- Model quantization and optimization for edge devices
- Comprehensive training pipeline with hyperparameter optimization
- Terminal-style CLI for training, testing, and deployment
- Cross-platform support (Linux, macOS, Windows)
- Hardware optimization (CPU, CUDA, microcontrollers)

Quick Start:
    >>> from wakeywakey import WakeWordDetector
    >>> detector = WakeWordDetector(model_path="model.pth")
    >>> detector.start_detection()

CLI Usage:
    $ wakeywakey train --data-dir ./data --model-type LightweightCNN
    $ wakeywakey detect --model ./models/model.pth --threshold 0.7
"""

__version__ = "0.1.0"
__author__ = "Serhat Kildaci"
__email__ = "serhat.kildaci@example.com"
__license__ = "MIT"
__description__ = "Lightweight wake word detection optimized for microcontrollers to Raspberry Pi"
__url__ = "https://github.com/serhatkildaci/wakeywakey"

# Core imports for public API
from .core import WakeWordDetector, AudioProcessor
from .models import (
    LightweightCNN,
    CompactRNN, 
    HybridCRNN,
    MobileWakeWord,
    ModelQuantizer,
    ModelOptimizer
)
from .training import Trainer, WakeWordDataset

# Version compatibility
import sys
if sys.version_info < (3, 7):
    raise ImportError("WakeyWakey requires Python 3.7 or later")

# Public API
__all__ = [
    # Core functionality
    "WakeWordDetector",
    "AudioProcessor",
    
    # Neural network models
    "LightweightCNN",
    "CompactRNN",
    "HybridCRNN", 
    "MobileWakeWord",
    
    # Model utilities
    "ModelQuantizer",
    "ModelOptimizer",
    
    # Training
    "Trainer",
    "WakeWordDataset",
    
    # Package metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__description__",
    "__url__",
]

# Package configuration
import logging

# Set up default logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Runtime configuration
_config = {
    "default_sample_rate": 16000,
    "default_model_type": "LightweightCNN",
    "default_threshold": 0.7,
    "verbose": False
}

def get_config(key=None):
    """Get package configuration."""
    if key is None:
        return _config.copy()
    return _config.get(key)

def set_config(key, value):
    """Set package configuration."""
    if key in _config:
        _config[key] = value
    else:
        raise KeyError(f"Unknown configuration key: {key}")

def get_version():
    """Get package version."""
    return __version__

def get_system_info():
    """Get system information for debugging."""
    import platform
    import torch
    
    info = {
        "wakeywakey_version": __version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pytorch_version": torch.__version__ if torch else "Not installed",
        "cuda_available": torch.cuda.is_available() if torch else False,
    }
    
    if torch and torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
    
    return info

# Module-level convenience functions
def quick_detect(model_path, threshold=0.7, verbose=False):
    """
    Quick start function for immediate wake word detection.
    
    Args:
        model_path: Path to trained model file
        threshold: Detection confidence threshold
        verbose: Enable verbose output
    
    Returns:
        WakeWordDetector instance ready for detection
    """
    if verbose:
        set_config("verbose", True)
        logging.basicConfig(level=logging.INFO)
    
    detector = WakeWordDetector(
        model_path=model_path,
        threshold=threshold
    )
    
    return detector

def list_models(model_dir="./models"):
    """
    List available models in directory.
    
    Args:
        model_dir: Directory containing model files
    
    Returns:
        List of model information dictionaries
    """
    from .core.detector import ModelLoader
    return ModelLoader.list_models(model_dir)

# Package health check
def _check_dependencies():
    """Check if all required dependencies are available."""
    missing = []
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import librosa
    except ImportError:
        missing.append("librosa")
    
    try:
        import sounddevice
    except ImportError:
        missing.append("sounddevice")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    if missing:
        import warnings
        warnings.warn(
            f"Missing optional dependencies: {', '.join(missing)}. "
            f"Some functionality may not be available.",
            UserWarning
        )
    
    return len(missing) == 0

# Run dependency check on import
_dependencies_ok = _check_dependencies()

# Cleanup
del sys  # Don't expose sys in package namespace 