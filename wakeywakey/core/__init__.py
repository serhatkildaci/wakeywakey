"""Core wake word detection functionality."""

from .audio import AudioProcessor
from .detector import WakeWordDetector

__all__ = ["AudioProcessor", "WakeWordDetector"] 