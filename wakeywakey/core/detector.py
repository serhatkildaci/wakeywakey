"""
Wake word detection module.

Provides the main WakeWordDetector class for real-time wake word detection
with model loading, confidence scoring, and performance monitoring.
"""

import os
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Tuple
import logging
import json
import glob

import numpy as np
import torch
import torch.nn as nn
from collections import deque

from .audio import AudioProcessor

logger = logging.getLogger(__name__)


class ModelLoader:
    """Utility class for loading and managing wake word models."""
    
    @staticmethod
    def load_model(model_path: str, device: str = 'cpu') -> Tuple[Optional[nn.Module], Optional[Dict]]:
        """
        Load a wake word model from file.
        
        Args:
            model_path: Path to model file
            device: Device to load model on
            
        Returns:
            Tuple of (model, metadata) or (None, None) if failed
        """
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return None, None
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            
            # Extract model type and configuration
            model_type = checkpoint.get('model_type', 'LightweightCNN')
            config = checkpoint.get('config', {})
            
            # Import model class dynamically
            from ..models.architectures import get_model_class
            model_class = get_model_class(model_type)
            
            if model_class is None:
                logger.error(f"Unknown model type: {model_type}")
                return None, None
            
            # Create model instance
            model = model_class(**config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            
            # Extract metadata
            metadata = {
                'model_type': model_type,
                'config': config,
                'accuracy': checkpoint.get('accuracy', 0.0),
                'training_loss': checkpoint.get('loss', 0.0),
                'epoch': checkpoint.get('epoch', 0),
                'created': checkpoint.get('timestamp', time.time()),
                'input_size': config.get('input_size', 26),
                'version': checkpoint.get('version', '0.1.0')
            }
            
            logger.info(f"Model loaded: {model_type} from {model_path}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            return None, None
    
    @staticmethod
    def list_models(model_dir: str) -> List[Dict[str, Any]]:
        """
        List available models in directory.
        
        Args:
            model_dir: Directory containing model files
            
        Returns:
            List of model information dictionaries
        """
        models = []
        
        if not os.path.exists(model_dir):
            return models
        
        # Find all .pth files
        model_files = glob.glob(os.path.join(model_dir, "*.pth"))
        
        for model_path in model_files:
            try:
                # Get basic file info
                stat = os.stat(model_path)
                size_mb = stat.st_size / (1024 * 1024)
                
                # Try to load metadata
                _, metadata = ModelLoader.load_model(model_path)
                
                model_info = {
                    'name': os.path.basename(model_path),
                    'path': model_path,
                    'size_mb': size_mb,
                    'created': stat.st_mtime,
                    'type': metadata['model_type'] if metadata else 'Unknown',
                    'accuracy': metadata['accuracy'] if metadata else 'Unknown',
                    'input_size': metadata['input_size'] if metadata else 'Unknown'
                }
                
                models.append(model_info)
                
            except Exception as e:
                logger.warning(f"Could not read model info for {model_path}: {e}")
        
        # Sort by creation time (newest first)
        models.sort(key=lambda x: x['created'], reverse=True)
        
        return models


class WakeWordDetector:
    """
    Main wake word detection class.
    
    Provides real-time wake word detection with configurable thresholds,
    smoothing, and performance monitoring.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.7,
        sensitivity: str = 'medium',
        smoothing_window: int = 3,
        debounce_time: float = 2.0,
        device: str = 'auto',
        audio_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize wake word detector.
        
        Args:
            model_path: Path to trained model file
            threshold: Detection confidence threshold (0.0-1.0)
            sensitivity: Sensitivity level ('low', 'medium', 'high', 'very_high')
            smoothing_window: Number of predictions to smooth over
            debounce_time: Minimum time between detections (seconds)
            device: Device for inference ('cpu', 'cuda', 'auto')
            audio_config: Audio processor configuration
        """
        self.model_path = model_path
        self.threshold = threshold
        self.sensitivity = sensitivity
        self.smoothing_window = smoothing_window
        self.debounce_time = debounce_time
        
        # Device setup
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Model and metadata
        self.model: Optional[nn.Module] = None
        self.model_metadata: Optional[Dict] = None
        
        # Audio processor
        audio_config = audio_config or {}
        self.audio_processor = AudioProcessor(**audio_config)
        
        # Detection state
        self.is_detecting = False
        self.detection_thread: Optional[threading.Thread] = None
        self.detection_callback: Optional[Callable] = None
        self.debug_callback: Optional[Callable] = None
        
        # Smoothing and debouncing
        self.prediction_history = deque(maxlen=smoothing_window)
        self.last_detection_time = 0.0
        
        # Performance monitoring
        self.stats = {
            'total_predictions': 0,
            'detections': 0,
            'avg_confidence': 0.0,
            'avg_inference_time': 0.0,
            'false_positives': 0,
            'session_start': time.time()
        }
        
        # Load model if provided
        if model_path:
            self.load_model(model_path)
        
        # Apply sensitivity mapping
        self._apply_sensitivity()
        
        logger.info(f"WakeWordDetector initialized: threshold={self.threshold}, "
                   f"device={self.device}, smoothing={smoothing_window}")
    
    def _apply_sensitivity(self):
        """Apply sensitivity level to threshold."""
        sensitivity_map = {
            'low': 0.9,
            'medium': 0.7,
            'high': 0.5,
            'very_high': 0.3
        }
        
        if self.sensitivity in sensitivity_map:
            self.threshold = sensitivity_map[self.sensitivity]
            logger.info(f"Sensitivity {self.sensitivity} applied: threshold={self.threshold}")
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a wake word model.
        
        Args:
            model_path: Path to model file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            self.model, self.model_metadata = ModelLoader.load_model(model_path, self.device)
            
            if self.model is None:
                logger.error(f"Failed to load model from {model_path}")
                return False
            
            self.model_path = model_path
            
            # Validate input size compatibility
            expected_input_size = self.model_metadata.get('input_size', 26)
            if expected_input_size != self.audio_processor.feature_dim:
                logger.warning(f"Model input size ({expected_input_size}) != "
                             f"audio processor output ({self.audio_processor.feature_dim})")
            
            logger.info(f"Model loaded successfully: {self.model_metadata['model_type']}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Make prediction on feature vector.
        
        Args:
            features: Input feature vector
            
        Returns:
            Tuple of (confidence, inference_time)
        """
        if self.model is None:
            return 0.0, 0.0
        
        try:
            start_time = time.time()
            
            # Convert to tensor
            if isinstance(features, np.ndarray):
                features_tensor = torch.from_numpy(features).float().unsqueeze(0)
            else:
                features_tensor = features.float().unsqueeze(0)
            
            features_tensor = features_tensor.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(features_tensor)
                
                # Handle different output formats
                if isinstance(output, torch.Tensor):
                    if output.dim() > 1:
                        confidence = torch.sigmoid(output).item()
                    else:
                        confidence = torch.sigmoid(output[0]).item()
                else:
                    confidence = float(output)
            
            inference_time = time.time() - start_time
            
            # Update stats
            self.stats['total_predictions'] += 1
            self.stats['avg_confidence'] = (
                (self.stats['avg_confidence'] * (self.stats['total_predictions'] - 1) + confidence) 
                / self.stats['total_predictions']
            )
            self.stats['avg_inference_time'] = (
                (self.stats['avg_inference_time'] * (self.stats['total_predictions'] - 1) + inference_time)
                / self.stats['total_predictions']
            )
            
            return confidence, inference_time
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.0, 0.0
    
    def _smooth_predictions(self, confidence: float) -> float:
        """
        Apply smoothing to predictions.
        
        Args:
            confidence: Current confidence score
            
        Returns:
            Smoothed confidence score
        """
        self.prediction_history.append(confidence)
        
        if len(self.prediction_history) < self.smoothing_window:
            return confidence
        
        # Simple moving average
        return float(np.mean(self.prediction_history))
    
    def _should_trigger_detection(self, smoothed_confidence: float) -> bool:
        """
        Determine if detection should be triggered.
        
        Args:
            smoothed_confidence: Smoothed confidence score
            
        Returns:
            True if detection should be triggered
        """
        current_time = time.time()
        
        # Check threshold
        if smoothed_confidence < self.threshold:
            return False
        
        # Check debounce time
        if current_time - self.last_detection_time < self.debounce_time:
            return False
        
        return True
    
    def start_detection(
        self,
        detection_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        debug_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Start real-time wake word detection.
        
        Args:
            detection_callback: Function called when wake word is detected
            debug_callback: Function called with debug information
        """
        if self.is_detecting:
            logger.warning("Detection already in progress")
            return
        
        if self.model is None:
            logger.error("No model loaded for detection")
            return
        
        self.detection_callback = detection_callback
        self.debug_callback = debug_callback
        self.is_detecting = True
        
        # Reset stats
        self.stats['session_start'] = time.time()
        self.stats['detections'] = 0
        
        def feature_callback(features):
            """Process features for detection."""
            if not self.is_detecting:
                return
            
            # Make prediction
            confidence, inference_time = self.predict(features)
            
            # Apply smoothing
            smoothed_confidence = self._smooth_predictions(confidence)
            
            # Debug callback
            if self.debug_callback:
                debug_info = {
                    'confidence': confidence,
                    'smoothed_confidence': smoothed_confidence,
                    'inference_time': inference_time,
                    'threshold': self.threshold,
                    'features': features[:5].tolist() if len(features) > 5 else features.tolist(),
                    'timestamp': time.time()
                }
                self.debug_callback(debug_info)
            
            # Check for detection
            if self._should_trigger_detection(smoothed_confidence):
                self.last_detection_time = time.time()
                self.stats['detections'] += 1
                
                if self.detection_callback:
                    detection_info = {
                        'confidence': smoothed_confidence,
                        'raw_confidence': confidence,
                        'inference_time': inference_time,
                        'timestamp': self.last_detection_time,
                        'model_type': self.model_metadata['model_type'] if self.model_metadata else 'Unknown',
                        'threshold': self.threshold
                    }
                    self.detection_callback(detection_info)
                
                logger.info(f"Wake word detected! Confidence: {smoothed_confidence:.3f}")
        
        # Start audio processing
        try:
            self.audio_processor.start_recording(callback=feature_callback)
            logger.info("Wake word detection started")
        except Exception as e:
            logger.error(f"Failed to start detection: {e}")
            self.is_detecting = False
            raise
    
    def stop_detection(self):
        """Stop wake word detection."""
        if not self.is_detecting:
            return
        
        self.is_detecting = False
        self.audio_processor.stop_recording()
        
        # Clear callbacks
        self.detection_callback = None
        self.debug_callback = None
        
        logger.info("Wake word detection stopped")
    
    def set_threshold(self, threshold: float):
        """
        Set detection threshold.
        
        Args:
            threshold: New threshold value (0.0-1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        
        self.threshold = threshold
        logger.info(f"Detection threshold set to {threshold}")
    
    def set_sensitivity(self, sensitivity: str):
        """
        Set sensitivity level.
        
        Args:
            sensitivity: Sensitivity level ('low', 'medium', 'high', 'very_high')
        """
        if sensitivity not in ['low', 'medium', 'high', 'very_high']:
            raise ValueError("Invalid sensitivity level")
        
        self.sensitivity = sensitivity
        self._apply_sensitivity()
        logger.info(f"Sensitivity set to {sensitivity} (threshold: {self.threshold})")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        current_time = time.time()
        session_duration = current_time - self.stats['session_start']
        
        stats = self.stats.copy()
        stats.update({
            'session_duration': session_duration,
            'detection_rate': self.stats['detections'] / max(session_duration, 1),
            'prediction_rate': self.stats['total_predictions'] / max(session_duration, 1),
            'model_info': self.model_metadata if self.model_metadata else {},
            'is_detecting': self.is_detecting,
            'current_threshold': self.threshold,
            'device': self.device
        })
        
        return stats
    
    def reset_stats(self):
        """Reset detection statistics."""
        self.stats = {
            'total_predictions': 0,
            'detections': 0,
            'avg_confidence': 0.0,
            'avg_inference_time': 0.0,
            'false_positives': 0,
            'session_start': time.time()
        }
        logger.info("Detection statistics reset")
    
    def test_model(self, test_audio_path: str) -> Dict[str, Any]:
        """
        Test model on audio file.
        
        Args:
            test_audio_path: Path to test audio file
            
        Returns:
            Test results dictionary
        """
        if self.model is None:
            return {'error': 'No model loaded'}
        
        try:
            # Extract features
            features = self.audio_processor.extract_features_from_file(test_audio_path)
            
            if features is None:
                return {'error': 'Failed to extract features'}
            
            # Make prediction
            confidence, inference_time = self.predict(features)
            
            result = {
                'file': test_audio_path,
                'confidence': confidence,
                'inference_time': inference_time,
                'predicted_class': 'wake_word' if confidence >= self.threshold else 'not_wake_word',
                'threshold': self.threshold,
                'features_shape': features.shape,
                'model_type': self.model_metadata['model_type'] if self.model_metadata else 'Unknown'
            }
            
            return result
            
        except Exception as e:
            return {'error': f'Test failed: {e}'}
    
    def save_config(self, config_path: str):
        """Save detector configuration to file."""
        config = {
            'model_path': self.model_path,
            'threshold': self.threshold,
            'sensitivity': self.sensitivity,
            'smoothing_window': self.smoothing_window,
            'debounce_time': self.debounce_time,
            'device': self.device,
            'audio_config': self.audio_processor.get_config(),
            'timestamp': time.time()
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")
    
    def load_config(self, config_path: str) -> bool:
        """
        Load detector configuration from file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Apply configuration
            self.threshold = config.get('threshold', self.threshold)
            self.sensitivity = config.get('sensitivity', self.sensitivity)
            self.smoothing_window = config.get('smoothing_window', self.smoothing_window)
            self.debounce_time = config.get('debounce_time', self.debounce_time)
            
            # Load model if specified
            model_path = config.get('model_path')
            if model_path and model_path != self.model_path:
                self.load_model(model_path)
            
            logger.info(f"Configuration loaded from {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.stop_detection()
        except:
            pass 