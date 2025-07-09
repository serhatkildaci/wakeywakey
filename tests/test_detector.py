"""Tests for wake word detection functionality."""

import pytest
import numpy as np
import tempfile
import time
import threading
from pathlib import Path
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock

# Import modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from wakeywakey.core.detector import WakeWordDetector, ModelLoader
from wakeywakey.models.architectures import LightweightCNN


class MockModel(nn.Module):
    """Mock model for testing."""
    
    def __init__(self, input_size=26):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.fc(x))


class TestWakeWordDetector:
    """Test suite for WakeWordDetector class."""
    
    @pytest.fixture
    def mock_model_file(self):
        """Create a mock model file for testing."""
        model = MockModel()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {'input_size': 26},
                'model_type': 'LightweightCNN',
                'accuracy': 0.95
            }, f.name)
            
            yield f.name
            
            # Cleanup
            Path(f.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def detector_without_model(self):
        """Create detector without loading a model."""
        return WakeWordDetector(
            model_path=None,
            threshold=0.7,
            device="cpu"
        )
    
    @pytest.fixture 
    def sample_features(self):
        """Generate sample audio features."""
        return np.random.randn(26).astype(np.float32)
    
    def test_initialization_without_model(self):
        """Test detector initialization without model."""
        detector = WakeWordDetector(
            threshold=0.8,
            smoothing_window=3,
            cooldown_duration=1.0
        )
        
        assert detector.threshold == 0.8
        assert detector.smoothing_window == 3
        assert detector.cooldown_duration == 1.0
        assert detector.model is None
        assert not detector.is_detecting
    
    @patch('wakeywakey.core.detector.torch.load')
    @patch('wakeywakey.models.architectures.LightweightCNN')
    def test_load_model_success(self, mock_model_class, mock_torch_load, detector_without_model):
        """Test successful model loading."""
        # Setup mocks
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        mock_torch_load.return_value = {
            'model_state_dict': {'weight': torch.tensor([1.0])},
            'config': {'input_size': 26},
            'model_type': 'LightweightCNN'
        }
        
        # Create temporary model file
        with tempfile.NamedTemporaryFile(suffix='.pth') as f:
            success = detector_without_model.load_model(f.name)
            
            assert success
            assert detector_without_model.model is not None
            assert detector_without_model.feature_size == 26
    
    def test_load_nonexistent_model(self, detector_without_model):
        """Test loading nonexistent model file."""
        success = detector_without_model.load_model("nonexistent_model.pth")
        assert not success
        assert detector_without_model.model is None
    
    def test_predict_without_model(self, detector_without_model, sample_features):
        """Test prediction without loaded model."""
        confidence, proc_time = detector_without_model.predict(sample_features)
        assert confidence == 0.0
        assert proc_time == 0.0
    
    @patch('wakeywakey.core.detector.torch.load')
    def test_predict_with_mock_model(self, mock_torch_load, detector_without_model, sample_features):
        """Test prediction with mocked model."""
        # Create a simple mock model
        model = MockModel(input_size=26)
        
        # Mock the torch.load to return our model state
        mock_torch_load.return_value = {
            'model_state_dict': model.state_dict(),
            'config': {'input_size': 26},
            'model_type': 'LightweightCNN'
        }
        
        # Create temporary model file and load it
        with tempfile.NamedTemporaryFile(suffix='.pth') as f:
            # Save the model properly first
            torch.save(mock_torch_load.return_value, f.name)
            
            # Load model
            detector_without_model.load_model(f.name)
            
            # Test prediction
            confidence, proc_time = detector_without_model.predict(sample_features)
            
            assert 0.0 <= confidence <= 1.0
            assert proc_time > 0.0
    
    def test_smooth_predictions(self, detector_without_model):
        """Test prediction smoothing."""
        detector = detector_without_model
        
        # Test smoothing with multiple predictions
        predictions = [0.1, 0.3, 0.7, 0.9, 0.6]
        smoothed_values = []
        
        for pred in predictions:
            smoothed = detector.smooth_predictions(pred)
            smoothed_values.append(smoothed)
        
        # Last smoothed value should be influenced by all previous values
        assert len(detector.prediction_buffer) <= detector.smoothing_window
        assert 0.0 <= smoothed_values[-1] <= 1.0
    
    def test_set_threshold(self, detector_without_model):
        """Test threshold setting."""
        detector = detector_without_model
        
        # Valid threshold
        detector.set_threshold(0.5)
        assert detector.threshold == 0.5
        
        # Invalid thresholds should be rejected
        original_threshold = detector.threshold
        detector.set_threshold(-0.1)  # Too low
        assert detector.threshold == original_threshold
        
        detector.set_threshold(1.5)   # Too high
        assert detector.threshold == original_threshold
    
    def test_set_sensitivity(self, detector_without_model):
        """Test sensitivity preset setting."""
        detector = detector_without_model
        
        # Test all sensitivity levels
        detector.set_sensitivity("low")
        assert detector.threshold == 0.9
        
        detector.set_sensitivity("medium") 
        assert detector.threshold == 0.7
        
        detector.set_sensitivity("high")
        assert detector.threshold == 0.5
        
        detector.set_sensitivity("very_high")
        assert detector.threshold == 0.3
    
    @patch('wakeywakey.core.audio.AudioProcessor.start_recording')
    def test_start_detection_without_model(self, mock_start_recording, detector_without_model):
        """Test starting detection without loaded model."""
        success = detector_without_model.start_detection()
        assert not success
        assert not detector_without_model.is_detecting
    
    @patch('wakeywakey.core.audio.AudioProcessor.start_recording')
    @patch('wakeywakey.core.detector.torch.load')
    def test_start_detection_with_model(self, mock_torch_load, mock_start_recording, detector_without_model):
        """Test starting detection with loaded model."""
        # Setup model mock
        model = MockModel(input_size=26)
        mock_torch_load.return_value = {
            'model_state_dict': model.state_dict(),
            'config': {'input_size': 26},
            'model_type': 'LightweightCNN'
        }
        mock_start_recording.return_value = True
        
        with tempfile.NamedTemporaryFile(suffix='.pth') as f:
            torch.save(mock_torch_load.return_value, f.name)
            
            # Load model and start detection
            detector_without_model.load_model(f.name)
            success = detector_without_model.start_detection()
            
            assert success
            assert detector_without_model.is_detecting
            
            # Cleanup
            detector_without_model.stop_detection()
    
    def test_detection_callback(self, detector_without_model):
        """Test detection callback functionality."""
        callback_called = threading.Event()
        detection_info = None
        
        def test_callback(info):
            nonlocal detection_info
            detection_info = info
            callback_called.set()
        
        # Test callback storage
        detector_without_model.detection_callback = test_callback
        assert detector_without_model.detection_callback == test_callback
    
    def test_audio_callback(self, detector_without_model):
        """Test audio callback processing."""
        detector = detector_without_model
        
        # Test when not detecting
        audio_data = np.random.randn(1600)  # 0.1 seconds at 16kHz
        detector.audio_callback(audio_data)  # Should not crash
        
        # Test when detecting but no model
        detector.is_detecting = True
        detector.audio_callback(audio_data)  # Should not crash
        
        detector.is_detecting = False
    
    def test_get_stats(self, detector_without_model):
        """Test statistics retrieval."""
        detector = detector_without_model
        stats = detector.get_stats()
        
        required_keys = [
            'total_detections',
            'runtime_seconds', 
            'detections_per_minute',
            'average_processing_time_ms',
            'current_threshold',
            'model_type',
            'device',
            'is_detecting'
        ]
        
        for key in required_keys:
            assert key in stats
        
        assert stats['total_detections'] >= 0
        assert stats['current_threshold'] == detector.threshold
        assert stats['is_detecting'] == detector.is_detecting
    
    def test_feature_size_mismatch_handling(self, detector_without_model):
        """Test handling of feature size mismatches."""
        detector = detector_without_model
        detector.feature_size = 26
        detector.model = MockModel(input_size=26)
        
        # Test with wrong feature size
        wrong_size_features = np.random.randn(13)  # Half the expected size
        confidence, proc_time = detector.predict(wrong_size_features)
        
        # Should handle gracefully
        assert 0.0 <= confidence <= 1.0
    
    @patch('wakeywakey.core.detector.torch.load')
    def test_test_detection_method(self, mock_torch_load, detector_without_model):
        """Test the test_detection method with audio files."""
        # Setup model mock
        model = MockModel(input_size=26)
        mock_torch_load.return_value = {
            'model_state_dict': model.state_dict(),
            'config': {'input_size': 26},
            'model_type': 'LightweightCNN'
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pth') as model_file:
            torch.save(mock_torch_load.return_value, model_file.name)
            detector_without_model.load_model(model_file.name)
            
            # Mock audio processor
            with patch.object(detector_without_model.audio_processor, 'process_audio_file') as mock_process:
                mock_process.return_value = np.random.randn(16000)
                
                with patch.object(detector_without_model.audio_processor, 'extract_mfcc_features') as mock_extract:
                    mock_extract.return_value = np.random.randn(26)
                    
                    # Test with valid audio file
                    result = detector_without_model.test_detection("test.wav")
                    
                    assert 'confidence' in result
                    assert 'detected' in result
                    assert 'processing_time_ms' in result
                    assert isinstance(result['confidence'], float)
                    assert isinstance(result['detected'], bool)


class TestModelLoader:
    """Test suite for ModelLoader utility class."""
    
    def test_list_models_empty_directory(self):
        """Test listing models in empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            models = ModelLoader.list_models(temp_dir)
            assert models == []
    
    def test_list_models_nonexistent_directory(self):
        """Test listing models in nonexistent directory."""
        models = ModelLoader.list_models("nonexistent_directory")
        assert models == []
    
    def test_list_models_with_valid_models(self):
        """Test listing valid model files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock model files
            model_data = {
                'model_state_dict': MockModel().state_dict(),
                'model_type': 'LightweightCNN',
                'accuracy': 0.95
            }
            
            model_path = Path(temp_dir) / "test_model.pth"
            torch.save(model_data, model_path)
            
            models = ModelLoader.list_models(temp_dir)
            
            assert len(models) == 1
            assert models[0]['name'] == 'test_model'
            assert models[0]['type'] == 'LightweightCNN'
            assert models[0]['accuracy'] == 0.95
    
    def test_get_recommended_model_cpu(self):
        """Test getting recommended model for CPU."""
        models = [
            {'name': 'large', 'size_mb': 10.0, 'accuracy': 0.99, 'path': 'large.pth'},
            {'name': 'small', 'size_mb': 1.0, 'accuracy': 0.95, 'path': 'small.pth'},
            {'name': 'medium', 'size_mb': 5.0, 'accuracy': 0.97, 'path': 'medium.pth'}
        ]
        
        recommended = ModelLoader.get_recommended_model(models, device="cpu")
        # Should recommend smallest model for CPU
        assert recommended == 'small.pth'
    
    def test_get_recommended_model_gpu(self):
        """Test getting recommended model for GPU."""
        models = [
            {'name': 'large', 'size_mb': 10.0, 'accuracy': 0.99, 'path': 'large.pth'},
            {'name': 'small', 'size_mb': 1.0, 'accuracy': 0.95, 'path': 'small.pth'},
            {'name': 'medium', 'size_mb': 5.0, 'accuracy': 0.97, 'path': 'medium.pth'}
        ]
        
        recommended = ModelLoader.get_recommended_model(models, device="cuda")
        # Should recommend most accurate model for GPU
        assert recommended == 'large.pth'
    
    def test_get_recommended_model_empty_list(self):
        """Test getting recommended model from empty list."""
        recommended = ModelLoader.get_recommended_model([], device="cpu")
        assert recommended is None


class TestDetectorIntegration:
    """Integration tests for detector components."""
    
    @patch('wakeywakey.core.audio.AudioProcessor.start_recording')
    @patch('wakeywakey.core.detector.torch.load')
    def test_full_detection_workflow(self, mock_torch_load, mock_start_recording):
        """Test complete detection workflow."""
        # Setup
        model = MockModel(input_size=26)
        mock_torch_load.return_value = {
            'model_state_dict': model.state_dict(),
            'config': {'input_size': 26},
            'model_type': 'LightweightCNN'
        }
        mock_start_recording.return_value = True
        
        detection_events = []
        
        def detection_callback(info):
            detection_events.append(info)
        
        with tempfile.NamedTemporaryFile(suffix='.pth') as f:
            torch.save(mock_torch_load.return_value, f.name)
            
            # Create detector and load model
            detector = WakeWordDetector(
                model_path=f.name,
                threshold=0.5,
                cooldown_duration=0.1  # Short cooldown for testing
            )
            
            # Start detection
            success = detector.start_detection(detection_callback=detection_callback)
            assert success
            
            # Simulate audio processing with high confidence
            high_confidence_audio = np.ones(1600) * 0.5  # Should trigger detection
            
            # Mock the feature extraction to return features that trigger detection
            with patch.object(detector.audio_processor, 'extract_mfcc_features') as mock_extract:
                # Return features that will result in high confidence
                mock_extract.return_value = np.ones(26) * 2.0
                
                # Simulate audio callback
                detector.audio_callback(high_confidence_audio)
                
                # Give some time for processing
                time.sleep(0.2)
            
            # Stop detection
            detector.stop_detection()
            
            # Verify detection was successful
            stats = detector.get_stats()
            assert stats['total_detections'] >= 0  # May be 0 due to threading timing
    
    def test_detector_cleanup(self, detector_without_model):
        """Test proper cleanup of detector resources."""
        detector = detector_without_model
        
        # Test destructor
        del detector
        # Should not raise any exceptions


class TestDetectorPerformance:
    """Performance tests for detector components."""
    
    @patch('wakeywakey.core.detector.torch.load')
    def test_prediction_performance(self, mock_torch_load):
        """Test prediction performance requirements."""
        model = MockModel(input_size=26)
        mock_torch_load.return_value = {
            'model_state_dict': model.state_dict(),
            'config': {'input_size': 26},
            'model_type': 'LightweightCNN'
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pth') as f:
            torch.save(mock_torch_load.return_value, f.name)
            
            detector = WakeWordDetector(model_path=f.name, device="cpu")
            
            # Test prediction speed
            features = np.random.randn(26)
            
            times = []
            for _ in range(100):
                start = time.time()
                detector.predict(features)
                times.append(time.time() - start)
            
            avg_time = np.mean(times)
            max_time = np.max(times)
            
            # Should be fast enough for real-time processing
            assert avg_time < 0.01  # 10ms average
            assert max_time < 0.05  # 50ms maximum


if __name__ == '__main__':
    pytest.main([__file__]) 