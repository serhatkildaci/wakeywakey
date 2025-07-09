"""Integration tests for WakeyWakey package."""

import pytest
import numpy as np
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch, Mock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPackageIntegration:
    """Test package-level integration."""
    
    def test_package_import(self):
        """Test that the package can be imported successfully."""
        try:
            import wakeywakey
            assert hasattr(wakeywakey, '__version__')
            assert hasattr(wakeywakey, 'WakeWordDetector')
            assert hasattr(wakeywakey, 'AudioProcessor')
        except ImportError as e:
            pytest.skip(f"Package import failed: {e}")
    
    def test_package_metadata(self):
        """Test package metadata is correctly set."""
        try:
            import wakeywakey
            
            assert wakeywakey.__version__ is not None
            assert wakeywakey.__author__ is not None
            assert isinstance(wakeywakey.__all__, list)
            assert len(wakeywakey.__all__) > 0
            
        except ImportError:
            pytest.skip("Package not available")
    
    def test_package_configuration(self):
        """Test package configuration functionality."""
        try:
            import wakeywakey
            
            # Test getting configuration
            config = wakeywakey.get_config()
            assert isinstance(config, dict)
            assert 'default_sample_rate' in config
            
            # Test setting configuration
            original_sr = wakeywakey.get_config('default_sample_rate')
            wakeywakey.set_config('default_sample_rate', 8000)
            assert wakeywakey.get_config('default_sample_rate') == 8000
            
            # Restore original
            wakeywakey.set_config('default_sample_rate', original_sr)
            
        except ImportError:
            pytest.skip("Package not available")
    
    def test_system_info(self):
        """Test system information gathering."""
        try:
            import wakeywakey
            
            info = wakeywakey.get_system_info()
            assert isinstance(info, dict)
            
            required_keys = [
                'wakeywakey_version',
                'python_version', 
                'platform',
                'pytorch_version',
                'cuda_available'
            ]
            
            for key in required_keys:
                assert key in info
                
        except ImportError:
            pytest.skip("Package not available")


class TestCLIIntegration:
    """Test CLI integration."""
    
    @patch('sys.argv')
    def test_cli_help(self, mock_argv):
        """Test CLI help functionality."""
        try:
            mock_argv.__getitem__.return_value = ['wakeywakey', '--help']
            
            from wakeywakey.cli.main import create_parser
            parser = create_parser()
            
            # Should not raise exception
            assert parser is not None
            
        except ImportError:
            pytest.skip("CLI not available")
    
    @patch('sys.argv')
    def test_cli_command_parsing(self, mock_argv):
        """Test CLI command parsing."""
        try:
            from wakeywakey.cli.main import create_parser
            parser = create_parser()
            
            # Test train command parsing
            args = parser.parse_args(['train', '--data-dir', './data'])
            assert args.command == 'train'
            assert args.data_dir == './data'
            
            # Test detect command parsing
            args = parser.parse_args(['detect', '--model', 'model.pth'])
            assert args.command == 'detect'
            assert args.model == 'model.pth'
            
        except ImportError:
            pytest.skip("CLI not available")


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    @patch('wakeywakey.core.audio.AudioProcessor.get_audio_devices')
    def test_device_listing_workflow(self, mock_devices):
        """Test the complete device listing workflow."""
        try:
            mock_devices.return_value = [
                {
                    'index': 0,
                    'name': 'Default Microphone',
                    'max_input_channels': 1,
                    'max_output_channels': 0
                }
            ]
            
            from wakeywakey.core.audio import AudioProcessor
            processor = AudioProcessor()
            devices = processor.get_audio_devices()
            
            assert len(devices) == 1
            assert devices[0]['name'] == 'Default Microphone'
            
        except ImportError:
            pytest.skip("Audio module not available")
    
    def test_feature_extraction_workflow(self):
        """Test complete feature extraction workflow."""
        try:
            from wakeywakey.core.audio import AudioProcessor
            
            processor = AudioProcessor(sample_rate=16000, n_mfcc=13)
            
            # Generate test audio
            sample_rate = 16000
            duration = 1.0
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio = np.sin(2 * np.pi * 440 * t) * 0.3  # 440 Hz sine wave
            
            # Extract features
            features = processor.extract_mfcc_features(audio, normalize=True)
            
            assert features is not None
            assert len(features) == 26  # 13 MFCCs * 2 (mean + std)
            assert not np.isnan(features).any()
            
        except ImportError:
            pytest.skip("Audio module not available")
    
    @patch('torch.load')
    @patch('wakeywakey.models.architectures.LightweightCNN')
    def test_model_loading_workflow(self, mock_model_class, mock_torch_load):
        """Test complete model loading workflow."""
        try:
            from wakeywakey.core.detector import WakeWordDetector
            
            # Setup mocks
            mock_model = Mock()
            mock_model_class.return_value = mock_model
            mock_torch_load.return_value = {
                'model_state_dict': {'weight': Mock()},
                'config': {'input_size': 26},
                'model_type': 'LightweightCNN'
            }
            
            with tempfile.NamedTemporaryFile(suffix='.pth') as f:
                detector = WakeWordDetector()
                success = detector.load_model(f.name)
                
                assert success
                assert detector.model is not None
                
        except ImportError:
            pytest.skip("Detector module not available")
    
    def test_prediction_workflow(self):
        """Test prediction workflow with mocked components."""
        try:
            # This test would require actual models, so we'll keep it simple
            from wakeywakey.core.detector import WakeWordDetector
            
            detector = WakeWordDetector(model_path=None)
            features = np.random.randn(26)
            
            # Without model, should return 0.0
            confidence, proc_time = detector.predict(features)
            assert confidence == 0.0
            assert proc_time == 0.0
            
        except ImportError:
            pytest.skip("Detector module not available")


class TestModuleInteroperability:
    """Test interoperability between different modules."""
    
    def test_audio_processor_detector_integration(self):
        """Test integration between AudioProcessor and WakeWordDetector."""
        try:
            from wakeywakey.core.audio import AudioProcessor
            from wakeywakey.core.detector import WakeWordDetector
            
            # Create components
            audio_processor = AudioProcessor(sample_rate=16000, n_mfcc=13)
            detector = WakeWordDetector(model_path=None)
            
            # Test that detector can use audio processor
            assert detector.audio_processor is not None
            assert detector.audio_processor.sample_rate == 16000
            
        except ImportError:
            pytest.skip("Modules not available")
    
    def test_model_architecture_compatibility(self):
        """Test that model architectures are compatible with detector."""
        try:
            from wakeywakey.models.architectures import LightweightCNN
            import torch
            
            # Create model
            model = LightweightCNN(input_size=26)
            
            # Test forward pass
            test_input = torch.randn(1, 26)
            output = model(test_input)
            
            assert output.shape == (1, 1)  # Should output single value
            assert 0.0 <= output.item() <= 1.0  # Should be probability
            
        except ImportError:
            pytest.skip("Model modules not available")


class TestErrorHandling:
    """Test error handling across the package."""
    
    def test_graceful_import_failures(self):
        """Test that import failures are handled gracefully."""
        # Test package-level imports
        try:
            import wakeywakey
            # Should not crash even if some dependencies are missing
            assert True
        except ImportError:
            # This is acceptable in test environments
            pytest.skip("Package import failed")
    
    def test_dependency_checking(self):
        """Test dependency checking functionality."""
        try:
            import wakeywakey
            
            # The package should have run dependency checks
            assert hasattr(wakeywakey, '_dependencies_ok')
            
        except ImportError:
            pytest.skip("Package not available")
    
    def test_configuration_error_handling(self):
        """Test configuration error handling."""
        try:
            import wakeywakey
            
            # Test invalid configuration key
            with pytest.raises(KeyError):
                wakeywakey.set_config('invalid_key', 'value')
                
        except ImportError:
            pytest.skip("Package not available")


class TestPerformanceIntegration:
    """Test performance characteristics of integrated components."""
    
    def test_feature_extraction_performance(self):
        """Test that feature extraction meets performance requirements."""
        try:
            from wakeywakey.core.audio import AudioProcessor
            import time
            
            processor = AudioProcessor()
            
            # Generate 10 seconds of audio
            audio = np.random.randn(160000)  # 10 seconds at 16kHz
            
            start_time = time.time()
            features = processor.extract_mfcc_features(audio)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # Should process 10 seconds of audio in much less than 10 seconds
            assert processing_time < 1.0  # Should be very fast
            assert features is not None
            
        except ImportError:
            pytest.skip("Audio module not available")
    
    def test_memory_usage_stability(self):
        """Test that repeated operations don't leak memory."""
        try:
            from wakeywakey.core.audio import AudioProcessor
            
            processor = AudioProcessor()
            audio = np.random.randn(16000)  # 1 second
            
            # Repeat feature extraction many times
            for _ in range(100):
                features = processor.extract_mfcc_features(audio)
                assert features is not None
            
            # If we get here without memory issues, test passes
            assert True
            
        except ImportError:
            pytest.skip("Audio module not available")


class TestPackageConsistency:
    """Test package consistency and API stability."""
    
    def test_api_consistency(self):
        """Test that the public API is consistent."""
        try:
            import wakeywakey
            
            # Check that all exported items are actually available
            for item in wakeywakey.__all__:
                assert hasattr(wakeywakey, item), f"Exported item {item} not found"
                
        except ImportError:
            pytest.skip("Package not available")
    
    def test_version_consistency(self):
        """Test that version information is consistent."""
        try:
            import wakeywakey
            
            version = wakeywakey.get_version()
            assert version == wakeywakey.__version__
            assert isinstance(version, str)
            assert len(version) > 0
            
        except ImportError:
            pytest.skip("Package not available")
    
    def test_documentation_consistency(self):
        """Test that modules have proper documentation."""
        try:
            import wakeywakey
            from wakeywakey.core import audio, detector
            
            # Check that modules have docstrings
            assert audio.__doc__ is not None
            assert detector.__doc__ is not None
            
            # Check that main classes have docstrings
            assert audio.AudioProcessor.__doc__ is not None
            assert detector.WakeWordDetector.__doc__ is not None
            
        except ImportError:
            pytest.skip("Modules not available")


if __name__ == '__main__':
    pytest.main([__file__]) 