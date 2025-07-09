"""Tests for audio processing functionality."""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
import wave

# Import the audio processor
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from wakeywakey.core.audio import AudioProcessor


class TestAudioProcessor:
    """Test suite for AudioProcessor class."""
    
    @pytest.fixture
    def audio_processor(self):
        """Create AudioProcessor instance for testing."""
        return AudioProcessor(
            sample_rate=16000,
            n_mfcc=13,
            window_length=1.0
        )
    
    @pytest.fixture
    def sample_audio(self):
        """Generate sample audio data for testing."""
        sample_rate = 16000
        duration = 1.0
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = np.sin(2 * np.pi * frequency * t) * 0.3
        
        return audio.astype(np.float32)
    
    @pytest.fixture
    def audio_file(self, sample_audio):
        """Create temporary audio file for testing."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
            # Write WAV file
            with wave.open(f.name, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)
                
                # Convert to 16-bit PCM
                audio_int16 = (sample_audio * 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())
            
            yield f.name
            
            # Cleanup
            os.unlink(f.name)
    
    def test_initialization(self):
        """Test AudioProcessor initialization."""
        processor = AudioProcessor(
            sample_rate=8000,
            n_mfcc=12,
            window_length=0.5
        )
        
        assert processor.sample_rate == 8000
        assert processor.n_mfcc == 12
        assert processor.window_length == 0.5
        assert processor.window_samples == 4000  # 8000 * 0.5
    
    def test_mfcc_extraction(self, audio_processor, sample_audio):
        """Test MFCC feature extraction."""
        features = audio_processor.extract_mfcc_features(sample_audio)
        
        # Should return concatenated mean and std features
        expected_size = audio_processor.n_mfcc * 2  # mean + std
        assert len(features) == expected_size
        assert isinstance(features, np.ndarray)
        assert not np.isnan(features).any()
        assert not np.isinf(features).any()
    
    def test_mfcc_with_delta_features(self, audio_processor, sample_audio):
        """Test MFCC extraction with delta features."""
        features = audio_processor.extract_mfcc_features(
            sample_audio, 
            delta_features=True
        )
        
        # Should include delta and delta-delta features (3x)
        expected_size = audio_processor.n_mfcc * 2 * 3  # (mfcc + delta + delta2) * (mean + std)
        assert len(features) == expected_size
    
    def test_spectral_features(self, audio_processor, sample_audio):
        """Test spectral feature extraction."""
        features = audio_processor.extract_spectral_features(sample_audio)
        
        expected_keys = [
            'spectral_centroid_mean', 'spectral_centroid_std',
            'spectral_rolloff_mean', 'spectral_rolloff_std',
            'zcr_mean', 'zcr_std',
            'rms_mean', 'rms_std',
            'spectral_bandwidth_mean', 'spectral_bandwidth_std'
        ]
        
        assert isinstance(features, dict)
        for key in expected_keys:
            assert key in features
            assert isinstance(features[key], (float, np.floating))
            assert not np.isnan(features[key])
    
    def test_preprocess_audio(self, audio_processor, sample_audio):
        """Test audio preprocessing."""
        # Add some noise
        noisy_audio = sample_audio + np.random.normal(0, 0.01, len(sample_audio))
        
        processed = audio_processor.preprocess_audio(
            noisy_audio,
            remove_noise=True,
            normalize_volume=True
        )
        
        assert isinstance(processed, np.ndarray)
        assert len(processed) == len(noisy_audio)
        # Should be normalized to [-1, 1] range approximately
        assert np.abs(processed).max() <= 1.0
    
    def test_process_audio_file(self, audio_processor, audio_file):
        """Test processing audio from file."""
        processed_audio = audio_processor.process_audio_file(audio_file)
        
        assert processed_audio is not None
        assert isinstance(processed_audio, np.ndarray)
        assert len(processed_audio) > 0
    
    def test_get_audio_devices(self, audio_processor):
        """Test getting available audio devices."""
        try:
            devices = audio_processor.get_audio_devices()
            assert isinstance(devices, list)
            
            if devices:  # If audio devices are available
                device = devices[0]
                assert 'index' in device
                assert 'name' in device
                assert 'max_input_channels' in device
                
        except Exception:
            # Skip if audio system not available (e.g., in CI)
            pytest.skip("Audio system not available")
    
    def test_test_microphone(self, audio_processor):
        """Test microphone testing functionality."""
        try:
            success, info = audio_processor.test_microphone(duration=0.1)
            
            # Should return success status and info dict
            assert isinstance(success, bool)
            assert isinstance(info, dict)
            
            if success:
                assert 'duration' in info
                assert 'sample_rate' in info
                assert 'rms_level' in info
                
        except Exception:
            # Skip if microphone not available (e.g., in CI)
            pytest.skip("Microphone not available")
    
    def test_feature_consistency(self, audio_processor, sample_audio):
        """Test that feature extraction is consistent."""
        features1 = audio_processor.extract_mfcc_features(sample_audio)
        features2 = audio_processor.extract_mfcc_features(sample_audio)
        
        # Should be identical for same input
        np.testing.assert_array_almost_equal(features1, features2)
    
    def test_different_audio_lengths(self, audio_processor):
        """Test handling of different audio lengths."""
        # Short audio
        short_audio = np.random.randn(1000)
        features_short = audio_processor.extract_mfcc_features(short_audio)
        
        # Long audio
        long_audio = np.random.randn(50000)
        features_long = audio_processor.extract_mfcc_features(long_audio)
        
        # Should return same feature size regardless of input length
        assert len(features_short) == len(features_long)
    
    def test_silent_audio(self, audio_processor):
        """Test handling of silent audio."""
        silent_audio = np.zeros(16000)  # 1 second of silence
        features = audio_processor.extract_mfcc_features(silent_audio)
        
        # Should not crash and return valid features
        assert len(features) == audio_processor.n_mfcc * 2
        assert not np.isnan(features).any()
    
    def test_normalize_parameter(self, audio_processor, sample_audio):
        """Test normalize parameter in MFCC extraction."""
        features_normalized = audio_processor.extract_mfcc_features(
            sample_audio, normalize=True
        )
        features_unnormalized = audio_processor.extract_mfcc_features(
            sample_audio, normalize=False
        )
        
        # Normalized features should have different statistics
        assert not np.allclose(features_normalized, features_unnormalized)
        
        # Normalized features should be roughly centered around 0
        assert abs(np.mean(features_normalized)) < 0.1
    
    def test_audio_buffer_management(self, audio_processor):
        """Test audio buffer operations."""
        # Test buffer initialization
        assert len(audio_processor.audio_buffer) == 0
        
        # Test buffer capacity
        max_samples = audio_processor.window_samples
        test_audio = np.random.randn(max_samples * 2)
        
        # Simulate filling buffer
        for sample in test_audio:
            audio_processor.audio_buffer.append(sample)
        
        # Should not exceed max capacity
        assert len(audio_processor.audio_buffer) <= max_samples


class TestAudioProcessorEdgeCases:
    """Test edge cases and error handling."""
    
    def test_invalid_sample_rate(self):
        """Test handling of invalid sample rates."""
        with pytest.warns(UserWarning):
            # Very low sample rate might cause warnings
            processor = AudioProcessor(sample_rate=100)
    
    def test_nonexistent_audio_file(self):
        """Test handling of nonexistent audio files."""
        processor = AudioProcessor()
        result = processor.process_audio_file("nonexistent_file.wav")
        assert result is None
    
    def test_empty_audio_array(self):
        """Test handling of empty audio arrays."""
        processor = AudioProcessor()
        empty_audio = np.array([])
        
        features = processor.extract_mfcc_features(empty_audio)
        # Should return zero features instead of crashing
        assert len(features) == processor.n_mfcc * 2
    
    def test_nan_audio_input(self):
        """Test handling of NaN values in audio."""
        processor = AudioProcessor()
        nan_audio = np.full(16000, np.nan)
        
        features = processor.extract_mfcc_features(nan_audio)
        # Should handle gracefully and return valid features
        assert len(features) == processor.n_mfcc * 2


class TestAudioProcessorPerformance:
    """Performance tests for audio processing."""
    
    def test_mfcc_extraction_speed(self, audio_processor, sample_audio):
        """Test MFCC extraction performance."""
        import time
        
        start_time = time.time()
        for _ in range(100):
            audio_processor.extract_mfcc_features(sample_audio)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        # Should process audio faster than real-time
        assert avg_time < 0.1  # 100ms for 1 second of audio
    
    def test_real_time_processing_simulation(self, audio_processor):
        """Simulate real-time processing requirements."""
        # Simulate 100ms audio chunks
        chunk_size = int(0.1 * audio_processor.sample_rate)
        
        processing_times = []
        for _ in range(50):  # Test 50 chunks
            chunk = np.random.randn(chunk_size)
            
            start_time = time.time()
            audio_processor.extract_mfcc_features(chunk)
            processing_time = time.time() - start_time
            
            processing_times.append(processing_time)
        
        avg_processing_time = np.mean(processing_times)
        max_processing_time = np.max(processing_times)
        
        # Should process much faster than real-time
        assert avg_processing_time < 0.01  # 10ms for 100ms audio
        assert max_processing_time < 0.05  # Max 50ms


if __name__ == '__main__':
    pytest.main([__file__]) 