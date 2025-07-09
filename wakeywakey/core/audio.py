"""
Audio processing module for wake word detection.

Provides real-time audio capture, MFCC feature extraction, and preprocessing
optimized for 16kHz sampling rate and microcontroller deployment.
"""

import numpy as np
import librosa
import sounddevice as sd
import threading
import time
import queue
from typing import Optional, Tuple, Callable, Dict, Any, List
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Audio processor for wake word detection with real-time MFCC extraction.
    
    Optimized for 16kHz sampling rate with sliding window processing for
    low-latency detection on edge devices.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 40,
        window_size: float = 1.0,
        stride: float = 0.5,
        normalize: bool = True,
        add_deltas: bool = True,
        device_id: Optional[int] = None
    ):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Audio sampling rate (default: 16000 Hz)
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Number of samples between frames
            n_mels: Number of mel filter banks
            window_size: Window size in seconds for feature extraction
            stride: Stride in seconds between windows
            normalize: Whether to normalize features
            add_deltas: Whether to add delta (velocity) features
            device_id: Audio device ID (None for default)
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize
        self.add_deltas = add_deltas
        self.device_id = device_id
        
        # Calculate derived parameters
        self.window_samples = int(window_size * sample_rate)
        self.stride_samples = int(stride * sample_rate)
        self.n_frames = 1 + (self.window_samples - n_fft) // hop_length
        
        # Feature dimensions
        self.base_features = n_mfcc
        self.feature_dim = n_mfcc * (2 if add_deltas else 1)
        
        # Real-time processing state
        self.is_recording = False
        self.audio_buffer = np.zeros(self.window_samples * 2)  # Double buffer
        self.buffer_lock = threading.Lock()
        self.audio_thread: Optional[threading.Thread] = None
        self.audio_queue = queue.Queue(maxsize=10)
        
        # Normalization statistics (computed from data or loaded)
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        
        logger.info(f"AudioProcessor initialized: {sample_rate}Hz, {n_mfcc} MFCC, "
                   f"{self.feature_dim}D features")
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio signal.
        
        Args:
            audio: Audio signal array
            
        Returns:
            MFCC feature vector of shape (feature_dim,)
        """
        try:
            # Ensure minimum length
            if len(audio) < self.n_fft:
                audio = np.pad(audio, (0, self.n_fft - len(audio)))
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio.astype(np.float32),
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmin=0,
                fmax=self.sample_rate // 2
            )
            
            # Aggregate across time dimension (mean)
            mfcc_features = np.mean(mfccs, axis=1)
            
            # Add delta features if requested
            if self.add_deltas:
                # Simple delta calculation (could be improved)
                deltas = np.gradient(mfcc_features)
                features = np.concatenate([mfcc_features, deltas])
            else:
                features = mfcc_features
            
            # Normalize if enabled
            if self.normalize and self.feature_mean is not None:
                features = (features - self.feature_mean) / (self.feature_std + 1e-8)
            
            return features.astype(np.float32)
            
        except Exception as e:
            logger.error(f"MFCC extraction failed: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)
    
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Preprocess audio signal.
        
        Args:
            audio: Raw audio signal
            
        Returns:
            Preprocessed audio signal
        """
        # Convert to float32 and normalize
        audio = audio.astype(np.float32)
        
        # Remove DC offset
        audio = audio - np.mean(audio)
        
        # Normalize amplitude
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        # Apply pre-emphasis filter
        pre_emphasis = 0.97
        audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        
        return audio
    
    def extract_features_from_file(self, file_path: str) -> np.ndarray:
        """
        Extract features from audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Feature vector
        """
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            audio = self.preprocess_audio(audio)
            
            # Ensure window size
            if len(audio) < self.window_samples:
                audio = np.pad(audio, (0, self.window_samples - len(audio)))
            else:
                audio = audio[:self.window_samples]
            
            return self.extract_mfcc(audio)
            
        except Exception as e:
            logger.error(f"Failed to extract features from {file_path}: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)
    
    def fit_normalization(self, audio_files: List[str], cache_path: Optional[str] = None):
        """
        Compute normalization statistics from training data.
        
        Args:
            audio_files: List of audio file paths
            cache_path: Path to cache statistics
        """
        logger.info(f"Computing normalization statistics from {len(audio_files)} files...")
        
        features_list = []
        for file_path in audio_files:
            # Extract features without normalization
            old_normalize = self.normalize
            self.normalize = False
            features = self.extract_features_from_file(file_path)
            self.normalize = old_normalize
            
            if features is not None and not np.any(np.isnan(features)):
                features_list.append(features)
        
        if features_list:
            features_array = np.array(features_list)
            self.feature_mean = np.mean(features_array, axis=0)
            self.feature_std = np.std(features_array, axis=0)
            
            # Prevent division by zero
            self.feature_std = np.maximum(self.feature_std, 1e-8)
            
            logger.info(f"Normalization stats computed: mean={self.feature_mean[:3]}, "
                       f"std={self.feature_std[:3]}")
            
            # Cache statistics if requested
            if cache_path:
                self.save_normalization_stats(cache_path)
        else:
            logger.warning("No valid features found for normalization")
    
    def save_normalization_stats(self, cache_path: str):
        """Save normalization statistics to file."""
        stats = {
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
            'feature_dim': self.feature_dim,
            'n_mfcc': self.n_mfcc,
            'add_deltas': self.add_deltas
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(stats, f)
        
        logger.info(f"Normalization stats saved to {cache_path}")
    
    def load_normalization_stats(self, cache_path: str) -> bool:
        """
        Load normalization statistics from file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(cache_path, 'rb') as f:
                stats = pickle.load(f)
            
            # Validate compatibility
            if (stats['feature_dim'] != self.feature_dim or 
                stats['n_mfcc'] != self.n_mfcc or
                stats['add_deltas'] != self.add_deltas):
                logger.warning("Cached stats incompatible with current configuration")
                return False
            
            self.feature_mean = stats['feature_mean']
            self.feature_std = stats['feature_std']
            
            logger.info(f"Normalization stats loaded from {cache_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load normalization stats: {e}")
            return False
    
    def start_recording(self, callback: Optional[Callable[[np.ndarray], None]] = None):
        """
        Start real-time audio recording.
        
        Args:
            callback: Function to call with extracted features
        """
        if self.is_recording:
            logger.warning("Recording already in progress")
            return
        
        self.is_recording = True
        
        def audio_callback(indata, frames, time, status):
            """Callback for audio input."""
            if status:
                logger.warning(f"Audio callback status: {status}")
            
            audio_data = indata[:, 0] if indata.ndim > 1 else indata
            
            with self.buffer_lock:
                # Shift buffer and add new data
                shift_samples = len(audio_data)
                self.audio_buffer[:-shift_samples] = self.audio_buffer[shift_samples:]
                self.audio_buffer[-shift_samples:] = audio_data
                
                # Extract features from current window
                if len(self.audio_buffer) >= self.window_samples:
                    window_audio = self.audio_buffer[-self.window_samples:].copy()
                    
                    # Process in separate thread to avoid blocking
                    try:
                        if not self.audio_queue.full():
                            self.audio_queue.put(window_audio, block=False)
                    except queue.Full:
                        pass  # Skip this frame if queue is full
        
        # Start audio stream
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                device=self.device_id,
                callback=audio_callback,
                blocksize=self.stride_samples
            )
            self.stream.start()
            
            # Start processing thread
            self.audio_thread = threading.Thread(target=self._audio_processing_thread, 
                                                args=(callback,))
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
            logger.info("Audio recording started")
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self.is_recording = False
            raise
    
    def _audio_processing_thread(self, callback: Optional[Callable[[np.ndarray], None]]):
        """Thread for processing audio data."""
        while self.is_recording:
            try:
                # Get audio data with timeout
                audio_data = self.audio_queue.get(timeout=0.1)
                
                # Preprocess and extract features
                processed_audio = self.preprocess_audio(audio_data)
                features = self.extract_mfcc(processed_audio)
                
                # Call callback if provided
                if callback and features is not None:
                    callback(features)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
    
    def stop_recording(self):
        """Stop real-time audio recording."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
        
        logger.info("Audio recording stopped")
    
    def test_microphone(self, duration: float = 5.0) -> Dict[str, Any]:
        """
        Test microphone setup and audio processing.
        
        Args:
            duration: Test duration in seconds
            
        Returns:
            Test results dictionary
        """
        logger.info(f"Testing microphone for {duration} seconds...")
        
        test_results = {
            'success': False,
            'duration': duration,
            'features_extracted': 0,
            'avg_amplitude': 0.0,
            'feature_stats': {},
            'errors': []
        }
        
        features_list = []
        amplitudes = []
        
        def test_callback(features):
            features_list.append(features)
            test_results['features_extracted'] += 1
        
        try:
            # Start recording with callback
            self.start_recording(callback=test_callback)
            
            # Record for specified duration
            start_time = time.time()
            while time.time() - start_time < duration:
                time.sleep(0.1)
                
                # Monitor audio buffer amplitude
                with self.buffer_lock:
                    if len(self.audio_buffer) > 0:
                        amplitudes.append(np.max(np.abs(self.audio_buffer)))
            
            # Stop recording
            self.stop_recording()
            
            # Analyze results
            if features_list:
                features_array = np.array(features_list)
                test_results['feature_stats'] = {
                    'mean': np.mean(features_array, axis=0)[:5].tolist(),
                    'std': np.std(features_array, axis=0)[:5].tolist(),
                    'shape': features_array.shape
                }
                test_results['success'] = True
            
            if amplitudes:
                test_results['avg_amplitude'] = float(np.mean(amplitudes))
            
            logger.info(f"Microphone test completed: {test_results['features_extracted']} "
                       f"features extracted, avg amplitude: {test_results['avg_amplitude']:.4f}")
            
        except Exception as e:
            error_msg = f"Microphone test failed: {e}"
            logger.error(error_msg)
            test_results['errors'].append(error_msg)
            
            # Ensure cleanup
            try:
                self.stop_recording()
            except:
                pass
        
        return test_results
    
    @staticmethod
    def list_audio_devices() -> List[Dict[str, Any]]:
        """
        List available audio input devices.
        
        Returns:
            List of device information dictionaries
        """
        try:
            devices = sd.query_devices()
            input_devices = []
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    input_devices.append({
                        'id': i,
                        'name': device['name'],
                        'channels': device['max_input_channels'],
                        'sample_rate': device['default_samplerate'],
                        'hostapi': sd.query_hostapis(device['hostapi'])['name']
                    })
            
            return input_devices
            
        except Exception as e:
            logger.error(f"Failed to list audio devices: {e}")
            return []
    
    def get_config(self) -> Dict[str, Any]:
        """Get processor configuration."""
        return {
            'sample_rate': self.sample_rate,
            'n_mfcc': self.n_mfcc,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'n_mels': self.n_mels,
            'window_size': self.window_size,
            'stride': self.stride,
            'normalize': self.normalize,
            'add_deltas': self.add_deltas,
            'feature_dim': self.feature_dim,
            'device_id': self.device_id
        }
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.stop_recording()
        except:
            pass 