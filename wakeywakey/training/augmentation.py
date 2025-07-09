"""
Audio augmentation utilities for training data enhancement.

Provides comprehensive audio augmentation including time/frequency domain
transformations, noise injection, and SpecAugment techniques.
"""

import random
from typing import Optional, Dict, Any, List, Union, Tuple
import logging

import numpy as np
import librosa
import torch

logger = logging.getLogger(__name__)


class AudioAugmentation:
    """
    Comprehensive audio augmentation for wake word detection training.
    
    Provides various augmentation techniques including time stretching,
    pitch shifting, noise injection, and frequency masking.
    """
    
    def __init__(
        self,
        augmentation_config: Optional[Dict[str, Any]] = None,
        sample_rate: int = 16000
    ):
        """
        Initialize audio augmentation.
        
        Args:
            augmentation_config: Augmentation configuration dictionary
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        
        # Default configuration
        default_config = {
            'time_stretch': {
                'enabled': True,
                'rate_range': (0.8, 1.2),
                'probability': 0.3
            },
            'pitch_shift': {
                'enabled': True,
                'n_steps_range': (-2, 2),
                'probability': 0.3
            },
            'noise_injection': {
                'enabled': True,
                'snr_range': (10, 30),
                'probability': 0.4
            },
            'volume_change': {
                'enabled': True,
                'gain_range': (-6, 6),
                'probability': 0.3
            },
            'spec_augment': {
                'enabled': True,
                'freq_mask_prob': 0.2,
                'time_mask_prob': 0.2,
                'freq_mask_width': 3,
                'time_mask_width': 5
            },
            'reverb': {
                'enabled': False,  # Requires room impulse responses
                'probability': 0.2
            }
        }
        
        # Update with user config
        self.config = default_config
        if augmentation_config:
            self._update_config(self.config, augmentation_config)
        
        logger.info("AudioAugmentation initialized")
    
    def _update_config(self, base_config: Dict, update_config: Dict):
        """Recursively update configuration."""
        for key, value in update_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._update_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def apply(self, features: np.ndarray, audio_file: Optional[str] = None) -> np.ndarray:
        """
        Apply augmentation to feature vector.
        
        Args:
            features: Input feature vector
            audio_file: Source audio file (for loading raw audio if needed)
            
        Returns:
            Augmented feature vector
        """
        # For MFCC features, we'll apply feature-space augmentations
        # For time-domain augmentations, we'd need the raw audio
        
        augmented_features = features.copy()
        
        # Feature-space augmentations
        augmented_features = self._apply_feature_noise(augmented_features)
        augmented_features = self._apply_feature_scaling(augmented_features)
        augmented_features = self._apply_spec_augment(augmented_features)
        
        return augmented_features
    
    def apply_to_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply time-domain augmentations to raw audio.
        
        Args:
            audio: Raw audio signal
            
        Returns:
            Augmented audio signal
        """
        augmented_audio = audio.copy()
        
        # Time stretching
        if self._should_apply('time_stretch'):
            augmented_audio = self._time_stretch(augmented_audio)
        
        # Pitch shifting
        if self._should_apply('pitch_shift'):
            augmented_audio = self._pitch_shift(augmented_audio)
        
        # Noise injection
        if self._should_apply('noise_injection'):
            augmented_audio = self._add_noise(augmented_audio)
        
        # Volume change
        if self._should_apply('volume_change'):
            augmented_audio = self._change_volume(augmented_audio)
        
        return augmented_audio
    
    def _should_apply(self, augmentation_type: str) -> bool:
        """Check if augmentation should be applied based on probability."""
        config = self.config.get(augmentation_type, {})
        if not config.get('enabled', False):
            return False
        
        probability = config.get('probability', 0.5)
        return random.random() < probability
    
    def _time_stretch(self, audio: np.ndarray) -> np.ndarray:
        """Apply time stretching."""
        try:
            config = self.config['time_stretch']
            rate_range = config['rate_range']
            rate = random.uniform(rate_range[0], rate_range[1])
            
            stretched = librosa.effects.time_stretch(audio, rate=rate)
            
            # Ensure same length by padding or truncating
            if len(stretched) > len(audio):
                stretched = stretched[:len(audio)]
            elif len(stretched) < len(audio):
                padding = len(audio) - len(stretched)
                stretched = np.pad(stretched, (0, padding), mode='constant')
            
            return stretched
            
        except Exception as e:
            logger.warning(f"Time stretch failed: {e}")
            return audio
    
    def _pitch_shift(self, audio: np.ndarray) -> np.ndarray:
        """Apply pitch shifting."""
        try:
            config = self.config['pitch_shift']
            n_steps_range = config['n_steps_range']
            n_steps = random.uniform(n_steps_range[0], n_steps_range[1])
            
            shifted = librosa.effects.pitch_shift(
                audio, sr=self.sample_rate, n_steps=n_steps
            )
            
            return shifted
            
        except Exception as e:
            logger.warning(f"Pitch shift failed: {e}")
            return audio
    
    def _add_noise(self, audio: np.ndarray) -> np.ndarray:
        """Add white noise to audio."""
        try:
            config = self.config['noise_injection']
            snr_range = config['snr_range']
            snr_db = random.uniform(snr_range[0], snr_range[1])
            
            # Calculate noise level
            signal_power = np.mean(audio ** 2)
            snr_linear = 10 ** (snr_db / 10)
            noise_power = signal_power / snr_linear
            
            # Generate and add noise
            noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
            noisy_audio = audio + noise
            
            return noisy_audio
            
        except Exception as e:
            logger.warning(f"Noise injection failed: {e}")
            return audio
    
    def _change_volume(self, audio: np.ndarray) -> np.ndarray:
        """Change audio volume."""
        try:
            config = self.config['volume_change']
            gain_range = config['gain_range']
            gain_db = random.uniform(gain_range[0], gain_range[1])
            
            # Convert dB to linear scale
            gain_linear = 10 ** (gain_db / 20)
            
            # Apply gain
            gained_audio = audio * gain_linear
            
            # Clip to prevent overflow
            gained_audio = np.clip(gained_audio, -1.0, 1.0)
            
            return gained_audio
            
        except Exception as e:
            logger.warning(f"Volume change failed: {e}")
            return audio
    
    def _apply_feature_noise(self, features: np.ndarray) -> np.ndarray:
        """Add noise to feature vector."""
        if not self._should_apply('noise_injection'):
            return features
        
        try:
            # Add small amount of gaussian noise to features
            noise_std = 0.01 * np.std(features)
            noise = np.random.normal(0, noise_std, features.shape)
            
            return features + noise
            
        except Exception as e:
            logger.warning(f"Feature noise failed: {e}")
            return features
    
    def _apply_feature_scaling(self, features: np.ndarray) -> np.ndarray:
        """Apply random scaling to features."""
        if not self._should_apply('volume_change'):
            return features
        
        try:
            # Random scaling factor
            scale_factor = random.uniform(0.8, 1.2)
            return features * scale_factor
            
        except Exception as e:
            logger.warning(f"Feature scaling failed: {e}")
            return features
    
    def _apply_spec_augment(self, features: np.ndarray) -> np.ndarray:
        """Apply SpecAugment-style masking to features."""
        config = self.config.get('spec_augment', {})
        if not config.get('enabled', False):
            return features
        
        try:
            augmented = features.copy()
            
            # Frequency masking (mask some feature dimensions)
            if random.random() < config.get('freq_mask_prob', 0.2):
                mask_width = config.get('freq_mask_width', 3)
                mask_width = min(mask_width, len(features) // 4)  # Limit mask size
                
                if mask_width > 0:
                    start_idx = random.randint(0, len(features) - mask_width)
                    augmented[start_idx:start_idx + mask_width] = 0
            
            return augmented
            
        except Exception as e:
            logger.warning(f"SpecAugment failed: {e}")
            return features
    
    def _apply_reverb(self, audio: np.ndarray) -> np.ndarray:
        """Apply simple reverb effect."""
        try:
            # Simple reverb simulation with decay
            delay_samples = int(0.1 * self.sample_rate)  # 100ms delay
            decay = 0.3
            
            reverb_audio = audio.copy()
            
            # Add delayed version
            if len(audio) > delay_samples:
                reverb_audio[delay_samples:] += audio[:-delay_samples] * decay
            
            return reverb_audio
            
        except Exception as e:
            logger.warning(f"Reverb failed: {e}")
            return audio
    
    def create_mixup_sample(
        self,
        features1: np.ndarray,
        features2: np.ndarray,
        label1: int,
        label2: int,
        alpha: float = 0.2
    ) -> Tuple[np.ndarray, float]:
        """
        Create mixup sample from two feature vectors.
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            label1: First label
            label2: Second label
            alpha: Mixup parameter
            
        Returns:
            Tuple of (mixed_features, mixed_label)
        """
        try:
            # Sample mixing ratio
            lam = np.random.beta(alpha, alpha) if alpha > 0 else 0.5
            
            # Mix features
            mixed_features = lam * features1 + (1 - lam) * features2
            
            # Mix labels
            mixed_label = lam * label1 + (1 - lam) * label2
            
            return mixed_features, mixed_label
            
        except Exception as e:
            logger.warning(f"Mixup failed: {e}")
            return features1, float(label1)
    
    def create_cutmix_sample(
        self,
        features1: np.ndarray,
        features2: np.ndarray,
        label1: int,
        label2: int,
        alpha: float = 1.0
    ) -> Tuple[np.ndarray, float]:
        """
        Create CutMix sample from two feature vectors.
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            label1: First label
            label2: Second label
            alpha: CutMix parameter
            
        Returns:
            Tuple of (mixed_features, mixed_label)
        """
        try:
            # Sample mixing ratio
            lam = np.random.beta(alpha, alpha) if alpha > 0 else 0.5
            
            # Determine cut region
            cut_ratio = np.sqrt(1 - lam)
            cut_w = int(len(features1) * cut_ratio)
            
            if cut_w > 0 and cut_w < len(features1):
                cut_start = random.randint(0, len(features1) - cut_w)
                cut_end = cut_start + cut_w
                
                # Create mixed features
                mixed_features = features1.copy()
                mixed_features[cut_start:cut_end] = features2[cut_start:cut_end]
                
                # Adjust label based on actual mixing ratio
                actual_lam = 1 - (cut_w / len(features1))
                mixed_label = actual_lam * label1 + (1 - actual_lam) * label2
                
                return mixed_features, mixed_label
            else:
                return features1, float(label1)
                
        except Exception as e:
            logger.warning(f"CutMix failed: {e}")
            return features1, float(label1)
    
    def get_augmentation_stats(self) -> Dict[str, Any]:
        """Get augmentation configuration and statistics."""
        stats = {
            'config': self.config,
            'enabled_augmentations': [],
            'sample_rate': self.sample_rate
        }
        
        for aug_type, config in self.config.items():
            if config.get('enabled', False):
                stats['enabled_augmentations'].append(aug_type)
        
        return stats
    
    def disable_augmentation(self, augmentation_type: str):
        """Disable specific augmentation type."""
        if augmentation_type in self.config:
            self.config[augmentation_type]['enabled'] = False
            logger.info(f"Disabled {augmentation_type} augmentation")
    
    def enable_augmentation(self, augmentation_type: str):
        """Enable specific augmentation type."""
        if augmentation_type in self.config:
            self.config[augmentation_type]['enabled'] = True
            logger.info(f"Enabled {augmentation_type} augmentation")
    
    def set_probability(self, augmentation_type: str, probability: float):
        """Set probability for specific augmentation type."""
        if augmentation_type in self.config:
            self.config[augmentation_type]['probability'] = probability
            logger.info(f"Set {augmentation_type} probability to {probability}")


# Convenience function for creating default augmentation
def create_default_augmentation(sample_rate: int = 16000) -> AudioAugmentation:
    """Create default audio augmentation configuration."""
    return AudioAugmentation(sample_rate=sample_rate) 