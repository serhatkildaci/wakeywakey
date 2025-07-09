"""
Dataset classes for wake word detection training.

Provides WakeWordDataset and AudioDataset for loading and preprocessing
audio data with feature caching and data validation.
"""

import os
import glob
import pickle
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging

import torch
from torch.utils.data import Dataset
import numpy as np
import librosa

from ..core.audio import AudioProcessor

logger = logging.getLogger(__name__)


class WakeWordDataset(Dataset):
    """
    Wake word dataset for binary classification.
    
    Loads audio files from positive and negative directories,
    extracts MFCC features, and provides caching capabilities.
    """
    
    def __init__(
        self,
        data_dir: str,
        audio_processor: Optional[AudioProcessor] = None,
        cache_features: bool = True,
        cache_path: Optional[str] = None,
        augmentation: Optional[object] = None,
        max_samples_per_class: Optional[int] = None,
        validation_split: float = 0.0
    ):
        """
        Initialize wake word dataset.
        
        Args:
            data_dir: Root directory containing 'positive' and 'negative' folders
            audio_processor: Audio processor for feature extraction
            cache_features: Whether to cache extracted features
            cache_path: Path to feature cache file
            augmentation: Audio augmentation object
            max_samples_per_class: Maximum samples per class (for debugging)
            validation_split: Fraction of data for validation
        """
        self.data_dir = Path(data_dir)
        self.audio_processor = audio_processor or AudioProcessor()
        self.cache_features = cache_features
        self.cache_path = cache_path or (self.data_dir / "features_cache.pkl")
        self.augmentation = augmentation
        self.max_samples_per_class = max_samples_per_class
        self.validation_split = validation_split
        
        # Data storage
        self.audio_files = []
        self.labels = []
        self.features_cache = {}
        
        # Load data
        self._load_data()
        
        # Load or create feature cache
        if cache_features:
            self._load_cache()
        
        logger.info(f"Dataset initialized: {len(self)} samples, "
                   f"{sum(self.labels)} positive, {len(self.labels) - sum(self.labels)} negative")
    
    def _load_data(self):
        """Load audio file paths and labels."""
        positive_dir = self.data_dir / "positive"
        negative_dir = self.data_dir / "negative"
        
        # Supported audio formats
        audio_extensions = ["*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg"]
        
        # Load positive samples
        positive_files = []
        if positive_dir.exists():
            for ext in audio_extensions:
                positive_files.extend(glob.glob(str(positive_dir / ext)))
        
        # Load negative samples
        negative_files = []
        if negative_dir.exists():
            for ext in audio_extensions:
                negative_files.extend(glob.glob(str(negative_dir / ext)))
        
        # Apply sample limits
        if self.max_samples_per_class:
            positive_files = positive_files[:self.max_samples_per_class]
            negative_files = negative_files[:self.max_samples_per_class]
        
        # Combine and shuffle
        all_files = positive_files + negative_files
        all_labels = [1] * len(positive_files) + [0] * len(negative_files)
        
        # Shuffle data
        combined = list(zip(all_files, all_labels))
        random.shuffle(combined)
        self.audio_files, self.labels = zip(*combined)
        
        self.audio_files = list(self.audio_files)
        self.labels = list(self.labels)
        
        logger.info(f"Loaded {len(positive_files)} positive and {len(negative_files)} negative samples")
    
    def _load_cache(self):
        """Load feature cache from disk."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # Validate cache compatibility
                if self._validate_cache(cache_data):
                    self.features_cache = cache_data.get('features', {})
                    logger.info(f"Loaded feature cache: {len(self.features_cache)} entries")
                else:
                    logger.warning("Cache incompatible, will rebuild")
                    self.features_cache = {}
                    
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.features_cache = {}
        else:
            logger.info("No feature cache found, will create new")
    
    def _validate_cache(self, cache_data: Dict) -> bool:
        """Validate cache compatibility."""
        if not isinstance(cache_data, dict):
            return False
        
        # Check audio processor config
        cached_config = cache_data.get('audio_config', {})
        current_config = self.audio_processor.get_config()
        
        # Key parameters that affect features
        key_params = ['sample_rate', 'n_mfcc', 'n_fft', 'hop_length', 'add_deltas']
        
        for param in key_params:
            if cached_config.get(param) != current_config.get(param):
                logger.warning(f"Cache config mismatch for {param}: "
                             f"{cached_config.get(param)} vs {current_config.get(param)}")
                return False
        
        return True
    
    def _save_cache(self):
        """Save feature cache to disk."""
        if not self.cache_features:
            return
        
        cache_data = {
            'features': self.features_cache,
            'audio_config': self.audio_processor.get_config(),
            'timestamp': os.path.getmtime(__file__),
            'version': '0.1.0'
        }
        
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Feature cache saved: {len(self.features_cache)} entries")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (features, label)
        """
        audio_file = self.audio_files[idx]
        label = self.labels[idx]
        
        # Check cache first
        if self.cache_features and audio_file in self.features_cache:
            features = self.features_cache[audio_file]
        else:
            # Extract features
            features = self.audio_processor.extract_features_from_file(audio_file)
            
            # Cache features
            if self.cache_features:
                self.features_cache[audio_file] = features
        
        # Apply augmentation if provided
        if self.augmentation and random.random() < 0.5:  # 50% chance
            features = self.augmentation.apply(features, audio_file)
        
        # Convert to tensors
        features_tensor = torch.from_numpy(features).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return features_tensor, label_tensor
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced training."""
        positive_count = sum(self.labels)
        negative_count = len(self.labels) - positive_count
        
        if positive_count == 0 or negative_count == 0:
            return torch.tensor([1.0, 1.0])
        
        total = len(self.labels)
        weight_negative = total / (2 * negative_count)
        weight_positive = total / (2 * positive_count)
        
        return torch.tensor([weight_negative, weight_positive])
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution."""
        positive_count = sum(self.labels)
        negative_count = len(self.labels) - positive_count
        
        return {
            'positive': positive_count,
            'negative': negative_count,
            'total': len(self.labels),
            'positive_ratio': positive_count / len(self.labels) if self.labels else 0
        }
    
    def split_dataset(self, val_ratio: float = 0.2, random_seed: int = 42) -> Tuple['WakeWordDataset', 'WakeWordDataset']:
        """
        Split dataset into train and validation sets.
        
        Args:
            val_ratio: Validation set ratio
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        if val_ratio <= 0 or val_ratio >= 1:
            raise ValueError("val_ratio must be between 0 and 1")
        
        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Get indices for each class
        positive_indices = [i for i, label in enumerate(self.labels) if label == 1]
        negative_indices = [i for i, label in enumerate(self.labels) if label == 0]
        
        # Shuffle indices
        random.shuffle(positive_indices)
        random.shuffle(negative_indices)
        
        # Split each class
        pos_val_size = int(len(positive_indices) * val_ratio)
        neg_val_size = int(len(negative_indices) * val_ratio)
        
        val_indices = positive_indices[:pos_val_size] + negative_indices[:neg_val_size]
        train_indices = positive_indices[pos_val_size:] + negative_indices[neg_val_size:]
        
        # Create subset datasets
        train_dataset = DatasetSubset(self, train_indices)
        val_dataset = DatasetSubset(self, val_indices)
        
        return train_dataset, val_dataset
    
    def save_features_cache(self):
        """Public method to save feature cache."""
        self._save_cache()
    
    def clear_cache(self):
        """Clear feature cache."""
        self.features_cache.clear()
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)
        logger.info("Feature cache cleared")


class DatasetSubset(Dataset):
    """Subset of a dataset for train/val splits."""
    
    def __init__(self, dataset: WakeWordDataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution for subset."""
        labels = [self.dataset.labels[i] for i in self.indices]
        positive_count = sum(labels)
        negative_count = len(labels) - positive_count
        
        return {
            'positive': positive_count,
            'negative': negative_count,
            'total': len(labels),
            'positive_ratio': positive_count / len(labels) if labels else 0
        }


class AudioDataset(Dataset):
    """
    Generic audio dataset for loading raw audio files.
    
    More flexible than WakeWordDataset, allows custom label mapping
    and preprocessing functions.
    """
    
    def __init__(
        self,
        audio_files: List[str],
        labels: List[int],
        audio_processor: Optional[AudioProcessor] = None,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None
    ):
        """
        Initialize audio dataset.
        
        Args:
            audio_files: List of audio file paths
            labels: List of corresponding labels
            audio_processor: Audio processor for feature extraction
            transform: Transform function for features
            target_transform: Transform function for labels
        """
        if len(audio_files) != len(labels):
            raise ValueError("Number of audio files must match number of labels")
        
        self.audio_files = audio_files
        self.labels = labels
        self.audio_processor = audio_processor or AudioProcessor()
        self.transform = transform
        self.target_transform = target_transform
        
        logger.info(f"AudioDataset initialized: {len(self)} samples")
    
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_file = self.audio_files[idx]
        label = self.labels[idx]
        
        # Extract features
        features = self.audio_processor.extract_features_from_file(audio_file)
        
        # Apply transforms
        if self.transform:
            features = self.transform(features)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        # Convert to tensors
        features_tensor = torch.from_numpy(features).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return features_tensor, label_tensor
    
    @classmethod
    def from_directory(
        cls,
        data_dir: str,
        label_map: Dict[str, int],
        audio_processor: Optional[AudioProcessor] = None
    ) -> 'AudioDataset':
        """
        Create dataset from directory structure.
        
        Args:
            data_dir: Root directory
            label_map: Mapping from subdirectory names to labels
            audio_processor: Audio processor
            
        Returns:
            AudioDataset instance
        """
        audio_files = []
        labels = []
        
        data_path = Path(data_dir)
        audio_extensions = ["*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg"]
        
        for subdir, label in label_map.items():
            subdir_path = data_path / subdir
            if subdir_path.exists():
                for ext in audio_extensions:
                    files = glob.glob(str(subdir_path / ext))
                    audio_files.extend(files)
                    labels.extend([label] * len(files))
        
        return cls(audio_files, labels, audio_processor)


def create_synthetic_dataset(
    num_positive: int = 1000,
    num_negative: int = 1000,
    audio_processor: Optional[AudioProcessor] = None,
    feature_dim: int = 26,
    noise_level: float = 0.1
) -> WakeWordDataset:
    """
    Create synthetic dataset for testing and debugging.
    
    Args:
        num_positive: Number of positive samples
        num_negative: Number of negative samples
        audio_processor: Audio processor (unused for synthetic data)
        feature_dim: Feature dimension
        noise_level: Noise level for synthetic features
        
    Returns:
        Synthetic dataset
    """
    
    class SyntheticDataset(Dataset):
        def __init__(self, num_pos, num_neg, feature_dim, noise_level):
            self.num_positive = num_pos
            self.num_negative = num_neg
            self.feature_dim = feature_dim
            self.noise_level = noise_level
            
            # Generate base patterns
            self.positive_pattern = np.random.randn(feature_dim)
            self.negative_pattern = np.random.randn(feature_dim)
            
        def __len__(self):
            return self.num_positive + self.num_negative
        
        def __getitem__(self, idx):
            if idx < self.num_positive:
                # Positive sample
                features = self.positive_pattern + np.random.randn(self.feature_dim) * self.noise_level
                label = 1
            else:
                # Negative sample
                features = self.negative_pattern + np.random.randn(self.feature_dim) * self.noise_level
                label = 0
            
            return torch.from_numpy(features).float(), torch.tensor(label, dtype=torch.long)
        
        def get_class_distribution(self):
            return {
                'positive': self.num_positive,
                'negative': self.num_negative,
                'total': self.num_positive + self.num_negative,
                'positive_ratio': self.num_positive / (self.num_positive + self.num_negative)
            }
    
    return SyntheticDataset(num_positive, num_negative, feature_dim, noise_level) 