"""
File utilities for wake word detection.

Provides file handling, validation, and directory management utilities.
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

import librosa

logger = logging.getLogger(__name__)


def ensure_dir(directory: str) -> str:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
        
    Returns:
        Absolute path to directory
    """
    path = Path(directory).absolute()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def list_audio_files(
    directory: str,
    extensions: Optional[List[str]] = None,
    recursive: bool = True
) -> List[str]:
    """
    List all audio files in directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to include
        recursive: Whether to search recursively
        
    Returns:
        List of audio file paths
    """
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']
    
    directory = Path(directory)
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []
    
    audio_files = []
    
    if recursive:
        for ext in extensions:
            pattern = f"**/*{ext}"
            files = directory.glob(pattern)
            audio_files.extend([str(f) for f in files])
    else:
        for ext in extensions:
            pattern = f"*{ext}"
            files = directory.glob(pattern)
            audio_files.extend([str(f) for f in files])
    
    # Sort for consistent ordering
    audio_files.sort()
    
    logger.info(f"Found {len(audio_files)} audio files in {directory}")
    return audio_files


def validate_audio_file(file_path: str) -> Dict[str, Any]:
    """
    Validate audio file and return metadata.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Dictionary with validation results and metadata
    """
    result = {
        'valid': False,
        'error': None,
        'duration': 0.0,
        'sample_rate': 0,
        'channels': 0,
        'file_size': 0
    }
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            result['error'] = 'File does not exist'
            return result
        
        # Get file size
        result['file_size'] = os.path.getsize(file_path)
        
        # Load and analyze audio
        y, sr = librosa.load(file_path, sr=None)
        
        result['duration'] = len(y) / sr
        result['sample_rate'] = sr
        result['channels'] = 1  # librosa loads as mono by default
        
        # Basic validation checks
        if result['duration'] < 0.1:
            result['error'] = 'Audio too short (< 0.1s)'
            return result
        
        if result['duration'] > 60.0:
            result['error'] = 'Audio too long (> 60s)'
            return result
        
        if sr < 8000:
            result['error'] = 'Sample rate too low (< 8kHz)'
            return result
        
        result['valid'] = True
        
    except Exception as e:
        result['error'] = f'Failed to load audio: {str(e)}'
    
    return result


def organize_dataset(
    source_dir: str,
    output_dir: str,
    positive_keywords: Optional[List[str]] = None,
    negative_keywords: Optional[List[str]] = None
) -> Dict[str, int]:
    """
    Organize audio files into positive/negative directories.
    
    Args:
        source_dir: Source directory with audio files
        output_dir: Output directory for organized dataset
        positive_keywords: Keywords to identify positive samples
        negative_keywords: Keywords to identify negative samples
        
    Returns:
        Dictionary with organization statistics
    """
    if positive_keywords is None:
        positive_keywords = ['wake', 'wakey', 'hey']
    
    if negative_keywords is None:
        negative_keywords = ['noise', 'background', 'other']
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    positive_dir = ensure_dir(output_path / 'positive')
    negative_dir = ensure_dir(output_path / 'negative')
    unknown_dir = ensure_dir(output_path / 'unknown')
    
    # Get all audio files
    audio_files = list_audio_files(source_dir)
    
    stats = {
        'positive': 0,
        'negative': 0,
        'unknown': 0,
        'errors': 0
    }
    
    for file_path in audio_files:
        try:
            file_name = Path(file_path).name.lower()
            
            # Determine category based on filename
            is_positive = any(keyword in file_name for keyword in positive_keywords)
            is_negative = any(keyword in file_name for keyword in negative_keywords)
            
            if is_positive and not is_negative:
                dest_dir = positive_dir
                category = 'positive'
            elif is_negative and not is_positive:
                dest_dir = negative_dir
                category = 'negative'
            else:
                dest_dir = unknown_dir
                category = 'unknown'
            
            # Copy file
            dest_path = Path(dest_dir) / Path(file_path).name
            import shutil
            shutil.copy2(file_path, dest_path)
            
            stats[category] += 1
            
        except Exception as e:
            logger.error(f"Error organizing {file_path}: {e}")
            stats['errors'] += 1
    
    logger.info(f"Dataset organized: {stats}")
    return stats


def clean_dataset(
    dataset_dir: str,
    min_duration: float = 0.5,
    max_duration: float = 10.0,
    min_sample_rate: int = 8000,
    remove_invalid: bool = False
) -> Dict[str, Any]:
    """
    Clean dataset by validating and optionally removing invalid files.
    
    Args:
        dataset_dir: Dataset directory
        min_duration: Minimum audio duration
        max_duration: Maximum audio duration
        min_sample_rate: Minimum sample rate
        remove_invalid: Whether to remove invalid files
        
    Returns:
        Cleaning results dictionary
    """
    dataset_path = Path(dataset_dir)
    
    # Find all audio files
    audio_files = []
    for subdir in ['positive', 'negative']:
        subdir_path = dataset_path / subdir
        if subdir_path.exists():
            audio_files.extend(list_audio_files(str(subdir_path)))
    
    results = {
        'total_files': len(audio_files),
        'valid_files': 0,
        'invalid_files': 0,
        'removed_files': 0,
        'issues': {
            'too_short': 0,
            'too_long': 0,
            'low_sample_rate': 0,
            'load_error': 0
        }
    }
    
    for file_path in audio_files:
        validation = validate_audio_file(file_path)
        
        is_valid = True
        issues = []
        
        if not validation['valid']:
            is_valid = False
            if 'too short' in validation.get('error', ''):
                results['issues']['too_short'] += 1
                issues.append('too_short')
            elif 'too long' in validation.get('error', ''):
                results['issues']['too_long'] += 1
                issues.append('too_long')
            elif 'sample rate' in validation.get('error', ''):
                results['issues']['low_sample_rate'] += 1
                issues.append('low_sample_rate')
            else:
                results['issues']['load_error'] += 1
                issues.append('load_error')
        else:
            # Additional validation
            if validation['duration'] < min_duration:
                is_valid = False
                results['issues']['too_short'] += 1
                issues.append('too_short')
            elif validation['duration'] > max_duration:
                is_valid = False
                results['issues']['too_long'] += 1
                issues.append('too_long')
            elif validation['sample_rate'] < min_sample_rate:
                is_valid = False
                results['issues']['low_sample_rate'] += 1
                issues.append('low_sample_rate')
        
        if is_valid:
            results['valid_files'] += 1
        else:
            results['invalid_files'] += 1
            
            if remove_invalid:
                try:
                    os.remove(file_path)
                    results['removed_files'] += 1
                    logger.info(f"Removed invalid file: {file_path} (issues: {issues})")
                except Exception as e:
                    logger.error(f"Failed to remove {file_path}: {e}")
    
    logger.info(f"Dataset cleaning completed: {results}")
    return results


def create_dataset_splits(
    dataset_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict[str, List[str]]:
    """
    Create train/validation/test splits from dataset.
    
    Args:
        dataset_dir: Dataset directory with positive/negative subdirs
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with file lists for each split
    """
    import random
    import numpy as np
    
    # Set random seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    dataset_path = Path(dataset_dir)
    splits = {'train': [], 'val': [], 'test': []}
    
    for class_name in ['positive', 'negative']:
        class_dir = dataset_path / class_name
        if not class_dir.exists():
            continue
        
        # Get all files for this class
        class_files = list_audio_files(str(class_dir), recursive=False)
        random.shuffle(class_files)
        
        # Calculate split sizes
        n_files = len(class_files)
        n_train = int(n_files * train_ratio)
        n_val = int(n_files * val_ratio)
        n_test = n_files - n_train - n_val
        
        # Split files
        train_files = class_files[:n_train]
        val_files = class_files[n_train:n_train + n_val]
        test_files = class_files[n_train + n_val:]
        
        # Add to splits
        splits['train'].extend(train_files)
        splits['val'].extend(val_files)
        splits['test'].extend(test_files)
        
        logger.info(f"{class_name}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    # Shuffle each split
    for split_name in splits:
        random.shuffle(splits[split_name])
    
    return splits


def get_dataset_info(dataset_dir: str) -> Dict[str, Any]:
    """
    Get comprehensive dataset information.
    
    Args:
        dataset_dir: Dataset directory
        
    Returns:
        Dataset information dictionary
    """
    dataset_path = Path(dataset_dir)
    info = {
        'total_files': 0,
        'classes': {},
        'total_duration': 0.0,
        'sample_rates': {},
        'file_sizes': [],
        'durations': []
    }
    
    for class_name in ['positive', 'negative']:
        class_dir = dataset_path / class_name
        if not class_dir.exists():
            continue
        
        class_files = list_audio_files(str(class_dir), recursive=False)
        class_info = {
            'count': len(class_files),
            'duration': 0.0,
            'avg_duration': 0.0
        }
        
        for file_path in class_files:
            validation = validate_audio_file(file_path)
            if validation['valid']:
                duration = validation['duration']
                sample_rate = validation['sample_rate']
                file_size = validation['file_size']
                
                class_info['duration'] += duration
                info['total_duration'] += duration
                info['durations'].append(duration)
                info['file_sizes'].append(file_size)
                
                # Track sample rates
                if sample_rate not in info['sample_rates']:
                    info['sample_rates'][sample_rate] = 0
                info['sample_rates'][sample_rate] += 1
        
        if class_info['count'] > 0:
            class_info['avg_duration'] = class_info['duration'] / class_info['count']
        
        info['classes'][class_name] = class_info
        info['total_files'] += class_info['count']
    
    # Calculate averages
    if info['durations']:
        info['avg_duration'] = np.mean(info['durations'])
        info['min_duration'] = np.min(info['durations'])
        info['max_duration'] = np.max(info['durations'])
        info['std_duration'] = np.std(info['durations'])
    
    if info['file_sizes']:
        info['avg_file_size'] = np.mean(info['file_sizes'])
        info['total_size_mb'] = sum(info['file_sizes']) / (1024 * 1024)
    
    return info 