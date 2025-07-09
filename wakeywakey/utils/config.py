"""
Configuration utilities for wake word detection.

Provides configuration loading, saving, and merging utilities
with support for YAML and JSON formats.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file (JSON or YAML)
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                if not YAML_AVAILABLE:
                    raise ImportError("PyYAML not available for YAML config files")
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                # Try to detect format
                content = f.read()
                f.seek(0)
                
                try:
                    config = json.load(f)
                except json.JSONDecodeError:
                    if YAML_AVAILABLE:
                        f.seek(0)
                        config = yaml.safe_load(f)
                    else:
                        raise ValueError("Could not parse config file format")
        
        logger.info(f"Configuration loaded from {config_path}")
        return config if config is not None else {}
        
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise


def save_config(config: Dict[str, Any], config_path: str, format: str = 'auto') -> bool:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
        format: Output format ('json', 'yaml', 'auto')
        
    Returns:
        True if saved successfully, False otherwise
    """
    config_path = Path(config_path)
    
    # Determine format
    if format == 'auto':
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            format = 'yaml'
        else:
            format = 'json'
    
    try:
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            if format == 'yaml':
                if not YAML_AVAILABLE:
                    logger.warning("PyYAML not available, saving as JSON")
                    json.dump(config, f, indent=2, ensure_ascii=False)
                else:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            else:
                json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Configuration saved to {config_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save configuration to {config_path}: {e}")
        return False


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    
    Later configs override earlier ones. Nested dictionaries are merged recursively.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    if not configs:
        return {}
    
    result = {}
    
    for config in configs:
        if not isinstance(config, dict):
            continue
        
        result = _deep_merge(result, config)
    
    return result


def _deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for wake word detection.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'audio': {
            'sample_rate': 16000,
            'n_mfcc': 13,
            'n_fft': 512,
            'hop_length': 160,
            'n_mels': 40,
            'window_size': 1.0,
            'stride': 0.5,
            'normalize': True,
            'add_deltas': True
        },
        'model': {
            'type': 'LightweightCNN',
            'input_size': 26,
            'hidden_dim': 64,
            'dropout': 0.2,
            'device': 'auto'
        },
        'training': {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'scheduler': 'cosine',
            'early_stopping_patience': 10,
            'grad_clip': 1.0,
            'weight_decay': 1e-5
        },
        'detection': {
            'threshold': 0.7,
            'sensitivity': 'medium',
            'smoothing_window': 3,
            'debounce_time': 2.0
        },
        'augmentation': {
            'enabled': True,
            'time_stretch': {
                'enabled': True,
                'rate_range': [0.8, 1.2],
                'probability': 0.3
            },
            'pitch_shift': {
                'enabled': True,
                'n_steps_range': [-2, 2],
                'probability': 0.3
            },
            'noise_injection': {
                'enabled': True,
                'snr_range': [10, 30],
                'probability': 0.4
            }
        },
        'data': {
            'cache_features': True,
            'max_samples_per_class': None,
            'validation_split': 0.2,
            'random_seed': 42
        }
    }


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration and return validation results.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Validation results with errors and warnings
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Get default config for reference
    default_config = get_default_config()
    
    # Validate audio config
    if 'audio' in config:
        audio_config = config['audio']
        
        # Check sample rate
        if 'sample_rate' in audio_config:
            sr = audio_config['sample_rate']
            if not isinstance(sr, int) or sr <= 0:
                results['errors'].append("audio.sample_rate must be a positive integer")
                results['valid'] = False
            elif sr < 8000:
                results['warnings'].append("audio.sample_rate below 8kHz may affect quality")
        
        # Check MFCC parameters
        if 'n_mfcc' in audio_config:
            n_mfcc = audio_config['n_mfcc']
            if not isinstance(n_mfcc, int) or n_mfcc <= 0:
                results['errors'].append("audio.n_mfcc must be a positive integer")
                results['valid'] = False
    
    # Validate model config
    if 'model' in config:
        model_config = config['model']
        
        # Check model type
        if 'type' in model_config:
            model_type = model_config['type']
            valid_types = ['LightweightCNN', 'CompactRNN', 'HybridCRNN', 'MobileWakeWord', 'AttentionWakeWord']
            if model_type not in valid_types:
                results['errors'].append(f"model.type must be one of {valid_types}")
                results['valid'] = False
        
        # Check input size
        if 'input_size' in model_config:
            input_size = model_config['input_size']
            if not isinstance(input_size, int) or input_size <= 0:
                results['errors'].append("model.input_size must be a positive integer")
                results['valid'] = False
    
    # Validate training config
    if 'training' in config:
        training_config = config['training']
        
        # Check epochs
        if 'epochs' in training_config:
            epochs = training_config['epochs']
            if not isinstance(epochs, int) or epochs <= 0:
                results['errors'].append("training.epochs must be a positive integer")
                results['valid'] = False
        
        # Check batch size
        if 'batch_size' in training_config:
            batch_size = training_config['batch_size']
            if not isinstance(batch_size, int) or batch_size <= 0:
                results['errors'].append("training.batch_size must be a positive integer")
                results['valid'] = False
        
        # Check learning rate
        if 'learning_rate' in training_config:
            lr = training_config['learning_rate']
            if not isinstance(lr, (int, float)) or lr <= 0:
                results['errors'].append("training.learning_rate must be a positive number")
                results['valid'] = False
    
    # Validate detection config
    if 'detection' in config:
        detection_config = config['detection']
        
        # Check threshold
        if 'threshold' in detection_config:
            threshold = detection_config['threshold']
            if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
                results['errors'].append("detection.threshold must be between 0 and 1")
                results['valid'] = False
        
        # Check sensitivity
        if 'sensitivity' in detection_config:
            sensitivity = detection_config['sensitivity']
            valid_sensitivities = ['low', 'medium', 'high', 'very_high']
            if sensitivity not in valid_sensitivities:
                results['errors'].append(f"detection.sensitivity must be one of {valid_sensitivities}")
                results['valid'] = False
    
    return results


def create_config_template(output_path: str, format: str = 'yaml') -> bool:
    """
    Create configuration template file.
    
    Args:
        output_path: Path to save template
        format: Output format ('yaml' or 'json')
        
    Returns:
        True if created successfully, False otherwise
    """
    template_config = get_default_config()
    
    # Add comments to the template
    if format == 'yaml':
        template_config['_comments'] = {
            'audio': 'Audio processing configuration',
            'model': 'Neural network model configuration',
            'training': 'Training hyperparameters',
            'detection': 'Real-time detection settings',
            'augmentation': 'Data augmentation configuration',
            'data': 'Dataset configuration'
        }
    
    return save_config(template_config, output_path, format)


def load_config_with_defaults(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration with fallback to defaults.
    
    Args:
        config_path: Path to configuration file (optional)
        
    Returns:
        Merged configuration dictionary
    """
    default_config = get_default_config()
    
    if config_path and os.path.exists(config_path):
        try:
            user_config = load_config(config_path)
            return merge_configs(default_config, user_config)
        except Exception as e:
            logger.warning(f"Failed to load user config, using defaults: {e}")
    
    return default_config


def update_config(
    config_path: str,
    updates: Dict[str, Any],
    create_backup: bool = True
) -> bool:
    """
    Update existing configuration file.
    
    Args:
        config_path: Path to configuration file
        updates: Updates to apply
        create_backup: Whether to create backup before updating
        
    Returns:
        True if updated successfully, False otherwise
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return False
    
    try:
        # Create backup if requested
        if create_backup:
            backup_path = config_path.with_suffix(config_path.suffix + '.bak')
            import shutil
            shutil.copy2(config_path, backup_path)
            logger.info(f"Backup created: {backup_path}")
        
        # Load existing config
        existing_config = load_config(str(config_path))
        
        # Merge with updates
        updated_config = merge_configs(existing_config, updates)
        
        # Save updated config
        return save_config(updated_config, str(config_path))
        
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        return False 