#!/usr/bin/env python3
"""
Setup script for WakeyWakey - Lightweight Wake Word Detection.

A lightweight, accurate wake word detection package optimized for deployment
from microcontrollers to Raspberry Pi.
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    requirements_path = this_directory / filename
    if requirements_path.exists():
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Version management
def get_version():
    """Get version from __init__.py"""
    version_file = this_directory / "wakeywakey" / "__init__.py"
    if version_file.exists():
        with open(version_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"').strip("'")
    return "0.1.0"

# Check Python version
if sys.version_info < (3, 7):
    sys.exit("Python 3.7 or later is required.")

# Main package data
setup(
    name="wakeywakey",
    version=get_version(),
    author="Serhat Kildaci",
    author_email="serhat.kildaci@example.com",  # Replace with actual email
    description="Lightweight wake word detection optimized for microcontrollers to Raspberry Pi",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/serhatkildaci/wakeywakey",
    project_urls={
        "Bug Reports": "https://github.com/serhatkildaci/wakeywakey/issues",
        "Source": "https://github.com/serhatkildaci/wakeywakey",
        "Documentation": "https://github.com/serhatkildaci/wakeywakey/blob/main/docs/",
    },
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    keywords=[
        "wake-word", "voice-detection", "speech-recognition", "tinyml", 
        "microcontroller", "raspberry-pi", "pytorch", "neural-network",
        "mfcc", "audio-processing", "embedded-systems", "edge-computing"
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "torchaudio>=0.9.0",
        "numpy>=1.19.0",
        "librosa>=0.8.0",
        "sounddevice>=0.4.0",
        "scikit-learn>=0.24.0",
        "scipy>=1.7.0",
        "colorama>=0.4.4",
        "tqdm>=4.62.0",
        "PyYAML>=5.4.0",
        "pathlib2>=2.3.0; python_version<'3.4'",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.900",
            "pre-commit>=2.15.0",
            "twine>=3.4.0",
            "wheel>=0.37.0",
        ],
        "training": [
            "wandb>=0.12.0",
            "optuna>=2.10.0",
            "tensorboard>=2.7.0",
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
        ],
        "optimization": [
            "onnx>=1.10.0",
            "onnxruntime>=1.9.0",
            "openvino>=2021.4.0",
            "tensorflow-lite>=2.6.0",
        ],
        "microcontroller": [
            "micropython-lib>=1.9.0",
            "circuitpython>=7.0.0",
        ],
        "all": [
            "wandb>=0.12.0",
            "optuna>=2.10.0",
            "tensorboard>=2.7.0",
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
            "onnx>=1.10.0",
            "onnxruntime>=1.9.0",
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.900",
        ]
    },
    entry_points={
        "console_scripts": [
            "wakeywakey=wakeywakey.cli.console:cli_entry_point",
            "ww=wakeywakey.cli.console:cli_entry_point",  # Short alias
        ],
    },
    include_package_data=True,
    package_data={
        "wakeywakey": [
            "data/models/*.pth",
            "data/configs/*.yaml",
            "data/samples/*.wav",
        ],
    },
    zip_safe=False,  # Needed for proper resource access
    platforms=["any"],
    license="MIT",
    
    # Additional metadata for PyPI
    download_url="https://github.com/serhatkildaci/wakeywakey/archive/v{}.tar.gz".format(get_version()),
    
    # Ensure compatibility
    setup_requires=[
        "wheel",
        "setuptools>=45.0.0",
    ],
    
    # Test configuration
    test_suite="tests",
    tests_require=[
        "pytest>=6.0.0",
        "pytest-cov>=2.12.0",
    ],
)

# Post-installation message
print("""
🎉 WakeyWakey installation complete!

Quick start:
  1. Train a model:     wakeywakey train --data-dir ./data --model-type LightweightCNN
  2. Test detection:    wakeywakey test --model ./models/model.pth --input ./audio/
  3. Real-time detect:  wakeywakey detect --model ./models/model.pth
  4. List devices:      wakeywakey list-devices

Documentation: https://github.com/serhatkildaci/wakeywakey
Issues: https://github.com/serhatkildaci/wakeywakey/issues
""") 