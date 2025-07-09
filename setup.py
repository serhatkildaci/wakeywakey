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
    name="wakeyworddetection",
    version=get_version(),
    author="Serhat KILDACI",
    author_email="taserdeveloper@gmail.com",
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
    # Dependencies moved to pyproject.toml
    # Optional dependencies moved to pyproject.toml
    # Entry points moved to pyproject.toml
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

    
    # Additional metadata for PyPI
    download_url="https://github.com/serhatkildaci/wakeywakey/archive/v{}.tar.gz".format(get_version()),
    
    # Ensure compatibility
    setup_requires=[
        "wheel",
        "setuptools>=45.0.0",
    ],
    
    # Test configuration moved to pyproject.toml
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