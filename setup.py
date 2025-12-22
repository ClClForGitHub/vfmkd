#!/usr/bin/env python3
"""
VFMKD - Vision Foundation Model Knowledge Distillation
"""

from setuptools import setup, find_packages
import os

# Read README
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="vfmkd",
    version="0.1.0",
    author="VFMKD Team",
    author_email="vfmkd@example.com",
    description="Vision Foundation Model Knowledge Distillation Framework",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/example/vfmkd",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "sam2": [
            "segment-anything2>=1.0.0",
        ],
        "mamba": [
            "mamba-ssm>=1.0.0",
            "causal-conv1d>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vfmkd-train=tools.train:main",
            "vfmkd-eval=tools.eval:main",
            "vfmkd-download=tools.download_models:main",
        ],
    },
    include_package_data=True,
    package_data={
        "vfmkd": ["configs/*.yaml", "configs/**/*.yaml"],
    },
)