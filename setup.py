"""
NextGen-UIA: Medical Ultrasound Image Analysis with Vision-Language Models

Setup script for package installation.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="nextgen-uia",
    version="0.1.0",
    description="Adapting Vision-Language Foundation Models for Medical Ultrasound Image Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/your-repo/NextGen-UIA",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.12",
    install_requires=[
        "torch>=2.9.0",
        "torchvision>=0.24.0",
        "open-clip-torch>=3.2.0",
        "transformers>=4.57.1",
        "numpy>=2.3.4",
        "pandas>=2.3.3",
        "pillow>=12.0.0",
        "scikit-learn>=1.7.2",
        "torchmetrics>=1.8.2",
        "monai>=1.5.1",
        "tensorboard>=2.20.0",
        "tqdm>=4.67.1",
        "matplotlib>=3.10.7",
        "optuna>=4.5.0",
        "nvitop>=1.6.0",
    ],
    extras_require={
        "dev": [
            "ruff>=0.14.1",
            "pytest>=7.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="medical-imaging ultrasound clip vision-language deep-learning",
    project_urls={
        "Source": "https://github.com/jinggqu/NextGen-UIA",
        "Bug Reports": "https://github.com/jinggqu/NextGen-UIA/issues",
    },
)
