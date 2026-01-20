"""
Maniscope: Efficient Neural Reranking via Geodesic Distances
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="maniscope",
    version="1.1.0",
    author="Wen G. Gong, Albert Gong",
    author_email="wen.gong.research@gmail.com",
    description="Efficient neural reranking via geodesic distances on k-NN manifolds",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/digital-duck/maniscope",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Indexing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "networkx>=2.6.0",
        "scikit-learn>=1.0.0",
        "sentence-transformers>=2.2.0",
        "torch>=1.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    keywords="information-retrieval reranking manifold-learning geodesic-distance neural-search rag",
)
