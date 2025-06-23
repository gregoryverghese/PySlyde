"""Setup configuration for PySlyde package."""

from setuptools import setup, find_packages

setup(
    name="PySlyde",
    version="0.1.0",
    author="Gregory Verghese",
    author_email="gregory.verghese@gmail.com",
    description="A Python package for preprocessing pathology whole slide images.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/gregoryverghese/PySlide",
    packages=find_packages(),
    install_requires=[
        "einops",
        "h5py",
        "huggingface_hub",
        "lmdb",
        "matplotlib",
        "numpy",
        "opencv-python",
        "openslide-python",
        "pandas",
        "rocksdb",
        "scipy",
        "seaborn",
        "scikit-image",
        "tensorflow",
        "timm",
        "torch",
        "webdataset"
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
            "pre-commit"
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    keywords="pathology, whole-slide-image, medical-imaging, computer-vision",
    project_urls={
        "Bug Reports": "https://github.com/gregoryverghese/PySlide/issues",
        "Source": "https://github.com/gregoryverghese/PySlide",
        "Documentation": "https://github.com/gregoryverghese/PySlide#readme",
    },
)

