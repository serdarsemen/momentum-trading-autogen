# setup.py
from setuptools import setup, find_packages

setup(
    name="momentum-trading-autogen",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "yfinance>=0.1.70",
        "pyautogen>=0.2.0",
        "jupyter>=1.0.0",
    ],
    entry_points={
        'console_scripts': [
            'momentum-analysis=examples.momentum_cli:main',
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="Momentum Trading Strategy Analysis with AutoGen",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/momentum-trading-autogen",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)