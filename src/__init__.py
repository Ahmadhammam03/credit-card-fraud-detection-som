"""
Credit Card Fraud Detection Package
Author: Ahmad Hammam
Version: 1.0.0

This package provides tools for credit card fraud detection using:
- Self-Organizing Maps (SOM) for unsupervised anomaly detection
- Artificial Neural Networks (ANN) for supervised classification
"""

__version__ = "1.0.0"
__author__ = "Ahmad Hammam"
__email__ = "ahmadhammam501@gmail.com"
__description__ = "Credit Card Fraud Detection using SOM and ANN"

from .data_preprocessing import DataPreprocessor
from .som_model import SOMFraudDetector
from .ann_model import ANNFraudClassifier

__all__ = [
    'DataPreprocessor',
    'SOMFraudDetector', 
    'ANNFraudClassifier'
]