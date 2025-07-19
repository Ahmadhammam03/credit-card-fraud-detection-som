"""
Data Preprocessing Module for Credit Card Fraud Detection
Author: Ahmad Hammam
Description: This module handles data loading, cleaning, and preprocessing
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreprocessor:
    """
    A class to handle data preprocessing for credit card fraud detection
    """
    
    def __init__(self):
        self.minmax_scaler = MinMaxScaler(feature_range=(0, 1))
        self.standard_scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load the credit card applications dataset
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            dataset = pd.read_csv(file_path)
            print(f"Dataset loaded successfully. Shape: {dataset.shape}")
            return dataset
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
            return None
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return None
    
    def explore_data(self, dataset: pd.DataFrame) -> None:
        """
        Perform basic data exploration
        
        Args:
            dataset (pd.DataFrame): Dataset to explore
        """
        print("=" * 50)
        print("DATA EXPLORATION SUMMARY")
        print("=" * 50)
        
        print(f"Dataset Shape: {dataset.shape}")
        print(f"Number of Features: {dataset.shape[1] - 1}")
        print(f"Number of Samples: {dataset.shape[0]}")
        
        print("\nDataset Info:")
        print(dataset.info())
        
        print("\nFirst 5 rows:")
        print(dataset.head())
        
        print("\nDataset Statistics:")
        print(dataset.describe())
        
        print("\nMissing Values:")
        print(dataset.isnull().sum())
        
        # Check class distribution
        if 'Class' in dataset.columns:
            print("\nClass Distribution:")
            print(dataset['Class'].value_counts())
            print(f"Class Balance Ratio: {dataset['Class'].value_counts().min() / dataset['Class'].value_counts().max():.3f}")
    
    def visualize_data_distribution(self, dataset: pd.DataFrame) -> None:
        """
        Create visualizations for data distribution
        
        Args:
            dataset (pd.DataFrame): Dataset to visualize
        """
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        
        # Class distribution
        if 'Class' in dataset.columns:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            dataset['Class'].value_counts().plot(kind='bar', color=['skyblue', 'lightcoral'])
            plt.title('Class Distribution')
            plt.xlabel('Class (0: Rejected, 1: Approved)')
            plt.ylabel('Count')
            plt.xticks(rotation=0)
            
            plt.subplot(1, 2, 2)
            plt.pie(dataset['Class'].value_counts(), labels=['Rejected', 'Approved'], 
                   autopct='%1.1f%%', colors=['lightcoral', 'skyblue'])
            plt.title('Class Distribution (%)')
            
            plt.tight_layout()
            plt.show()
        
        # Feature correlation heatmap
        plt.figure(figsize=(12, 10))
        correlation_matrix = dataset.corr()
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    def prepare_features_target(self, dataset: pd.DataFrame, 
                              target_column: str = 'Class') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features (X) and target (y) arrays
        
        Args:
            dataset (pd.DataFrame): Input dataset
            target_column (str): Name of the target column
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and target arrays
        """
        # Separate features and target
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        
        # Store feature names for later use
        self.feature_names = list(dataset.columns[:-1])
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        return X, y
    
    def scale_features_minmax(self, X: np.ndarray, 
                            fit_transform: bool = True) -> np.ndarray:
        """
        Scale features using MinMaxScaler (for SOM)
        
        Args:
            X (np.ndarray): Feature array
            fit_transform (bool): Whether to fit and transform or just transform
            
        Returns:
            np.ndarray: Scaled features
        """
        if fit_transform:
            X_scaled = self.minmax_scaler.fit_transform(X)
            print("MinMax scaling applied (fitted and transformed)")
        else:
            X_scaled = self.minmax_scaler.transform(X)
            print("MinMax scaling applied (transformed only)")
            
        return X_scaled
    
    def scale_features_standard(self, X: np.ndarray, 
                              fit_transform: bool = True) -> np.ndarray:
        """
        Scale features using StandardScaler (for ANN)
        
        Args:
            X (np.ndarray): Feature array
            fit_transform (bool): Whether to fit and transform or just transform
            
        Returns:
            np.ndarray: Scaled features
        """
        if fit_transform:
            X_scaled = self.standard_scaler.fit_transform(X)
            print("Standard scaling applied (fitted and transformed)")
        else:
            X_scaled = self.standard_scaler.transform(X)
            print("Standard scaling applied (transformed only)")
            
        return X_scaled
    
    def inverse_transform_minmax(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform MinMax scaled data
        
        Args:
            X_scaled (np.ndarray): Scaled feature array
            
        Returns:
            np.ndarray: Original scale features
        """
        return self.minmax_scaler.inverse_transform(X_scaled)
    
    def get_preprocessing_summary(self) -> dict:
        """
        Get summary of preprocessing steps
        
        Returns:
            dict: Preprocessing summary
        """
        summary = {
            'minmax_scaler_fitted': hasattr(self.minmax_scaler, 'scale_'),
            'standard_scaler_fitted': hasattr(self.standard_scaler, 'scale_'),
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names) if self.feature_names else None
        }
        
        return summary

def main():
    """
    Example usage of the DataPreprocessor class
    """
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data
    dataset = preprocessor.load_data('data/Credit_Card_Applications.csv')
    
    if dataset is not None:
        # Explore data
        preprocessor.explore_data(dataset)
        
        # Visualize data
        preprocessor.visualize_data_distribution(dataset)
        
        # Prepare features and target
        X, y = preprocessor.prepare_features_target(dataset)
        
        # Scale features for SOM
        X_minmax = preprocessor.scale_features_minmax(X)
        
        # Scale features for ANN
        X_standard = preprocessor.scale_features_standard(X)
        
        # Get preprocessing summary
        summary = preprocessor.get_preprocessing_summary()
        print("\nPreprocessing Summary:")
        print(summary)

if __name__ == "__main__":
    main()