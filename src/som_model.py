"""
Self-Organizing Map (SOM) Model Module for Fraud Detection
Author: Ahmad Hammam
Description: Implementation of SOM for unsupervised anomaly detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show
from typing import Tuple, List, Optional
import seaborn as sns

class SOMFraudDetector:
    """
    Self-Organizing Map implementation for credit card fraud detection
    """
    
    def __init__(self, x: int = 10, y: int = 10, input_len: int = 15, 
                 sigma: float = 1.0, learning_rate: float = 0.5, 
                 random_seed: Optional[int] = None):
        """
        Initialize SOM parameters
        
        Args:
            x (int): Width of the SOM grid
            y (int): Height of the SOM grid
            input_len (int): Number of input features
            sigma (float): Spread of the neighborhood function
            learning_rate (float): Learning rate
            random_seed (int, optional): Random seed for reproducibility
        """
        self.x = x
        self.y = y
        self.input_len = input_len
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        
        # Initialize SOM
        self.som = MiniSom(x=x, y=y, input_len=input_len, 
                          sigma=sigma, learning_rate=learning_rate,
                          random_seed=random_seed)
        
        # Training metrics
        self.training_history = []
        self.is_trained = False
        
    def train_som(self, X: np.ndarray, num_iterations: int = 100, 
                  verbose: bool = True) -> None:
        """
        Train the Self-Organizing Map
        
        Args:
            X (np.ndarray): Training data (scaled)
            num_iterations (int): Number of training iterations
            verbose (bool): Whether to print training progress
        """
        if verbose:
            print(f"Training SOM with {num_iterations} iterations...")
            print(f"Grid size: {self.x}x{self.y}")
            print(f"Input dimensions: {self.input_len}")
        
        # Initialize weights randomly
        self.som.random_weights_init(X)
        
        # Train the SOM
        self.som.train_random(data=X, num_iteration=num_iterations, verbose=verbose)
        
        self.is_trained = True
        
        if verbose:
            print("SOM training completed!")
    
    def get_som_output(self, X: np.ndarray) -> np.ndarray:
        """
        Get SOM output for input data
        
        Args:
            X (np.ndarray): Input data
            
        Returns:
            np.ndarray: SOM output coordinates
        """
        if not self.is_trained:
            raise ValueError("SOM must be trained before getting output")
        
        output = []
        for x in X:
            winner = self.som.winner(x)
            output.append(winner)
        
        return np.array(output)
    
    def visualize_som(self, X: np.ndarray, y: np.ndarray, 
                     title: str = "SOM Fraud Detection Map") -> None:
        """
        Visualize the SOM with data points
        
        Args:
            X (np.ndarray): Input data (scaled)
            y (np.ndarray): Target labels
            title (str): Plot title
        """
        if not self.is_trained:
            raise ValueError("SOM must be trained before visualization")
        
        # Create the SOM visualization
        plt.figure(figsize=(12, 8))
        
        # Set bone colormap for background
        bone()
        
        # Plot the distance map
        pcolor(self.som.distance_map().T)
        colorbar()
        
        # Define markers and colors for different classes
        markers = ['o', 's']  # circle for rejected, square for approved
        colors = ['r', 'g']   # red for rejected, green for approved
        
        # Plot each data point
        for i, x in enumerate(X):
            w = self.som.winner(x)
            plot(w[0] + 0.5, w[1] + 0.5,
                 markers[y[i]],
                 markeredgecolor=colors[y[i]],
                 markerfacecolor='None',
                 markersize=10,
                 markeredgewidth=2)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('SOM Grid X-coordinate')
        plt.ylabel('SOM Grid Y-coordinate')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='None',
                   markeredgecolor='r', markersize=10, label='Rejected Applications'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='None',
                   markeredgecolor='g', markersize=10, label='Approved Applications')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        show()
    
    def find_fraud_clusters(self, X: np.ndarray, fraud_threshold: float = 0.8) -> Tuple[np.ndarray, List[Tuple]]:
        """
        Find potential fraud clusters in the SOM
        
        Args:
            X (np.ndarray): Input data (scaled)
            fraud_threshold (float): Threshold for identifying fraud clusters
            
        Returns:
            Tuple[np.ndarray, List[Tuple]]: Fraud samples and cluster coordinates
        """
        if not self.is_trained:
            raise ValueError("SOM must be trained before finding fraud clusters")
        
        # Get the distance map
        distance_map = self.som.distance_map()
        
        # Find high-distance regions (potential fraud areas)
        fraud_coords = []
        threshold = np.percentile(distance_map, fraud_threshold * 100)
        
        for i in range(self.x):
            for j in range(self.y):
                if distance_map[i, j] > threshold:
                    fraud_coords.append((i, j))
        
        # Get winner map
        mappings = self.som.win_map(X)
        
        # Extract fraud samples
        fraud_samples = []
        for coord in fraud_coords:
            if coord in mappings:
                fraud_samples.extend(mappings[coord])
        
        if fraud_samples:
            fraud_array = np.array(fraud_samples)
            print(f"Found {len(fraud_samples)} potential fraud cases in {len(fraud_coords)} clusters")
            return fraud_array, fraud_coords
        else:
            print("No fraud clusters found with current threshold")
            return np.array([]), fraud_coords
    
    def identify_specific_frauds(self, X: np.ndarray, 
                               fraud_coordinates: List[Tuple]) -> np.ndarray:
        """
        Identify frauds from specific SOM coordinates
        
        Args:
            X (np.ndarray): Input data (scaled)
            fraud_coordinates (List[Tuple]): Specific coordinates to check
            
        Returns:
            np.ndarray: Identified fraud samples
        """
        if not self.is_trained:
            raise ValueError("SOM must be trained before identifying frauds")
        
        mappings = self.som.win_map(X)
        fraud_samples = []
        
        for coord in fraud_coordinates:
            if coord in mappings:
                fraud_samples.extend(mappings[coord])
                print(f"Found {len(mappings[coord])} samples at coordinate {coord}")
        
        if fraud_samples:
            return np.array(fraud_samples)
        else:
            return np.array([])
    
    def analyze_som_topology(self) -> dict:
        """
        Analyze the SOM topology and provide insights
        
        Returns:
            dict: Analysis results
        """
        if not self.is_trained:
            raise ValueError("SOM must be trained before analysis")
        
        distance_map = self.som.distance_map()
        
        analysis = {
            'mean_distance': np.mean(distance_map),
            'std_distance': np.std(distance_map),
            'max_distance': np.max(distance_map),
            'min_distance': np.min(distance_map),
            'high_distance_regions': np.sum(distance_map > np.percentile(distance_map, 90))
        }
        
        return analysis
    
    def plot_distance_map(self, title: str = "SOM Distance Map") -> None:
        """
        Plot the SOM distance map
        
        Args:
            title (str): Plot title
        """
        if not self.is_trained:
            raise ValueError("SOM must be trained before plotting")
        
        plt.figure(figsize=(10, 8))
        
        distance_map = self.som.distance_map()
        
        plt.imshow(distance_map.T, cmap='bone', interpolation='nearest')
        plt.colorbar(label='Average Distance to Neighbors')
        plt.title(title)
        plt.xlabel('SOM Grid X-coordinate')
        plt.ylabel('SOM Grid Y-coordinate')
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained SOM model
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("SOM must be trained before saving")
        
        # Note: MiniSom doesn't have built-in save functionality
        # This would need to be implemented based on specific requirements
        print(f"Model saving functionality to be implemented for: {filepath}")
    
    def get_fraud_customer_ids(self, fraud_samples: np.ndarray, 
                             scaler, original_dataset: pd.DataFrame) -> List[int]:
        """
        Get customer IDs for identified fraud samples
        
        Args:
            fraud_samples (np.ndarray): Scaled fraud samples
            scaler: Scaler used for preprocessing
            original_dataset (pd.DataFrame): Original dataset with customer IDs
            
        Returns:
            List[int]: List of customer IDs identified as potential frauds
        """
        if len(fraud_samples) == 0:
            return []
        
        # Inverse transform to original scale
        fraud_original = scaler.inverse_transform(fraud_samples)
        
        # Extract customer IDs (assuming first column is CustomerID)
        customer_ids = []
        for fraud_sample in fraud_original:
            customer_id = int(fraud_sample[0])
            customer_ids.append(customer_id)
        
        return customer_ids

def main():
    """
    Example usage of the SOMFraudDetector class
    """
    # Example data loading and preprocessing
    print("SOM Fraud Detection Example")
    print("=" * 40)
    
    # This would typically be called after data preprocessing
    # from data_preprocessing import DataPreprocessor
    
    # preprocessor = DataPreprocessor()
    # dataset = preprocessor.load_data('data/Credit_Card_Applications.csv')
    # X, y = preprocessor.prepare_features_target(dataset)
    # X_scaled = preprocessor.scale_features_minmax(X)
    
    # Initialize SOM
    som_detector = SOMFraudDetector(
        x=10, y=10, input_len=15,
        sigma=1.0, learning_rate=0.5,
        random_seed=42
    )
    
    print("SOM initialized successfully!")
    print(f"Grid size: {som_detector.x}x{som_detector.y}")
    print(f"Input dimensions: {som_detector.input_len}")
    
    # Training would happen here:
    # som_detector.train_som(X_scaled, num_iterations=100)
    
    # Visualization would happen here:
    # som_detector.visualize_som(X_scaled, y)
    
    # Fraud detection would happen here:
    # fraud_samples, fraud_coords = som_detector.find_fraud_clusters(X_scaled)
    
    print("\nSOM setup completed. Ready for training with actual data.")

if __name__ == "__main__":
    main()