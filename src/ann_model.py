"""
Artificial Neural Network (ANN) Model Module for Fraud Detection
Author: Ahmad Hammam
Description: Implementation of ANN for supervised fraud classification
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, List

class ANNFraudClassifier:
    """
    Artificial Neural Network implementation for credit card fraud classification
    """
    
    def __init__(self, input_dim: int, random_seed: Optional[int] = 42):
        """
        Initialize ANN parameters
        
        Args:
            input_dim (int): Number of input features
            random_seed (int, optional): Random seed for reproducibility
        """
        self.input_dim = input_dim
        self.random_seed = random_seed
        
        # Set random seeds for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
            tf.random.set_seed(random_seed)
        
        self.model = None
        self.history = None
        self.is_trained = False
        
    def build_model(self, hidden_units: List[int] = [2], 
                   activation: str = 'relu',
                   output_activation: str = 'sigmoid',
                   optimizer: str = 'adam',
                   loss: str = 'binary_crossentropy',
                   metrics: List[str] = ['accuracy']) -> None:
        """
        Build the neural network architecture
        
        Args:
            hidden_units (List[int]): Number of units in each hidden layer
            activation (str): Activation function for hidden layers
            output_activation (str): Activation function for output layer
            optimizer (str): Optimizer for training
            loss (str): Loss function
            metrics (List[str]): Metrics to track during training
        """
        print("Building ANN model...")
        
        # Initialize sequential model
        self.model = keras.Sequential()
        
        # Add hidden layers
        for i, units in enumerate(hidden_units):
            if i == 0:
                # First hidden layer with input shape
                self.model.add(keras.layers.Dense(
                    units=units, 
                    activation=activation,
                    input_shape=(self.input_dim,),
                    name=f'hidden_layer_{i+1}'
                ))
            else:
                # Additional hidden layers
                self.model.add(keras.layers.Dense(
                    units=units, 
                    activation=activation,
                    name=f'hidden_layer_{i+1}'
                ))
        
        # Add output layer
        self.model.add(keras.layers.Dense(
            units=1, 
            activation=output_activation,
            name='output_layer'
        ))
        
        # Compile the model
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        print("Model architecture:")
        self.model.summary()
        
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   batch_size: int = 1, epochs: int = 10,
                   validation_split: float = 0.2,
                   verbose: int = 1) -> None:
        """
        Train the neural network
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
            validation_split (float): Fraction of data to use for validation
            verbose (int): Verbosity level
        """
        if self.model is None:
            raise ValueError("Model must be built before training")
        
        print(f"Training ANN for {epochs} epochs...")
        print(f"Training data shape: {X_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            verbose=verbose
        )
        
        self.is_trained = True
        print("ANN training completed!")
        
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data
        
        Args:
            X (np.ndarray): Input features
            threshold (float): Classification threshold
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Probabilities and binary predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get prediction probabilities
        y_pred_proba = self.model.predict(X)
        
        # Convert probabilities to binary predictions
        y_pred = (y_pred_proba > threshold).astype(int)
        
        return y_pred_proba, y_pred
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate the model performance
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred_proba, y_pred = self.predict(X_test)
        
        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Additional metrics
        try:
            auc_score = roc_auc_score(y_test, y_pred_proba)
        except:
            auc_score = None
        
        evaluation_results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'auc_score': auc_score,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        # Print classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return evaluation_results
    
    def plot_training_history(self) -> None:
        """
        Plot training history (loss and accuracy)
        """
        if self.history is None:
            raise ValueError("Model must be trained before plotting history")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training & validation loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot training & validation accuracy
        ax2.plot(self.history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in self.history.history:
            ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Plot confusion matrix
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Rejected', 'Approved'],
                   yticklabels=['Rejected', 'Approved'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()
        
        # Print confusion matrix statistics
        tn, fp, fn, tp = cm.ravel()
        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Positives: {tp}")
        
        # Calculate additional metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1_score:.4f}")
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> None:
        """
        Plot ROC curve
        
        Args:
            y_true (np.ndarray): True labels
            y_pred_proba (np.ndarray): Predicted probabilities
        """
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            auc_score = roc_auc_score(y_true, y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC Curve (AUC = {auc_score:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                    label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not plot ROC curve: {e}")
    
    def predict_fraud_probability(self, customers: np.ndarray) -> np.ndarray:
        """
        Predict fraud probability for customers
        
        Args:
            customers (np.ndarray): Customer feature data
            
        Returns:
            np.ndarray: Fraud probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        probabilities = self.model.predict(customers)
        return probabilities.flatten()
    
    def rank_customers_by_fraud_risk(self, customers: np.ndarray, 
                                   customer_ids: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Rank customers by fraud risk
        
        Args:
            customers (np.ndarray): Customer feature data
            customer_ids (np.ndarray, optional): Customer IDs
            
        Returns:
            pd.DataFrame: Ranked customers with fraud probabilities
        """
        probabilities = self.predict_fraud_probability(customers)
        
        if customer_ids is None:
            customer_ids = np.arange(len(customers))
        
        # Create DataFrame with results
        results_df = pd.DataFrame({
            'CustomerID': customer_ids.flatten(),
            'FraudProbability': probabilities
        })
        
        # Sort by fraud probability (descending)
        results_df = results_df.sort_values('FraudProbability', ascending=False)
        results_df = results_df.reset_index(drop=True)
        
        return results_df
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        self.model.save(filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a pre-trained model
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        self.is_trained = True
        print(f"Model loaded from: {filepath}")

def main():
    """
    Example usage of the ANNFraudClassifier class
    """
    print("ANN Fraud Classification Example")
    print("=" * 40)
    
    # Initialize ANN classifier
    ann_classifier = ANNFraudClassifier(input_dim=15, random_seed=42)
    
    # Build model architecture
    ann_classifier.build_model(
        hidden_units=[2],  # Single hidden layer with 2 units
        activation='relu',
        output_activation='sigmoid',
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("ANN model built successfully!")
    
    # Training would happen here with actual data:
    # ann_classifier.train_model(X_train, y_train, epochs=10)
    
    # Evaluation would happen here:
    # results = ann_classifier.evaluate_model(X_test, y_test)
    
    # Plotting would happen here:
    # ann_classifier.plot_training_history()
    
    print("\nANN setup completed. Ready for training with actual data.")

if __name__ == "__main__":
    main()