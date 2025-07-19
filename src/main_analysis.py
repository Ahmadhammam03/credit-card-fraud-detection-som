"""
Main Analysis Script for Credit Card Fraud Detection
Author: Ahmad Hammam
Description: Complete pipeline combining SOM and ANN for fraud detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data_preprocessing import DataPreprocessor
from src.som_model import SOMFraudDetector
from src.ann_model import ANNFraudClassifier
import warnings
warnings.filterwarnings('ignore')

def main():
    """
    Main analysis pipeline for credit card fraud detection
    """
    print("=" * 60)
    print("CREDIT CARD FRAUD DETECTION ANALYSIS")
    print("=" * 60)
    print("Author: Ahmad Hammam")
    print("Method: Self-Organizing Maps + Artificial Neural Networks")
    print("=" * 60)
    
    # Step 1: Data Loading and Preprocessing
    print("\n🔄 STEP 1: DATA LOADING AND PREPROCESSING")
    print("-" * 50)
    
    preprocessor = DataPreprocessor()
    
    # Load the dataset
    dataset = preprocessor.load_data('data/Credit_Card_Applications.csv')
    
    if dataset is None:
        print("❌ Error: Could not load dataset. Please check the file path.")
        return
    
    # Explore the data
    preprocessor.explore_data(dataset)
    
    # Visualize data distribution
    preprocessor.visualize_data_distribution(dataset)
    
    # Prepare features and target
    X, y = preprocessor.prepare_features_target(dataset)
    
    # Scale features for SOM (MinMax scaling)
    X_som = preprocessor.scale_features_minmax(X, fit_transform=True)
    
    print("✅ Data preprocessing completed successfully!")
    
    # Step 2: SOM Analysis for Unsupervised Anomaly Detection
    print("\n🧠 STEP 2: SELF-ORGANIZING MAP ANALYSIS")
    print("-" * 50)
    
    # Initialize SOM
    som_detector = SOMFraudDetector(
        x=10, y=10, input_len=15,
        sigma=1.0, learning_rate=0.5,
        random_seed=42
    )
    
    # Train SOM
    som_detector.train_som(X_som, num_iterations=100, verbose=True)
    
    # Visualize SOM results
    som_detector.visualize_som(X_som, y, title="SOM Fraud Detection Map")
    
    # Analyze SOM topology
    topology_analysis = som_detector.analyze_som_topology()
    print("\n📊 SOM Topology Analysis:")
    for key, value in topology_analysis.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Find fraud clusters automatically
    fraud_samples_auto, fraud_coords_auto = som_detector.find_fraud_clusters(X_som, fraud_threshold=0.8)
    
    print(f"\n🔍 Automatic fraud detection found {len(fraud_samples_auto)} potential fraud cases")
    
    # Manual fraud identification (based on visual inspection)
    # These coordinates should be identified from the SOM visualization
    manual_fraud_coords = [(8, 1), (6, 9)]  # Example coordinates from your analysis
    fraud_samples_manual = som_detector.identify_specific_frauds(X_som, manual_fraud_coords)
    
    print(f"🎯 Manual fraud detection found {len(fraud_samples_manual)} potential fraud cases")
    
    # Get customer IDs for manual detection
    if len(fraud_samples_manual) > 0:
        fraud_customer_ids = som_detector.get_fraud_customer_ids(
            fraud_samples_manual, preprocessor.minmax_scaler, dataset
        )
        
        print("\n🆔 Fraud Customer IDs:")
        for customer_id in fraud_customer_ids:
            print(f"  Customer ID: {customer_id}")
    
    print("✅ SOM analysis completed successfully!")
    
    # Step 3: Prepare data for supervised learning
    print("\n⚡ STEP 3: PREPARING DATA FOR SUPERVISED LEARNING")
    print("-" * 50)
    
    # Create fraud labels based on SOM results
    is_fraud = np.zeros(len(dataset))
    
    if len(fraud_samples_manual) > 0:
        # Find indices of fraud samples in original dataset
        fraud_original = preprocessor.inverse_transform_minmax(fraud_samples_manual)
        
        for fraud_sample in fraud_original:
            customer_id = int(fraud_sample[0])
            # Find the index of this customer in the original dataset
            customer_index = dataset[dataset.iloc[:, 0] == customer_id].index
            if len(customer_index) > 0:
                is_fraud[customer_index[0]] = 1
    
    print(f"📈 Created fraud labels: {np.sum(is_fraud)} fraud cases out of {len(is_fraud)} total")
    
    # Scale features for ANN (Standard scaling)
    customers = dataset.iloc[:, 1:].values  # Exclude customer ID
    X_ann = preprocessor.scale_features_standard(customers, fit_transform=True)
    
    print("✅ Data preparation for supervised learning completed!")
    
    # Step 4: ANN Training and Evaluation
    print("\n🤖 STEP 4: ARTIFICIAL NEURAL NETWORK TRAINING")
    print("-" * 50)
    
    # Initialize ANN
    ann_classifier = ANNFraudClassifier(input_dim=X_ann.shape[1], random_seed=42)
    
    # Build model
    ann_classifier.build_model(
        hidden_units=[2],
        activation='relu',
        output_activation='sigmoid',
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model
    ann_classifier.train_model(
        X_ann, is_fraud,
        batch_size=1,
        epochs=10,
        validation_split=0.2,
        verbose=1
    )
    
    # Plot training history
    ann_classifier.plot_training_history()
    
    print("✅ ANN training completed successfully!")
    
    # Step 5: Model Evaluation and Results
    print("\n📊 STEP 5: MODEL EVALUATION AND RESULTS")
    print("-" * 50)
    
    # Make predictions on all data
    y_pred_proba, y_pred = ann_classifier.predict(X_ann)
    
    # Evaluate model
    evaluation_results = ann_classifier.evaluate_model(X_ann, is_fraud)
    
    # Plot confusion matrix
    ann_classifier.plot_confusion_matrix(is_fraud, y_pred.flatten())
    
    # Plot ROC curve
    ann_classifier.plot_roc_curve(is_fraud, y_pred_proba.flatten())
    
    # Rank customers by fraud risk
    customer_rankings = ann_classifier.rank_customers_by_fraud_risk(
        X_ann, dataset.iloc[:, 0].values
    )
    
    print("\n🏆 TOP 10 CUSTOMERS BY FRAUD RISK:")
    print(customer_rankings.head(10).to_string(index=False))
    
    print("\n📈 FINAL RESULTS SUMMARY:")
    print("-" * 30)
    print(f"  • Total customers analyzed: {len(dataset)}")
    print(f"  • Fraud cases identified by SOM: {len(fraud_samples_manual)}")
    print(f"  • ANN training accuracy: {evaluation_results['test_accuracy']:.4f}")
    if evaluation_results['auc_score'] is not None:
        print(f"  • ANN AUC score: {evaluation_results['auc_score']:.4f}")
    print(f"  • High-risk customers (>50% fraud probability): {np.sum(y_pred_proba > 0.5)}")
    
    print("\n✅ Analysis completed successfully!")
    print("\n💡 INSIGHTS:")
    print("  • SOM effectively identified anomalous patterns in credit card applications")
    print("  • ANN successfully learned to classify fraud based on SOM-identified patterns")
    print("  • The combined approach provides both unsupervised discovery and supervised classification")
    print("  • Visual SOM maps help understand the fraud detection decision process")
    
    return {
        'dataset': dataset,
        'som_detector': som_detector,
        'ann_classifier': ann_classifier,
        'fraud_customer_ids': fraud_customer_ids if len(fraud_samples_manual) > 0 else [],
        'customer_rankings': customer_rankings,
        'evaluation_results': evaluation_results
    }

if __name__ == "__main__":
    results = main()