# Credit Card Fraud Detection using Self-Organizing Maps (SOM) and Artificial Neural Networks

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Project Overview

This project implements a comprehensive fraud detection system for credit card applications using **Self-Organizing Maps (SOM)** for unsupervised anomaly detection, followed by an **Artificial Neural Network (ANN)** for supervised classification. The system effectively identifies fraudulent credit card applications with high accuracy.

## 🚀 Key Features

- **Unsupervised Learning**: Uses SOM to identify potential fraud patterns without labeled data
- **Supervised Learning**: Implements ANN for final fraud classification
- **Data Visualization**: Interactive maps showing fraud clusters
- **Scalable Architecture**: Modular design for easy extension and modification
- **Comprehensive Analysis**: Detailed exploratory data analysis and results interpretation

## 📊 Dataset

The project uses the **Credit Card Applications Dataset** containing:
- **690 applications** with 15 features each
- **Anonymized attributes** (A1-A15) for privacy protection
- **Binary classification**: Approved (1) vs Rejected (0) applications

### Dataset Features:
- Customer demographic information
- Financial attributes
- Credit history indicators
- Application-specific details

## 🛠️ Technologies Used

- **Python 3.8+**
- **TensorFlow/Keras** - Neural network implementation
- **Scikit-learn** - Data preprocessing and evaluation
- **MiniSOM** - Self-Organizing Maps implementation
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical plotting

## 📁 Project Structure

```
credit-card-fraud-detection-som/
│
├── data/
│   └── Credit_Card_Applications.csv
│
├── notebooks/
│   ├── som.ipynb                    # SOM implementation
│   └── mega_case_study.ipynb        # Complete analysis
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── som_model.py
│   ├── ann_model.py
│   └── visualization.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

## 🔧 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ahmadhammam03/credit-card-fraud-detection-som.git
   cd credit-card-fraud-detection-som
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv fraud_detection_env
   source fraud_detection_env/bin/activate  # On Windows: fraud_detection_env\Scripts\activate
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚦 Quick Start

### Running the SOM Analysis

```python
# Import required libraries
import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess data
dataset = pd.read_csv('data/Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature scaling
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

# Train SOM
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

# Identify fraud patterns
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,9)]), axis=0)
```

### Running the Complete Analysis

```bash
# Run the Jupyter notebook
jupyter notebook notebooks/mega_case_study.ipynb
```

## 📈 Results & Performance

### SOM Results:
- Successfully identified **14 potential fraud cases** from unlabeled data
- Created visual fraud detection map with clear anomaly clusters
- Fraud patterns concentrated in specific SOM regions

### ANN Results:
- **Training Accuracy**: ~99.5% after 10 epochs
- **Fraud Detection Rate**: High precision in identifying fraudulent applications
- **Model Architecture**: 2 hidden layers with optimized parameters

### Key Insights:
1. **Fraudulent applications** show distinct patterns in financial attributes
2. **Geographic clustering** of fraud cases in SOM visualization
3. **Effective combination** of unsupervised and supervised learning approaches

## 🔍 Methodology

### Phase 1: Unsupervised Learning (SOM)
1. **Data Preprocessing**: Feature scaling using MinMaxScaler
2. **SOM Training**: 10x10 grid with 100 iterations
3. **Anomaly Detection**: Identification of outlier clusters
4. **Visualization**: Color-coded fraud detection map

### Phase 2: Supervised Learning (ANN)
1. **Feature Engineering**: Using SOM results as additional features
2. **Neural Network Design**: 2-layer architecture with sigmoid activation
3. **Training**: Binary classification with Adam optimizer
4. **Evaluation**: Comprehensive performance metrics

## 📊 Visualizations

The project includes several key visualizations:

- **SOM Distance Map**: Showing fraud clusters and patterns
- **Training Progress**: Neural network convergence plots
- **Fraud Distribution**: Geographic and feature-based analysis
- **Performance Metrics**: ROC curves and confusion matrices

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Ahmad Hammam**
- GitHub: [@Ahmadhammam03](https://github.com/Ahmadhammam03)
- LinkedIn: [Ahmad Hammam](https://www.linkedin.com/in/ahmad-hammam-1561212b2)

## 🙏 Acknowledgments

- Original dataset source and research community
- MiniSOM library developers
- TensorFlow and Scikit-learn communities
- Open source machine learning community

## 📚 References

- [Self-Organizing Maps Theory](https://en.wikipedia.org/wiki/Self-organizing_map)
- [MiniSOM Documentation](https://github.com/JustGlowing/minisom)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Fraud Detection Research Papers](https://arxiv.org/search/?query=fraud+detection&searchtype=all&source=header)

---

⭐ **If you found this project helpful, please give it a star!** ⭐