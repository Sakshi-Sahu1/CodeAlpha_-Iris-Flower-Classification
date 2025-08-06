# 🌸 Iris Flower Classification - Complete Machine Learning Tutorial

A comprehensive machine learning tutorial that demonstrates the complete workflow of building a classification model using the famous Iris dataset. This project covers everything from data exploration to model deployment.

## 🎯 Project Overview

This project implements multiple machine learning algorithms to classify iris flowers into three species:
- **Setosa** 🌺
- **Versicolor** 🌸
- **Virginica** 🌷

The tutorial demonstrates best practices in machine learning including data preprocessing, model comparison, hyperparameter tuning, and model persistence.

## 📊 Dataset

The Iris dataset contains 150 samples with 4 features:
- **Sepal Length** (cm)
- **Sepal Width** (cm)
- **Petal Length** (cm)
- **Petal Width** (cm)

**Dataset Characteristics:**
- 150 samples total
- 3 classes (50 samples each)
- 4 numerical features
- No missing values
- Perfectly balanced dataset

## 🚀 Features

### Machine Learning Pipeline
- ✅ **Data Loading & Exploration** - Comprehensive EDA with visualizations
- ✅ **Data Preprocessing** - Feature scaling and label encoding
- ✅ **Model Training** - Multiple algorithms comparison
- ✅ **Model Evaluation** - Cross-validation and performance metrics
- ✅ **Hyperparameter Tuning** - Grid search optimization
- ✅ **Feature Importance** - Analysis of most predictive features
- ✅ **Model Persistence** - Save/load trained models

### Algorithms Implemented
1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**
4. **Support Vector Machine (SVM)**
5. **K-Nearest Neighbors (KNN)**
6. **Naive Bayes**

### Visualizations
- Feature distribution plots
- Correlation heatmap
- Feature importance charts
- Model performance comparisons

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/iris-classification-tutorial.git
   cd iris-classification-tutorial
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

### Running the Complete Tutorial
```bash
python iris_classification_tutorial.py
```

### Using the Trained Model
```python
import joblib
import numpy as np

# Load saved model components
model = joblib.load('iris_classifier_random_forest_20241206.pkl')
scaler = joblib.load('iris_scaler_20241206.pkl')
encoder = joblib.load('iris_label_encoder_20241206.pkl')

# Make prediction on new data
new_flower = np.array([[5.8, 2.7, 5.1, 1.9]])
new_flower_scaled = scaler.transform(new_flower)
prediction = model.predict(new_flower_scaled)
species = encoder.inverse_transform(prediction)

print(f"Predicted species: {species[0]}")
```

### Interactive Jupyter Notebook
```bash
jupyter notebook iris_classification_notebook.ipynb
```

## 📈 Model Performance

| Algorithm | Test Accuracy | CV Score |
|-----------|---------------|----------|
| Random Forest | 96.67% | 95.83% ± 0.04 |
| SVM | 96.67% | 95.83% ± 0.04 |
| Logistic Regression | 96.67% | 95.00% ± 0.05 |
| KNN | 96.67% | 95.00% ± 0.05 |
| Decision Tree | 93.33% | 93.33% ± 0.05 |
| Naive Bayes | 96.67% | 95.00% ± 0.05 |

## 📁 Project Structure

```
iris-classification-tutorial/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── .gitignore                         # Git ignore rules
├── LICENSE                            # MIT License
├── iris_classification_tutorial.py    # Main tutorial script
├── iris_classification_notebook.ipynb # Jupyter notebook version
├── src/                               # Source code modules
│   ├── __init__.py
│   ├── data_loader.py                 # Data loading utilities
│   ├── preprocessor.py                # Data preprocessing
│   ├── models.py                      # Model definitions
│   ├── evaluator.py                   # Model evaluation
│   └── utils.py                       # Utility functions
├── data/                              # Data directory
│   ├── raw/                           # Raw datasets
│   └── processed/                     # Processed datasets
├── models/                            # Saved model files
├── results/                           # Results and outputs
│   ├── figures/                       # Generated plots
│   └── reports/                       # Analysis reports
├── tests/                             # Unit tests
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   └── test_models.py
├── docs/                              # Documentation
│   ├── api_reference.md
│   └── tutorial.md
├── scripts/                           # Utility scripts
│   ├── train_model.py
│   └── predict.py
├── .github/                           # GitHub workflows
│   └── workflows/
│       └── ci.yml
├── docker/                            # Docker configuration
│   ├── Dockerfile
│   └── docker-compose.yml
└── config/                            # Configuration files
    ├── config.yaml
    └── logging.conf
```

## 🤖 API Reference

### Core Functions

#### `load_and_explore_data()`
Loads the Iris dataset and performs exploratory data analysis.

**Returns:**
- `DataFrame`: Preprocessed iris dataset
- `dict`: Dataset statistics and information

#### `train_models(X_train, y_train, X_test, y_test)`
Trains multiple classification models and compares performance.

**Parameters:**
- `X_train`: Training features
- `y_train`: Training labels
- `X_test`: Test features
- `y_test`: Test labels

**Returns:**
- `dict`: Model performance results

#### `tune_hyperparameters(model, param_grid, X_train, y_train)`
Performs grid search hyperparameter tuning.

**Parameters:**
- `model`: Base model to tune
- `param_grid`: Parameter search space
- `X_train`: Training features
- `y_train`: Training labels

**Returns:**
- `object`: Best tuned model

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test
python -m pytest tests/test_models.py
```

## 🐳 Docker Support

### Build and run with Docker:
```bash
# Build the image
docker build -t iris-classifier .

# Run the container
docker run -it iris-classifier

# Run with Docker Compose
docker-compose up
```

## 📚 Learning Objectives

After completing this tutorial, you will understand:

### Machine Learning Fundamentals
- Data exploration and visualization techniques
- Feature engineering and preprocessing
- Train/validation/test split strategies
- Cross-validation for model selection
- Overfitting and underfitting concepts

### Classification Algorithms
- How different algorithms work
- When to use each algorithm
- Hyperparameter tuning strategies
- Model evaluation metrics

### Best Practices
- Code organization and modularity
- Version control with Git
- Documentation and reproducibility
- Model deployment preparation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style
- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions
- Write unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Iris Dataset**: Originally collected by Edgar Anderson and made famous by Ronald Fisher
- **Scikit-learn**: For providing excellent machine learning tools
- **Matplotlib & Seaborn**: For beautiful data visualizations
- **Pandas & NumPy**: For data manipulation and numerical computing

## 📞 Contact

**Sakshi Sahu ** - sakshi100sahu@gmail.com

Project Link: [https://github.com/Sakshi-Sahu1/CodeAlpha_-Iris-Flower-Classification](https://github.com/Sakshi-Sahu1/CodeAlpha_-Iris-Flower-Classification)

---

⭐ **If you found this project helpful, please give it a star!** ⭐

*Happy Learning! 🎓*
