# Iris Flower Classification - Complete Machine Learning Tutorial.  .
# This tutorial covers all aspects of building a classification model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, precision_recall_fscore_support)
import warnings
warnings.filterwarnings('ignore')

print("🌸 Iris Flower Classification Tutorial 🌸")
print("=" * 50)

# ============================================================================
# STEP 1: DATA LOADING AND EXPLORATION
# ============================================================================

print("\n📊 STEP 1: Loading and Exploring the Dataset")
print("-" * 40)

# Method 1: Load from sklearn (built-in dataset)
iris_sklearn = datasets.load_iris()
X_sklearn = iris_sklearn.data
y_sklearn = iris_sklearn.target
feature_names = iris_sklearn.feature_names
target_names = iris_sklearn.target_names

# Create DataFrame for better visualization
df = pd.DataFrame(X_sklearn, columns=feature_names)
df['species'] = [target_names[i] for i in y_sklearn]

print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")
print(f"Features: {feature_names}")
print(f"Target classes: {target_names}")

print("\n📋 First 5 rows of the dataset:")
print(df.head())

print("\n📈 Dataset Statistics:")
print(df.describe())

print("\n🎯 Class Distribution:")
class_counts = df['species'].value_counts()
print(class_counts)
print(f"Dataset is balanced: {class_counts.std() < 1}")

# Check for missing values
print(f"\n❓ Missing values: {df.isnull().sum().sum()}")

# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n🔍 STEP 2: Exploratory Data Analysis")
print("-" * 40)

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

# Create comprehensive visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Iris Dataset - Exploratory Data Analysis', fontsize=16, fontweight='bold')

# 1. Pairplot equivalent - Feature distributions
for i, feature in enumerate(feature_names):
    ax = axes[0, i] if i < 3 else axes[1, i-3]
    for species in target_names:
        species_data = df[df['species'] == species][feature]
        ax.hist(species_data, alpha=0.7, label=species, bins=15)
    ax.set_title(f'{feature} Distribution')
    ax.set_xlabel(feature)
    ax.set_ylabel('Frequency')
    ax.legend()

# Correlation heatmap
ax = axes[1, 2]
correlation_matrix = df[feature_names].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
ax.set_title('Feature Correlation Matrix')

plt.tight_layout()
plt.show()

# Statistical analysis of features by species
print("\n📊 Feature Statistics by Species:")
for species in target_names:
    print(f"\n{species.upper()}:")
    species_stats = df[df['species'] == species][feature_names].describe()
    print(species_stats.round(2))

# ============================================================================
# STEP 3: DATA PREPROCESSING
# ============================================================================

print("\n🔧 STEP 3: Data Preprocessing")
print("-" * 40)

# Prepare features and target
X = df[feature_names].values
y = df['species'].values

# Encode target labels to numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Original classes:", np.unique(y))
print("Encoded classes:", np.unique(y_encoded))
print("Encoding mapping:", dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nDataset split:")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nFeature scaling applied:")
print(f"Original feature ranges:")
for i, feature in enumerate(feature_names):
    print(f"  {feature}: [{X_train[:, i].min():.2f}, {X_train[:, i].max():.2f}]")

print(f"\nScaled feature ranges:")
for i, feature in enumerate(feature_names):
    print(f"  {feature}: [{X_train_scaled[:, i].min():.2f}, {X_train_scaled[:, i].max():.2f}]")

# ============================================================================
# STEP 4: MODEL TRAINING AND COMPARISON
# ============================================================================

print("\n🤖 STEP 4: Training Multiple Classification Models")
print("-" * 40)

# Define multiple classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=200),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Support Vector Machine': SVC(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3),
    'Naive Bayes': GaussianNB()
}

# Train and evaluate each model
model_results = {}

for name, classifier in classifiers.items():
    print(f"\n🔬 Training {name}...")
    
    # Train the model
    classifier.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_train = classifier.predict(X_train_scaled)
    y_pred_test = classifier.predict(X_test_scaled)
    
    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    # Cross-validation
    cv_scores = cross_val_score(classifier, X_train_scaled, y_train, cv=5)
    
    # Store results
    model_results[name] = {
        'model': classifier,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred_test
    }
    
    print(f"✅ {name} Results:")
    print(f"   Training Accuracy: {train_accuracy:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    print(f"   CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# ============================================================================
# STEP 5: MODEL EVALUATION AND COMPARISON
# ============================================================================

print("\n📈 STEP 5: Model Evaluation and Comparison")
print("-" * 40)

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Model': list(model_results.keys()),
    'Train Accuracy': [results['train_accuracy'] for results in model_results.values()],
    'Test Accuracy': [results['test_accuracy'] for results in model_results.values()],
    'CV Mean': [results['cv_mean'] for results in model_results.values()],
    'CV Std': [results['cv_std'] for results in model_results.values()]
})

print("🏆 Model Performance Comparison:")
print(comparison_df.round(4))

# Find the best model
best_model_name = comparison_df.loc[comparison_df['Test Accuracy'].idxmax(), 'Model']
best_model = model_results[best_model_name]['model']
best_predictions = model_results[best_model_name]['predictions']

print(f"\n🥇 Best Model: {best_model_name}")
print(f"   Test Accuracy: {comparison_df[comparison_df['Model'] == best_model_name]['Test Accuracy'].values[0]:.4f}")

# ============================================================================
# STEP 6: DETAILED EVALUATION OF BEST MODEL
# ============================================================================

print(f"\n🔍 STEP 6: Detailed Evaluation of {best_model_name}")
print("-" * 40)

# Classification Report
print("📊 Classification Report:")
class_names = label_encoder.classes_
print(classification_report(y_test, best_predictions, target_names=class_names))

# Confusion Matrix
print("\n🎯 Confusion Matrix:")
cm = confusion_matrix(y_test, best_predictions)
print("Actual vs Predicted:")
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
print(cm_df)

# Calculate additional metrics
precision, recall, f1, support = precision_recall_fscore_support(y_test, best_predictions, average=None)

print(f"\n📏 Detailed Metrics by Class:")
for i, class_name in enumerate(class_names):
    print(f"{class_name}:")
    print(f"  Precision: {precision[i]:.4f}")
    print(f"  Recall: {recall[i]:.4f}")
    print(f"  F1-Score: {f1[i]:.4f}")
    print(f"  Support: {support[i]}")

# ============================================================================
# STEP 7: HYPERPARAMETER TUNING
# ============================================================================

print(f"\n⚙️ STEP 7: Hyperparameter Tuning for {best_model_name}")
print("-" * 40)

# Define parameter grids for different models
param_grids = {
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10]
    },
    'Support Vector Machine': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear']
    },
    'K-Nearest Neighbors': {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'Decision Tree': {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
}

if best_model_name in param_grids:
    print(f"🔍 Searching for optimal hyperparameters...")
    
    # Recreate the best model class
    model_classes = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Support Vector Machine': SVC(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    base_model = model_classes[best_model_name]
    param_grid = param_grids[best_model_name]
    
    # Perform Grid Search
    grid_search = GridSearchCV(
        base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    # Get the best model
    tuned_model = grid_search.best_estimator_
    tuned_predictions = tuned_model.predict(X_test_scaled)
    tuned_accuracy = accuracy_score(y_test, tuned_predictions)
    
    print(f"✅ Hyperparameter tuning completed!")
    print(f"🎯 Best parameters: {grid_search.best_params_}")
    print(f"📈 Best CV score: {grid_search.best_score_:.4f}")
    print(f"🏆 Tuned model test accuracy: {tuned_accuracy:.4f}")
    
    original_accuracy = model_results[best_model_name]['test_accuracy']
    improvement = tuned_accuracy - original_accuracy
    print(f"📊 Improvement: {improvement:.4f} ({improvement/original_accuracy*100:+.2f}%)")
    
else:
    print(f"Hyperparameter tuning not implemented for {best_model_name}")
    tuned_model = best_model
    tuned_accuracy = model_results[best_model_name]['test_accuracy']

# ============================================================================
# STEP 8: FEATURE IMPORTANCE ANALYSIS
# ============================================================================

print(f"\n🔬 STEP 8: Feature Importance Analysis")
print("-" * 40)

# Get feature importance (if available)
if hasattr(tuned_model, 'feature_importances_'):
    feature_importance = tuned_model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    print("🌟 Feature Importance Ranking:")
    for _, row in importance_df.iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(importance_df['Feature'], importance_df['Importance'])
    plt.title(f'Feature Importance - {best_model_name}')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
elif hasattr(tuned_model, 'coef_'):
    # For linear models, use coefficient magnitudes
    coef_importance = np.abs(tuned_model.coef_[0])
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': coef_importance
    }).sort_values('Importance', ascending=False)
    
    print("🌟 Feature Coefficient Magnitudes:")
    for _, row in importance_df.iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")
else:
    print("Feature importance not available for this model type.")

# ============================================================================
# STEP 9: MAKING PREDICTIONS ON NEW DATA
# ============================================================================

print(f"\n🔮 STEP 9: Making Predictions on New Data")
print("-" * 40)

# Create some example new data points
new_data_examples = [
    [5.1, 3.5, 1.4, 0.2],  # Likely Setosa
    [6.2, 2.8, 4.8, 1.8],  # Likely Versicolor  
    [7.7, 3.0, 6.1, 2.3],  # Likely Virginica
]

print("🧪 Making predictions on new flower measurements:")
for i, measurements in enumerate(new_data_examples):
    # Scale the new data
    new_data_scaled = scaler.transform([measurements])
    
    # Make prediction
    prediction = tuned_model.predict(new_data_scaled)[0]
    prediction_proba = tuned_model.predict_proba(new_data_scaled)[0] if hasattr(tuned_model, 'predict_proba') else None
    
    predicted_species = label_encoder.inverse_transform([prediction])[0]
    
    print(f"\nSample {i+1}: {measurements}")
    print(f"  Predicted Species: {predicted_species}")
    
    if prediction_proba is not None:
        print("  Prediction Probabilities:")
        for j, prob in enumerate(prediction_proba):
            species = label_encoder.inverse_transform([j])[0]
            print(f"    {species}: {prob:.4f} ({prob*100:.1f}%)")

# ============================================================================
# STEP 10: MODEL PERSISTENCE AND DEPLOYMENT PREPARATION
# ============================================================================

print(f"\n💾 STEP 10: Model Persistence")
print("-" * 40)

import joblib
from datetime import datetime

# Save the trained model and preprocessing components
model_filename = f'iris_classifier_{best_model_name.lower().replace(" ", "_")}_{datetime.now().strftime("%Y%m%d")}.pkl'
scaler_filename = f'iris_scaler_{datetime.now().strftime("%Y%m%d")}.pkl'
encoder_filename = f'iris_label_encoder_{datetime.now().strftime("%Y%m%d")}.pkl'

# Save models
joblib.dump(tuned_model, model_filename)
joblib.dump(scaler, scaler_filename)
joblib.dump(label_encoder, encoder_filename)

print(f"✅ Model saved as: {model_filename}")
print(f"✅ Scaler saved as: {scaler_filename}")
print(f"✅ Label encoder saved as: {encoder_filename}")

# Example of how to load and use the saved model
print(f"\n📖 Example of loading and using the saved model:")
print(f"""
# Load the saved components
loaded_model = joblib.load('{model_filename}')
loaded_scaler = joblib.load('{scaler_filename}')
loaded_encoder = joblib.load('{encoder_filename}')

# Make prediction on new data
new_flower = [[5.8, 2.7, 5.1, 1.9]]
new_flower_scaled = loaded_scaler.transform(new_flower)
prediction = loaded_model.predict(new_flower_scaled)
predicted_species = loaded_encoder.inverse_transform(prediction)
print(f"Predicted species: {{predicted_species[0]}}")
""")

# ============================================================================
# SUMMARY AND KEY TAKEAWAYS
# ============================================================================

print(f"\n🎓 TUTORIAL SUMMARY AND KEY TAKEAWAYS")
print("=" * 50)

print(f"""
🌸 IRIS CLASSIFICATION PROJECT COMPLETED! 🌸

📊 DATASET OVERVIEW:
• 150 samples across 3 iris species
• 4 features: sepal/petal length & width
• Perfectly balanced dataset (50 samples per class)
• No missing values - clean dataset

🤖 MODELS TESTED:
• Logistic Regression
• Decision Tree
• Random Forest  
• Support Vector Machine
• K-Nearest Neighbors
• Naive Bayes

🏆 BEST MODEL: {best_model_name}
• Test Accuracy: {tuned_accuracy:.4f} ({tuned_accuracy*100:.1f}%)
• This means the model correctly classifies {tuned_accuracy*100:.1f}% of iris flowers

🔍 KEY MACHINE LEARNING CONCEPTS LEARNED:
1. Data Exploration & Visualization
2. Train/Test Split & Cross-Validation
3. Feature Scaling & Preprocessing
4. Multiple Algorithm Comparison
5. Hyperparameter Tuning
6. Model Evaluation Metrics
7. Feature Importance Analysis
8. Model Persistence & Deployment

💡 PRACTICAL INSIGHTS:
• Iris dataset is excellent for learning - it's clean and well-separated
• Multiple algorithms perform well (>90% accuracy)
• Feature scaling improves some models (SVM, KNN)
• Cross-validation prevents overfitting
• Hyperparameter tuning can improve performance

🚀 NEXT STEPS:
1. Try this approach on other datasets
2. Experiment with ensemble methods
3. Learn about deep learning for classification
4. Explore more advanced evaluation metrics
5. Build a web app for model deployment

Congratulations! You've successfully built and evaluated a complete machine learning classification system! 🎉
""")
