# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_auc_score)
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def train_and_evaluate():
    print("Loading data...")
    # Load data
    df = pd.read_csv('telco.csv')
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Display first few rows
    print("\nFirst few rows of data:")
    print(df.head())
    
    # Check for target column
    # Assuming 'Churn' is the target variable. If it has a different name, change it here.
    # Common churn column names: 'Churn', 'churn', 'Exited', 'Attrition'
    target_column = None
    possible_targets = ['Churn', 'churn', 'Exited', 'Attrition', 'is_churn']
    
    for col in possible_targets:
        if col in df.columns:
            target_column = col
            break
    
    if not target_column:
        # Try to find any binary column
        for col in df.columns:
            if df[col].nunique() == 2:
                target_column = col
                print(f"Assuming '{col}' as target variable (binary column)")
                break
    
    if not target_column:
        raise ValueError("Could not find target column. Please specify the churn column name.")
    
    print(f"\nUsing '{target_column}' as target variable")
    
    # Check for unique values in target
    print(f"Target distribution:\n{df[target_column].value_counts()}")
    
    # Handle missing values in target
    df = df.dropna(subset=[target_column])
    
    # Convert target to binary (0/1) if needed
    if df[target_column].dtype == 'object':
        # Map Yes/No or similar to 1/0
        unique_vals = df[target_column].unique()
        print(f"Unique target values: {unique_vals}")
        
        # Common mappings
        if set(unique_vals).issubset({'Yes', 'No', 'yes', 'no'}):
            df[target_column] = df[target_column].map({'Yes': 1, 'yes': 1, 'No': 0, 'no': 0})
        elif set(unique_vals).issubset({'True', 'False', 'true', 'false'}):
            df[target_column] = df[target_column].map({'True': 1, 'true': 1, 'False': 0, 'false': 0})
        else:
            # Use label encoding for other categorical values
            df[target_column] = pd.factorize(df[target_column])[0]
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column].astype(int)
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"\nNumerical columns ({len(numerical_cols)}): {numerical_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    
    # Handle columns that might look numerical but are actually categorical
    for col in numerical_cols:
        if X[col].nunique() < 10:  # Few unique values might indicate categorical
            print(f"Warning: '{col}' has only {X[col].nunique()} unique values")
    
    # Preprocessing pipelines
    # In train_model.py, update the preprocessing section:

# Numerical pipeline - use mean instead of median to avoid issues with new data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Changed from 'median'
    ('scaler', StandardScaler())
])

# Categorical pipeline - use most_frequent for better handling of new categories
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Changed from 'constant'
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    print("\nPreprocessing data...")
    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names after one-hot encoding
    feature_names = []
    
    # Numerical feature names
    feature_names.extend(numerical_cols)
    
    # Categorical feature names (from one-hot encoding)
    categorical_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    categorical_features = categorical_encoder.get_feature_names_out(categorical_cols)
    feature_names.extend(categorical_features)
    
    print(f"Processed features shape: {X_processed.shape}")
    print(f"Number of features after preprocessing: {len(feature_names)}")
    
    # Split data
    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    model.fit(X_train, y_train)
    print("Model training completed!")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Save model
    print("\nSaving model and artifacts...")
    with open('churn_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save feature information
    with open('features.pkl', 'wb') as f:
        pickle.dump({
            'feature_names': feature_names,
            'categorical_columns': categorical_cols,
            'numerical_columns': numerical_cols,
            'preprocessor': preprocessor,
            'original_columns': X.columns.tolist(),
            'target_column': target_column
        }, f)
    
    # Save evaluation results
    with open('model_metrics.pkl', 'wb') as f:
        pickle.dump({
            'metrics': metrics,
            'confusion_matrix': cm,
            'feature_importance': feature_importance,
            'y_test': y_test.values,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'test_indices': y_test.index.tolist()  # Save indices for reference
        }, f)
    
    # Print results
    print("\n" + "="*50)
    print("MODEL TRAINING COMPLETE")
    print("="*50)
    
    print("\nEvaluation Metrics:")
    print("-"*30)
    for metric, value in metrics.items():
        print(f"{metric.capitalize():12}: {value:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(cm)
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"\nTop 10 Most Important Features:")
    print("-"*40)
    for i, row in feature_importance.head(10).iterrows():
        print(f"{i+1:2}. {row['feature'][:50]:50} : {row['importance']:.4f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    plt.figure(figsize=(10, 6))
    top_20 = feature_importance.head(20)
    plt.barh(range(len(top_20)), top_20['importance'][::-1])
    plt.yticks(range(len(top_20)), top_20['feature'][::-1])
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=100)
    print("Saved 'feature_importance.png'")
    
    # Confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=100)
    print("Saved 'confusion_matrix.png'")
    
    print("\n" + "="*50)
    print("All files saved successfully!")
    print("="*50)
    
    return model, metrics, cm, feature_importance

if __name__ == "__main__":
    train_and_evaluate()