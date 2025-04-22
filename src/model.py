"""
Model Module

This module contains the model architecture and training functions
for the ITSCPaper project.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib
import time
import os
from pathlib import Path
from src.config import (
    RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE,
    PROCESSED_DATA_DIR
)

def split_data(X, y, test_size=TEST_SIZE, val_size=VALIDATION_SIZE):
    """
    Split data into train, validation, and test sets.
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Target
    test_size : float, default=from config
        Proportion of data to use for testing
    val_size : float, default=from config
        Proportion of training data to use for validation
    
    Returns:
    --------
    dict
        Dictionary containing the split data
    """
    # First split out the test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    
    # Then split the remaining data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, 
        random_state=RANDOM_STATE, stratify=y_train_val
    )
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }

def train_random_forest(X_train, y_train, n_estimators=100):
    """
    Train a Random Forest classifier.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training targets
    n_estimators : int, default=100
        Number of trees in the forest
    
    Returns:
    --------
    sklearn.ensemble.RandomForestClassifier
        The trained model
    """
    print(f"Training Random Forest with {n_estimators} estimators...")
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        random_state=RANDOM_STATE, 
        n_jobs=-1
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model

def evaluate_model(model, X_val, y_val, X_test=None, y_test=None):
    """
    Evaluate the model on validation and optionally test data.
    
    Parameters:
    -----------
    model : estimator
        The trained model
    X_val : array-like
        Validation features
    y_val : array-like
        Validation targets
    X_test : array-like, optional
        Test features
    y_test : array-like, optional
        Test targets
    
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    results = {}
    
    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    
    results['val_accuracy'] = val_accuracy
    results['val_f1'] = val_f1
    
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation F1-Score: {val_f1:.4f}")
    print("\nValidation Classification Report:")
    print(classification_report(y_val, y_val_pred))
    
    # Evaluate on test set if provided
    if X_test is not None and y_test is not None:
        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        results['test_accuracy'] = test_accuracy
        results['test_f1'] = test_f1
        
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        print(f"Test F1-Score: {test_f1:.4f}")
        print("\nTest Classification Report:")
        print(classification_report(y_test, y_test_pred))
    
    return results

def save_model(model, model_name, output_dir=None):
    """
    Save the trained model to disk.
    
    Parameters:
    -----------
    model : estimator
        The trained model
    model_name : str
        Name for the saved model
    output_dir : str or Path, optional
        Directory to save the model. Defaults to a 'models' folder in the
        project root directory.
    
    Returns:
    --------
    str
        Path where the model was saved
    """
    if output_dir is None:
        # Create a models directory in the project root
        output_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "models"
        os.makedirs(output_dir, exist_ok=True)
    
    model_path = Path(output_dir) / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    return model_path

def train_model_pipeline(X, y, model_type='random_forest', model_params=None, save=True):
    """
    Complete model training pipeline.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    model_type : str, default='random_forest'
        Type of model to train
    model_params : dict, optional
        Parameters for the model
    save : bool, default=True
        Whether to save the trained model
    
    Returns:
    --------
    dict
        Dictionary containing the model, data splits, and evaluation results
    """
    # Set default model parameters if not provided
    if model_params is None:
        model_params = {}
    
    # Ensure y is encoded if it's not numeric
    if not np.issubdtype(np.array(y).dtype, np.number):
        print("Encoding target variable...")
        le = LabelEncoder()
        y = le.fit_transform(y)
        label_encoder = le
    else:
        label_encoder = None
    
    # Split the data
    print("Splitting data into train, validation, and test sets...")
    data_splits = split_data(X, y)
    
    # Train the model
    if model_type == 'random_forest':
        n_estimators = model_params.get('n_estimators', 100)
        model = train_random_forest(
            data_splits['X_train'], 
            data_splits['y_train'], 
            n_estimators=n_estimators
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Evaluate the model
    print("\nEvaluating model...")
    evaluation = evaluate_model(
        model, 
        data_splits['X_val'], 
        data_splits['y_val'],
        data_splits['X_test'], 
        data_splits['y_test']
    )
    
    # Save the model if requested
    if save:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_name = f"{model_type}_{timestamp}"
        model_path = save_model(model, model_name)
    else:
        model_path = None
    
    # Return a dictionary with all relevant outputs
    return {
        'model': model,
        'data_splits': data_splits,
        'evaluation': evaluation,
        'model_path': model_path,
        'label_encoder': label_encoder
    }

if __name__ == "__main__":
    # Example usage
    from src.feature_fusion import get_fused_features
    
    print("Getting fused features...")
    fusion_result = get_fused_features(fusion_method='weighted', power_weight=0.7)
    
    if fusion_result:
        X = fusion_result['X_fused']
        
        # Use one of the target variables (you could also use a combined target if appropriate)
        y = fusion_result['y_power']
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        print("\nStarting model training...")
        result = train_model_pipeline(
            X, y, 
            model_type='random_forest', 
            model_params={'n_estimators': 200}
        )
        
        print("\nTraining completed!")
        print(f"Model saved to: {result.get('model_path')}")
        print(f"Validation accuracy: {result['evaluation']['val_accuracy']:.4f}")
        if 'test_accuracy' in result['evaluation']:
            print(f"Test accuracy: {result['evaluation']['test_accuracy']:.4f}")
    else:
        print("Failed to get fused features. Check that data preprocessing has been completed.")