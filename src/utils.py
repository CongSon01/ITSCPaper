"""
Utility Functions Module

This module contains utility and helper functions for the project.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from pathlib import Path
from datetime import datetime
from src.config import BASE_DIR

# Configure logging
def setup_logging(log_level=logging.INFO):
    """
    Set up logging for the project.
    
    Parameters:
    -----------
    log_level : int, default=logging.INFO
        Logging level
    
    Returns:
    --------
    logging.Logger
        Configured logger
    """
    # Create logs directory if it doesn't exist
    log_dir = BASE_DIR / 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"log_{timestamp}.txt"
    
    # Create logger
    logger = logging.getLogger('itscpaper')
    logger.setLevel(log_level)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Create formatter and add to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Visualization functions
def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(10, 8), save_path=None):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    class_names : list, optional
        List of class names
    figsize : tuple, default=(10, 8)
        Figure size
    save_path : str or Path, optional
        Path to save the figure. If None, the figure is not saved.
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot confusion matrix
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names if class_names else 'auto',
        yticklabels=class_names if class_names else 'auto'
    )
    
    # Add labels
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_roc_curve(y_true, y_score, n_classes, class_names=None, figsize=(10, 8), save_path=None):
    """
    Plot ROC curve for multi-class classification.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_score : array-like
        Predicted probabilities
    n_classes : int
        Number of classes
    class_names : list, optional
        List of class names
    figsize : tuple, default=(10, 8)
        Figure size
    save_path : str or Path, optional
        Path to save the figure. If None, the figure is not saved.
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Binarize labels for ROC calculation
    y_test_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    plt.figure(figsize=figsize)
    
    colors = plt.cm.get_cmap('tab10', n_classes)
    
    for i in range(n_classes):
        class_name = class_names[i] if class_names else f"Class {i}"
        plt.plot(
            fpr[i], 
            tpr[i], 
            color=colors(i),
            lw=2,
            label=f'{class_name} (AUC = {roc_auc[i]:.2f})'
        )
    
    # Add diagonal line for reference (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Add labels and legend
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_feature_importance(model, feature_names, top_n=20, figsize=(12, 8), save_path=None):
    """
    Plot feature importance for tree-based models.
    
    Parameters:
    -----------
    model : estimator
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    top_n : int, default=20
        Number of top features to display
    figsize : tuple, default=(12, 8)
        Figure size
    save_path : str or Path, optional
        Path to save the figure. If None, the figure is not saved.
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Check if model has feature_importances_
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    
    # Select top N features
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_feature_names = [feature_names[i] for i in top_indices]
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot feature importances
    plt.bar(range(len(top_importances)), top_importances, align='center')
    plt.xticks(range(len(top_importances)), top_feature_names, rotation=45, ha='right')
    
    # Add labels
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def create_results_dir():
    """
    Create a directory for storing results.
    
    Returns:
    --------
    pathlib.Path
        Path to the results directory
    """
    # Create results directory if it doesn't exist
    results_dir = BASE_DIR / 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Create timestamped subdirectory for current run
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = results_dir / f"run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Create subdirectories for plots and data
    plots_dir = run_dir / 'plots'
    data_dir = run_dir / 'data'
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    return run_dir

if __name__ == "__main__":
    # Example usage
    logger = setup_logging()
    logger.info("Utility functions module loaded")
    
    # Create a results directory
    results_dir = create_results_dir()
    logger.info(f"Results directory created at {results_dir}")