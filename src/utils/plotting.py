"""
Plotting Utilities

This module provides general plotting and visualization functions for the project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(10, 8), save_path=None):
    """
    Plot a confusion matrix for classifier results
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    class_names : list, optional
        List of class names
    figsize : tuple, default=(10, 8)
        Figure size in inches
    save_path : str, optional
        Path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_training_history(history, figsize=(12, 5), save_path=None):
    """
    Plot training history for model training
    
    Parameters:
    -----------
    history : dict
        Dictionary containing training metrics
    figsize : tuple, default=(12, 5)
        Figure size in inches
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=figsize)
    
    # Plot accuracy
    if 'accuracy' in history and 'val_accuracy' in history:
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Train')
        plt.plot(history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
    
    # Plot loss
    if 'loss' in history and 'val_loss' in history:
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Train')
        plt.plot(history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_latent_space(features, labels, figsize=(10, 8), save_path=None):
    """
    Visualize data in 2D using t-SNE
    
    Parameters:
    -----------
    features : array-like
        Features to visualize
    labels : array-like
        Labels for coloring points
    figsize : tuple, default=(10, 8)
        Figure size in inches
    save_path : str, optional
        Path to save the figure
    """
    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features)
    
    # Plot the results
    plt.figure(figsize=figsize)
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], 
                        c=labels.reshape(-1), cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Encoded Features')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    
    if save_path:
        # Create directory if it doesn't exist
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()