"""
Visualization Utilities

This module provides functions for visualizing MMD-GAN model outputs and training metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_latent_distributions(x_latent, y_latent, title=None, save_path=None):
    """
    Plot the distribution of latent variables.
    
    Parameters:
    -----------
    x_latent : numpy.ndarray
        Latent space representation of dataset 1
    y_latent : numpy.ndarray
        Latent space representation of dataset 2
    title : str, optional
        Plot title
    save_path : str or Path, optional
        Path to save the visualization
    """
    plt.figure(figsize=(12, 6))
    
    # Plot density histograms
    bins = 30
    for i in range(min(5, x_latent.shape[1])):  # Show up to 5 dimensions
        plt.subplot(1, min(5, x_latent.shape[1]), i + 1)
        plt.hist(x_latent[:, i], bins=bins, alpha=0.5, label='Dataset 1')
        plt.hist(y_latent[:, i], bins=bins, alpha=0.5, label='Dataset 2')
        plt.title(f'Dimension {i+1}')
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    if title:
        plt.suptitle(title, y=1.05)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_training_history_mmd(history, save_path=None):
    """
    Plot the training history for MMD-GAN.
    
    Parameters:
    -----------
    history : dict
        Dictionary containing training metrics
    save_path : str or Path, optional
        Path to save the figure
    """
    plt.figure(figsize=(15, 10))
    
    # Plot losses
    plt.subplot(2, 1, 1)
    plt.plot(history['triplet_loss'], label='Triplet Loss')
    plt.plot(history['discriminator_loss'], label='Discriminator Loss')
    plt.plot(history['generator_loss'], label='Generator Loss')
    plt.title('MMD-GAN Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot discriminator accuracy
    plt.subplot(2, 1, 2)
    plt.plot(history['discriminator_accuracy'], label='Discriminator Accuracy')
    plt.title('Discriminator Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        
def plot_latent_space_comparison(encoders, datasets, labels, titles=None, save_path=None):
    """
    Compare latent spaces from multiple encoders.
    
    Parameters:
    -----------
    encoders : list
        List of encoder models
    datasets : list
        List of datasets to encode
    labels : list
        List of labels for each dataset
    titles : list, optional
        Titles for each subplot
    save_path : str or Path, optional
        Path to save the visualization
    """
    from sklearn.manifold import TSNE
    
    n_encoders = len(encoders)
    n_datasets = len(datasets)
    
    if titles is None:
        titles = [f"Encoder {i+1}" for i in range(n_encoders)]
    
    plt.figure(figsize=(5 * n_encoders, 5 * n_datasets))
    
    # For each encoder and dataset combination
    for i, encoder in enumerate(encoders):
        for j, (data, label) in enumerate(zip(datasets, labels)):
            # Encode the data
            encoded = encoder.encode(data)
            
            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            encoded_tsne = tsne.fit_transform(encoded)
            
            # Create subplot
            plt.subplot(n_datasets, n_encoders, j * n_encoders + i + 1)
            scatter = plt.scatter(encoded_tsne[:, 0], encoded_tsne[:, 1], 
                                c=label.reshape(-1), cmap='viridis', alpha=0.7)
            plt.colorbar(scatter)
            plt.title(f"{titles[i]} - Dataset {j+1}")
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()