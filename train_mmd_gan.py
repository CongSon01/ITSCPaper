"""
MMD-GAN Encoder Training Script

This script demonstrates how to train the MMD-GAN encoder for mapping features
from diverse datasets into a shared latent space.
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import PROCESSED_DATA_DIR
from src.models.mmd_gan import MMDGANEncoder
from src.models.trainer import MMDFusionTrainer
# Import directly from src.utils instead of src.utils.visualization
from src.utils import setup_logging, visualize_latent_space
from src.utils.visualization import plot_training_history_mmd

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train MMD-GAN Encoder for feature fusion'
    )
    
    parser.add_argument(
        '--latent-dim', 
        type=int, 
        default=32,
        help='Dimension of latent space in the encoder'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=100,
        help='Number of epochs to train'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=64,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--triplet-margin', 
        type=float, 
        default=1.0,
        help='Margin parameter for triplet loss'
    )
    
    parser.add_argument(
        '--visualize', 
        action='store_true',
        help='Generate visualizations'
    )
    
    parser.add_argument(
        '--save-model', 
        action='store_true',
        help='Save the trained model'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='models',
        help='Directory to save model and results'
    )
    
    return parser.parse_args()

def load_data():
    """
    Load preprocessed data for training.
    
    Returns:
    --------
    tuple
        (X_hpc, y_hpc, X_power, y_power) - Features and labels for both datasets
    """
    try:
        # File paths
        HPC_PATH = Path(PROCESSED_DATA_DIR) / "HPC_processed.csv"
        POWER_PATH = Path(PROCESSED_DATA_DIR) / "Power_processed.csv"
        
        # Load datasets
        hpc_df = pd.read_csv(HPC_PATH)
        power_df = pd.read_csv(POWER_PATH)
        
        # Extract features and labels
        X_hpc = hpc_df.iloc[:, :-1].values
        y_hpc = hpc_df.iloc[:, -1].values
        
        X_power = power_df.iloc[:, :-1].values
        y_power = power_df.iloc[:, -1].values
        
        print(f"Loaded HPC data: {X_hpc.shape}, {y_hpc.shape}")
        print(f"Loaded Power data: {X_power.shape}, {y_power.shape}")
        
        return X_hpc, y_hpc, X_power, y_power
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting MMD-GAN encoder training")
    
    # Load data
    X_hpc, y_hpc, X_power, y_power = load_data()
    if X_hpc is None:
        logger.error("Failed to load data. Exiting.")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create and train MMD-GAN encoder
    trainer = MMDFusionTrainer(
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        triplet_margin=args.triplet_margin
    )
    
    encoder, history = trainer.train_mmd_encoder(
        X_hpc=X_hpc,
        y_hpc=y_hpc,
        X_power=X_power,
        y_power=y_power
    )
    
    # Process data with the trained encoder
    processed_data = trainer.process_data(
        encoder=encoder,
        X_hpc=X_hpc,
        y_hpc=y_hpc,
        X_power=X_power,
        y_power=y_power
    )
    
    # Create visualizations if requested
    if args.visualize:
        logger.info("Generating visualizations")
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True, parents=True)
        
        # Visualize latent space for HPC data
        visualize_latent_space(
            processed_data['X_hpc'], 
            processed_data['y_hpc'],
            save_path=vis_dir / "hpc_latent_space.png"
        )
        
        # Visualize latent space for Power data
        visualize_latent_space(
            processed_data['X_power'], 
            processed_data['y_power'],
            save_path=vis_dir / "power_latent_space.png"
        )
        
        # Visualize combined latent space
        visualize_latent_space(
            processed_data['X_combined'], 
            processed_data['y_combined'],
            save_path=vis_dir / "combined_latent_space.png"
        )
        
        # Plot training losses
        if history:
            plt.figure(figsize=(15, 10))
            
            # Plot losses
            plt.subplot(2, 1, 1)
            plt.plot(history['triplet_loss'], label='Triplet Loss')
            plt.plot(history['discriminator_loss'], label='Discriminator Loss')
            plt.plot(history['generator_loss'], label='Generator Loss')
            plt.title('Training Losses')
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
            plt.savefig(vis_dir / "training_history.png")
            plt.close()
    
    # Save model if requested
    if args.save_model:
        logger.info("Saving model")
        encoder.save_model(output_dir)
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main()