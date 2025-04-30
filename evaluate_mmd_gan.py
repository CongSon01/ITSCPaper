"""
MMD-GAN Encoder Evaluation Script

This script evaluates a trained MMD-GAN encoder by visualizing the latent space
and testing how well it aligns data from different domains.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
import os

from src.config import PROCESSED_DATA_DIR
from src.models.mmd_gan import MMDGANEncoder
from src.models.trainer import MMDFusionTrainer
from src.utils.visualization import (
    plot_latent_distributions, 
    plot_latent_space_comparison
)
from src.utils import setup_logging, visualize_latent_space

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate MMD-GAN Encoder for feature fusion'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to the trained encoder model'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/visualizations',
        help='Directory to save evaluation results'
    )
    
    parser.add_argument(
        '--classifier',
        action='store_true',
        help='Train and evaluate a classifier on encoded features'
    )
    
    return parser.parse_args()

def load_data():
    """
    Load preprocessed data for evaluation.
    
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
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting MMD-GAN encoder evaluation")
    
    # Load data
    X_hpc, y_hpc, X_power, y_power = load_data()
    if X_hpc is None:
        logger.error("Failed to load data. Exiting.")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the trained encoder
    try:
        model_path = Path(args.model_path)
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return
        
        # Load the encoder model
        encoder = tf.keras.models.load_model(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        # Wrap the loaded model in our MMDGANEncoder class if needed
        if not isinstance(encoder, MMDGANEncoder):
            input_dim = X_hpc.shape[1]
            latent_dim = encoder.output_shape[1]
            
            wrapped_encoder = MMDGANEncoder(
                input_dim=input_dim,
                latent_dim=latent_dim
            )
            wrapped_encoder.encoder = encoder
            encoder = wrapped_encoder
            
        logger.info(f"Encoder architecture: {encoder.encoder.summary()}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Create trainer for evaluation
    trainer = MMDFusionTrainer()
    
    # Process data with the trained encoder
    processed_data = trainer.process_data(
        encoder=encoder,
        X_hpc=X_hpc,
        y_hpc=y_hpc,
        X_power=X_power,
        y_power=y_power
    )
    
    # Generate visualizations
    logger.info("Generating visualizations")
    
    # Visualize latent space distributions
    plot_latent_distributions(
        processed_data['X_hpc'],
        processed_data['X_power'],
        title='Latent Space Distributions',
        save_path=output_dir / "latent_distributions.png"
    )
    
    # Visualize latent spaces
    visualize_latent_space(
        processed_data['X_combined'],
        processed_data['y_combined'],
        save_path=output_dir / "combined_latent_space.png"
    )
    
    # Train and evaluate classifier if requested
    if args.classifier:
        logger.info("Training classifier on encoded features")
        
        # Train classifier on combined data
        classifier, history = trainer.evaluate(
            processed_data['X_combined'],
            processed_data['y_combined']
        )
        
        # Save classifier
        classifier_path = output_dir / "classifier_model.h5"
        classifier.save(classifier_path)
        logger.info(f"Classifier saved to {classifier_path}")
    
    logger.info("Evaluation completed successfully")

if __name__ == "__main__":
    main()