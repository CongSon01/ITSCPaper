"""
Example Script for Training TensorFlow Model

This script demonstrates how to train a model using the TensorFlow-based
approach with attention encoders or MMD-GAN encoder.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import argparse
from pathlib import Path

from src.data_preprocessing import DataProcessor
from src.model import AttentionEncoder, DatasetMerger, ModelTrainer, Visualizer
from src.mmd_gan_encoder import MMDGANEncoder, MMDFusionTrainer
from src.utils import setup_logging
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train a model using TensorFlow and attention mechanisms or MMD-GAN'
    )
    
    parser.add_argument(
        '--latent-dim', 
        type=int, 
        default=16,
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
        default=32,
        help='Batch size for training'
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
        '--encoder-type', 
        type=str, 
        default='attention',
        choices=['attention', 'mmd-gan'],
        help='Type of encoder to use (attention or mmd-gan)'
    )
    
    parser.add_argument(
        '--triplet-margin', 
        type=float, 
        default=1.0,
        help='Margin parameter for triplet loss (only used with mmd-gan)'
    )
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logging()
    logger.info(f"Starting TensorFlow model training with {args.encoder_type} encoder...")
    
    try:
        # File paths
        HPC_PATH = Path(PROCESSED_DATA_DIR) / "HPC_processed.csv"
        POWER_PATH = Path(PROCESSED_DATA_DIR) / "Power_processed.csv"
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        data = DataProcessor.load_data(hpc_path=HPC_PATH, power_path=POWER_PATH)
        
        # Create encoder based on chosen type
        if args.encoder_type == 'attention':
            # Create attention-based encoders
            encoder_hpc = AttentionEncoder.create_encoder(
                input_dim=data['X_hpc'].shape[1],
                latent_dim=args.latent_dim
            )
            encoder_power = AttentionEncoder.create_encoder(
                input_dim=data['X_power'].shape[1],
                latent_dim=args.latent_dim
            )
            
            # Print model summaries for debugging
            logger.info("HPC Encoder Summary:")
            encoder_hpc.summary()
            logger.info("\nPower Encoder Summary:")
            encoder_power.summary()
            
            # Encode datasets
            latent_hpc = encoder_hpc.predict(data['X_hpc'])
            latent_power = encoder_power.predict(data['X_power'])
            
            # Print shapes for verification
            logger.info(f"\nLatent HPC Shape: {latent_hpc.shape}")
            logger.info(f"Latent Power Shape: {latent_power.shape}")
            
            # Merge datasets
            X_merged, y_merged, common_labels, label_encoder = DatasetMerger.merge_datasets(
                latent_hpc, data['y_hpc'],
                latent_power, data['y_power']
            )
            
        else:  # mmd-gan
            # Create MMD-GAN fusion trainer
            trainer = MMDFusionTrainer(
                latent_dim=args.latent_dim,
                batch_size=args.batch_size,
                epochs=args.epochs,
                triplet_margin=args.triplet_margin
            )
            
            # Train the MMD-GAN encoder
            logger.info("Training MMD-GAN encoder...")
            encoder, history = trainer.train_mmd_encoder(
                X_hpc=data['X_hpc'],
                y_hpc=data['y_hpc'],
                X_power=data['X_power'],
                y_power=data['y_power']
            )
            
            # Process data with the trained encoder
            processed_data = trainer.process_data(
                encoder=encoder,
                X_hpc=data['X_hpc'],
                y_hpc=data['y_hpc'],
                X_power=data['X_power'],
                y_power=data['y_power']
            )
            
            # Extract merged data
            X_merged = processed_data['X_combined']
            y_merged = processed_data['y_combined']
            common_labels = processed_data['common_labels']
            
            # Create a label encoder for compatibility
            label_encoder = LabelEncoder()
            label_encoder.fit(y_merged)
            
            # Print information
            logger.info(f"MMD-GAN encoder trained successfully")
            logger.info(f"Merged data shape: {X_merged.shape}")
        
        logger.info(f"Common labels: {common_labels}")
        logger.info(f"Merged data shape: {X_merged.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_merged, y_merged, test_size=0.2, random_state=42
        )
        
        # Create and train classifier
        classifier = ModelTrainer.create_classifier(
            input_dim=X_merged.shape[1],
            num_classes=len(np.unique(y_merged))
        )
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        logger.info("Training model...")
        history = classifier.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Predictions
        logger.info("Making predictions...")
        y_pred = classifier.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Visualization
        if args.visualize:
            logger.info("Generating visualizations...")
            # Create output directory for visualizations
            vis_dir = Path("results/visualizations")
            vis_dir.mkdir(parents=True, exist_ok=True)
            
            # Plot latent space
            Visualizer.plot_latent_space(
                X_merged, y_merged,
                save_path=vis_dir / "latent_space.png"
            )
            
            # Plot training history
            Visualizer.plot_training_history(
                history,
                save_path=vis_dir / "training_history.png"
            )
        
        # Classification Report
        logger.info("\nClassification Report:")
        print(classification_report(
            y_test,
            y_pred_classes,
            target_names=label_encoder.classes_ if hasattr(label_encoder, 'classes_') else None
        ))
        
        # Save models if requested
        if args.save_model:
            logger.info("Saving models...")
            # Create output directory
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True)
            
            # Save models based on encoder type
            if args.encoder_type == 'attention':
                encoder_hpc.save(model_dir / "encoder_hpc_model.h5")
                encoder_power.save(model_dir / "encoder_power_model.h5")
            else:
                encoder.save_model(model_dir)
                
            classifier.save(model_dir / "classifier_model.h5")
            
            logger.info(f"Models saved to {model_dir}")
    
    except Exception as e:
        logger.error(f"An error occurred in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()