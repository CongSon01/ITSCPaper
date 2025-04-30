"""
Main Pipeline Script

This script runs the complete data processing and model training pipeline
for the ITSCPaper project using TensorFlow and attention mechanisms.
"""

import argparse
import os
import logging
import numpy as np
from pathlib import Path
import tensorflow as tf

from src.data_preprocessing import DataProcessor, preprocess_datasets
from src.model import AttentionEncoder, DatasetMerger, ModelTrainer, Visualizer, train_model_pipeline
from src.utils import setup_logging, create_results_dir
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ITSCPaper - Run data processing and model training pipeline with TensorFlow'
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
        '--preprocess-only', 
        action='store_true',
        help='Run only the preprocessing steps'
    )
    
    parser.add_argument(
        '--skip-preprocessing', 
        action='store_true',
        help='Skip preprocessing and use existing processed data'
    )
    
    parser.add_argument(
        '--save-model', 
        action='store_true',
        help='Save the trained model'
    )
    
    return parser.parse_args()

def run_pipeline():
    """Run the complete pipeline."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logging()
    logger.info(f"Starting pipeline with arguments: {args}")
    
    # Create results directory
    results_dir = create_results_dir()
    logger.info(f"Results will be saved to {results_dir}")
    
    # Ensure TensorFlow is available
    try:
        tf_version = tf.__version__
        logger.info(f"TensorFlow version: {tf_version}")
        
        # Check for GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"GPU is available. Found {len(gpus)} GPU(s)")
            for gpu in gpus:
                logger.info(f"  {gpu}")
        else:
            logger.info("No GPU found. Using CPU")
            
    except Exception as e:
        logger.error(f"Error checking TensorFlow: {e}")
        logger.error("Make sure TensorFlow is installed correctly")
        return
    
    # Check for data files
    if not args.skip_preprocessing:
        power_path = Path(RAW_DATA_DIR) / "Power.csv"
        hpc_path = Path(RAW_DATA_DIR) / "HPC.csv"
        
        if not power_path.exists() or not hpc_path.exists():
            logger.error(f"Raw data files not found in {RAW_DATA_DIR}")
            logger.error("Please ensure Power.csv and HPC.csv are in the raw data directory")
            return
    
    # Run preprocessing if not skipped
    if not args.skip_preprocessing:
        logger.info("Running data preprocessing...")
        data_dict = preprocess_datasets()
        if data_dict is None:
            logger.error("Preprocessing failed")
            return
        logger.info("Preprocessing complete")
    else:
        logger.info("Loading preprocessed data...")
        try:
            # Try to load processed data from files
            X_hpc = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_hpc.npy'))
            y_hpc = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_hpc.npy'))
            X_power = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_power.npy'))
            y_power = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_power.npy'))
            
            data_dict = {
                'X_hpc': X_hpc,
                'y_hpc': y_hpc,
                'X_power': X_power,
                'y_power': y_power
            }
            
            # If numpy files don't exist, try loading from CSV
            logger.info(f"Loaded preprocessed data: X_hpc shape {X_hpc.shape}, X_power shape {X_power.shape}")
        except Exception as e:
            logger.warning(f"Could not load preprocessed NumPy data: {e}")
            logger.info("Trying to load from CSV files...")
            
            try:
                # Load processed CSV files
                hpc_processed = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'HPC_processed.csv'))
                power_processed = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'Power_processed.csv'))
                
                # Extract features and targets
                X_hpc = hpc_processed.drop(columns=['Scenario']).values
                y_hpc = hpc_processed['Scenario'].values
                X_power = power_processed.drop(columns=['Attack-Group']).values
                y_power = power_processed['Attack-Group'].values
                
                data_dict = {
                    'X_hpc': X_hpc,
                    'y_hpc': y_hpc,
                    'X_power': X_power,
                    'y_power': y_power
                }
                
                logger.info(f"Loaded preprocessed CSV data: X_hpc shape {X_hpc.shape}, X_power shape {X_power.shape}")
            except Exception as e2:
                logger.error(f"Failed to load processed data: {e2}")
                logger.error("Please run preprocessing first")
                return
    
    # Exit if preprocess only
    if args.preprocess_only:
        logger.info("Preprocessing complete. Exiting as --preprocess-only was specified.")
        return
    
    # Train model using the new TensorFlow approach
    logger.info(f"Training model with latent dimension {args.latent_dim}...")
    model_results = train_model_pipeline(
        X_hpc=data_dict['X_hpc'],
        y_hpc=data_dict['y_hpc'],
        X_power=data_dict['X_power'],
        y_power=data_dict['y_power'],
        latent_dim=args.latent_dim,
        save=args.save_model,
        output_dir=results_dir
    )
    
    if model_results is None:
        logger.error("Model training failed")
        return
    
    # Generate visualizations (already done in train_model_pipeline)
    logger.info(f"Model training complete!")
    logger.info(f"Test accuracy: {model_results['test_accuracy']:.4f}")
    
    logger.info(f"All results saved to {results_dir}")
    logger.info("Pipeline complete.")

if __name__ == "__main__":
    run_pipeline()