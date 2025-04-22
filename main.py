"""
Main Pipeline Script

This script runs the complete data processing and model training pipeline
for the ITSCPaper project.
"""

import argparse
import os
import logging
import numpy as np
from pathlib import Path

from src.data_preprocessing import preprocess_datasets
from src.feature_fusion import get_fused_features
from src.model import train_model_pipeline
from src.utils import setup_logging, create_results_dir, plot_confusion_matrix, plot_feature_importance
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ITSCPaper - Run data processing and model training pipeline'
    )
    
    parser.add_argument(
        '--fusion-method', 
        type=str, 
        choices=['concatenation', 'weighted', 'pca'],
        default='weighted',
        help='Feature fusion method to use'
    )
    
    parser.add_argument(
        '--power-weight', 
        type=float, 
        default=0.5,
        help='Weight for PowerCombined features (0-1) when using weighted fusion'
    )
    
    parser.add_argument(
        '--model-type', 
        type=str, 
        choices=['random_forest'],
        default='random_forest',
        help='Type of model to train'
    )
    
    parser.add_argument(
        '--n-estimators', 
        type=int, 
        default=100,
        help='Number of estimators for random forest model'
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
        preproc_results = preprocess_datasets()
        logger.info("Preprocessing complete")
    
    # Exit if preprocess only
    if args.preprocess_only:
        logger.info("Preprocessing complete. Exiting as --preprocess-only was specified.")
        return
    
    # Get fused features
    logger.info(f"Fusing features with method: {args.fusion_method}")
    fusion_results = get_fused_features(
        fusion_method=args.fusion_method,
        power_weight=args.power_weight
    )
    
    if fusion_results is None:
        logger.error("Feature fusion failed. Please ensure preprocessed data exists.")
        return
    
    logger.info(f"Feature fusion complete. Fused feature shape: {fusion_results['X_fused'].shape}")
    
    # Extract the appropriate portion of the fused features to match with y_power
    # Note: X_fused contains data from both Power and HPC datasets
    if len(fusion_results['X_fused']) != len(fusion_results['y_power']):
        logger.info("Adjusting features and targets to match dimensions...")
        logger.info(f"X_fused shape: {fusion_results['X_fused'].shape}, y_power length: {len(fusion_results['y_power'])}")
        
        if len(fusion_results['X_fused']) < len(fusion_results['y_power']):
            # X_fused is smaller, so we need to truncate y_power to match
            logger.info("X_fused is smaller than y_power. Truncating y_power to match...")
            n_samples = len(fusion_results['X_fused'])
            y_power_truncated = fusion_results['y_power'][:n_samples]
            logger.info(f"Using truncated y_power. New length: {len(y_power_truncated)}")
            X_power_fused = fusion_results['X_fused']
            y_power = y_power_truncated
        else:
            # y_power is smaller, so we need to truncate X_fused to match
            logger.info("y_power is smaller than X_fused. Truncating X_fused to match...")
            n_samples = len(fusion_results['y_power'])
            X_power_fused = fusion_results['X_fused'][:n_samples]
            logger.info(f"Using truncated X_fused. New shape: {X_power_fused.shape}")
            y_power = fusion_results['y_power']
    else:
        X_power_fused = fusion_results['X_fused']
        y_power = fusion_results['y_power']
    
    # Train model
    logger.info(f"Training {args.model_type} model...")
    model_results = train_model_pipeline(
        X_power_fused,  # Using correctly sized fused features
        y_power,  # Using correctly sized Power dataset targets
        model_type=args.model_type,
        model_params={'n_estimators': args.n_estimators},
        save=args.save_model
    )
    
    # Generate and save visualizations
    logger.info("Generating visualizations...")
    
    # Confusion matrix
    if 'model' in model_results and 'data_splits' in model_results:
        y_pred = model_results['model'].predict(model_results['data_splits']['X_test'])
        
        cm_path = results_dir / 'plots' / 'confusion_matrix.png'
        plot_confusion_matrix(
            model_results['data_splits']['y_test'],
            y_pred,
            figsize=(10, 8),
            save_path=cm_path
        )
        logger.info(f"Confusion matrix saved to {cm_path}")
        
        # Feature importance (if applicable)
        if hasattr(model_results['model'], 'feature_importances_'):
            # Get feature names - this depends on how features were created
            if isinstance(fusion_results['X_fused'], tuple):
                feature_names = [f"Feature_{i}" for i in range(model_results['data_splits']['X_train'].shape[1])]
            else:
                # If X_fused is a DataFrame with column names
                try:
                    feature_names = fusion_results['X_fused'].columns.tolist()
                except:
                    feature_names = [f"Feature_{i}" for i in range(model_results['data_splits']['X_train'].shape[1])]
            
            fi_path = results_dir / 'plots' / 'feature_importance.png'
            plot_feature_importance(
                model_results['model'],
                feature_names,
                top_n=20,
                figsize=(12, 8),
                save_path=fi_path
            )
            logger.info(f"Feature importance plot saved to {fi_path}")
    
    logger.info(f"Model training complete!")
    logger.info(f"Validation accuracy: {model_results['evaluation']['val_accuracy']:.4f}")
    if 'test_accuracy' in model_results['evaluation']:
        logger.info(f"Test accuracy: {model_results['evaluation']['test_accuracy']:.4f}")
    
    # Save model path
    if args.save_model and model_results.get('model_path'):
        logger.info(f"Model saved to: {model_results['model_path']}")
    
    logger.info(f"All results saved to {results_dir}")
    logger.info("Pipeline complete.")

if __name__ == "__main__":
    run_pipeline()