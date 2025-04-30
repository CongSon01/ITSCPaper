"""
Data Preprocessing Module

This module contains functions for preprocessing the PowerCombined and HPC-Kernel-Events
datasets using the DataProcessor class approach.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import logging
from datetime import datetime
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/log_{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Class for preprocessing datasets for the ITSCPaper project.
    """
    
    @staticmethod
    def ensure_directories():
        """Create necessary directories if they don't exist."""
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
    
    @staticmethod
    def load_data(hpc_path=None, power_path=None):
        """  
        Load and preprocess datasets  
        """
        try:
            # Use default paths if not provided
            if hpc_path is None:
                hpc_path = Path(RAW_DATA_DIR) / "HPC.csv"
            if power_path is None:
                power_path = Path(RAW_DATA_DIR) / "Power.csv"
                
            # Load datasets  
            hpc_df = pd.read_csv(hpc_path)  
            power_df = pd.read_csv(power_path)
            
            logger.info(f"Loaded Power dataset with shape: {power_df.shape}")
            logger.info(f"Loaded HPC dataset with shape: {hpc_df.shape}")
            
            # Prepare HPC data  
            X_hpc = hpc_df.drop(columns=['Scenario']).values  
            y_hpc = hpc_df['Scenario'].values  
            
            # Prepare Power data  
            X_power = power_df.drop(columns=['Attack-Group']).values  
            y_power = power_df['Attack-Group'].values  
            
            # Scale data  
            scaler_hpc = StandardScaler()  
            scaler_power = StandardScaler()  
            
            X_hpc_scaled = scaler_hpc.fit_transform(X_hpc)  
            X_power_scaled = scaler_power.fit_transform(X_power)

            # Save scalers
            scaler_hpc_path = os.path.join(PROCESSED_DATA_DIR, 'hpc_scaler.joblib')
            scaler_power_path = os.path.join(PROCESSED_DATA_DIR, 'power_scaler.joblib')
            joblib.dump(scaler_hpc, scaler_hpc_path)
            joblib.dump(scaler_power, scaler_power_path)
            logger.info(f"Saved scalers to {PROCESSED_DATA_DIR}")
            
            return {  
                'X_hpc': X_hpc_scaled,  
                'y_hpc': y_hpc,  
                'X_power': X_power_scaled,  
                'y_power': y_power,
                'scaler_hpc': scaler_hpc,
                'scaler_power': scaler_power
            }
            
        except Exception as e:  
            logger.error(f"Error in data loading: {e}")  
            raise

    @staticmethod
    def save_processed_data(data_dict, save_arrays=True, save_dataframes=True):
        """
        Save processed data to files
        
        Parameters:
        -----------
        data_dict : dict
            Dictionary containing processed data
        save_arrays : bool, default=True
            Whether to save numpy arrays
        save_dataframes : bool, default=True
            Whether to save as CSV files
        """
        try:
            if save_dataframes:
                # Save HPC data
                hpc_df = pd.DataFrame(data_dict['X_hpc'])
                hpc_df['Scenario'] = data_dict['y_hpc']
                hpc_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'HPC_processed.csv'), index=False)
                
                # Save Power data
                power_df = pd.DataFrame(data_dict['X_power'])
                power_df['Attack-Group'] = data_dict['y_power']
                power_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'Power_processed.csv'), index=False)
                
                logger.info("Saved processed data as CSV files")
            
            if save_arrays:
                # Save numpy arrays
                np.save(os.path.join(PROCESSED_DATA_DIR, 'X_hpc.npy'), data_dict['X_hpc'])
                np.save(os.path.join(PROCESSED_DATA_DIR, 'y_hpc.npy'), data_dict['y_hpc'])
                np.save(os.path.join(PROCESSED_DATA_DIR, 'X_power.npy'), data_dict['X_power'])
                np.save(os.path.join(PROCESSED_DATA_DIR, 'y_power.npy'), data_dict['y_power'])
                
                logger.info("Saved processed data as NumPy arrays")
                
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise

    @staticmethod
    def plot_data_distribution(data_dict, save_dir=None):
        """
        Plot the distribution of data
        
        Parameters:
        -----------
        data_dict : dict
            Dictionary containing processed data
        save_dir : str or Path, optional
            Directory to save plots
        """
        if save_dir is None:
            save_dir = os.path.join(PROCESSED_DATA_DIR, 'visualizations')
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot HPC data class distribution
        plt.figure(figsize=(10, 6))
        unique_hpc, counts_hpc = np.unique(data_dict['y_hpc'], return_counts=True)
        plt.bar(unique_hpc.astype(str), counts_hpc)
        plt.title('HPC Data Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.savefig(os.path.join(save_dir, 'hpc_class_distribution.png'))
        plt.close()
        
        # Plot Power data class distribution
        plt.figure(figsize=(10, 6))
        unique_power, counts_power = np.unique(data_dict['y_power'], return_counts=True)
        plt.bar(unique_power.astype(str), counts_power)
        plt.title('Power Data Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.savefig(os.path.join(save_dir, 'power_class_distribution.png'))
        plt.close()
        
        logger.info(f"Saved data distribution plots to {save_dir}")

def preprocess_datasets():
    """
    Main function to preprocess both datasets.
    
    Returns:
    --------
    dict
        Dictionary containing processed data
    """
    # Ensure directories exist
    DataProcessor.ensure_directories()
    
    try:
        # Load and preprocess data
        data_dict = DataProcessor.load_data()
        
        # Save processed data
        DataProcessor.save_processed_data(data_dict)
        
        # Plot data distribution
        DataProcessor.plot_data_distribution(data_dict)
        
        logger.info("Preprocessing complete")
        
        return data_dict
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        return None

if __name__ == "__main__":
    # Run preprocessing pipeline
    preprocess_datasets()