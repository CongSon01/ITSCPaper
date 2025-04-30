"""
Data Preprocessing Module

This module contains functions for preprocessing the PowerCombined and HPC-Kernel-Events
datasets using the DataProcessor class approach.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, HPC_RAW_FILE, POWER_RAW_FILE

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
                hpc_path = HPC_RAW_FILE
                
            if power_path is None:
                power_path = POWER_RAW_FILE
                
            # Load datasets - convert Path objects to strings if needed
            hpc_df = pd.read_csv(str(hpc_path))  
            power_df = pd.read_csv(str(power_path))
            
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
            
            # Save scalers for later use
            joblib.dump(scaler_hpc, os.path.join(PROCESSED_DATA_DIR, "hpc_scaler.joblib"))
            joblib.dump(scaler_power, os.path.join(PROCESSED_DATA_DIR, "power_scaler.joblib"))
            
            # Return preprocessed data
            return {
                'X_hpc': X_hpc_scaled,
                'y_hpc': y_hpc,
                'X_power': X_power_scaled,
                'y_power': y_power,
                'hpc_df': hpc_df,
                'power_df': power_df,
                'scaler_hpc': scaler_hpc,
                'scaler_power': scaler_power
            }
            
        except Exception as e:  
            logger.error(f"Error loading data: {e}")
            return None

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
            # Save as CSV files
            if save_dataframes:
                # Create DataFrames from arrays and save
                if 'X_hpc' in data_dict and 'y_hpc' in data_dict:
                    hpc_df = pd.DataFrame(data_dict['X_hpc'])
                    hpc_df['Scenario'] = data_dict['y_hpc']
                    hpc_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "HPC_processed.csv"), index=False)
                    logger.info(f"Saved processed HPC data to {os.path.join(PROCESSED_DATA_DIR, 'HPC_processed.csv')}")
                
                if 'X_power' in data_dict and 'y_power' in data_dict:
                    power_df = pd.DataFrame(data_dict['X_power'])
                    power_df['Attack-Group'] = data_dict['y_power']
                    power_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "Power_processed.csv"), index=False)
                    logger.info(f"Saved processed Power data to {os.path.join(PROCESSED_DATA_DIR, 'Power_processed.csv')}")
            
            # Save numpy arrays
            if save_arrays:
                for key, value in data_dict.items():
                    if isinstance(value, np.ndarray):
                        np.save(os.path.join(PROCESSED_DATA_DIR, f"{key}.npy"), value)
                        logger.info(f"Saved array {key} to {os.path.join(PROCESSED_DATA_DIR, f'{key}.npy')}")
                
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")

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
            save_dir = os.path.join(PROCESSED_DATA_DIR, "visualizations")
            
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