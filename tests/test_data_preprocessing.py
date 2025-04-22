"""
Tests for data preprocessing module.
"""

import unittest
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the parent directory to the path so we can import the src module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preprocessing import (
    load_dataset, 
    clean_dataset, 
    preprocess_power_combined,
    preprocess_hpc_kernel_events
)

class TestDataPreprocessing(unittest.TestCase):
    """Test cases for data preprocessing functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a sample Power dataset
        self.power_data = {
            'shunt_voltage': np.random.rand(100),
            'bus_voltage_V': np.random.rand(100),
            'current_mA': np.random.rand(100),
            'power_mW': np.random.rand(100),
            'State': np.random.choice(['S1', 'S2', 'S3'], 100),
            'interface': np.random.choice(['eth0', 'wlan0'], 100),
            'Attack-Group': np.random.choice(['host-attack', 'none', 'recon'], 100),
            'Label': np.random.choice(['Normal', 'Attack'], 100)
        }
        self.power_df = pd.DataFrame(self.power_data)
        
        # Create a sample HPC dataset
        self.hpc_data = {
            'Feature1': np.random.rand(100),
            'Feature2': np.random.rand(100),
            'Feature3': np.random.rand(100),
            'Scenario': np.random.choice(['Scenario1', 'Scenario2', 'writeback:writeback_write_inode_start', '0'], 100)
        }
        self.hpc_df = pd.DataFrame(self.hpc_data)
        
        # Create temporary CSV files
        self.temp_dir = Path('tests/temp')
        os.makedirs(self.temp_dir, exist_ok=True)
        
        self.power_filepath = self.temp_dir / 'power_test.csv'
        self.hpc_filepath = self.temp_dir / 'hpc_test.csv'
        
        self.power_df.to_csv(self.power_filepath, index=False)
        self.hpc_df.to_csv(self.hpc_filepath, index=False)
    
    def tearDown(self):
        """Tear down test fixtures."""
        if os.path.exists(self.power_filepath):
            os.remove(self.power_filepath)
        if os.path.exists(self.hpc_filepath):
            os.remove(self.hpc_filepath)
    
    def test_load_dataset(self):
        """Test the load_dataset function."""
        # Test loading an existing file
        df = load_dataset(self.power_filepath)
        self.assertIsNotNone(df)
        self.assertEqual(df.shape[0], 100)
        
        # Test loading a non-existent file
        df = load_dataset('non_existent_file.csv')
        self.assertIsNone(df)
    
    def test_clean_dataset(self):
        """Test the clean_dataset function."""
        # Create a dataset with duplicates and missing values
        dirty_data = self.power_data.copy()
        df = pd.DataFrame(dirty_data)
        df = pd.concat([df, df.iloc[0:10]], ignore_index=True)  # Add duplicates
        df.iloc[20:30, 0] = np.nan  # Add missing values
        
        # Clean the dataset
        cleaned_df = clean_dataset(df)
        
        # Check that duplicates were removed
        self.assertEqual(cleaned_df.shape[0], 100)
        
        # Check that missing values were handled
        self.assertEqual(cleaned_df.isnull().sum().sum(), 0)
    
    def test_preprocess_power_combined(self):
        """Test the preprocess_power_combined function."""
        X_scaled, y_resampled, features, power_processed = preprocess_power_combined(self.power_filepath)
        
        # Check that the function returned the expected outputs
        self.assertIsNotNone(X_scaled)
        self.assertIsNotNone(y_resampled)
        self.assertIsNotNone(features)
        self.assertIsNotNone(power_processed)
        
        # Check that the feature names match
        expected_features = ['shunt_voltage', 'bus_voltage_V', 'current_mA', 'power_mW', 'State', 'interface']
        self.assertEqual(set(features), set(expected_features))
    
    def test_preprocess_hpc_kernel_events(self):
        """Test the preprocess_hpc_kernel_events function."""
        X_pca, y, pca_columns, hpc_processed = preprocess_hpc_kernel_events(
            self.hpc_filepath, n_components=2
        )
        
        # Check that the function returned the expected outputs
        self.assertIsNotNone(X_pca)
        self.assertIsNotNone(y)
        self.assertIsNotNone(pca_columns)
        self.assertIsNotNone(hpc_processed)
        
        # Check the number of PCA components
        self.assertEqual(len(pca_columns), 2)
        self.assertEqual(X_pca.shape[1], 2)

if __name__ == '__main__':
    unittest.main()