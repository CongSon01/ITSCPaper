"""
Tests for feature fusion module.
"""

import unittest
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the parent directory to the path so we can import the src module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_fusion import (
    load_processed_data,
    extract_features,
    feature_concatenation,
    weighted_feature_fusion,
    fusion_with_dim_reduction
)

class TestFeatureFusion(unittest.TestCase):
    """Test cases for feature fusion functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample datasets
        self.power_data = pd.DataFrame({
            'shunt_voltage': np.random.rand(100),
            'bus_voltage_V': np.random.rand(100),
            'current_mA': np.random.rand(100),
            'power_mW': np.random.rand(100),
            'State': np.random.randint(0, 3, 100),
            'interface': np.random.randint(0, 2, 100),
            'Attack-Group': np.random.randint(0, 3, 100)
        })
        
        self.hpc_data = pd.DataFrame({
            'PC1': np.random.rand(100),
            'PC2': np.random.rand(100),
            'PC3': np.random.rand(100),
            'Scenario': np.random.randint(0, 5, 100)
        })
        
        # Create temporary directory for saving
        self.temp_dir = Path('tests/temp')
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def test_extract_features(self):
        """Test the extract_features function."""
        X_power, y_power, X_hpc, y_hpc = extract_features(self.power_data, self.hpc_data)
        
        # Check shapes
        self.assertEqual(X_power.shape[0], 100)
        self.assertEqual(X_power.shape[1], 6)  # All columns except Attack-Group
        self.assertEqual(y_power.shape[0], 100)
        
        self.assertEqual(X_hpc.shape[0], 100)
        self.assertEqual(X_hpc.shape[1], 3)  # All columns except Scenario
        self.assertEqual(y_hpc.shape[0], 100)
    
    def test_feature_concatenation(self):
        """Test the feature concatenation function."""
        # Extract features
        X_power, _, X_hpc, _ = extract_features(self.power_data, self.hpc_data)
        
        # Concatenate features
        X_combined = feature_concatenation(X_power, X_hpc)
        
        # Check shape
        self.assertEqual(X_combined.shape[0], 100)
        self.assertEqual(X_combined.shape[1], X_power.shape[1] + X_hpc.shape[1])
    
    def test_weighted_feature_fusion(self):
        """Test the weighted feature fusion function."""
        # Extract features
        X_power, _, X_hpc, _ = extract_features(self.power_data, self.hpc_data)
        
        # Apply weighted fusion
        X_combined, power_idx, hpc_idx = weighted_feature_fusion(X_power, X_hpc, power_weight=0.7)
        
        # Check shape
        self.assertEqual(X_combined.shape[0], 100)
        self.assertEqual(X_combined.shape[1], X_power.shape[1] + X_hpc.shape[1])
        
        # Check indices
        self.assertEqual(len(power_idx), X_power.shape[1])
        self.assertEqual(len(hpc_idx), X_hpc.shape[1])
    
    def test_fusion_with_dim_reduction(self):
        """Test the fusion with dimensionality reduction function."""
        # Extract features
        X_power, _, X_hpc, _ = extract_features(self.power_data, self.hpc_data)
        
        # Apply fusion with PCA
        X_reduced = fusion_with_dim_reduction(X_power, X_hpc, n_components=3)
        
        # Check shape
        self.assertEqual(X_reduced.shape[0], 100)
        self.assertEqual(X_reduced.shape[1], 3)  # 3 components

if __name__ == '__main__':
    unittest.main()