"""
Tests for model module.
"""

import unittest
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

# Add the parent directory to the path so we can import the src module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import (
    split_data,
    train_random_forest,
    evaluate_model,
    train_model_pipeline
)

class TestModel(unittest.TestCase):
    """Test cases for model module functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample dataset for testing
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        
        # Create random features
        self.X = np.random.rand(n_samples, n_features)
        
        # Create random target (3 classes)
        self.y = np.random.randint(0, 3, n_samples)
        
        # Create temporary directory for saving
        self.temp_dir = Path('tests/temp')
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def test_split_data(self):
        """Test the split_data function."""
        # Split the data
        splits = split_data(self.X, self.y, test_size=0.2, val_size=0.2)
        
        # Check that all expected keys are present
        expected_keys = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']
        for key in expected_keys:
            self.assertIn(key, splits)
        
        # Check shapes
        # train: 0.6 * 200 = 120, val: 0.2 * 200 = 40, test: 0.2 * 200 = 40
        self.assertEqual(splits['X_train'].shape[0], 128)
        self.assertEqual(splits['X_val'].shape[0], 32)
        self.assertEqual(splits['X_test'].shape[0], 40)
        
        # Check that the number of samples adds up
        total_samples = (
            splits['X_train'].shape[0] + 
            splits['X_val'].shape[0] + 
            splits['X_test'].shape[0]
        )
        self.assertEqual(total_samples, self.X.shape[0])
    
    def test_train_random_forest(self):
        """Test the train_random_forest function."""
        # Split the data
        splits = split_data(self.X, self.y)
        
        # Train a random forest
        model = train_random_forest(splits['X_train'], splits['y_train'], n_estimators=10)
        
        # Check that model is trained
        self.assertIsInstance(model, RandomForestClassifier)
        self.assertEqual(len(model.estimators_), 10)
        
        # Try making predictions
        y_pred = model.predict(splits['X_val'])
        self.assertEqual(y_pred.shape[0], splits['X_val'].shape[0])
    
    def test_evaluate_model(self):
        """Test the evaluate_model function."""
        # Split the data
        splits = split_data(self.X, self.y)
        
        # Train a model
        model = train_random_forest(splits['X_train'], splits['y_train'], n_estimators=10)
        
        # Evaluate on validation data only
        eval_results = evaluate_model(model, splits['X_val'], splits['y_val'])
        
        # Check that evaluation metrics are present
        self.assertIn('val_accuracy', eval_results)
        self.assertIn('val_f1', eval_results)
        
        # Evaluate on validation and test data
        eval_results = evaluate_model(
            model, 
            splits['X_val'], 
            splits['y_val'], 
            splits['X_test'], 
            splits['y_test']
        )
        
        # Check that test metrics are present
        self.assertIn('val_accuracy', eval_results)
        self.assertIn('val_f1', eval_results)
        self.assertIn('test_accuracy', eval_results)
        self.assertIn('test_f1', eval_results)
    
    def test_train_model_pipeline(self):
        """Test the train_model_pipeline function."""
        # Run the model pipeline
        result = train_model_pipeline(
            self.X, 
            self.y,
            model_type='random_forest',
            model_params={'n_estimators': 10},
            save=False
        )
        
        # Check that the result has the expected keys
        expected_keys = ['model', 'data_splits', 'evaluation', 'model_path', 'label_encoder']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check that the model is trained
        self.assertIsInstance(result['model'], RandomForestClassifier)
        
        # Check that data_splits has the expected keys
        for key in ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']:
            self.assertIn(key, result['data_splits'])
        
        # Check that evaluation has metrics
        self.assertIn('val_accuracy', result['evaluation'])
        self.assertIn('val_f1', result['evaluation'])
        self.assertIn('test_accuracy', result['evaluation'])
        self.assertIn('test_f1', result['evaluation'])

if __name__ == '__main__':
    unittest.main()