"""
Feature Fusion Module

This module contains functions for combining heterogeneous data features
from the PowerCombined and HPC-Kernel-Events datasets.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.config import PROCESSED_DATA_DIR

def load_processed_data():
    """
    Load the processed datasets.
    
    Returns:
    --------
    tuple
        (power_df, hpc_df) - The processed dataframes
    """
    try:
        power_df = pd.read_csv(f"{PROCESSED_DATA_DIR}/Power_processed.csv")
        hpc_df = pd.read_csv(f"{PROCESSED_DATA_DIR}/HPC_processed.csv")
        return power_df, hpc_df
    except FileNotFoundError as e:
        print(f"Error loading processed data: {e}")
        print("Make sure preprocessing has been run first.")
        return None, None

def align_datasets(power_df, hpc_df):
    """
    Align datasets by common samples or timestamps if possible.
    
    Parameters:
    -----------
    power_df : pd.DataFrame
        The processed PowerCombined dataframe
    hpc_df : pd.DataFrame
        The processed HPC-Kernel-Events dataframe
    
    Returns:
    --------
    tuple
        (aligned_power_df, aligned_hpc_df) - The aligned dataframes
    """
    # This is a placeholder for actual alignment logic
    # In a real implementation, you would need a common key to align the datasets
    # such as timestamps or sample IDs
    
    print("Note: Dataset alignment should be customized for your specific data.")
    print(f"Power dataset shape: {power_df.shape}")
    print(f"HPC dataset shape: {hpc_df.shape}")
    
    # For the sake of the example, we'll just return the original dataframes
    # In practice, you would implement appropriate alignment logic
    return power_df, hpc_df

def extract_features(power_df, hpc_df):
    """
    Extract features from both datasets.
    
    Parameters:
    -----------
    power_df : pd.DataFrame
        The aligned PowerCombined dataframe
    hpc_df : pd.DataFrame
        The aligned HPC-Kernel-Events dataframe
    
    Returns:
    --------
    tuple
        (X_power, y_power, X_hpc, y_hpc) - Features and targets from both datasets
    """
    # Extract PowerCombined features and target
    if 'Attack-Group' in power_df.columns:
        y_power = power_df['Attack-Group']
        X_power = power_df.drop(columns=['Attack-Group'])
    else:
        print("Warning: 'Attack-Group' not found in Power dataset")
        # Try to identify the target column (likely the last column)
        y_power = power_df.iloc[:, -1]
        X_power = power_df.iloc[:, :-1]
    
    # Extract HPC-Kernel-Events features and target
    if 'Scenario' in hpc_df.columns:
        y_hpc = hpc_df['Scenario']
        X_hpc = hpc_df.drop(columns=['Scenario'])
    else:
        print("Warning: 'Scenario' not found in HPC dataset")
        # Try to identify the target column (likely the last column)
        y_hpc = hpc_df.iloc[:, -1]
        X_hpc = hpc_df.iloc[:, :-1]
    
    return X_power, y_power, X_hpc, y_hpc

def feature_concatenation(X_power, X_hpc):
    """
    Simple concatenation of features from both datasets.
    
    Parameters:
    -----------
    X_power : pd.DataFrame
        Features from the PowerCombined dataset
    X_hpc : pd.DataFrame
        Features from the HPC-Kernel-Events dataset
    
    Returns:
    --------
    np.ndarray
        Combined feature matrix
    """
    # Convert to numpy arrays if they are DataFrames
    if isinstance(X_power, pd.DataFrame):
        X_power = X_power.values
    if isinstance(X_hpc, pd.DataFrame):
        X_hpc = X_hpc.values
    
    # Make sure we have the same number of samples
    n_power = X_power.shape[0]
    n_hpc = X_hpc.shape[0]
    
    if n_power != n_hpc:
        print(f"Warning: Datasets have different numbers of samples ({n_power} vs {n_hpc})")
        print("Using only the first min(n_power, n_hpc) samples from each dataset")
        min_samples = min(n_power, n_hpc)
        X_power = X_power[:min_samples]
        X_hpc = X_hpc[:min_samples]
    
    # Concatenate along feature dimension
    X_combined = np.hstack((X_power, X_hpc))
    print(f"Combined feature matrix shape: {X_combined.shape}")
    
    return X_combined

def weighted_feature_fusion(X_power, X_hpc, power_weight=0.5):
    """
    Weighted fusion of features from both datasets.
    
    Parameters:
    -----------
    X_power : pd.DataFrame or np.ndarray
        Features from the PowerCombined dataset
    X_hpc : pd.DataFrame or np.ndarray
        Features from the HPC-Kernel-Events dataset
    power_weight : float, default=0.5
        Weight for PowerCombined features (0-1)
    
    Returns:
    --------
    tuple
        (X_combined, power_features_idx, hpc_features_idx) - Combined features and indices
    """
    # Convert to numpy arrays if they are DataFrames
    if isinstance(X_power, pd.DataFrame):
        X_power = X_power.values
    if isinstance(X_hpc, pd.DataFrame):
        X_hpc = X_hpc.values
    
    # Make sure we have the same number of samples
    n_power = X_power.shape[0]
    n_hpc = X_hpc.shape[0]
    
    if n_power != n_hpc:
        print(f"Warning: Datasets have different numbers of samples ({n_power} vs {n_hpc})")
        print("Using only the first min(n_power, n_hpc) samples from each dataset")
        min_samples = min(n_power, n_hpc)
        X_power = X_power[:min_samples]
        X_hpc = X_hpc[:min_samples]
    
    # Standardize each dataset separately
    power_scaler = StandardScaler()
    hpc_scaler = StandardScaler()
    
    X_power_scaled = power_scaler.fit_transform(X_power)
    X_hpc_scaled = hpc_scaler.fit_transform(X_hpc)
    
    # Apply weights
    hpc_weight = 1.0 - power_weight
    X_power_weighted = X_power_scaled * power_weight
    X_hpc_weighted = X_hpc_scaled * hpc_weight
    
    # Concatenate along feature dimension
    X_combined = np.hstack((X_power_weighted, X_hpc_weighted))
    
    # Keep track of feature indices for each dataset
    power_features_idx = np.arange(X_power.shape[1])
    hpc_features_idx = np.arange(X_hpc.shape[1]) + X_power.shape[1]
    
    print(f"Combined feature matrix shape: {X_combined.shape}")
    
    return X_combined, power_features_idx, hpc_features_idx

def fusion_with_dim_reduction(X_power, X_hpc, n_components=0.95):
    """
    Feature fusion with dimensionality reduction using PCA.
    
    Parameters:
    -----------
    X_power : pd.DataFrame or np.ndarray
        Features from the PowerCombined dataset
    X_hpc : pd.DataFrame or np.ndarray
        Features from the HPC-Kernel-Events dataset
    n_components : int or float, default=0.95
        Number of components or variance to keep in PCA
    
    Returns:
    --------
    np.ndarray
        Reduced feature matrix
    """
    # First concatenate the features
    X_combined = feature_concatenation(X_power, X_hpc)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_combined)
    
    print(f"Number of components after PCA: {X_reduced.shape[1]}")
    print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    return X_reduced

def get_fused_features(fusion_method='concatenation', power_weight=0.5, n_components=0.95):
    """
    Get fused features from both datasets using the specified method.
    
    Parameters:
    -----------
    fusion_method : str, default='concatenation'
        The fusion method to use. One of:
        - 'concatenation': Simple concatenation
        - 'weighted': Weighted fusion
        - 'pca': PCA-based fusion
    power_weight : float, default=0.5
        Weight for PowerCombined features (0-1) when using weighted fusion
    n_components : int or float, default=0.95
        Number of components or variance to keep when using PCA
    
    Returns:
    --------
    dict
        Dictionary containing the fused features and related information
    """
    # Load the processed data
    power_df, hpc_df = load_processed_data()
    if power_df is None or hpc_df is None:
        return None
    
    # Align datasets if needed
    power_aligned, hpc_aligned = align_datasets(power_df, hpc_df)
    
    # Extract features
    X_power, y_power, X_hpc, y_hpc = extract_features(power_aligned, hpc_aligned)
    
    # Apply the specified fusion method
    result = {
        'X_power': X_power,
        'y_power': y_power,
        'X_hpc': X_hpc,
        'y_hpc': y_hpc
    }
    
    if fusion_method == 'concatenation':
        X_fused = feature_concatenation(X_power, X_hpc)
        result['X_fused'] = X_fused
        result['fusion_method'] = 'concatenation'
    
    elif fusion_method == 'weighted':
        X_fused, power_idx, hpc_idx = weighted_feature_fusion(X_power, X_hpc, power_weight)
        result['X_fused'] = X_fused
        result['power_features_idx'] = power_idx
        result['hpc_features_idx'] = hpc_idx
        result['power_weight'] = power_weight
        result['fusion_method'] = 'weighted'
    
    elif fusion_method == 'pca':
        X_fused = fusion_with_dim_reduction(X_power, X_hpc, n_components)
        result['X_fused'] = X_fused
        result['n_components'] = X_fused.shape[1]
        result['fusion_method'] = 'pca'
    
    else:
        print(f"Unknown fusion method: {fusion_method}")
        print("Using default concatenation method")
        X_fused = feature_concatenation(X_power, X_hpc)
        result['X_fused'] = X_fused
        result['fusion_method'] = 'concatenation'
    
    return result

if __name__ == "__main__":
    # Example usage
    fusion_result = get_fused_features(fusion_method='weighted', power_weight=0.7)
    if fusion_result:
        print("\nFusion complete!")
        for key, value in fusion_result.items():
            if isinstance(value, np.ndarray):
                print(f"{key} shape: {value.shape}")
            elif isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
                print(f"{key} shape: {value.shape}")
            else:
                print(f"{key}: {value}")