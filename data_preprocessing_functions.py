"""
Data Preprocessing Functions for PowerCombined and HPC-Kernel-Events Datasets

This module contains functions to preprocess two independent datasets:
1. PowerCombined dataset
2. HPC-Kernel-Events dataset

Functions include data loading, cleaning, encoding, standardization,
dimensionality reduction, and handling class imbalance.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def load_dataset(filepath):
    """
    Load a dataset from the specified filepath.
    
    Parameters:
    -----------
    filepath : str
        Path to the csv file
    
    Returns:
    --------
    pd.DataFrame or None
        The loaded dataframe or None if file not found
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded with shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"File not found at {filepath}. Please check the path.")
        return None

def clean_dataset(df):
    """
    Clean the dataset by removing duplicates and handling missing values.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to clean
    
    Returns:
    --------
    pd.DataFrame
        The cleaned dataframe
    """
    if df is None:
        return None
    
    # Make a copy of the original data
    clean_df = df.copy()
    
    # Drop duplicates if any
    original_shape = clean_df.shape
    clean_df = clean_df.drop_duplicates()
    print(f"Removed {original_shape[0] - clean_df.shape[0]} duplicate rows")
    
    # Handle missing values
    for col in clean_df.columns:
        if clean_df[col].isnull().sum() > 0:
            if clean_df[col].dtype in ['int64', 'float64']:
                # Fill numeric columns with median
                clean_df[col] = clean_df[col].fillna(clean_df[col].median())
            else:
                # Fill categorical columns with mode
                clean_df[col] = clean_df[col].fillna(clean_df[col].mode()[0])
    
    print(f"After cleaning, dataset shape: {clean_df.shape}")
    return clean_df

def preprocess_power_combined(filepath):
    """
    Preprocess the PowerCombined dataset.
    
    Parameters:
    -----------
    filepath : str
        Path to the PowerCombined csv file
    
    Returns:
    --------
    tuple
        (X_scaled, y_resampled, feature_names, power_processed_df)
        - X_scaled: Standardized feature matrix
        - y_resampled: Balanced target vector
        - feature_names: List of feature names
        - power_processed_df: Complete processed dataframe
    """
    # Load data
    df = load_dataset(filepath)
    if df is None:
        return None, None, None, None
    
    # Clean data
    clean_df = clean_dataset(df)
    
    # Data Cleaning - Rename attack labels
    if "Attack-Group" in clean_df.columns:
        clean_df["Attack-Group"] = clean_df["Attack-Group"].replace({
            "host-attack": "Other",
            "none": "Begin",
            "recon": "Recon"
        })
    
    # Encode categorical features
    label_encoders = {}
    for col in ['Attack-Group', 'State', 'interface', 'Label']:
        if col in clean_df.columns:
            le = LabelEncoder()
            clean_df[col] = le.fit_transform(clean_df[col])
            label_encoders[col] = le
            print(f"Encoded column: {col}")
    
    # Define features and target
    features = ['shunt_voltage', 'bus_voltage_V', 'current_mA', 'power_mW', 'State', 'interface']
    target = 'Attack-Group'
    
    # Filter out features that don't exist in the dataframe
    features = [f for f in features if f in clean_df.columns]
    
    if target not in clean_df.columns:
        print(f"Target column '{target}' not found in the dataset.")
        return None, None, None, None
    
    X = clean_df[features]
    y = clean_df[target]
    
    # Apply SMOTE for class balancing
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    print(f"After SMOTE, features shape: {X_balanced.shape}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_balanced)
    print(f"Scaled features shape: {X_scaled.shape}")
    
    # Create final processed dataframe
    power_processed = pd.DataFrame(X_scaled, columns=features)
    power_processed[target] = y_balanced
    
    return X_scaled, y_balanced, features, power_processed

def preprocess_hpc_kernel_events(filepath, n_components=30):
    """
    Preprocess the HPC-Kernel-Events dataset.
    
    Parameters:
    -----------
    filepath : str
        Path to the HPC-Kernel-Events csv file
    n_components : int, default=30
        Number of components to use for PCA
    
    Returns:
    --------
    tuple
        (X_pca, y, pca_columns, hpc_processed_df)
        - X_pca: PCA transformed feature matrix
        - y: Encoded target vector
        - pca_columns: List of PCA component names
        - hpc_processed_df: Complete processed dataframe
    """
    # Load data
    df = load_dataset(filepath)
    if df is None:
        return None, None, None, None
    
    # Clean data
    clean_df = clean_dataset(df)
    
    # Data cleaning - filter out specific values
    if 'Scenario' in clean_df.columns:
        clean_df = clean_df[~clean_df['Scenario'].isin(['writeback:writeback_write_inode_start', '0', 0.0])]
    
    # Rename columns if needed
    if 'Cryptojacking' in clean_df.columns:
        clean_df.rename(columns={'Cryptojacking': 'Other'}, inplace=True)
    
    # Identify numeric columns for standardization
    numeric_cols = clean_df.select_dtypes(include=['float64', 'int']).columns.tolist()
    
    # Standardize numeric features
    scaler = StandardScaler()
    clean_df[numeric_cols] = scaler.fit_transform(clean_df[numeric_cols])
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(clean_df[numeric_cols])
    print(f"After PCA, features shape: {X_pca.shape}")
    print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # Create column names for PCA components
    pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    
    # Encode target label
    target_col = 'Scenario'
    if target_col in clean_df.columns:
        le = LabelEncoder()
        y = le.fit_transform(clean_df[target_col])
        print(f"Encoded {target_col} with {len(le.classes_)} unique values")
    else:
        print(f"Target column '{target_col}' not found in the dataset.")
        return None, None, None, None
    
    # Create final processed dataframe
    hpc_processed = pd.DataFrame(X_pca, columns=pca_columns)
    hpc_processed[target_col] = y
    
    return X_pca, y, pca_columns, hpc_processed

def save_processed_data(df, filepath):
    """
    Save processed dataframe to a CSV file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to save
    filepath : str
        Path where to save the csv file
    
    Returns:
    --------
    bool
        True if saved successfully, False otherwise
    """
    if df is None:
        return False
    
    try:
        df.to_csv(filepath, index=False)
        print(f"Processed dataset saved to {filepath}")
        return True
    except Exception as e:
        print(f"Error saving dataset: {e}")
        return False

def main():
    """
    Main function to demonstrate the usage of the preprocessing functions.
    """
    # File paths
    power_combined_path = "datasets/EVSE-B-PowerCombined.csv"
    hpc_kernel_events_path = "datasets/EVSE-B-HPC-Kernel-Events-cleaned.csv"
    
    # Output paths
    power_output_path = "datasets/PowerCombined_processed.csv"
    hpc_output_path = "datasets/HPC-Kernel-Events_processed.csv"
    
    print("Processing PowerCombined dataset...")
    X_power, y_power, power_features, power_processed = preprocess_power_combined(power_combined_path)
    if power_processed is not None:
        save_processed_data(power_processed, power_output_path)
    
    print("\nProcessing HPC-Kernel-Events dataset...")
    X_hpc, y_hpc, pca_cols, hpc_processed = preprocess_hpc_kernel_events(hpc_kernel_events_path)
    if hpc_processed is not None:
        save_processed_data(hpc_processed, hpc_output_path)
    
    print("\nPreprocessing complete!")

if __name__ == "__main__":
    main()