"""
Data Preprocessing Module

This module contains functions for preprocessing the PowerCombined and HPC-Kernel-Events
datasets. It handles loading raw data, cleaning, feature extraction, and normalization.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
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

def ensure_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
def load_raw_data():
    """
    Load raw data from CSV files.
    
    Returns:
    --------
    tuple
        (power_df, hpc_df) - DataFrames containing raw data
    """
    try:
        # Load PowerCombined dataset
        power_path = Path(RAW_DATA_DIR) / "Power.csv"
        power_df = pd.read_csv(power_path)
        logger.info(f"Loaded Power dataset with shape: {power_df.shape}")
        
        # Load HPC-Kernel-Events dataset
        hpc_path = Path(RAW_DATA_DIR) / "HPC.csv"
        hpc_df = pd.read_csv(hpc_path)
        logger.info(f"Loaded HPC dataset with shape: {hpc_df.shape}")
        
        return power_df, hpc_df
    
    except FileNotFoundError as e:
        logger.error(f"Error loading raw data: {e}")
        logger.info("Please make sure the data files exist in the data/raw directory")
        return None, None
    
    except Exception as e:
        logger.error(f"Unexpected error loading raw data: {e}")
        return None, None

def analyze_dataset(df, name):
    """
    Analyze dataset characteristics and print summary statistics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to analyze
    name : str
        Name of the dataset for logging
    """
    logger.info(f"\nAnalyzing {name} dataset:")
    logger.info(f"Shape: {df.shape}")
    
    # Basic info
    logger.info("\nColumn types:")
    for col, dtype in df.dtypes.items():
        logger.info(f"  {col}: {dtype}")
    
    # Missing values
    missing = df.isna().sum()
    if missing.sum() > 0:
        logger.info("\nMissing values:")
        for col, count in missing[missing > 0].items():
            logger.info(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
    else:
        logger.info("\nNo missing values found.")
    
    # Statistics for numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        logger.info("\nNumeric columns statistics:")
        stats = df[numeric_cols].describe().T
        stats['range'] = stats['max'] - stats['min']
        logger.info(f"\n{stats}")
    
    # Categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        logger.info("\nCategorical columns:")
        for col in cat_cols:
            unique_vals = df[col].nunique()
            logger.info(f"  {col}: {unique_vals} unique values")
            if unique_vals < 10:  # Only show value counts for columns with few unique values
                logger.info(f"  {col} value counts:\n{df[col].value_counts(normalize=True)}")
    
    return numeric_cols, cat_cols

def encode_labels(labels):
    """
    Encode labels according to the defined rules:
    0: 'Benign' or 'none'
    1: 'DoS'
    2: 'Recon'
    3: 'Other' or 'Cryptojacking' or 'host-attack'
    
    Parameters:
    -----------
    labels : pd.Series
        Original labels
    
    Returns:
    --------
    pd.Series
        Encoded numeric labels
    """
    encoded_labels = pd.Series(index=labels.index, dtype='int')
    
    for i, label in enumerate(labels):
        label_str = str(label).lower()
        
        if 'benign' in label_str or 'none' in label_str or 'normal' in label_str:
            encoded_labels[i] = 0
        elif 'dos' in label_str or 'denial' in label_str:
            encoded_labels[i] = 1
        elif 'recon' in label_str or 'reconnaissance' in label_str or 'probe' in label_str or 'scan' in label_str:
            encoded_labels[i] = 2
        elif ('other' in label_str or 'crypto' in label_str or 'mining' in label_str or 
              'host-attack' in label_str or 'host_attack' in label_str or 'attack' in label_str):
            encoded_labels[i] = 3
        else:
            # Default to "Other" category for unrecognized labels
            encoded_labels[i] = 3
            logger.warning(f"Unrecognized label '{label}' mapped to category 3 (Other)")
    
    # Log the mapping for verification
    mapping_counts = encoded_labels.value_counts().sort_index()
    logger.info(f"Label encoding distribution: {mapping_counts}")
    
    return encoded_labels

def preprocess_power_dataset(df, output_scaler=True):
    """
    Preprocess the PowerCombined dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw PowerCombined dataset
    output_scaler : bool, default=True
        Whether to output the scaler for future use
    
    Returns:
    --------
    tuple
        (processed_df, scaler) - Processed DataFrame and scaler
    """
    logger.info("\nPreprocessing PowerCombined dataset...")
    
    # Make a copy of the dataframe to avoid modifying the original
    df_processed = df.copy()
    
    # Check for target column - Usually 'Attack-Group' in this dataset
    if 'Attack-Group' in df_processed.columns:
        logger.info(f"Found target column: 'Attack-Group' with {df_processed['Attack-Group'].nunique()} unique values")
        # Extract target and keep it for later
        target = df_processed['Attack-Group']
        df_processed = df_processed.drop(columns=['Attack-Group'])
    else:
        # Try to identify the target column (likely the last column)
        logger.warning("'Attack-Group' column not found. Assuming the last column is the target.")
        target = df_processed.iloc[:, -1]
        df_processed = df_processed.iloc[:, :-1]
    
    # Handle categorical features
    cat_cols = df_processed.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        logger.info(f"One-hot encoding categorical column: {col}")
        # One-hot encode categorical columns
        df_dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
        df_processed = pd.concat([df_processed, df_dummies], axis=1)
        df_processed = df_processed.drop(columns=[col])
    
    # Handle missing values in numeric columns
    numeric_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
    if df_processed[numeric_cols].isna().sum().sum() > 0:
        logger.info("Imputing missing numeric values with median")
        imputer = SimpleImputer(strategy='median')
        df_processed[numeric_cols] = imputer.fit_transform(df_processed[numeric_cols])
    
    # Remove constant columns
    constant_cols = [col for col in df_processed.columns if df_processed[col].nunique() <= 1]
    if constant_cols:
        logger.info(f"Removing {len(constant_cols)} constant columns")
        df_processed = df_processed.drop(columns=constant_cols)
    
    # Apply normalization
    logger.info("Applying StandardScaler for normalization")
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_processed),
        columns=df_processed.columns,
        index=df_processed.index
    )
    
    # Save the scaler if requested
    if output_scaler:
        scaler_path = os.path.join(PROCESSED_DATA_DIR, 'power_scaler.joblib')
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")
    
    # Encode target labels according to defined rules
    encoded_labels = encode_labels(target)
    
    # Add target back to the processed dataframe
    df_scaled['Attack-Group'] = encoded_labels.values
    
    # Log the final shape
    logger.info(f"Processed PowerCombined dataset shape: {df_scaled.shape}")
    
    return df_scaled, scaler

def preprocess_hpc_dataset(df, output_scaler=True, apply_pca=True, n_components=0.95):
    """
    Preprocess the HPC-Kernel-Events dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw HPC-Kernel-Events dataset
    output_scaler : bool, default=True
        Whether to output the scaler for future use
    apply_pca : bool, default=True
        Whether to apply PCA for dimensionality reduction
    n_components : int or float, default=0.95
        Number of components or variance to keep in PCA
    
    Returns:
    --------
    tuple
        (processed_df, scaler, pca) - Processed DataFrame, scaler, and PCA model if applied
    """
    logger.info("\nPreprocessing HPC-Kernel-Events dataset...")
    
    # Make a copy of the dataframe to avoid modifying the original
    df_processed = df.copy()
    
    # Check for target column - Usually 'Scenario' in HPC dataset
    if 'Scenario' in df_processed.columns:
        logger.info(f"Found target column: 'Scenario' with {df_processed['Scenario'].nunique()} unique values")
        # Extract target and keep it for later
        target = df_processed['Scenario']
        df_processed = df_processed.drop(columns=['Scenario'])
    else:
        # Try to identify the target column (likely the last column)
        logger.warning("'Scenario' column not found. Assuming the last column is the target.")
        target = df_processed.iloc[:, -1]
        df_processed = df_processed.iloc[:, :-1]
    
    # Handle categorical features
    cat_cols = df_processed.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        logger.info(f"One-hot encoding categorical column: {col}")
        # One-hot encode categorical columns
        df_dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
        df_processed = pd.concat([df_processed, df_dummies], axis=1)
        df_processed = df_processed.drop(columns=[col])
    
    # Handle missing values in numeric columns
    numeric_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
    if df_processed[numeric_cols].isna().sum().sum() > 0:
        logger.info("Imputing missing numeric values with median")
        imputer = SimpleImputer(strategy='median')
        df_processed[numeric_cols] = imputer.fit_transform(df_processed[numeric_cols])
    
    # Remove constant columns
    constant_cols = [col for col in df_processed.columns if df_processed[col].nunique() <= 1]
    if constant_cols:
        logger.info(f"Removing {len(constant_cols)} constant columns")
        df_processed = df_processed.drop(columns=constant_cols)
    
    # Apply normalization
    logger.info("Applying StandardScaler for normalization")
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_processed),
        columns=df_processed.columns,
        index=df_processed.index
    )
    
    # Apply PCA if requested
    pca = None
    if apply_pca and df_scaled.shape[1] > 10:  # Only apply PCA if we have many features
        logger.info(f"Applying PCA with n_components={n_components}")
        pca = PCA(n_components=n_components, random_state=42)
        df_pca = pd.DataFrame(
            pca.fit_transform(df_scaled),
            index=df_scaled.index
        )
        
        # Name PCA components
        df_pca.columns = [f'PC{i+1}' for i in range(df_pca.shape[1])]
        
        logger.info(f"PCA reduced dimensions from {df_scaled.shape[1]} to {df_pca.shape[1]}")
        logger.info(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
        
        # Save PCA model if requested
        if output_scaler:
            pca_path = os.path.join(PROCESSED_DATA_DIR, 'hpc_pca.joblib')
            joblib.dump(pca, pca_path)
            logger.info(f"Saved PCA model to {pca_path}")
        
        # Use PCA results as the processed dataframe
        df_scaled = df_pca
    
    # Save the scaler if requested
    if output_scaler:
        scaler_path = os.path.join(PROCESSED_DATA_DIR, 'hpc_scaler.joblib')
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")
    
    # Encode target labels according to defined rules
    encoded_labels = encode_labels(target)
    
    # Add target back to the processed dataframe
    df_scaled['Scenario'] = encoded_labels.values
    
    # Log the final shape
    logger.info(f"Processed HPC-Kernel-Events dataset shape: {df_scaled.shape}")
    
    return df_scaled, scaler, pca

def generate_visualization(df, name, output_dir=None):
    """
    Generate visualizations for the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to visualize
    name : str
        Name of the dataset for saving files
    output_dir : str or Path, optional
        Directory to save visualizations
    """
    if output_dir is None:
        output_dir = os.path.join(PROCESSED_DATA_DIR, 'visualizations')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract features and target
    if name == 'Power' and 'Attack-Group' in df.columns:
        features = df.drop(columns=['Attack-Group'])
        target = df['Attack-Group']
    elif name == 'HPC' and 'Scenario' in df.columns:
        features = df.drop(columns=['Scenario'])
        target = df['Scenario'] 
    else:
        features = df.iloc[:, :-1]
        target = df.iloc[:, -1]
    
    # 1. Correlation heatmap (limited to first 20 features if there are many)
    plt.figure(figsize=(12, 10))
    feature_subset = features.iloc[:, :min(20, features.shape[1])]
    sns.heatmap(feature_subset.corr(), annot=False, cmap='coolwarm')
    plt.title(f'{name} Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{name}_correlation.png'))
    plt.close()
    
    # 2. PCA visualization (if applicable)
    if features.shape[1] > 2:
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=pd.factorize(target)[0], cmap='viridis', alpha=0.6)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title(f'{name} - PCA Visualization')
        
        # Add legend if there aren't too many classes
        unique_targets = target.unique()
        if len(unique_targets) <= 10:
            handles, labels = scatter.legend_elements()
            target_labels = {i: label for i, label in enumerate(unique_targets)}
            legend_labels = [target_labels[int(label.split('{')[1].split('}')[0])] for label in labels]
            plt.legend(handles, legend_labels, title="Classes")
        
        plt.savefig(os.path.join(output_dir, f'{name}_pca.png'))
        plt.close()
    
    # 3. Class distribution
    plt.figure(figsize=(10, 6))
    target_counts = target.value_counts().sort_values(ascending=False)
    bars = sns.barplot(x=target_counts.index.astype(str), y=target_counts.values)
    plt.title(f'{name} - Class Distribution')
    plt.xticks(rotation=90 if len(target_counts) > 5 else 45)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    
    # Add counts as text on bars
    for i, v in enumerate(target_counts.values):
        bars.text(i, v + 0.1, str(v), ha='center')
    
    plt.savefig(os.path.join(output_dir, f'{name}_class_distribution.png'))
    plt.close()
    
    logger.info(f"Saved visualizations for {name} dataset to {output_dir}")

def split_data(df, target_col, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split data into training, validation, and test sets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to split
    target_col : str
        Name of the target column
    test_size : float, default=0.2
        Proportion of data for testing
    val_size : float, default=0.2
        Proportion of data for validation
    random_state : int, default=42
        Random state for reproducibility
    
    Returns:
    --------
    dict
        Dictionary containing the data splits
    """
    # Extract features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # First split to separate test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split to separate validation set from training set
    # Adjust validation size to account for the reduced dataset size
    adjusted_val_size = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=adjusted_val_size, 
        random_state=random_state, stratify=y_train_val
    )
    
    logger.info(f"Data split:")
    logger.info(f"  Training set: {X_train.shape[0]} samples")
    logger.info(f"  Validation set: {X_val.shape[0]} samples")
    logger.info(f"  Test set: {X_test.shape[0]} samples")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }

def create_synthetic_samples(df_processed, target_col, minority_classes=None, multiplier=2):
    """
    Create synthetic samples for minority classes using SMOTE.
    
    Parameters:
    -----------
    df_processed : pd.DataFrame
        Processed DataFrame
    target_col : str
        Name of the target column
    minority_classes : list, optional
        List of minority classes to oversample
    multiplier : int, default=2
        Multiplier for minority class samples
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with synthetic samples added
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        logger.warning("imblearn not installed. Install with pip install imbalanced-learn")
        return df_processed
    
    logger.info("Creating synthetic samples for minority classes...")
    
    # Extract features and target
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]
    
    # Get class distribution
    class_counts = y.value_counts()
    logger.info(f"Original class distribution:\n{class_counts}")
    
    # Determine minority classes if not provided
    if minority_classes is None:
        # Use classes with count less than 50% of the mean as minority classes
        threshold = class_counts.mean() * 0.5
        minority_classes = class_counts[class_counts < threshold].index.tolist()
    
    if not minority_classes:
        logger.info("No minority classes identified. Skipping SMOTE.")
        return df_processed
    
    logger.info(f"Identified {len(minority_classes)} minority classes: {minority_classes}")
    
    # Set sampling strategy
    sampling_strategy = {
        cls: min(int(class_counts[cls] * multiplier), class_counts.max()) 
        for cls in minority_classes
    }
    
    # Apply SMOTE
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Create DataFrame with synthetic samples
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[target_col] = y_resampled
    
    logger.info(f"New class distribution:\n{df_resampled[target_col].value_counts()}")
    logger.info(f"Added {len(df_resampled) - len(df_processed)} synthetic samples")
    
    return df_resampled

def preprocess_datasets(balance_classes=True, apply_pca_to_hpc=True):
    """
    Preprocess both datasets.
    
    Parameters:
    -----------
    balance_classes : bool, default=True
        Whether to balance classes using synthetic samples
    apply_pca_to_hpc : bool, default=True
        Whether to apply PCA to the HPC dataset
    
    Returns:
    --------
    tuple
        (power_processed, hpc_processed) - Processed DataFrames
    """
    # Ensure directories exist
    ensure_directories()
    
    # Load raw data
    power_df, hpc_df = load_raw_data()
    if power_df is None or hpc_df is None:
        logger.error("Failed to load raw data. Exiting.")
        return None, None
    
    # Analyze datasets
    analyze_dataset(power_df, "Power")
    analyze_dataset(hpc_df, "HPC")
    
    # Preprocess datasets
    power_processed, power_scaler = preprocess_power_dataset(power_df)
    hpc_processed, hpc_scaler, hpc_pca = preprocess_hpc_dataset(hpc_df, apply_pca=apply_pca_to_hpc)
    
    # Balance classes if requested
    if balance_classes:
        logger.info("Balancing classes...")
        power_processed = create_synthetic_samples(power_processed, 'Attack-Group')
        hpc_processed = create_synthetic_samples(hpc_processed, 'Scenario')
    
    # Save processed datasets
    power_processed.to_csv(os.path.join(PROCESSED_DATA_DIR, 'Power_processed.csv'), index=False)
    hpc_processed.to_csv(os.path.join(PROCESSED_DATA_DIR, 'HPC_processed.csv'), index=False)
    
    logger.info("Preprocessing complete. Saved processed datasets.")
    
    return power_processed, hpc_processed

if __name__ == "__main__":
    # Run preprocessing pipeline
    preprocess_datasets(balance_classes=True, apply_pca_to_hpc=True)