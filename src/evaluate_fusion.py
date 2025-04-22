"""
Evaluate Feature Fusion Techniques

This script evaluates the effectiveness of different techniques for transforming and
combining heterogeneous datasets (Power and HPC-Kernel-Events) into the same latent space
while addressing concept drift.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import umap
import os
from pathlib import Path

# Import custom modules
from src.feature_fusion import feature_concatenation, weighted_feature_fusion, fusion_with_dim_reduction
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.utils import setup_logging, create_results_dir

# Set up logging
logger = setup_logging()

def load_processed_data():
    """
    Load processed Power and HPC datasets.
    
    Returns:
    --------
    tuple
        (power_df, hpc_df) - The processed dataframes
    """
    try:
        power_path = Path(PROCESSED_DATA_DIR) / "Power_processed.csv"
        hpc_path = Path(PROCESSED_DATA_DIR) / "HPC_processed.csv"
        
        power_df = pd.read_csv(power_path)
        hpc_df = pd.read_csv(hpc_path)
        
        logger.info(f"Loaded processed Power data with shape {power_df.shape}")
        logger.info(f"Loaded processed HPC data with shape {hpc_df.shape}")
        
        return power_df, hpc_df
    except FileNotFoundError as e:
        logger.error(f"Error loading processed data: {e}")
        logger.info("Trying to load raw data and process it...")
        
        # If processed data not found, load and process raw data
        from src.data_preprocessing import preprocess_power_combined, preprocess_hpc_kernel_events
        
        _, _, _, power_df = preprocess_power_combined()
        _, _, _, hpc_df = preprocess_hpc_kernel_events()
        
        if power_df is None or hpc_df is None:
            logger.error("Failed to load or process data")
            return None, None
            
        return power_df, hpc_df

def extract_features_labels(power_df, hpc_df):
    """
    Extract features and labels from the dataframes.
    
    Parameters:
    -----------
    power_df : pd.DataFrame
        Power dataset
    hpc_df : pd.DataFrame
        HPC dataset
        
    Returns:
    --------
    tuple
        (X_power, y_power, X_hpc, y_hpc)
    """
    # Extract Power features and labels
    if 'Attack-Group' in power_df.columns:
        y_power = power_df['Attack-Group']
        X_power = power_df.drop(columns=['Attack-Group'])
    else:
        # Assume the last column is the target
        y_power = power_df.iloc[:, -1]
        X_power = power_df.iloc[:, :-1]
    
    # Extract HPC features and labels
    if 'Scenario' in hpc_df.columns:
        y_hpc = hpc_df['Scenario']
        X_hpc = hpc_df.drop(columns=['Scenario'])
    else:
        # Assume the last column is the target
        y_hpc = hpc_df.iloc[:, -1]
        X_hpc = hpc_df.iloc[:, :-1]
    
    logger.info(f"Power features shape: {X_power.shape}, labels shape: {y_power.shape}")
    logger.info(f"HPC features shape: {X_hpc.shape}, labels shape: {y_hpc.shape}")
    
    return X_power, y_power, X_hpc, y_hpc

def evaluate_clustering(X, y_true, results_dir=None):
    """
    Evaluate clustering quality using various metrics.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y_true : array-like
        True labels
    results_dir : Path, optional
        Directory to save results
        
    Returns:
    --------
    dict
        Dictionary of clustering metrics
    """
    # Ensure we have numeric labels for evaluation
    unique_labels = np.unique(y_true)
    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    y_numeric = np.array([label_mapping[label] for label in y_true])
    
    # Run K-means with the same number of clusters as classes
    n_clusters = len(unique_labels)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X)
    
    # Calculate clustering metrics
    metrics = {}
    metrics['silhouette'] = silhouette_score(X, y_pred)
    metrics['adjusted_rand'] = adjusted_rand_score(y_numeric, y_pred)
    metrics['davies_bouldin'] = davies_bouldin_score(X, y_pred)
    metrics['calinski_harabasz'] = calinski_harabasz_score(X, y_pred)
    
    logger.info(f"Clustering metrics: {metrics}")
    
    # Visualize clustering results if dimensions allow
    if X.shape[1] > 2:
        # Use t-SNE for visualization
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
        plt.colorbar(label='Cluster')
        plt.title('t-SNE Visualization of K-means Clusters')
        
        if results_dir:
            plt.savefig(results_dir / 'plots' / 'kmeans_clusters.png', dpi=300)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_numeric, cmap='tab10', alpha=0.7)
        plt.colorbar(label='True Class')
        plt.title('t-SNE Visualization of True Classes')
        
        if results_dir:
            plt.savefig(results_dir / 'plots' / 'true_classes.png', dpi=300)
    
    return metrics

def evaluate_classification(X_fused, y, results_dir=None):
    """
    Evaluate classification performance on fused features.
    
    Parameters:
    -----------
    X_fused : array-like
        Fused feature matrix
    y : array-like
        Target labels
    results_dir : Path, optional
        Directory to save results
        
    Returns:
    --------
    dict
        Classification metrics
    """
    # Convert labels to numeric if they aren't already
    if not np.issubdtype(np.array(y).dtype, np.number):
        unique_labels = np.unique(y)
        label_mapping = {label: i for i, label in enumerate(unique_labels)}
        y = np.array([label_mapping[label] for label in y])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_fused, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    logger.info(f"Classification accuracy: {accuracy:.4f}")
    logger.info(f"Classification F1-score: {f1:.4f}")
    
    # Save detailed report if results_dir is provided
    if results_dir:
        report = classification_report(y_test, y_pred)
        with open(results_dir / 'classification_report.txt', 'w') as f:
            f.write(report)
    
    return {'accuracy': accuracy, 'f1_score': f1}

def visualize_latent_space(X_fused, y, method_name, results_dir=None):
    """
    Visualize the latent space using dimensionality reduction techniques.
    
    Parameters:
    -----------
    X_fused : array-like
        Fused feature matrix
    y : array-like
        Labels
    method_name : str
        Name of the fusion method
    results_dir : Path, optional
        Directory to save visualizations
    """
    # Get unique labels for coloring
    unique_labels = np.unique(y)
    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    y_numeric = np.array([label_mapping[label] for label in y])
    
    # Create a directory for visualizations
    if results_dir:
        vis_dir = results_dir / 'plots' / 'latent_space'
        os.makedirs(vis_dir, exist_ok=True)
    
    # Visualize using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_fused)
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_numeric, cmap='tab10', alpha=0.7)
    
    # Add a legend
    if len(unique_labels) <= 10:  # Only show legend if not too many classes
        handles = scatter.legend_elements()[0]
        labels = [list(label_mapping.keys())[list(label_mapping.values()).index(i)] 
                 for i in range(len(unique_labels))]
        plt.legend(handles, labels, title="Classes")
    
    plt.title(f't-SNE Visualization of {method_name}')
    
    if results_dir:
        plt.savefig(vis_dir / f'tsne_{method_name}.png', dpi=300)
    plt.close()
    
    # Visualize using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_fused)
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_numeric, cmap='tab10', alpha=0.7)
    
    # Add a legend
    if len(unique_labels) <= 10:
        handles = scatter.legend_elements()[0]
        labels = [list(label_mapping.keys())[list(label_mapping.values()).index(i)] 
                 for i in range(len(unique_labels))]
        plt.legend(handles, labels, title="Classes")
    
    plt.title(f'PCA Visualization of {method_name}')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    
    if results_dir:
        plt.savefig(vis_dir / f'pca_{method_name}.png', dpi=300)
    plt.close()
    
    # Visualize using UMAP if available
    try:
        reducer = umap.UMAP(random_state=42)
        X_umap = reducer.fit_transform(X_fused)
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y_numeric, cmap='tab10', alpha=0.7)
        
        # Add a legend
        if len(unique_labels) <= 10:
            handles = scatter.legend_elements()[0]
            labels = [list(label_mapping.keys())[list(label_mapping.values()).index(i)] 
                     for i in range(len(unique_labels))]
            plt.legend(handles, labels, title="Classes")
        
        plt.title(f'UMAP Visualization of {method_name}')
        
        if results_dir:
            plt.savefig(vis_dir / f'umap_{method_name}.png', dpi=300)
        plt.close()
    except ImportError:
        logger.warning("UMAP not installed. Skipping UMAP visualization.")

def detect_concept_drift(X_power, y_power, X_hpc, y_hpc, results_dir=None):
    """
    Detect and visualize concept drift between the datasets.
    
    Parameters:
    -----------
    X_power : array-like
        Power features
    y_power : array-like
        Power labels
    X_hpc : array-like
        HPC features
    y_hpc : array-like
        HPC labels
    results_dir : Path, optional
        Directory to save results
        
    Returns:
    --------
    float
        Drift score (higher indicates more drift)
    """
    # We'll use a classifier to detect concept drift
    # If we can easily distinguish between datasets, there's significant drift
    
    # Create a combined dataset with source labels
    X_power_sample = X_power.sample(min(1000, len(X_power)), random_state=42).values
    X_hpc_sample = X_hpc.sample(min(1000, len(X_hpc)), random_state=42).values
    
    # Ensure same dimensionality for concept drift detection
    if X_power_sample.shape[1] != X_hpc_sample.shape[1]:
        # Use PCA to reduce to the same dimensionality
        min_dim = min(X_power_sample.shape[1], X_hpc_sample.shape[1])
        pca_power = PCA(n_components=min_dim)
        pca_hpc = PCA(n_components=min_dim)
        
        X_power_sample = pca_power.fit_transform(X_power_sample)
        X_hpc_sample = pca_hpc.fit_transform(X_hpc_sample)
    
    X_combined = np.vstack([X_power_sample, X_hpc_sample])
    y_source = np.array(['Power'] * len(X_power_sample) + ['HPC'] * len(X_hpc_sample))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_source, test_size=0.3, random_state=42, stratify=y_source
    )
    
    # Train a classifier to distinguish between sources
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate - higher accuracy means more concept drift
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    drift_score = accuracy  # Higher means easier to distinguish = more drift
    
    logger.info(f"Concept drift score: {drift_score:.4f}")
    
    if results_dir:
        with open(results_dir / 'concept_drift.txt', 'w') as f:
            f.write(f"Concept Drift Score: {drift_score:.4f}\n")
            f.write("Higher score indicates more drift between datasets.\n")
            f.write("A score near 0.5 would indicate minimal drift (random guessing).\n")
            f.write("A score near 1.0 indicates significant drift (easy to distinguish).\n\n")
            f.write(classification_report(y_test, y_pred))
    
    # Visualize datasets in common space using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_combined)
    
    plt.figure(figsize=(12, 10))
    source_map = {'Power': 0, 'HPC': 1}
    y_numeric = np.array([source_map[source] for source in y_source])
    
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_numeric, cmap='coolwarm', alpha=0.7)
    plt.colorbar(label='Dataset Source')
    plt.title('t-SNE Visualization of Dataset Sources (Concept Drift)')
    plt.legend(handles=scatter.legend_elements()[0], labels=['Power', 'HPC'])
    
    if results_dir:
        plt.savefig(results_dir / 'plots' / 'concept_drift.png', dpi=300)
    plt.close()
    
    return drift_score

def domain_adaptation_fusion(X_power, y_power, X_hpc, y_hpc, method='coral', lam=1.0):
    """
    Apply domain adaptation to align feature spaces before fusion.
    
    Parameters:
    -----------
    X_power : array-like
        Power features
    y_power : array-like
        Power labels
    X_hpc : array-like
        HPC features
    y_hpc : array-like
        HPC labels
    method : str, default='coral'
        Domain adaptation method ('coral', 'mmd')
    lam : float, default=1.0
        Regularization parameter
        
    Returns:
    --------
    tuple
        (X_fused, y_fused) - Fused features and corresponding labels
    """
    # Ensure consistent dimensionality
    if X_power.shape[1] != X_hpc.shape[1]:
        min_dim = min(X_power.shape[1], X_hpc.shape[1])
        pca_power = PCA(n_components=min_dim)
        pca_hpc = PCA(n_components=min_dim)
        
        X_power = pca_power.fit_transform(X_power)
        X_hpc = pca_hpc.fit_transform(X_hpc)
    
    # Apply Correlation Alignment (CORAL)
    if method == 'coral':
        # Calculate covariance matrices
        power_cov = np.cov(X_power, rowvar=False) + np.eye(X_power.shape[1]) * 1e-4
        hpc_cov = np.cov(X_hpc, rowvar=False) + np.eye(X_hpc.shape[1]) * 1e-4
        
        # Whitening power data
        power_cov_sqrt_inv = np.linalg.inv(scipy.linalg.sqrtm(power_cov))
        X_power_white = X_power @ power_cov_sqrt_inv
        
        # Re-coloring with HPC covariance
        hpc_cov_sqrt = scipy.linalg.sqrtm(hpc_cov)
        X_power_aligned = X_power_white @ hpc_cov_sqrt
        
        # Combine aligned data
        X_fused = np.vstack([X_power_aligned, X_hpc])
        y_fused = np.concatenate([y_power, y_hpc])
    
    else:
        # If method not implemented, just concatenate
        logger.warning(f"Domain adaptation method {method} not implemented. Using simple concatenation.")
        X_fused = np.vstack([X_power, X_hpc])
        y_fused = np.concatenate([y_power, y_hpc])
    
    return X_fused, y_fused

def adversarial_domain_adaptation(X_power, y_power, X_hpc, y_hpc, n_epochs=100):
    """
    Perform adversarial domain adaptation between Power and HPC datasets.
    This is a simplified version - in practice, you'd use a deep learning framework.
    
    Parameters:
    -----------
    X_power : array-like
        Power features
    y_power : array-like
        Power labels
    X_hpc : array-like
        HPC features
    y_hpc : array-like
        HPC labels
    n_epochs : int, default=100
        Number of training epochs
        
    Returns:
    --------
    tuple
        (X_fused, y_fused) - Fused features and corresponding labels
    """
    logger.info("Note: This is a simplified adversarial adaptation. For best results, implement with PyTorch/TensorFlow")
    
    # Make sure both datasets have the same dimensionality
    if X_power.shape[1] != X_hpc.shape[1]:
        common_dim = min(X_power.shape[1], X_hpc.shape[1])
        pca = PCA(n_components=common_dim)
        X_power = pca.fit_transform(X_power)
        X_hpc = pca.fit_transform(X_hpc)
    
    # Initialize feature embeddings (this would normally be a neural network)
    X_power_aligned = X_power.copy()
    X_hpc_aligned = X_hpc.copy()
    
    # In a real implementation, you would:
    # 1. Train a feature extractor to minimize task loss
    # 2. Train a domain classifier to distinguish between domains
    # 3. Update feature extractor to confuse domain classifier (adversarial)
    
    # For this simplified version, we'll just use a weighted average to move domains closer
    alpha = 0.3  # Controls adaptation strength
    
    # Compute domain means
    power_mean = np.mean(X_power, axis=0)
    hpc_mean = np.mean(X_hpc, axis=0)
    
    # Align domains by nudging toward common center
    common_center = (power_mean + hpc_mean) / 2
    X_power_aligned = X_power + alpha * (common_center - power_mean)
    X_hpc_aligned = X_hpc + alpha * (common_center - hpc_mean)
    
    # Combine the aligned datasets
    X_fused = np.vstack([X_power_aligned, X_hpc_aligned])
    y_fused = np.concatenate([y_power, y_hpc])
    
    return X_fused, y_fused

def evaluate_fusion_methods(X_power, y_power, X_hpc, y_hpc, results_dir=None):
    """
    Evaluate different fusion methods and compare their performance.
    
    Parameters:
    -----------
    X_power : array-like
        Power features
    y_power : array-like
        Power labels
    X_hpc : array-like
        HPC features
    y_hpc : array-like
        HPC labels
    results_dir : Path, optional
        Directory to save results
        
    Returns:
    --------
    dict
        Results for each fusion method
    """
    fusion_methods = {
        'simple_concat': lambda: feature_concatenation(X_power.values, X_hpc.values),
        'weighted_fusion': lambda: weighted_feature_fusion(X_power.values, X_hpc.values, power_weight=0.5)[0],
        'pca_fusion': lambda: fusion_with_dim_reduction(X_power.values, X_hpc.values, n_components=0.95),
        'adversarial': lambda: adversarial_domain_adaptation(X_power.values, y_power.values, X_hpc.values, y_hpc.values)
    }
    
    results = {}
    
    # Create a comparison table
    comparison = pd.DataFrame(columns=['Method', 'Classification Accuracy', 'F1 Score', 
                                       'Silhouette Score', 'Drift Score'])
    
    drift_score = detect_concept_drift(X_power, y_power, X_hpc, y_hpc, results_dir)
    
    for method_name, fusion_func in fusion_methods.items():
        logger.info(f"Evaluating {method_name}...")
        
        try:
            if method_name == 'adversarial':
                X_fused, y_fused = fusion_func()
            else:
                X_fused = fusion_func()
                y_fused = y_power  # Use one of the label sets
            
            # Calculate clustering metrics
            clustering_metrics = evaluate_clustering(X_fused, y_fused, results_dir)
            
            # Calculate classification metrics
            classification_metrics = evaluate_classification(X_fused, y_fused, results_dir)
            
            # Visualize the latent space
            visualize_latent_space(X_fused, y_fused, method_name, results_dir)
            
            results[method_name] = {
                'clustering': clustering_metrics,
                'classification': classification_metrics
            }
            
            # Add to comparison table
            comparison.loc[len(comparison)] = [
                method_name,
                classification_metrics['accuracy'],
                classification_metrics['f1_score'],
                clustering_metrics['silhouette'],
                drift_score
            ]
            
        except Exception as e:
            logger.error(f"Error evaluating {method_name}: {e}")
    
    # Save comparison table
    if results_dir:
        comparison.to_csv(results_dir / 'fusion_comparison.csv', index=False)
        
        # Plot comparison
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Method', y='Classification Accuracy', data=comparison)
        plt.title('Classification Accuracy by Fusion Method')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(results_dir / 'plots' / 'accuracy_comparison.png', dpi=300)
        plt.close()
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Method', y='Silhouette Score', data=comparison)
        plt.title('Clustering Quality by Fusion Method')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(results_dir / 'plots' / 'silhouette_comparison.png', dpi=300)
        plt.close()
    
    return results

def main():
    """Main function to run the evaluation."""
    # Create results directory
    results_dir = create_results_dir()
    logger.info(f"Results will be saved to {results_dir}")
    
    # Load data
    power_df, hpc_df = load_processed_data()
    if power_df is None or hpc_df is None:
        logger.error("Failed to load data. Exiting.")
        return
    
    # Extract features and labels
    X_power, y_power, X_hpc, y_hpc = extract_features_labels(power_df, hpc_df)
    
    # Detect concept drift
    logger.info("Detecting concept drift between datasets...")
    drift_score = detect_concept_drift(X_power, y_power, X_hpc, y_hpc, results_dir)
    
    # Evaluate fusion methods
    logger.info("Evaluating different fusion methods...")
    fusion_results = evaluate_fusion_methods(X_power, y_power, X_hpc, y_hpc, results_dir)
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    # Try to import scipy for some advanced transformations
    try:
        import scipy
    except ImportError:
        logger.warning("scipy not installed. Some advanced transformations may not work.")
    
    # Try to import umap for visualization
    try:
        import umap
    except ImportError:
        logger.warning("umap-learn not installed. UMAP visualizations will be skipped.")
    
    main()