"""
Feature Fusion Module

This module contains functions for combining heterogeneous data features
from the PowerCombined and HPC-Kernel-Events datasets, with specific
techniques to transform them into a common latent space while addressing
concept drift issues.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
from sklearn.cross_decomposition import CCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from src.config import PROCESSED_DATA_DIR

# Adding necessary imports for neural encoder models
# Make PyTorch imports optional with fallback mechanisms
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Neural network-based fusion methods will fall back to PCA-based methods.")

from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        print(f"Loaded Power dataset with shape: {power_df.shape}")
        print(f"Loaded HPC dataset with shape: {hpc_df.shape}")
        return power_df, hpc_df
    except FileNotFoundError as e:
        print(f"Error loading processed data: {e}")
        print("Make sure preprocessing has been run first.")
        return None, None

def load_preprocessing_models():
    """
    Load saved preprocessing models.
    
    Returns:
    --------
    dict
        Dictionary containing the loaded models
    """
    models = {}
    model_files = {
        'power_scaler': 'power_scaler.joblib',
        'hpc_scaler': 'hpc_scaler.joblib',
        'hpc_pca': 'hpc_pca.joblib'
    }
    
    for name, filename in model_files.items():
        path = Path(PROCESSED_DATA_DIR) / filename
        if path.exists():
            try:
                models[name] = joblib.load(path)
                print(f"Loaded {name} from {path}")
            except Exception as e:
                print(f"Error loading {name}: {e}")
        else:
            print(f"Model file not found: {path}")
    
    return models

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
    
    # For the sake of demonstration, implement simple alignment by label matching
    # Find common labels between datasets
    if 'Attack-Group' in power_df.columns and 'Scenario' in hpc_df.columns:
        power_labels = power_df['Attack-Group'].unique()
        hpc_labels = hpc_df['Scenario'].unique()
        
        # Check if there's any commonality in labels (may not be exact matches)
        # Here using a simple substring matching approach
        common_labels = []
        for p_label in power_labels:
            for h_label in hpc_labels:
                if str(p_label).lower() in str(h_label).lower() or str(h_label).lower() in str(p_label).lower():
                    common_labels.append((p_label, h_label))
        
        if common_labels:
            print(f"Found potential common labels between datasets: {common_labels}")
            # You could filter datasets based on these common labels
    
    # In practical applications, alignment might be based on:
    # 1. Temporal alignment if timestamps are available
    # 2. Exact label matching if labels are consistent
    # 3. Manual mapping between labels
    
    # For now, we'll just return the original dataframes
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
    
    # Remove is_synthetic column if present
    if 'is_synthetic' in X_power.columns:
        X_power = X_power.drop(columns=['is_synthetic'])
    if 'is_synthetic' in X_hpc.columns:
        X_hpc = X_hpc.drop(columns=['is_synthetic'])
    
    print(f"Extracted Power features: {X_power.shape}, labels: {y_power.shape}")
    print(f"Extracted HPC features: {X_hpc.shape}, labels: {y_hpc.shape}")
    
    return X_power, y_power, X_hpc, y_hpc

# Neural network model to encode features into a common latent space
class EncoderModel(nn.Module):
    """
    Neural network encoder model to transform features into a common latent space.
    
    This model maps input features to a latent space of specified dimension
    using a multi-layer perceptron architecture with domain adaptation capabilities.
    """
    
    def __init__(self, input_dim, latent_dim=32, hidden_dims=[128, 64], dropout_rate=0.2):
        """
        Initialize the encoder model.
        
        Parameters:
        -----------
        input_dim : int
            Dimensionality of the input features
        latent_dim : int, default=32
            Dimensionality of the latent space
        hidden_dims : list, default=[128, 64]
            Dimensions of hidden layers
        dropout_rate : float, default=0.2
            Rate of dropout for regularization
        """
        super(EncoderModel, self).__init__()
        
        # Build layers dynamically
        layers = []
        
        # Input layer
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer (latent space)
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        # Combine layers
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
        
        Returns:
        --------
        torch.Tensor
            Encoded representation in the latent space
        """
        return self.encoder(x)

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer that reverses gradients during backpropagation.
    Used for domain adaptation.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversalLayer(nn.Module):
    """
    Wrapper module for the gradient reversal function.
    """
    def __init__(self):
        super(GradientReversalLayer, self).__init__()
    
    def forward(self, x, alpha):
        return GradientReversalFunction.apply(x, alpha)

class DomainAdversarialEncoder(nn.Module):
    """
    Domain adversarial neural network encoder for feature fusion.
    Implements a feature extractor, class classifier, and domain discriminator.
    
    Parameters:
    -----------
    input_dim : int
        Dimension of input features
    latent_dim : int
        Dimension of the shared latent space
    hidden_dim : int, default=128
        Dimension of hidden layers
    num_classes : int, default=2
        Number of output classes
    """
    def __init__(self, input_dim, latent_dim, hidden_dim=128, num_classes=2):
        super(DomainAdversarialEncoder, self).__init__()
        
        # Feature extractor network (encoder)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, latent_dim),
        )
        
        # Class classifier (for supervised learning)
        self.class_classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.gradient_reversal = GradientReversalLayer()
        
    def forward(self, x, alpha=1.0):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
        alpha : float, default=1.0
            Gradient reversal scaling factor
            
        Returns:
        --------
        features : torch.Tensor
            Extracted features in the shared latent space
        class_outputs : torch.Tensor
            Class predictions
        domain_outputs : torch.Tensor
            Domain predictions (0=target, 1=source)
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Class prediction
        class_outputs = self.class_classifier(features)
        
        # Domain prediction with gradient reversal
        reversed_features = self.gradient_reversal(features, alpha)
        domain_outputs = self.domain_classifier(reversed_features)
        
        return features, class_outputs, domain_outputs
    
    def encode(self, x):
        """
        Encode inputs to the shared latent space.
        
        Parameters:
        -----------
        x : numpy.ndarray or torch.Tensor
            Input data
            
        Returns:
        --------
        numpy.ndarray
            Encoded features
        """
        self.eval()
        
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        
        device = next(self.parameters()).device
        x = x.to(device)
        
        with torch.no_grad():
            features = self.feature_extractor(x)
            
        return features.cpu().numpy()

def train_encoder(X, y, input_dim, latent_dim=32, batch_size=64, epochs=100, 
                  learning_rate=0.001, validation_split=0.2, device=None):
    """
    Train an encoder model on the dataset.
    
    Parameters:
    -----------
    X : np.ndarray or pd.DataFrame
        Input features
    y : np.ndarray or pd.Series
        Target labels
    input_dim : int
        Dimensionality of the input features
    latent_dim : int, default=32
        Dimensionality of the latent space
    batch_size : int, default=64
        Batch size for training
    epochs : int, default=100
        Number of epochs to train
    learning_rate : float, default=0.001
        Learning rate for optimizer
    validation_split : float, default=0.2
        Proportion of data to use for validation
    device : str, default=None
        Device to use for training ('cuda', 'cpu', or None for auto-detection)
    
    Returns:
    --------
    EncoderModel
        Trained encoder model
    """
    # Convert inputs to numpy arrays if they are not
    if isinstance(X, pd.DataFrame) or isinstance(y, pd.Series):
        X = X.values
        y = y.values
    
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=42, stratify=y
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = EncoderModel(input_dim, latent_dim).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Add classification head for training
    classifier = nn.Linear(latent_dim, len(np.unique(y))).to(device)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 10  # Early stopping patience
    counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            encoded = model(inputs)
            outputs = classifier(encoded)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss /= total
        train_acc = correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                # Forward pass
                encoded = model(inputs)
                outputs = classifier(encoded)
                
                # Compute loss
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= total
        val_acc = correct / total
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Check if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
        
        # Early stopping
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    
    return model

def train_adversarial_encoder(source_data, target_data, source_labels, 
                         input_dim, latent_dim=32, num_epochs=100,
                         batch_size=64, lr=0.001, weight_decay=1e-5,
                         early_stopping_patience=10, alpha=1.0):
    """
    Train a domain adversarial encoder to transform features from both source and target
    domains into a common latent space with domain adaptation capabilities.
    
    Parameters:
    -----------
    source_data : numpy.ndarray
        Source domain features
    target_data : numpy.ndarray
        Target domain features
    source_labels : numpy.ndarray
        Labels for source domain (used in task classifier)
    input_dim : int
        Dimensionality of input features
    latent_dim : int, default=32
        Dimensionality of latent space
    num_epochs : int, default=100
        Number of training epochs
    batch_size : int, default=64
        Batch size for training
    lr : float, default=0.001
        Learning rate
    weight_decay : float, default=1e-5
        L2 regularization strength
    early_stopping_patience : int, default=10
        Number of epochs to wait before early stopping
    alpha : float, default=1.0
        Domain confusion strength (gradient reversal layer parameter)
    
    Returns:
    --------
    model : DomainAdversarialEncoder
        Trained domain adversarial encoder model
    history : dict
        Training history
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    # Convert data to PyTorch tensors
    source_data_tensor = torch.FloatTensor(source_data)
    source_labels_tensor = torch.LongTensor(source_labels)
    target_data_tensor = torch.FloatTensor(target_data)
    
    # Create domain labels: 0 for source, 1 for target
    source_domain_labels = torch.zeros(source_data.shape[0])
    target_domain_labels = torch.ones(target_data.shape[0])
    
    # Split source data into train and validation sets
    source_train_data, source_val_data, source_train_labels, source_val_labels, \
    source_train_domains, source_val_domains = train_test_split(
        source_data_tensor, source_labels_tensor, source_domain_labels,
        test_size=0.2, random_state=42
    )
    
    # Create data loaders
    # Training data from both domains
    source_train_dataset = TensorDataset(source_train_data, source_train_labels, source_train_domains)
    source_train_loader = DataLoader(source_train_dataset, batch_size=batch_size, shuffle=True)
    
    # Validation data (source domain)
    source_val_dataset = TensorDataset(source_val_data, source_val_labels, source_val_domains)
    source_val_loader = DataLoader(source_val_dataset, batch_size=batch_size, shuffle=False)
    
    # Target domain data for domain adaptation
    target_dataset = TensorDataset(target_data_tensor, 
                                  torch.zeros(target_data.shape[0]).long(), # dummy labels
                                  target_domain_labels)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model, optimizer and loss functions
    model = DomainAdversarialEncoder(input_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    classifier_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCELoss()
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Initialize history dictionary to track metrics
    history = {
        'train_loss': [], 'val_loss': [],
        'train_class_acc': [], 'val_class_acc': [],
        'train_domain_acc': [], 'val_domain_acc': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_class_correct = 0
        train_class_total = 0
        train_domain_correct = 0
        train_domain_total = 0
        
        # Create an iterator for target data to sync with source data batches
        target_iter = iter(target_loader)
        
        # Gradually increase alpha from 0 to the specified value
        current_alpha = min(alpha, alpha * (epoch / (num_epochs * 0.1)))
        
        for source_batch in source_train_loader:
            # Source data forward pass
            source_inputs, source_class_labels, source_domain_labels = source_batch
            source_features, source_class_outputs, source_domain_outputs = model(source_inputs, current_alpha)
            
            # Try to get a batch from target data, wrap around if necessary
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch = next(target_iter)
                
            target_inputs, _, target_domain_labels = target_batch
            target_features, _, target_domain_outputs = model(target_inputs, current_alpha)
            
            # Calculate losses
            # Classification loss (only on source domain)
            classification_loss = classifier_criterion(source_class_outputs, source_class_labels)
            
            # Domain loss (on both domains)
            source_domain_loss = domain_criterion(source_domain_outputs.squeeze(), source_domain_labels.float())
            target_domain_loss = domain_criterion(target_domain_outputs.squeeze(), target_domain_labels.float())
            domain_loss = source_domain_loss + target_domain_loss
            
            # Total loss with weighted domain loss
            total_loss = classification_loss + domain_loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            
            # Calculate classification accuracy
            _, predicted_classes = torch.max(source_class_outputs, 1)
            train_class_correct += (predicted_classes == source_class_labels).sum().item()
            train_class_total += source_class_labels.size(0)
            
            # Calculate domain classification accuracy
            predicted_source_domains = (source_domain_outputs > 0.5).float().squeeze()
            predicted_target_domains = (target_domain_outputs > 0.5).float().squeeze()
            
            train_domain_correct += (predicted_source_domains == source_domain_labels).sum().item()
            train_domain_correct += (predicted_target_domains == target_domain_labels).sum().item()
            train_domain_total += source_domain_labels.size(0) + target_domain_labels.size(0)
        
        # Evaluation on validation set
        model.eval()
        val_loss = 0.0
        val_class_correct = 0
        val_class_total = 0
        val_domain_correct = 0
        val_domain_total = 0
        
        with torch.no_grad():
            for val_batch in source_val_loader:
                val_inputs, val_class_labels, val_domain_labels = val_batch
                val_features, val_class_outputs, val_domain_outputs = model(val_inputs)
                
                val_class_loss = classifier_criterion(val_class_outputs, val_class_labels)
                val_domain_loss = domain_criterion(val_domain_outputs.squeeze(), val_domain_labels.float())
                val_batch_loss = val_class_loss + val_domain_loss
                
                val_loss += val_batch_loss.item()
                
                _, predicted_classes = torch.max(val_class_outputs, 1)
                val_class_correct += (predicted_classes == val_class_labels).sum().item()
                val_class_total += val_class_labels.size(0)
                
                predicted_domains = (val_domain_outputs > 0.5).float().squeeze()
                val_domain_correct += (predicted_domains == val_domain_labels).sum().item()
                val_domain_total += val_domain_labels.size(0)
                
            # Check domain accuracy on target data
            for target_batch in target_loader:
                target_inputs, _, target_domain_labels = target_batch
                _, _, target_domain_outputs = model(target_inputs)
                
                predicted_domains = (target_domain_outputs > 0.5).float().squeeze()
                val_domain_correct += (predicted_domains == target_domain_labels).sum().item()
                val_domain_total += target_domain_labels.size(0)
        
        # Calculate epoch metrics
        epoch_train_loss = train_loss / len(source_train_loader)
        epoch_val_loss = val_loss / len(source_val_loader)
        epoch_train_class_acc = train_class_correct / train_class_total
        epoch_val_class_acc = val_class_correct / val_class_total
        epoch_train_domain_acc = train_domain_correct / train_domain_total
        epoch_val_domain_acc = val_domain_correct / val_domain_total
        
        # Update history
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_class_acc'].append(epoch_train_class_acc)
        history['val_class_acc'].append(epoch_val_class_acc)
        history['train_domain_acc'].append(epoch_train_domain_acc)
        history['val_domain_acc'].append(epoch_val_domain_acc)
        
        # Early stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Print progress
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch}/{num_epochs}:")
            print(f"  Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
            print(f"  Train Class Acc: {epoch_train_class_acc:.4f}, Val Class Acc: {epoch_val_class_acc:.4f}")
            print(f"  Train Domain Acc: {epoch_train_domain_acc:.4f}, Val Domain Acc: {epoch_val_domain_acc:.4f}")
    
    return model, history

def train_domain_adversarial_encoder(X_source, y_source, X_target, latent_dim=32, 
                                 num_epochs=100, batch_size=64, lr=0.001,
                                 weight_decay=1e-5, alpha=1.0, device=None):
    """
    Train a domain adversarial encoder for cross-domain feature fusion.
    
    This function takes source and target domain data and trains a domain adversarial
    neural network that maps both domains to a common latent space where domain
    discrimination is difficult, while maintaining class discrimination ability.
    
    Parameters:
    -----------
    X_source : numpy.ndarray
        Source domain features
    y_source : numpy.ndarray
        Source domain labels
    X_target : numpy.ndarray
        Target domain features
    latent_dim : int, default=32
        Dimensionality of the latent space
    num_epochs : int, default=100
        Number of training epochs
    batch_size : int, default=64
        Batch size for training
    lr : float, default=0.001
        Learning rate
    weight_decay : float, default=1e-5
        L2 regularization strength
    alpha : float, default=1.0
        Domain confusion strength (gradient reversal layer parameter)
    device : str, default=None
        Device to use for training ('cuda', 'cpu', or None for auto-detection)
    
    Returns:
    --------
    model : DomainAdversarialEncoder
        Trained domain adversarial encoder model
    history : dict
        Training history dictionary containing metrics over epochs
    """
    if not TORCH_AVAILABLE:
        print("PyTorch is not available. Cannot perform domain adversarial fusion.")
        return None, None
    
    # Convert inputs to numpy arrays if they are not
    if isinstance(X_source, pd.DataFrame):
        X_source = X_source.values
    if isinstance(y_source, pd.Series):
        y_source = y_source.values
    if isinstance(X_target, pd.DataFrame):
        X_target = X_target.values
    
    # Check for NaN values and replace if necessary
    if np.isnan(X_source).any():
        print("Warning: Source data contains NaN values. Replacing with zeros.")
        X_source = np.nan_to_num(X_source)
    if np.isnan(X_target).any():
        print("Warning: Target data contains NaN values. Replacing with zeros.")
        X_target = np.nan_to_num(X_target)
    
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Get input dimensionality 
    input_dim = X_source.shape[1]
    
    # Convert labels to numeric if they're not already
    from sklearn.preprocessing import LabelEncoder
    if not np.issubdtype(y_source.dtype, np.number):
        label_encoder = LabelEncoder()
        y_source = label_encoder.fit_transform(y_source)
        print(f"Label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    # Get number of classes
    num_classes = len(np.unique(y_source))
    
    # Initialize the model
    model = DomainAdversarialEncoder(input_dim, latent_dim, 
                                    hidden_dim=128, 
                                    num_classes=num_classes).to(device)
    
    # Train the model
    print(f"Training domain adversarial encoder (input_dim={input_dim}, latent_dim={latent_dim}, num_classes={num_classes})")
    print(f"Source data shape: {X_source.shape}, Target data shape: {X_target.shape}")
    
    history = train_adversarial_encoder(
        X_source, X_target, y_source,
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        alpha=alpha
    )
    
    return model, history

def encoder_feature_fusion(X_power, y_power, X_hpc, y_hpc, latent_dim=32, save_models=True):
    """
    Use neural network encoders to transform both datasets into a common latent space.
    
    Parameters:
    -----------
    X_power : pd.DataFrame or np.ndarray
        Features from the PowerCombined dataset
    y_power : pd.Series or np.ndarray
        Labels from the PowerCombined dataset
    X_hpc : pd.DataFrame or np.ndarray
        Features from the HPC-Kernel-Events dataset
    y_hpc : pd.Series or np.ndarray
        Labels from the HPC-Kernel-Events dataset
    latent_dim : int, default=32
        Dimensionality of the common latent space
    save_models : bool, default=True
        Whether to save the trained encoder models
    
    Returns:
    --------
    tuple
        (X_power_latent, X_hpc_latent, power_encoder, hpc_encoder)
    """
    # Convert to numpy arrays if they are DataFrames
    if isinstance(X_power, pd.DataFrame):
        X_power = X_power.values
    if isinstance(X_hpc, pd.DataFrame):
        X_hpc = X_hpc.values
    if isinstance(y_power, pd.Series):
        y_power = y_power.values
    if isinstance(y_hpc, pd.Series):
        y_hpc = y_hpc.values
    
    # Define input dimensions
    power_input_dim = X_power.shape[1]
    hpc_input_dim = X_hpc.shape[1]
    
    print(f"Training Power dataset encoder (input dim: {power_input_dim}, latent dim: {latent_dim})")
    # Train encoder for Power dataset
    power_encoder = train_encoder(
        X_power, y_power,
        input_dim=power_input_dim,
        latent_dim=latent_dim
    )
    
    print(f"Training HPC dataset encoder (input dim: {hpc_input_dim}, latent dim: {latent_dim})")
    # Train encoder for HPC dataset
    hpc_encoder = train_encoder(
        X_hpc, y_hpc,
        input_dim=hpc_input_dim,
        latent_dim=latent_dim
    )
    
    # Transform datasets to latent space
    device = next(power_encoder.parameters()).device
    
    power_encoder.eval()
    hpc_encoder.eval()
    
    with torch.no_grad():
        X_power_tensor = torch.FloatTensor(X_power).to(device)
        X_hpc_tensor = torch.FloatTensor(X_hpc).to(device)
        
        X_power_latent = power_encoder(X_power_tensor).cpu().numpy()
        X_hpc_latent = hpc_encoder(X_hpc_tensor).cpu().numpy()
    
    print(f"Transformed Power dataset to latent space: {X_power_latent.shape}")
    print(f"Transformed HPC dataset to latent space: {X_hpc_latent.shape}")
    
    # Save the encoder models if requested
    if save_models:
        os.makedirs(f"{PROCESSED_DATA_DIR}/models", exist_ok=True)
        torch.save(power_encoder.state_dict(), f"{PROCESSED_DATA_DIR}/models/power_encoder.pth")
        torch.save(hpc_encoder.state_dict(), f"{PROCESSED_DATA_DIR}/models/hpc_encoder.pth")
        print(f"Saved encoder models to {PROCESSED_DATA_DIR}/models/")
    
    # Save the transformed datasets
    power_latent_df = pd.DataFrame(
        X_power_latent,
        columns=[f'latent_{i+1}' for i in range(latent_dim)]
    )
    power_latent_df['Attack-Group'] = y_power
    
    hpc_latent_df = pd.DataFrame(
        X_hpc_latent,
        columns=[f'latent_{i+1}' for i in range(latent_dim)]
    )
    hpc_latent_df['Scenario'] = y_hpc
    
    power_latent_df.to_csv(f"{PROCESSED_DATA_DIR}/power_fused_encoder.csv", index=False)
    hpc_latent_df.to_csv(f"{PROCESSED_DATA_DIR}/hpc_fused_encoder.csv", index=False)
    
    return X_power_latent, X_hpc_latent, power_encoder, hpc_encoder

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
    
    # Apply adaptive weighting based on feature importance if power_weight is None
    if power_weight is None:
        # Calculate variance of each feature as a simple measure of importance
        power_var = np.var(X_power_scaled, axis=0).mean()
        hpc_var = np.var(X_hpc_scaled, axis=0).mean()
        total_var = power_var + hpc_var
        
        if total_var > 0:
            power_weight = power_var / total_var
            print(f"Calculated adaptive power_weight: {power_weight:.4f}")
        else:
            power_weight = 0.5
            print(f"Using default power_weight: {power_weight}")
    
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
    print(f"Power weight: {power_weight:.4f}, HPC weight: {hpc_weight:.4f}")
    
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
    tuple
        (X_reduced, pca) - Reduced feature matrix and PCA model
    """
    # First concatenate the features
    X_combined = feature_concatenation(X_power, X_hpc)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_combined)
    
    print(f"Number of components after PCA: {X_reduced.shape[1]}")
    print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    return X_reduced, pca

def cca_fusion(X_power, X_hpc, n_components=2):
    """
    Feature fusion using Canonical Correlation Analysis (CCA).
    CCA finds linear combinations of features from both datasets 
    that are maximally correlated.
    
    Parameters:
    -----------
    X_power : pd.DataFrame or np.ndarray
        Features from the PowerCombined dataset
    X_hpc : pd.DataFrame or np.ndarray
        Features from the HPC-Kernel-Events dataset
    n_components : int, default=2
        Number of components for CCA
    
    Returns:
    --------
    tuple
        (X_cca_power, X_cca_hpc, cca) - CCA transformed features and model
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
    
    # Determine maximum possible components
    max_components = min(X_power.shape[1], X_hpc.shape[1], X_power.shape[0])
    if n_components > max_components:
        print(f"Reducing n_components from {n_components} to {max_components}")
        n_components = max_components
    
    # Apply CCA
    cca = CCA(n_components=n_components)
    X_cca_power, X_cca_hpc = cca.fit_transform(X_power, X_hpc)
    
    print(f"CCA transformed Power shape: {X_cca_power.shape}")
    print(f"CCA transformed HPC shape: {X_cca_hpc.shape}")
    
    # Combine the CCA features if needed
    X_cca_combined = np.hstack((X_cca_power, X_cca_hpc))
    print(f"Combined CCA features shape: {X_cca_combined.shape}")
    
    return X_cca_power, X_cca_hpc, X_cca_combined, cca

def kernel_fusion(X_power, X_hpc, kernel='rbf', n_components=0.95):
    """
    Feature fusion using Kernel PCA to handle non-linear relationships.
    
    Parameters:
    -----------
    X_power : pd.DataFrame or np.ndarray
        Features from the PowerCombined dataset
    X_hpc : pd.DataFrame or np.ndarray
        Features from the HPC-Kernel-Events dataset
    kernel : str, default='rbf'
        Kernel type for KernelPCA
    n_components : int or float, default=0.95
        Number of components or variance to keep
    
    Returns:
    --------
    tuple
        (X_kpca, kpca) - Transformed features and KernelPCA model
    """
    # First concatenate the features
    X_combined = feature_concatenation(X_power, X_hpc)
    
    # Apply Kernel PCA
    kpca = KernelPCA(n_components=n_components, kernel=kernel, fit_inverse_transform=True)
    X_kpca = kpca.fit_transform(X_combined)
    
    print(f"Kernel PCA transformed shape: {X_kpca.shape}")
    
    return X_kpca, kpca

def supervised_fusion(X_power, X_hpc, y_power, y_hpc):
    """
    Supervised fusion using LDA to find discriminant components.
    This is useful when labels are available for both datasets.
    
    Parameters:
    -----------
    X_power : pd.DataFrame or np.ndarray
        Features from the PowerCombined dataset
    X_hpc : pd.DataFrame or np.ndarray
        Features from the HPC-Kernel-Events dataset
    y_power : pd.Series or np.ndarray
        Labels for PowerCombined dataset
    y_hpc : pd.Series or np.ndarray
        Labels for HPC-Kernel-Events dataset
    
    Returns:
    --------
    tuple
        (X_lda_power, X_lda_hpc) - LDA-transformed features for both datasets
    """
    # Convert to numpy arrays if needed
    if isinstance(X_power, pd.DataFrame):
        X_power = X_power.values
    if isinstance(X_hpc, pd.DataFrame):
        X_hpc = X_hpc.values
    if isinstance(y_power, pd.Series):
        y_power = y_power.values
    if isinstance(y_hpc, pd.Series):
        y_hpc = y_hpc.values
    
    # Apply LDA to each dataset separately
    lda_power = LDA()
    lda_hpc = LDA()
    
    try:
        X_lda_power = lda_power.fit_transform(X_power, y_power)
        print(f"LDA transformed Power shape: {X_lda_power.shape}")
    except Exception as e:
        print(f"Error applying LDA to Power data: {e}")
        print("Using PCA instead")
        pca = PCA(n_components=min(5, X_power.shape[1]))
        X_lda_power = pca.fit_transform(X_power)
    
    try:
        X_lda_hpc = lda_hpc.fit_transform(X_hpc, y_hpc)
        print(f"LDA transformed HPC shape: {X_lda_hpc.shape}")
    except Exception as e:
        print(f"Error applying LDA to HPC data: {e}")
        print("Using PCA instead")
        pca = PCA(n_components=min(5, X_hpc.shape[1]))
        X_lda_hpc = pca.fit_transform(X_hpc)
    
    # Normalize the LDA outputs
    X_lda_power = StandardScaler().fit_transform(X_lda_power)
    X_lda_hpc = StandardScaler().fit_transform(X_lda_hpc)
    
    return X_lda_power, X_lda_hpc

def domain_adaptation_fusion(X_power, X_hpc, y_power=None, y_hpc=None):
    """
    Feature fusion with domain adaptation to address concept drift.
    This method attempts to align the feature distributions across datasets.
    
    Parameters:
    -----------
    X_power : pd.DataFrame or np.ndarray
        Features from the PowerCombined dataset
    X_hpc : pd.DataFrame or np.ndarray
        Features from the HPC-Kernel-Events dataset
    y_power : pd.Series or np.ndarray, optional
        Labels for PowerCombined dataset
    y_hpc : pd.Series or np.ndarray, optional
        Labels for HPC-Kernel-Events dataset
    
    Returns:
    --------
    np.ndarray
        Combined features in a common latent space
    """
    # Convert to numpy arrays if needed
    if isinstance(X_power, pd.DataFrame):
        X_power = X_power.values
    if isinstance(X_hpc, pd.DataFrame):
        X_hpc = X_hpc.values
    
    # Step 1: Dimensionality reduction to a common dimensionality if needed
    power_dim = X_power.shape[1]
    hpc_dim = X_hpc.shape[1]
    
    if power_dim != hpc_dim:
        common_dim = min(power_dim, hpc_dim)
        print(f"Reducing dimensions to common size: {common_dim}")
        
        pca_power = PCA(n_components=common_dim)
        pca_hpc = PCA(n_components=common_dim)
        
        X_power_reduced = pca_power.fit_transform(X_power)
        X_hpc_reduced = pca_hpc.fit_transform(X_hpc)
    else:
        X_power_reduced = X_power
        X_hpc_reduced = X_hpc
        common_dim = power_dim
    
    # Step 2: Feature space alignment
    # Calculate mean vectors
    power_mean = np.mean(X_power_reduced, axis=0)
    hpc_mean = np.mean(X_hpc_reduced, axis=0)
    
    # Center the data
    X_power_centered = X_power_reduced - power_mean
    X_hpc_centered = X_hpc_reduced - hpc_mean
    
    # Calculate transformation matrices
    U_power, _, Vt_power = np.linalg.svd(X_power_centered, full_matrices=False)
    U_hpc, _, Vt_hpc = np.linalg.svd(X_hpc_centered, full_matrices=False)
    
    # Define transformation
    R = Vt_power.T @ Vt_hpc
    
    # Transform HPC features to power feature space
    X_hpc_aligned = X_hpc_centered @ R + power_mean
    
    # Step 3: Combine datasets
    # We can either concatenate or blend
    # Here we'll use a simple averaging approach for features
    X_combined = np.vstack((X_power_reduced, X_hpc_aligned))
    
    print(f"Combined feature matrix shape after domain adaptation: {X_combined.shape}")
    
    return X_combined

def detect_concept_drift(X_source, X_target):
    """
    Detect concept drift between source and target datasets.
    
    Parameters:
    -----------
    X_source : np.ndarray
        Features from the source dataset
    X_target : np.ndarray
        Features from the target dataset
    
    Returns:
    --------
    float
        Drift score (higher indicates more drift)
    """
    # Calculate distribution statistics
    source_mean = np.mean(X_source, axis=0)
    target_mean = np.mean(X_target, axis=0)
    
    source_std = np.std(X_source, axis=0)
    target_std = np.std(X_target, axis=0)
    
    # Calculate distributional distance
    mean_diff = np.linalg.norm(source_mean - target_mean)
    std_diff = np.linalg.norm(source_std - target_std)
    
    # Compute MMD (Maximum Mean Discrepancy) - simplified version
    n_source = X_source.shape[0]
    n_target = X_target.shape[0]
    
    # Sample a subset if datasets are large
    max_samples = 1000
    if n_source > max_samples:
        indices = np.random.choice(n_source, max_samples, replace=False)
        X_source_sample = X_source[indices]
    else:
        X_source_sample = X_source
    
    if n_target > max_samples:
        indices = np.random.choice(n_target, max_samples, replace=False)
        X_target_sample = X_target[indices]
    else:
        X_target_sample = X_target
    
    # Calculate cosine similarity between distributions
    cosine_sim = cosine_similarity(
        X_source_sample.mean(axis=0).reshape(1, -1), 
        X_target_sample.mean(axis=0).reshape(1, -1)
    )[0, 0]
    
    # Calculate average distance between samples from both sets
    dist = cdist(X_source_sample[:100] if len(X_source_sample) > 100 else X_source_sample, 
                X_target_sample[:100] if len(X_target_sample) > 100 else X_target_sample, 
                metric='euclidean')
    avg_dist = np.mean(dist)
    
    # Combine metrics into a drift score
    drift_score = (mean_diff + std_diff + (1 - cosine_sim) + avg_dist/10) / 4
    
    print(f"Concept Drift Analysis:")
    print(f"  Mean difference: {mean_diff:.4f}")
    print(f"  Std difference: {std_diff:.4f}")
    print(f"  Cosine similarity: {cosine_sim:.4f}")
    print(f"  Average distance: {avg_dist:.4f}")
    print(f"  Overall drift score: {drift_score:.4f}")
    
    return drift_score

def create_tsne_visualization(X_power, X_hpc, y_power=None, y_hpc=None, output_path=None):
    """
    Create a t-SNE visualization of both datasets.
    
    Parameters:
    -----------
    X_power : pd.DataFrame or np.ndarray
        Features from the PowerCombined dataset
    X_hpc : pd.DataFrame or np.ndarray
        Features from the HPC-Kernel-Events dataset
    y_power : pd.Series or np.ndarray, optional
        Labels for PowerCombined dataset
    y_hpc : pd.Series or np.ndarray, optional
        Labels for HPC-Kernel-Events dataset
    output_path : str, optional
        Path to save the visualization
    """
    
    # Limit to 1000 samples max for each dataset for visualization
    max_samples = 1000
    if X_power.shape[0] > max_samples:
        indices = np.random.choice(X_power.shape[0], max_samples, replace=False)
        X_power_sample = X_power[indices]
        y_power_sample = y_power[indices] if y_power is not None else None
    else:
        X_power_sample = X_power
        y_power_sample = y_power
    
    if X_hpc.shape[0] > max_samples:
        indices = np.random.choice(X_hpc.shape[0], max_samples, replace=False)
        X_hpc_sample = X_hpc[indices]
        y_hpc_sample = y_hpc[indices] if y_hpc is not None else None
    else:
        X_hpc_sample = X_hpc
        y_hpc_sample = y_hpc
    
    # Combine datasets for t-SNE
    X_combined = np.vstack((X_power_sample, X_hpc_sample))
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_combined)
    
    # Split back into separate datasets
    X_power_tsne = X_tsne[:len(X_power_sample)]
    X_hpc_tsne = X_tsne[len(X_power_sample):]
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Plot Power dataset
    plt.scatter(X_power_tsne[:, 0], X_power_tsne[:, 1], 
                alpha=0.7, label='Power', marker='o')
    
    # Plot HPC dataset
    plt.scatter(X_hpc_tsne[:, 0], X_hpc_tsne[:, 1], 
                alpha=0.7, label='HPC', marker='x')
    
    plt.title('t-SNE Visualization of Power and HPC Datasets')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    else:
        plt.tight_layout()
        plt.show()

def get_fused_features(fusion_method='domain_adaptation', power_weight=0.5, 
                     n_components=0.95, visualize=False):
    """
    Get fused features from both datasets using the specified method.
    
    Parameters:
    -----------
    fusion_method : str, default='domain_adaptation'
        The fusion method to use. One of:
        - 'concatenation': Simple concatenation
        - 'weighted': Weighted fusion
        - 'pca': PCA-based fusion
        - 'cca': Canonical Correlation Analysis fusion
        - 'kernel': Kernel PCA fusion
        - 'supervised': Supervised fusion with LDA
        - 'domain_adaptation': Domain adaptation for concept drift
    power_weight : float or None, default=0.5
        Weight for PowerCombined features (0-1) when using weighted fusion.
        If None, the weight will be calculated adaptively.
    n_components : int or float, default=0.95
        Number of components or variance to keep when using PCA
    visualize : bool, default=False
        Whether to create and save t-SNE visualization
    
    Returns:
    --------
    dict
        Dictionary containing the fused features and related information
    """
    # Load the processed data
    power_df, hpc_df = load_processed_data()
    if power_df is None or hpc_df is None:
        return None
    
    # Load preprocessing models if available
    models = load_preprocessing_models()
    
    # Align datasets if needed
    power_aligned, hpc_aligned = align_datasets(power_df, hpc_df)
    
    # Extract features
    X_power, y_power, X_hpc, y_hpc = extract_features(power_aligned, hpc_aligned)
    
    # Detect concept drift between datasets
    print("\nAnalyzing concept drift between datasets...")
    
    # Convert both datasets to a common dimensionality for drift detection
    if X_power.shape[1] != X_hpc.shape[1]:
        common_dim = min(X_power.shape[1], X_hpc.shape[1])
        pca_power = PCA(n_components=common_dim)
        pca_hpc = PCA(n_components=common_dim)
        
        X_power_common = pca_power.fit_transform(X_power)
        X_hpc_common = pca_hpc.fit_transform(X_hpc)
        
        drift_score = detect_concept_drift(X_power_common, X_hpc_common)
    else:
        drift_score = detect_concept_drift(X_power.values, X_hpc.values)
    
    # Apply the specified fusion method
    result = {
        'X_power': X_power,
        'y_power': y_power,
        'X_hpc': X_hpc,
        'y_hpc': y_hpc,
        'drift_score': drift_score
    }
    
    print(f"\nApplying fusion method: {fusion_method}")
    
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
        X_fused, pca_model = fusion_with_dim_reduction(X_power, X_hpc, n_components)
        result['X_fused'] = X_fused
        result['n_components'] = X_fused.shape[1]
        result['pca_model'] = pca_model
        result['fusion_method'] = 'pca'
    
    elif fusion_method == 'cca':
        X_cca_power, X_cca_hpc, X_cca_combined, cca_model = cca_fusion(X_power, X_hpc)
        result['X_power_cca'] = X_cca_power
        result['X_hpc_cca'] = X_cca_hpc
        result['X_fused'] = X_cca_combined
        result['cca_model'] = cca_model
        result['fusion_method'] = 'cca'
    
    elif fusion_method == 'kernel':
        X_fused, kpca_model = kernel_fusion(X_power, X_hpc)
        result['X_fused'] = X_fused
        result['kpca_model'] = kpca_model
        result['fusion_method'] = 'kernel'
    
    elif fusion_method == 'supervised':
        X_lda_power, X_lda_hpc = supervised_fusion(X_power, X_hpc, y_power, y_hpc)
        # Combine the LDA outputs
        X_fused = np.hstack((X_lda_power, X_lda_hpc))
        result['X_power_lda'] = X_lda_power
        result['X_hpc_lda'] = X_lda_hpc
        result['X_fused'] = X_fused
        result['fusion_method'] = 'supervised'
    
    elif fusion_method == 'domain_adaptation':
        X_fused = domain_adaptation_fusion(X_power, X_hpc, y_power, y_hpc)
        result['X_fused'] = X_fused
        result['fusion_method'] = 'domain_adaptation'
    
    else:
        print(f"Unknown fusion method: {fusion_method}")
        print("Using domain adaptation method")
        X_fused = domain_adaptation_fusion(X_power, X_hpc, y_power, y_hpc)
        result['X_fused'] = X_fused
        result['fusion_method'] = 'domain_adaptation'
    
    # Create visualization if requested
    if visualize:
        try:
            # Create output directory if it doesn't exist
            vis_dir = Path(PROCESSED_DATA_DIR) / "visualizations"
            os.makedirs(vis_dir, exist_ok=True)
            
            # Create t-SNE visualization
            output_path = vis_dir / f"tsne_{fusion_method}.png"
            create_tsne_visualization(X_power, X_hpc, y_power, y_hpc, output_path)
            
            # If we have fused features, visualize those too
            if 'X_fused' in result:
                # Split the fused features back into power and hpc for visualization
                n_power = X_power.shape[0]
                X_fused_power = result['X_fused'][:n_power]
                X_fused_hpc = result['X_fused'][n_power:]
                
                output_path = vis_dir / f"tsne_fused_{fusion_method}.png"
                create_tsne_visualization(X_fused_power, X_fused_hpc, y_power, y_hpc, output_path)
                
            result['visualizations_path'] = str(vis_dir)
        except Exception as e:
            print(f"Error creating visualizations: {e}")
    
    # Save the fused features
    fused_filename = f"fused_{fusion_method}.csv"
    fused_path = Path(PROCESSED_DATA_DIR) / fused_filename
    
    try:
        if 'X_fused' in result:
            # Save fused data with labels
            n_power = X_power.shape[0]
            n_hpc = X_hpc.shape[0]
            
            # Create a combined target array
            if n_power == n_hpc:
                # If sizes are the same, we can directly use both labels
                combined_labels = np.concatenate([y_power, y_hpc])
                
                # Create a source indicator
                source_indicator = np.concatenate([
                    np.full(n_power, 'power'),
                    np.full(n_hpc, 'hpc')
                ])
                
                # Create a dataframe with features and metadata
                fused_df = pd.DataFrame(result['X_fused'])
                fused_df.columns = [f'feature_{i}' for i in range(fused_df.shape[1])]
                fused_df['source'] = source_indicator
                fused_df['label'] = combined_labels
                
                # Save to CSV
                fused_df.to_csv(fused_path, index=False)
                print(f"Saved fused features to {fused_path}")
            else:
                # If sizes are different, save separate files
                fused_df_power = pd.DataFrame(result['X_fused'][:n_power])
                fused_df_power.columns = [f'feature_{i}' for i in range(fused_df_power.shape[1])]
                fused_df_power['label'] = y_power
                fused_df_power['source'] = 'power'
                
                fused_df_hpc = pd.DataFrame(result['X_fused'][n_power:])
                fused_df_hpc.columns = [f'feature_{i}' for i in range(fused_df_hpc.shape[1])]
                fused_df_hpc['label'] = y_hpc
                fused_df_hpc['source'] = 'hpc'
                
                # Save to CSV
                power_path = Path(PROCESSED_DATA_DIR) / f"power_fused_{fusion_method}.csv"
                hpc_path = Path(PROCESSED_DATA_DIR) / f"hpc_fused_{fusion_method}.csv"
                
                fused_df_power.to_csv(power_path, index=False)
                fused_df_hpc.to_csv(hpc_path, index=False)
                
                print(f"Saved power fused features to {power_path}")
                print(f"Saved hpc fused features to {hpc_path}")
                
                result['fused_power_path'] = str(power_path)
                result['fused_hpc_path'] = str(hpc_path)
    except Exception as e:
        print(f"Error saving fused features: {e}")
    
    return result

def feature_concatenation(X_power, X_hpc):
    """
    Concatenate features from both datasets, handling datasets with different feature sets.
    
    Parameters:
    -----------
    X_power : pd.DataFrame or np.ndarray
        Features from the PowerCombined dataset
    X_hpc : pd.DataFrame or np.ndarray
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
    
    # Standardize each dataset separately to normalize scales
    power_scaler = StandardScaler()
    hpc_scaler = StandardScaler()
    
    X_power_scaled = power_scaler.fit_transform(X_power)
    X_hpc_scaled = hpc_scaler.fit_transform(X_hpc)
    
    # Concatenate along feature dimension
    X_combined = np.hstack((X_power_scaled, X_hpc_scaled))
    
    print(f"Combined feature matrix shape: {X_combined.shape}")
    print(f"Number of Power features: {X_power.shape[1]}")
    print(f"Number of HPC features: {X_hpc.shape[1]}")
    
    return X_combined

def train_domain_adversarial_encoder(X_source, y_source, X_target, latent_dim=32, 
                                 num_epochs=100, batch_size=64, lr=0.001,
                                 weight_decay=1e-5, alpha=1.0, device=None):
    """
    Train a domain adversarial encoder for cross-domain feature fusion.
    
    This function takes source and target domain data and trains a domain adversarial
    neural network that maps both domains to a common latent space where domain
    discrimination is difficult, while maintaining class discrimination ability.
    
    Parameters:
    -----------
    X_source : numpy.ndarray
        Source domain features
    y_source : numpy.ndarray
        Source domain labels
    X_target : numpy.ndarray
        Target domain features
    latent_dim : int, default=32
        Dimensionality of the latent space
    num_epochs : int, default=100
        Number of training epochs
    batch_size : int, default=64
        Batch size for training
    lr : float, default=0.001
        Learning rate
    weight_decay : float, default=1e-5
        L2 regularization strength
    alpha : float, default=1.0
        Domain confusion strength (gradient reversal layer parameter)
    device : str, default=None
        Device to use for training ('cuda', 'cpu', or None for auto-detection)
    
    Returns:
    --------
    model : DomainAdversarialEncoder
        Trained domain adversarial encoder model
    history : dict
        Training history dictionary containing metrics over epochs
    """
    if not TORCH_AVAILABLE:
        print("PyTorch is not available. Cannot perform domain adversarial fusion.")
        return None, None
    
    # Convert inputs to numpy arrays if they are not
    if isinstance(X_source, pd.DataFrame):
        X_source = X_source.values
    if isinstance(y_source, pd.Series):
        y_source = y_source.values
    if isinstance(X_target, pd.DataFrame):
        X_target = X_target.values
    
    # Check for NaN values and replace if necessary
    if np.isnan(X_source).any():
        print("Warning: Source data contains NaN values. Replacing with zeros.")
        X_source = np.nan_to_num(X_source)
    if np.isnan(X_target).any():
        print("Warning: Target data contains NaN values. Replacing with zeros.")
        X_target = np.nan_to_num(X_target)
    
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Get input dimensionality 
    input_dim = X_source.shape[1]
    
    # Convert labels to numeric if they're not already
    from sklearn.preprocessing import LabelEncoder
    if not np.issubdtype(y_source.dtype, np.number):
        label_encoder = LabelEncoder()
        y_source = label_encoder.fit_transform(y_source)
        print(f"Label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    # Get number of classes
    num_classes = len(np.unique(y_source))
    
    # Initialize the model
    model = DomainAdversarialEncoder(input_dim, latent_dim, 
                                    hidden_dim=128, 
                                    num_classes=num_classes).to(device)
    
    # Train the model
    print(f"Training domain adversarial encoder (input_dim={input_dim}, latent_dim={latent_dim}, num_classes={num_classes})")
    print(f"Source data shape: {X_source.shape}, Target data shape: {X_target.shape}")
    
    history = train_adversarial_encoder(
        X_source, X_target, y_source,
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        alpha=alpha
    )
    
    return model, history


def apply_domain_adversarial_fusion(X_power, X_hpc, model_path=None, latent_dim=32, 
                                   save_model=True, model=None, history=None):
    """
    Apply domain adversarial fusion to power and HPC datasets.
    
    Parameters:
    -----------
    X_power : numpy.ndarray or pd.DataFrame
        Power domain features
    X_hpc : numpy.ndarray or pd.DataFrame
        HPC domain features
    model_path : str, default=None
        Path to save/load the model
    latent_dim : int, default=32
        Dimensionality of the latent space
    save_model : bool, default=True
        Whether to save the trained model
    model : DomainAdversarialEncoder, default=None
        Pre-trained model (if available)
    history : dict, default=None
        Training history (if available)
        
    Returns:
    --------
    dict 
        Dictionary containing:
        - 'power_latent': Encoded power features
        - 'hpc_latent': Encoded HPC features
        - 'model': The trained or loaded model
        - 'history': Training history
    """
    if not TORCH_AVAILABLE:
        print("PyTorch is not available. Cannot perform domain adversarial fusion.")
        return None
    
    # Convert inputs to numpy arrays if they are not
    if isinstance(X_power, pd.DataFrame):
        X_power = X_power.values
    if isinstance(X_hpc, pd.DataFrame):
        X_hpc = X_hpc.values
    
    # Check for NaN values and replace if necessary
    if np.isnan(X_power).any():
        print("Warning: Power data contains NaN values. Replacing with zeros.")
        X_power = np.nan_to_num(X_power)
    if np.isnan(X_hpc).any():
        print("Warning: HPC data contains NaN values. Replacing with zeros.")
        X_hpc = np.nan_to_num(X_hpc)
    
    # If no model is provided, try to load one or return error
    if model is None:
        if model_path and os.path.exists(model_path):
            print(f"Loading domain adversarial encoder from {model_path}")
            model = torch.load(model_path)
        else:
            print("Error: No model provided and no model found at specified path.")
            print("Please train a model first using train_domain_adversarial_encoder().")
            return None
    
    # Encode both datasets
    print("Encoding power and HPC datasets to latent space...")
    power_latent = model.encode(X_power)
    hpc_latent = model.encode(X_hpc)
    
    print(f"Encoded power shape: {power_latent.shape}")
    print(f"Encoded HPC shape: {hpc_latent.shape}")
    
    # Save model if requested
    if save_model and model_path:
        print(f"Saving domain adversarial encoder to {model_path}")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model, model_path)
    
    return {
        'power_latent': power_latent,
        'hpc_latent': hpc_latent,
        'model': model,
        'history': history
    }

if __name__ == "__main__":
    # Example usage
    fusion_result = get_fused_features(
        fusion_method='domain_adaptation',  # Using domain adaptation to address concept drift
        power_weight=None,                  # Use adaptive weighting
        n_components=0.95,
        visualize=True                      # Create visualizations
    )
    
    if fusion_result:
        print("\nFusion complete!")
        print(f"Fusion method: {fusion_result['fusion_method']}")
        print(f"Drift score: {fusion_result['drift_score']:.4f}")
        
        for key, value in fusion_result.items():
            if isinstance(value, np.ndarray):
                print(f"{key} shape: {value.shape}")
            elif isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
                print(f"{key} shape: {value.shape}")
            elif key not in ['drift_score', 'fusion_method', 'power_weight', 
                            'pca_model', 'cca_model', 'kpca_model',
                            'power_features_idx', 'hpc_features_idx',
                            'fused_power_path', 'fused_hpc_path', 
                            'visualizations_path']:
                print(f"{key}: {value}")