"""
Model Module

This module contains the model architecture and training functions
for the ITSCPaper project, using TensorFlow/Keras and attention mechanisms.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import joblib
import time
import logging
from pathlib import Path
from src.config import RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE

logger = logging.getLogger(__name__)

class AttentionEncoder:
    @staticmethod
    def create_encoder(input_dim, latent_dim=16):
        """
        Create an encoder with a more robust architecture
        """
        inputs = Input(shape=(input_dim,))
        
        # Initial projection
        x = Dense(64, activation='relu')(inputs)
        
        # Additional dense layers
        x = Dense(32, activation='relu')(x)
        
        # Latent space representation
        latent = Dense(latent_dim, activation='linear')(x)
        
        return Model(inputs=inputs, outputs=latent)

class DatasetMerger:
    @staticmethod
    def merge_datasets(latent_hpc, y_hpc, latent_power, y_power):
        """
        Merge encoded datasets with common labels
        """
        try:
            # Convert labels to strings
            y_hpc_str = y_hpc.astype(str)
            y_power_str = y_power.astype(str)
            
            # Find common labels
            common_labels = np.intersect1d(
                np.unique(y_hpc_str), 
                np.unique(y_power_str)
            )
            
            # Validate common labels
            if len(common_labels) == 0:
                raise ValueError("No common labels found between datasets")
            
            # Filter datasets
            mask_hpc = np.isin(y_hpc_str, common_labels)
            mask_power = np.isin(y_power_str, common_labels)
            
            X_merged = np.vstack((
                latent_hpc[mask_hpc],
                latent_power[mask_power]
            ))
            y_merged = np.concatenate((
                y_hpc_str[mask_hpc],
                y_power_str[mask_power]
            ))
            
            # Encode labels
            label_encoder = LabelEncoder()
            y_merged_encoded = label_encoder.fit_transform(y_merged)
            
            return X_merged, y_merged_encoded, common_labels, label_encoder
        
        except Exception as e:
            logger.error(f"Error in dataset merging: {e}")
            raise

class ModelTrainer:
    @staticmethod
    def create_classifier(input_dim, num_classes):
        """
        Create a joint classifier with more robust architecture
        """
        inputs = Input(shape=(input_dim,))
        
        # Multiple dense layers
        x = Dense(64, activation='relu')(inputs)
        x = Dense(32, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        
        # Output layer
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    @staticmethod
    def train_model(model, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """
        Train the model with early stopping
        
        Parameters:
        -----------
        model : tf.keras.Model
            Model to train
        X_train : array-like
            Training features
        y_train : array-like
            Training targets
        X_val : array-like, optional
            Validation features
        y_val : array-like, optional
            Validation targets
        epochs : int, default=100
            Number of epochs to train
        batch_size : int, default=32
            Batch size for training
            
        Returns:
        --------
        tuple
            (trained_model, history)
        """
        # Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Prepare validation data if provided
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Train the model
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            validation_data=validation_data,
            validation_split=0.2 if validation_data is None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return model, history

class Visualizer:
    @staticmethod
    def plot_latent_space(X_merged, y_merged, title='Latent Space', save_path=None):
        """
        Visualize latent space using t-SNE
        """
        try:
            plt.figure(figsize=(10, 8))
            tsne = TSNE(n_components=2, random_state=42)
            X_tsne = tsne.fit_transform(X_merged)
            
            scatter = plt.scatter(
                X_tsne[:, 0],
                X_tsne[:, 1],
                c=y_merged,
                cmap='viridis',
                alpha=0.7
            )
            plt.colorbar(scatter)
            plt.title(f't-SNE Visualization of {title}')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Saved t-SNE visualization to {save_path}")
            else:
                plt.show()
                
            plt.close()
            
        except Exception as e:
            logger.error(f"Error in visualization: {e}")
            
    @staticmethod
    def plot_training_history(history, save_path=None):
        """
        Plot training and validation metrics
        
        Parameters:
        -----------
        history : tf.keras.callbacks.History
            Training history object
        save_path : str, optional
            Path to save the plot
        """
        try:
            plt.figure(figsize=(12, 5))
            
            # Plot accuracy
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Train')
            if 'val_accuracy' in history.history:
                plt.plot(history.history['val_accuracy'], label='Validation')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            # Plot loss
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Train')
            if 'val_loss' in history.history:
                plt.plot(history.history['val_loss'], label='Validation')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Saved training history plot to {save_path}")
            else:
                plt.show()
                
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting training history: {e}")

def train_model_pipeline(X_hpc, y_hpc, X_power, y_power, latent_dim=16, save=True, output_dir=None):
    """
    Complete model training pipeline using attention encoders
    
    Parameters:
    -----------
    X_hpc : array-like
        HPC features
    y_hpc : array-like
        HPC targets
    X_power : array-like
        Power features
    y_power : array-like
        Power targets
    latent_dim : int, default=16
        Dimension of latent space
    save : bool, default=True
        Whether to save the trained models
    output_dir : str or Path, optional
        Directory to save models and visualizations
        
    Returns:
    --------
    dict
        Dictionary containing models, encoders, and evaluation results
    """
    try:
        logger.info("Starting model training pipeline with attention encoders")
        
        # Set up output directory
        if output_dir is None:
            output_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "models"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualizations directory
        vis_dir = Path(output_dir) / "visualizations"
        os.makedirs(vis_dir, exist_ok=True)
        
        # Create encoders
        logger.info(f"Creating encoders with latent dimension {latent_dim}")
        encoder_hpc = AttentionEncoder.create_encoder(input_dim=X_hpc.shape[1], latent_dim=latent_dim)
        encoder_power = AttentionEncoder.create_encoder(input_dim=X_power.shape[1], latent_dim=latent_dim)
        
        # Encode datasets
        logger.info("Encoding datasets")
        latent_hpc = encoder_hpc.predict(X_hpc)
        latent_power = encoder_power.predict(X_power)
        
        logger.info(f"Latent HPC shape: {latent_hpc.shape}")
        logger.info(f"Latent Power shape: {latent_power.shape}")
        
        # Merge datasets
        logger.info("Merging datasets")
        X_merged, y_merged, common_labels, label_encoder = DatasetMerger.merge_datasets(
            latent_hpc, y_hpc, latent_power, y_power
        )
        
        logger.info(f"Merged dataset shape: {X_merged.shape}")
        logger.info(f"Common labels: {common_labels}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_merged, y_merged, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_merged
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE, stratify=y_train
        )
        
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Validation set: {X_val.shape}")
        logger.info(f"Test set: {X_test.shape}")
        
        # Create and train classifier
        logger.info("Creating classifier model")
        classifier = ModelTrainer.create_classifier(
            input_dim=X_merged.shape[1], num_classes=len(common_labels)
        )
        
        # Train the model
        logger.info("Training classifier model")
        trained_model, history = ModelTrainer.train_model(
            classifier, X_train, y_train, X_val, y_val
        )
        
        # Evaluate on test set
        logger.info("Evaluating model on test set")
        test_loss, test_accuracy = trained_model.evaluate(X_test, y_test)
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        
        # Get predictions
        y_pred = trained_model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Generate classification report
        class_report = classification_report(
            y_test, y_pred_classes, target_names=label_encoder.classes_
        )
        logger.info(f"Classification Report:\n{class_report}")
        
        # Visualizations
        logger.info("Generating visualizations")
        Visualizer.plot_latent_space(
            X_merged, y_merged, title='Merged Dataset Latent Space',
            save_path=vis_dir / 'latent_space.png'
        )
        
        Visualizer.plot_training_history(
            history, save_path=vis_dir / 'training_history.png'
        )
        
        # Save models if requested
        if save:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            
            # Save encoders
            encoder_hpc_path = Path(output_dir) / f"encoder_hpc_{timestamp}.h5"
            encoder_power_path = Path(output_dir) / f"encoder_power_{timestamp}.h5"
            encoder_hpc.save(encoder_hpc_path)
            encoder_power.save(encoder_power_path)
            
            # Save classifier
            classifier_path = Path(output_dir) / f"classifier_{timestamp}.h5"
            trained_model.save(classifier_path)
            
            # Save label encoder
            label_encoder_path = Path(output_dir) / f"label_encoder_{timestamp}.joblib"
            joblib.dump(label_encoder, label_encoder_path)
            
            logger.info(f"Saved models to {output_dir}")
        
        # Return results
        return {
            'encoder_hpc': encoder_hpc,
            'encoder_power': encoder_power,
            'classifier': trained_model,
            'history': history,
            'test_accuracy': test_accuracy,
            'classification_report': class_report,
            'label_encoder': label_encoder,
            'common_labels': common_labels,
            'X_merged': X_merged,
            'y_merged': y_merged,
            'latent_hpc': latent_hpc,
            'latent_power': latent_power
        }
        
    except Exception as e:
        logger.error(f"Error in model training pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None