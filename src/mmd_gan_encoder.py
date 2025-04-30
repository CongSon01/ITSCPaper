"""
MMD-GAN Encoder Module

This module implements a GAN-based encoder for mapping features from diverse datasets
into a shared latent space using Maximum Mean Discrepancy (MMD) techniques.
The encoder is trained with both triplet loss and adversarial loss.
"""

import os
# Configure TensorFlow to disable CPU instruction optimizations
# This must be set before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_DISABLE_MKL'] = '1'

import numpy as np
import tensorflow as tf

# Further TensorFlow configuration for CPU compatibility
try:
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
except:
    pass

from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

class MMDGANEncoder:
    """
    MMD-GAN Encoder for mapping features from diverse datasets into a shared latent space.
    
    This class implements an encoder model that is trained using:
    1. Triplet Loss - to make samples with same labels from different datasets closer
    2. Adversarial Loss - to ensure the latent space is well-structured
    """
    
    def __init__(self, input_dim, latent_dim=16, hidden_dims=[64, 32], 
                 dropout_rate=0.2, learning_rate=0.001, triplet_margin=1.0):
        """
        Initialize the MMD-GAN Encoder.
        
        Parameters:
        -----------
        input_dim : int
            Dimension of the input features
        latent_dim : int, default=16
            Dimension of the latent space
        hidden_dims : list, default=[64, 32]
            Dimensions of hidden layers
        dropout_rate : float, default=0.2
            Dropout rate for regularization
        learning_rate : float, default=0.001
            Learning rate for optimizer
        triplet_margin : float, default=1.0
            Margin parameter for triplet loss
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.triplet_margin = triplet_margin
        
        # Build the encoder model
        self.encoder = self._build_encoder()
        
        # Build the discriminator model
        self.discriminator = self._build_discriminator()
        
        # Create the combined model for adversarial training
        self.combined = self._build_combined_model()
    
    def _build_encoder(self):
        """
        Build the encoder model.
        
        Returns:
        --------
        tf.keras.Model
            The encoder model
        """
        inputs = Input(shape=(self.input_dim,))
        
        # Initial dense layer
        x = Dense(self.hidden_dims[0], activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Hidden layers
        for dim in self.hidden_dims[1:]:
            x = Dense(dim, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)
        
        # Latent space representation
        latent = Dense(self.latent_dim, activation='linear', name='latent')(x)
        
        # Create the encoder model
        encoder = Model(inputs=inputs, outputs=latent, name='encoder')
        return encoder
    
    def _build_discriminator(self):
        """
        Build the discriminator model for the adversarial component.
        
        Returns:
        --------
        tf.keras.Model
            The discriminator model
        """
        latent_input = Input(shape=(self.latent_dim,))
        class_input = Input(shape=(1,))  # Class label as input
        
        # Embedding for the class
        class_embedding = Dense(8, activation='relu')(tf.cast(class_input, tf.float32))
        
        # Concatenate latent code and class embedding
        x = tf.concat([latent_input, class_embedding], axis=-1)
        
        # Discriminator layers
        x = Dense(32, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        
        # Output layer (binary classification)
        validity = Dense(1, activation='sigmoid', name='validity')(x)
        
        # Create the discriminator model
        discriminator = Model(inputs=[latent_input, class_input], outputs=validity, name='discriminator')
        discriminator.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=['accuracy']
        )
        return discriminator
    
    def _build_combined_model(self):
        """
        Build the combined model for adversarial training.
        
        Returns:
        --------
        tf.keras.Model
            The combined model
        """
        # Freeze the discriminator for generator training
        self.discriminator.trainable = False
        
        # Generator inputs
        gen_input = Input(shape=(self.input_dim,))
        class_input = Input(shape=(1,))
        
        # Generate latent code
        latent_code = self.encoder(gen_input)
        
        # Get validity prediction from discriminator
        validity = self.discriminator([latent_code, class_input])
        
        # The combined model takes the generator input and outputs validity
        combined = Model(inputs=[gen_input, class_input], outputs=validity, name='combined')
        combined.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=self.learning_rate)
        )
        return combined
    
    def euclidean_distance(self, x, y):
        """
        Calculate Euclidean distance between two tensors.
        
        Parameters:
        -----------
        x : tf.Tensor
            First tensor
        y : tf.Tensor
            Second tensor
            
        Returns:
        --------
        tf.Tensor
            Euclidean distance
        """
        sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
        return tf.sqrt(tf.maximum(sum_square, 1e-10))
    
    def triplet_loss(self, anchor, positive, negative):
        """
        Calculate triplet loss.
        
        Parameters:
        -----------
        anchor : tf.Tensor
            Anchor samples
        positive : tf.Tensor
            Positive samples (same class as anchor)
        negative : tf.Tensor
            Negative samples (different class from anchor)
            
        Returns:
        --------
        tf.Tensor
            Triplet loss value
        """
        pos_dist = self.euclidean_distance(anchor, positive)
        neg_dist = self.euclidean_distance(anchor, negative)
        
        # Triplet loss formula: max(0, d(a,p) - d(a,n) + margin)
        basic_loss = pos_dist - neg_dist + self.triplet_margin
        loss = tf.maximum(basic_loss, 0.0)
        return tf.reduce_mean(loss)
    
    def train_step_triplet(self, anchors, positives, negatives):
        """
        Perform one training step for triplet loss.
        
        Parameters:
        -----------
        anchors : tf.Tensor
            Anchor samples
        positives : tf.Tensor
            Positive samples
        negatives : tf.Tensor
            Negative samples
            
        Returns:
        --------
        float
            Triplet loss value
        """
        with tf.GradientTape() as tape:
            # Encode samples
            anchor_encodings = self.encoder(anchors)
            positive_encodings = self.encoder(positives)
            negative_encodings = self.encoder(negatives)
            
            # Calculate triplet loss
            loss = self.triplet_loss(anchor_encodings, positive_encodings, negative_encodings)
        
        # Get gradients and update weights
        gradients = tape.gradient(loss, self.encoder.trainable_variables)
        self.encoder_optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables))
        
        return loss.numpy()
    
    def train_step_gan(self, batch_features, batch_labels, real_samples=None):
        """
        Perform one training step for the GAN.
        
        Parameters:
        -----------
        batch_features : tf.Tensor
            Batch of feature vectors
        batch_labels : tf.Tensor
            Batch of labels
        real_samples : tf.Tensor, optional
            Real samples for the discriminator (if None, generated samples are used)
            
        Returns:
        --------
        tuple
            (discriminator_loss, generator_loss)
        """
        # Generate fake samples
        fake_latent = self.encoder(batch_features)
        
        # Use real samples if provided, otherwise use generated ones as "real"
        if real_samples is None:
            real_latent = fake_latent.numpy()
            np.random.shuffle(real_latent)  # Shuffle to create "different" samples
        else:
            real_latent = real_samples
            
        # Labels for GAN training
        real_labels = np.ones((batch_features.shape[0], 1))
        fake_labels = np.zeros((batch_features.shape[0], 1))
        
        # ----- Train Discriminator -----
        # The discriminator receives the class label along with the latent representation
        d_loss_real = self.discriminator.train_on_batch([real_latent, batch_labels], real_labels)
        d_loss_fake = self.discriminator.train_on_batch([fake_latent, batch_labels], fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # ----- Train Generator (Encoder) -----
        # The generator wants the discriminator to classify fake samples as real
        g_loss = self.combined.train_on_batch([batch_features, batch_labels], real_labels)
        
        return d_loss, g_loss
    
    def compile_model(self):
        """
        Compile the encoder model.
        """
        self.encoder_optimizer = Adam(learning_rate=self.learning_rate)
        self.encoder.compile(optimizer=self.encoder_optimizer)
    
    def train(self, X_hpc, y_hpc, X_power, y_power, epochs=100, batch_size=32, sample_weight=None):
        """
        Train the MMD-GAN encoder.
        
        Parameters:
        -----------
        X_hpc : np.ndarray
            HPC dataset features
        y_hpc : np.ndarray
            HPC dataset labels
        X_power : np.ndarray
            Power dataset features
        y_power : np.ndarray
            Power dataset labels
        epochs : int, default=100
            Number of epochs to train
        batch_size : int, default=32
            Batch size for training
        sample_weight : np.ndarray, optional
            Sample weights for training
            
        Returns:
        --------
        dict
            Training history
        """
        self.compile_model()
        
        # Convert y values to proper format if they're not already
        y_hpc = y_hpc.reshape(-1, 1) if len(y_hpc.shape) == 1 else y_hpc
        y_power = y_power.reshape(-1, 1) if len(y_power.shape) == 1 else y_power
        
        # Create mapping of labels between datasets
        unique_hpc_labels = np.unique(y_hpc)
        unique_power_labels = np.unique(y_power)
        common_labels = np.intersect1d(unique_hpc_labels, unique_power_labels)
        
        if len(common_labels) == 0:
            logger.error("No common labels found between datasets")
            return None
        
        logger.info(f"Found {len(common_labels)} common labels between datasets")
        
        # History for recording metrics
        history = {
            'triplet_loss': [],
            'discriminator_loss': [],
            'generator_loss': [],
            'discriminator_accuracy': []
        }
        
        # Start training
        start_time = time.time()
        logger.info("Starting MMD-GAN encoder training")
        
        for epoch in range(epochs):
            epoch_triplet_losses = []
            epoch_d_losses = []
            epoch_g_losses = []
            epoch_d_accs = []
            
            # Randomly select batches
            for batch_start in range(0, len(X_hpc), batch_size):
                # Get batch for HPC data
                batch_end = min(batch_start + batch_size, len(X_hpc))
                batch_hpc_features = X_hpc[batch_start:batch_end]
                batch_hpc_labels = y_hpc[batch_start:batch_end]
                
                # For each HPC sample, find matching and non-matching samples from Power data
                triplet_batches = self._create_triplet_batches(
                    batch_hpc_features, batch_hpc_labels, X_power, y_power, common_labels
                )
                
                if triplet_batches is None:
                    continue
                
                anchors, positives, negatives = triplet_batches
                
                # Train on triplet loss
                triplet_loss = self.train_step_triplet(anchors, positives, negatives)
                epoch_triplet_losses.append(triplet_loss)
                
                # Train on GAN loss
                d_loss, g_loss = self.train_step_gan(batch_hpc_features, batch_hpc_labels)
                epoch_d_losses.append(d_loss[0])  # Loss is first element, accuracy is second
                epoch_d_accs.append(d_loss[1])
                epoch_g_losses.append(g_loss)
            
            # Calculate average losses for the epoch
            avg_triplet_loss = np.mean(epoch_triplet_losses)
            avg_d_loss = np.mean(epoch_d_losses)
            avg_g_loss = np.mean(epoch_g_losses)
            avg_d_acc = np.mean(epoch_d_accs)
            
            history['triplet_loss'].append(avg_triplet_loss)
            history['discriminator_loss'].append(avg_d_loss)
            history['generator_loss'].append(avg_g_loss)
            history['discriminator_accuracy'].append(avg_d_acc)
            
            # Log progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Triplet Loss: {avg_triplet_loss:.4f}, "
                    f"Discriminator Loss: {avg_d_loss:.4f}, "
                    f"Generator Loss: {avg_g_loss:.4f}, "
                    f"Discriminator Accuracy: {avg_d_acc:.4f}"
                )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return history
    
    def _create_triplet_batches(self, anchors, anchor_labels, power_features, power_labels, common_labels):
        """
        Create triplet batches for training.
        
        Parameters:
        -----------
        anchors : np.ndarray
            Anchor samples (HPC)
        anchor_labels : np.ndarray
            Labels for anchor samples
        power_features : np.ndarray
            Power dataset features
        power_labels : np.ndarray
            Power dataset labels
        common_labels : np.ndarray
            Common labels between datasets
            
        Returns:
        --------
        tuple or None
            (anchors, positives, negatives) or None if not enough samples
        """
        positives = []
        negatives = []
        valid_anchors = []
        
        for i, (anchor, anchor_label) in enumerate(zip(anchors, anchor_labels)):
            # Skip if label is not in common labels
            if anchor_label[0] not in common_labels:
                continue
                
            # Find positive samples (same label in Power dataset)
            positive_indices = np.where(power_labels == anchor_label[0])[0]
            
            # Find negative samples (different label in Power dataset)
            negative_indices = np.where(power_labels != anchor_label[0])[0]
            
            # Skip if not enough samples
            if len(positive_indices) == 0 or len(negative_indices) == 0:
                continue
                
            # Randomly select one positive and one negative
            positive_idx = np.random.choice(positive_indices)
            negative_idx = np.random.choice(negative_indices)
            
            positives.append(power_features[positive_idx])
            negatives.append(power_features[negative_idx])
            valid_anchors.append(anchor)
        
        # Return None if not enough valid triplets
        if len(valid_anchors) == 0:
            return None
            
        return np.array(valid_anchors), np.array(positives), np.array(negatives)
    
    def encode(self, features):
        """
        Encode features using the trained encoder.
        
        Parameters:
        -----------
        features : np.ndarray
            Input features to encode
            
        Returns:
        --------
        np.ndarray
            Encoded features
        """
        return self.encoder.predict(features)
    
    def save_model(self, model_dir):
        """
        Save the encoder and discriminator models.
        
        Parameters:
        -----------
        model_dir : str or Path
            Directory to save models
        """
        # Create model directory if it doesn't exist
        Path(model_dir).mkdir(exist_ok=True, parents=True)
        
        # Generate timestamp for unique filenames
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Save encoder
        encoder_path = Path(model_dir) / f"mmd_gan_encoder_{timestamp}.h5"
        self.encoder.save(encoder_path)
        
        # Save discriminator
        discriminator_path = Path(model_dir) / f"mmd_gan_discriminator_{timestamp}.h5"
        self.discriminator.save(discriminator_path)
        
        logger.info(f"Models saved to {model_dir}")
        
    def plot_training_history(self, history, save_path=None):
        """
        Plot the training history.
        
        Parameters:
        -----------
        history : dict
            Training history dictionary
        save_path : str or Path, optional
            Path to save the plot
        """
        plt.figure(figsize=(15, 10))
        
        # Plot losses
        plt.subplot(2, 1, 1)
        plt.plot(history['triplet_loss'], label='Triplet Loss')
        plt.plot(history['discriminator_loss'], label='Discriminator Loss')
        plt.plot(history['generator_loss'], label='Generator Loss')
        plt.title('Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot discriminator accuracy
        plt.subplot(2, 1, 2)
        plt.plot(history['discriminator_accuracy'], label='Discriminator Accuracy')
        plt.title('Discriminator Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            # Create directory if it doesn't exist
            Path(save_path).parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(save_path)
            logger.info(f"Training history plot saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
        
    def visualize_latent_space(self, X, y, save_path=None):
        """
        Visualize the latent space using t-SNE.
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
        y : np.ndarray
            Labels for coloring
        save_path : str or Path, optional
            Path to save the visualization
        """
        from sklearn.manifold import TSNE
        
        # Encode the features
        encoded_features = self.encode(X)
        
        # Apply t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        encoded_tsne = tsne.fit_transform(encoded_features)
        
        # Plot the results
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(encoded_tsne[:, 0], encoded_tsne[:, 1], c=y.reshape(-1), cmap='viridis', alpha=0.7)
        plt.colorbar(scatter)
        plt.title('t-SNE Visualization of MMD-GAN Encoded Features')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        
        if save_path:
            # Create directory if it doesn't exist
            Path(save_path).parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(save_path)
            logger.info(f"Latent space visualization saved to {save_path}")
        else:
            plt.show()
            
        plt.close()


class MMDFusionTrainer:
    """
    Trainer class for the MMD-GAN Encoder model.
    
    This class manages the training process for the MMD-GAN Encoder,
    including data preparation, training, evaluation, and visualization.
    """
    
    def __init__(self, latent_dim=16, batch_size=32, epochs=100, triplet_margin=1.0):
        """
        Initialize the MMD Fusion Trainer.
        
        Parameters:
        -----------
        latent_dim : int, default=16
            Dimension of the latent space
        batch_size : int, default=32
            Batch size for training
        epochs : int, default=100
            Number of epochs to train
        triplet_margin : float, default=1.0
            Margin parameter for triplet loss
        """
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.triplet_margin = triplet_margin
        
    def train_mmd_encoder(self, X_hpc, y_hpc, X_power, y_power):
        """
        Train a MMD-GAN Encoder on the HPC and Power datasets.
        
        Parameters:
        -----------
        X_hpc : np.ndarray
            HPC dataset features
        y_hpc : np.ndarray
            HPC dataset labels
        X_power : np.ndarray
            Power dataset features
        y_power : np.ndarray
            Power dataset labels
            
        Returns:
        --------
        tuple
            (encoder, history)
        """
        # Create encoder for HPC data
        logger.info(f"Creating MMD-GAN encoder with latent dimension {self.latent_dim}")
        encoder = MMDGANEncoder(
            input_dim=X_hpc.shape[1],
            latent_dim=self.latent_dim,
            triplet_margin=self.triplet_margin
        )
        
        # Train the encoder
        logger.info("Training MMD-GAN encoder")
        history = encoder.train(
            X_hpc=X_hpc,
            y_hpc=y_hpc,
            X_power=X_power,
            y_power=y_power,
            epochs=self.epochs,
            batch_size=self.batch_size
        )
        
        return encoder, history
    
    def process_data(self, encoder, X_hpc, y_hpc, X_power, y_power):
        """
        Process data using the trained encoder.
        
        Parameters:
        -----------
        encoder : MMDGANEncoder
            Trained encoder
        X_hpc : np.ndarray
            HPC dataset features
        y_hpc : np.ndarray
            HPC dataset labels
        X_power : np.ndarray
            Power dataset features
        y_power : np.ndarray
            Power dataset labels
            
        Returns:
        --------
        dict
            Dictionary containing the processed data
        """
        # Encode datasets
        logger.info("Encoding datasets with MMD-GAN encoder")
        latent_hpc = encoder.encode(X_hpc)
        latent_power = encoder.encode(X_power)
        
        # Find common labels
        common_labels = np.intersect1d(np.unique(y_hpc), np.unique(y_power))
        
        # Filter datasets by common labels
        mask_hpc = np.isin(y_hpc, common_labels)
        mask_power = np.isin(y_power, common_labels)
        
        X_hpc_filtered = latent_hpc[mask_hpc]
        y_hpc_filtered = y_hpc[mask_hpc]
        X_power_filtered = latent_power[mask_power]
        y_power_filtered = y_power[mask_power]
        
        # Combine datasets
        X_combined = np.vstack([X_hpc_filtered, X_power_filtered])
        y_combined = np.concatenate([y_hpc_filtered, y_power_filtered])
        
        return {
            'X_hpc': X_hpc_filtered,
            'y_hpc': y_hpc_filtered,
            'X_power': X_power_filtered,
            'y_power': y_power_filtered,
            'X_combined': X_combined,
            'y_combined': y_combined,
            'common_labels': common_labels
        }
    
    def evaluate(self, X, y, X_val=None, y_val=None):
        """
        Evaluate the encoded representations using a simple classifier.
        
        Parameters:
        -----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training labels
        X_val : np.ndarray, optional
            Validation features
        y_val : np.ndarray, optional
            Validation labels
            
        Returns:
        --------
        tuple
            (model, history)
        """
        from sklearn.model_selection import train_test_split
        
        # Split data if validation set not provided
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, y_train = X, y
        
        # Build a simple classifier
        num_classes = len(np.unique(y))
        
        inputs = Input(shape=(X_train.shape[1],))
        x = Dense(32, activation='relu')(inputs)
        x = Dense(16, activation='relu')(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train the classifier
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate on validation set
        val_loss, val_acc = model.evaluate(X_val, y_val)
        logger.info(f"Validation accuracy: {val_acc:.4f}")
        
        return model, history
    
    def save_results(self, encoder, history, output_dir, visualize=True):
        """
        Save the model and results.
        
        Parameters:
        -----------
        encoder : MMDGANEncoder
            Trained encoder
        history : dict
            Training history
        output_dir : str or Path
            Directory to save results
        visualize : bool, default=True
            Whether to generate visualizations
        """
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save encoder
        encoder.save_model(output_dir)
        
        # Visualizations
        if visualize and history is not None:
            vis_dir = output_dir / "visualizations"
            vis_dir.mkdir(exist_ok=True, parents=True)
            
            # Plot training history
            encoder.plot_training_history(
                history,
                save_path=vis_dir / "mmd_gan_training_history.png"
            )
            
            logger.info(f"Results saved to {output_dir}")