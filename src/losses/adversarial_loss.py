"""
Adversarial Loss Module

This module implements adversarial loss for GANs, particularly for training encoders
to generate latent representations that match a certain distribution.
"""

import tensorflow as tf
import numpy as np

def binary_adversarial_loss(is_real_pred, is_real_target=True):
    """
    Calculate binary adversarial loss.
    
    Parameters:
    -----------
    is_real_pred : tf.Tensor
        The prediction from the discriminator (0 to 1)
    is_real_target : bool, default=True
        Whether the target is real (True) or fake (False)
        
    Returns:
    --------
    tf.Tensor
        Binary cross entropy loss
    """
    if is_real_target:
        target = tf.ones_like(is_real_pred)
    else:
        target = tf.zeros_like(is_real_pred)
        
    return tf.keras.losses.binary_crossentropy(target, is_real_pred)

def wasserstein_loss(y_true, y_pred):
    """
    Calculate Wasserstein loss.
    
    Parameters:
    -----------
    y_true : tf.Tensor
        Real or fake labels (-1 for fake, 1 for real)
    y_pred : tf.Tensor
        Critic output
        
    Returns:
    --------
    tf.Tensor
        Wasserstein loss
    """
    return tf.reduce_mean(y_true * y_pred)

def gradient_penalty(discriminator, real_samples, fake_samples, labels=None):
    """
    Calculate gradient penalty for improved Wasserstein GAN.
    
    Parameters:
    -----------
    discriminator : tf.keras.Model
        The discriminator model
    real_samples : tf.Tensor
        Real data samples
    fake_samples : tf.Tensor
        Generated/fake data samples
    labels : tf.Tensor, optional
        Class labels if the discriminator is conditional
        
    Returns:
    --------
    tf.Tensor
        Gradient penalty term
    """
    batch_size = real_samples.shape[0]
    alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
    
    # Get random interpolation between real and fake samples
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        if labels is not None:
            pred = discriminator([interpolated, labels])
        else:
            pred = discriminator(interpolated)
            
    # Calculate gradients w.r.t. interpolated samples
    gradients = tape.gradient(pred, interpolated)
    
    # Compute the gradient norm
    gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
    
    # Return the gradient penalty
    return tf.reduce_mean(tf.square(gradient_norm - 1.0))

def mmd_loss(x, y, kernel='rbf', sigma=0.1):
    """
    Calculate Maximum Mean Discrepancy (MMD) loss between two distributions.
    
    Parameters:
    -----------
    x : tf.Tensor
        First distribution samples
    y : tf.Tensor
        Second distribution samples
    kernel : str, default='rbf'
        Kernel function ('rbf' or 'imq' for inverse multi-quadratic)
    sigma : float, default=0.1
        Kernel bandwidth parameter
        
    Returns:
    --------
    tf.Tensor
        MMD loss
    """
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    
    def _kernel(x, y):
        if kernel == 'rbf':
            # RBF kernel
            dist = tf.reduce_sum(tf.square(x[:, None] - y[None, :]), axis=2)
            return tf.exp(-dist / (2 * sigma * sigma))
        elif kernel == 'imq':
            # Inverse Multi-Quadratic kernel
            dist = tf.reduce_sum(tf.square(x[:, None] - y[None, :]), axis=2)
            return 1.0 / tf.sqrt(1.0 + dist / (2 * sigma * sigma))
        else:
            raise ValueError(f"Unknown kernel: {kernel}")
    
    # Calculate kernel matrices
    xx = _kernel(x, x)
    xy = _kernel(x, y)
    yy = _kernel(y, y)
    
    # Calculate MMD loss
    x_terms = tf.reduce_sum(xx) / tf.cast(x_size * x_size, tf.float32)
    y_terms = tf.reduce_sum(yy) / tf.cast(y_size * y_size, tf.float32)
    xy_terms = tf.reduce_sum(xy) / tf.cast(x_size * y_size, tf.float32)
    
    return x_terms + y_terms - 2 * xy_terms