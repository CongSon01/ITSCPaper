"""
Loss functions for the MMD-GAN model.

This module implements various loss functions used in the MMD-GAN model,
including triplet loss and Maximum Mean Discrepancy (MMD) loss.
"""

import tensorflow as tf
import numpy as np

def euclidean_distance(x, y):
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

def triplet_loss(anchor, positive, negative, margin=1.0):
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
    margin : float, default=1.0
        Margin for triplet loss
        
    Returns:
    --------
    tf.Tensor
        Triplet loss value
    """
    pos_dist = euclidean_distance(anchor, positive)
    neg_dist = euclidean_distance(anchor, negative)
    
    # Triplet loss formula: max(0, d(a,p) - d(a,n) + margin)
    basic_loss = pos_dist - neg_dist + margin
    loss = tf.maximum(basic_loss, 0.0)
    return tf.reduce_mean(loss)

def compute_mmd(x, y, kernel='rbf', sigma_list=None):
    """
    Calculate Maximum Mean Discrepancy (MMD) between two sets of samples.
    
    Parameters:
    -----------
    x : tf.Tensor
        First set of samples
    y : tf.Tensor
        Second set of samples
    kernel : str, default='rbf'
        Kernel function to use ('rbf' or 'imq')
    sigma_list : list, optional
        List of sigma values for the kernel
        
    Returns:
    --------
    tf.Tensor
        MMD value
    """
    if sigma_list is None:
        sigma_list = [1.0, 5.0, 10.0]  # Default sigmas
    
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    
    # Concatenate all samples
    all_samples = tf.concat([x, y], axis=0)
    
    # Calculate pairwise distances
    xx = tf.matmul(x, x, transpose_b=True)
    xy = tf.matmul(x, y, transpose_b=True)
    yy = tf.matmul(y, y, transpose_b=True)
    
    # Calculate self-dot products
    x_sq = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
    y_sq = tf.reduce_sum(tf.square(y), axis=1, keepdims=True)
    
    # Calculate pairwise squared distances
    x_sq_dist = x_sq + tf.transpose(x_sq) - 2 * xx
    xy_sq_dist = x_sq + tf.transpose(y_sq) - 2 * xy
    y_sq_dist = y_sq + tf.transpose(y_sq) - 2 * yy
    
    # Apply kernel to distances
    def _kernel(dist, sigma):
        if kernel == 'rbf':
            return tf.exp(-dist / (2 * sigma ** 2))
        elif kernel == 'imq':
            return tf.pow(dist + sigma ** 2, -0.5)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")
    
    # Apply multiple kernels and sum results
    total_mmd = 0.0
    for sigma in sigma_list:
        # Apply kernel to all pairwise distances
        xx_kernel = _kernel(x_sq_dist, sigma)
        xy_kernel = _kernel(xy_sq_dist, sigma)
        yy_kernel = _kernel(y_sq_dist, sigma)
        
        # Calculate MMD
        x_factor = 1.0 / tf.cast(x_size * (x_size - 1), tf.float32)
        y_factor = 1.0 / tf.cast(y_size * (y_size - 1), tf.float32)
        xy_factor = -2.0 / tf.cast(x_size * y_size, tf.float32)
        
        # Sum off-diagonal terms
        xx_sum = tf.reduce_sum(xx_kernel) - tf.reduce_sum(tf.linalg.diag_part(xx_kernel))
        yy_sum = tf.reduce_sum(yy_kernel) - tf.reduce_sum(tf.linalg.diag_part(yy_kernel))
        xy_sum = tf.reduce_sum(xy_kernel)
        
        # Calculate MMD for this kernel
        kernel_mmd = x_factor * xx_sum + y_factor * yy_sum + xy_factor * xy_sum
        
        total_mmd += kernel_mmd
    
    return total_mmd

def adversarial_loss(real_logits, fake_logits, smoothing=0.1):
    """
    Calculate adversarial loss for GAN training.
    
    Parameters:
    -----------
    real_logits : tf.Tensor
        Discriminator logits for real samples
    fake_logits : tf.Tensor
        Discriminator logits for fake samples
    smoothing : float, default=0.1
        Label smoothing factor
        
    Returns:
    --------
    tuple
        (discriminator_loss, generator_loss)
    """
    # Label smoothing
    real_labels = tf.ones_like(real_logits) * (1.0 - smoothing)
    fake_labels = tf.zeros_like(fake_logits) + smoothing
    
    # Discriminator loss
    d_loss_real = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
        real_labels, real_logits, from_logits=False
    ))
    d_loss_fake = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
        fake_labels, fake_logits, from_logits=False
    ))
    d_loss = 0.5 * (d_loss_real + d_loss_fake)
    
    # Generator loss
    g_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
        tf.ones_like(fake_logits), fake_logits, from_logits=False
    ))
    
    return d_loss, g_loss

def create_triplet_batches(anchors, anchor_labels, dataset_features, dataset_labels, common_labels):
    """
    Create triplet batches for training.
    
    Parameters:
    -----------
    anchors : np.ndarray
        Anchor samples (e.g., HPC)
    anchor_labels : np.ndarray
        Labels for anchor samples
    dataset_features : np.ndarray
        Features from the other dataset (e.g., Power)
    dataset_labels : np.ndarray
        Labels for the other dataset
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
        anchor_label_val = anchor_label[0] if isinstance(anchor_label, np.ndarray) else anchor_label
        if anchor_label_val not in common_labels:
            continue
            
        # Find positive samples (same label in the other dataset)
        positive_indices = np.where(dataset_labels == anchor_label_val)[0]
        
        # Find negative samples (different label in the other dataset)
        negative_indices = np.where(dataset_labels != anchor_label_val)[0]
        
        # Skip if not enough samples
        if len(positive_indices) == 0 or len(negative_indices) == 0:
            continue
            
        # Randomly select one positive and one negative
        positive_idx = np.random.choice(positive_indices)
        negative_idx = np.random.choice(negative_indices)
        
        positives.append(dataset_features[positive_idx])
        negatives.append(dataset_features[negative_idx])
        valid_anchors.append(anchor)
    
    # Return None if not enough valid triplets
    if len(valid_anchors) == 0:
        return None
        
    return np.array(valid_anchors), np.array(positives), np.array(negatives)