"""
Triplet Loss Module

This module implements triplet loss for learning embeddings that map similar samples closer
together and dissimilar samples farther apart in the embedding space.
"""

import tensorflow as tf

def euclidean_distance(x, y):
    """
    Calculate Euclidean distance between two tensors.
    
    Parameters:
    -----------
    x : tf.Tensor
        First tensor of shape (batch_size, embedding_dim)
    y : tf.Tensor
        Second tensor of shape (batch_size, embedding_dim)
        
    Returns:
    --------
    tf.Tensor
        Euclidean distance of shape (batch_size, 1)
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
        Margin parameter for triplet loss
            
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