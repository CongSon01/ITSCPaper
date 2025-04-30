"""
Visualization Utilities Subpackage

This subpackage contains specialized visualization utilities for different model types.
"""

# Import visualization utilities
from .mmd_gan import (
    plot_latent_distributions,
    plot_training_history_mmd,
    plot_latent_space_comparison
)

# Define what's publicly available when using "from src.utils.visualization import *"
__all__ = [
    # MMD-GAN visualization utilities
    'plot_latent_distributions',
    'plot_training_history_mmd',
    'plot_latent_space_comparison',
]