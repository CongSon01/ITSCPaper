"""
Utilities Package

This package contains various utility functions and modules for the ITSCPaper project.
"""

# Import core utilities for direct access from src.utils
from .logging_utils import setup_logging
from .plotting import (
    plot_confusion_matrix,
    plot_training_history,
    visualize_latent_space
)
from .file_utils import create_results_dir

# Define what's publicly available when using "from src.utils import *"
__all__ = [
    # Logging utilities
    'setup_logging',
    
    # General plotting utilities
    'plot_confusion_matrix',
    'plot_training_history', 
    'visualize_latent_space',
    
    # File utilities
    'create_results_dir',
]