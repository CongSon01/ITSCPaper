"""
File and Directory Utilities

This module provides functions for file and directory operations.
"""

import os
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def create_results_dir(base_dir='results', prefix='run'):
    """
    Create a timestamped results directory
    
    Parameters:
    -----------
    base_dir : str, default='results'
        Base directory for results
    prefix : str, default='run'
        Prefix for the results directory name
        
    Returns:
    --------
    Path
        Path to the created directory
    """
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Create timestamped directory
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    results_dir = f"{prefix}_{timestamp}"
    
    # Create full path
    full_path = Path(base_dir) / results_dir
    os.makedirs(full_path, exist_ok=True)
    
    logger.info(f"Created results directory: {full_path}")
    
    return full_path