"""
Logging Utilities

This module provides functions for setting up and configuring logging for the project.
"""

import os
import logging
from datetime import datetime
from pathlib import Path

def setup_logging(log_dir='logs', level=logging.INFO):
    """
    Set up logging for the application
    
    Parameters:
    -----------
    log_dir : str, default='logs'
        Directory to store log files
    level : int, default=logging.INFO
        Logging level
        
    Returns:
    --------
    logging.Logger
        Configured logger
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a timestamped log filename
    log_filename = f"log_{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
    log_filepath = os.path.join(log_dir, log_filename)
    
    # Set up logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers to prevent duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_filepath}")
    return logger