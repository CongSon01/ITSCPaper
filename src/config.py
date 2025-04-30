"""
Project Configuration File

This module contains all the configuration settings for the project,
including file paths, model hyperparameters, and other constants.
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# File paths
POWER_RAW_FILE = RAW_DATA_DIR / "Power.csv"
HPC_RAW_FILE = RAW_DATA_DIR / "HPC.csv"
POWER_PROCESSED_FILE = PROCESSED_DATA_DIR / "Power_processed.csv"
HPC_PROCESSED_FILE = PROCESSED_DATA_DIR / "HPC_processed.csv"

# Data preprocessing parameters
RANDOM_STATE = 42
N_COMPONENTS_PCA = 30
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Model hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100