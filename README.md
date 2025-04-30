# Multi-domain Intrusion Detection with Feature Fusion

This project implements a multi-domain intrusion detection approach that combines data from two different sources: power system data and HPC (High-Performance Computing) kernel events data. It uses advanced feature fusion techniques including MMD-GAN to map heterogeneous data features into a common latent space.

## Project Structure

```
NewCode/
├── data/                        # Data directory
│   ├── processed/               # Preprocessed datasets
│   └── raw/                     # Raw datasets (HPC.csv, Power.csv)
├── logs/                        # Log files
├── models/                      # Trained models
├── notebooks/                   # Jupyter notebooks for analysis
├── results/                     # Experiment results
├── src/                         # Source code
│   ├── config.py                # Configuration settings
│   ├── data_preprocessing.py    # Data preprocessing utilities
│   ├── evaluate_fusion.py       # Evaluation utilities
│   ├── feature_fusion.py        # Feature fusion implementations
│   ├── model.py                 # Model architecture & training logic
│   ├── utils.py                 # Helper functions
│   └── models/                  # Model implementations
│       └── mmd_gan.py           # MMD-GAN encoder implementation
├── data_preprocessing_functions.py  # Helper functions for data preprocessing
├── data_preprocessing.ipynb     # Interactive data preprocessing notebook
├── evaluate_mmd_gan.py          # Script to evaluate MMD-GAN performance
├── main.py                      # Main entry point
├── train_mmd_gan.py             # Script to train MMD-GAN encoder
├── train_tf_model.py            # Script to train TensorFlow model
└── requirements.txt             # Project dependencies
```

## Installation

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Setup

1. Clone the repository and navigate to the project directory:

```bash
git clone <repository-url>
cd NewCode
```

2. Create and activate a virtual environment:

```bash
# On Windows
python -m venv ITSS
ITSS\Scripts\activate

# On Linux/Mac
python -m venv ITSS
source ITSS/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

1. Place your raw datasets in the `data/raw/` directory:
   - `HPC.csv`: High-Performance Computing kernel events data
   - `Power.csv`: Power system data

2. Preprocess the data:

```bash
python data_preprocessing.py
```

Alternatively, you can run the Jupyter notebook for interactive preprocessing:

```bash
jupyter notebook data_preprocessing.ipynb
```

## Training Models

### Training MMD-GAN Encoder

The MMD-GAN encoder is used to map features from diverse datasets into a shared latent space:

```bash
python train_mmd_gan.py --epochs 100 --batch-size 32 --latent-dim 16
```

Optional arguments:
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size for training (default: 32)
- `--latent-dim`: Dimensionality of latent space (default: 16)
- `--learning-rate`: Learning rate (default: 0.001)

### Training the Classification Model

Train the TensorFlow classification model:

```bash
python train_tf_model.py --fusion-method domain_adaptation
```

Optional arguments:
- `--fusion-method`: Feature fusion method to use (options: 'concatenation', 'weighted', 'pca', 'cca', 'kernel', 'supervised', 'domain_adaptation')
- `--latent-dim`: Dimensionality of latent space (default: 32)

## Evaluation

Evaluate the trained MMD-GAN encoder:

```bash
python evaluate_mmd_gan.py --model-path models/mmd_gan_encoder_<timestamp>.h5
```

## Running the Complete Pipeline

To run the complete pipeline from data preprocessing to model evaluation:

```bash
python main.py
```

## Visualizing Results

The training process generates various visualizations in the `results/` directory, including:
- t-SNE plots of the latent space
- Training and validation metrics
- Feature importance plots

You can also explore the data and results using the provided Jupyter notebooks in the `notebooks/` directory.

## Troubleshooting

### Common Issues

1. **GPU-related errors**: If you encounter CUDA errors, try running with CPU only:
   ```bash
   export CUDA_VISIBLE_DEVICES=-1  # On Linux/Mac
   set CUDA_VISIBLE_DEVICES=-1     # On Windows
   ```

2. **Memory errors**: Try reducing batch size:
   ```bash
   python train_mmd_gan.py --batch-size 16
   ```

3. **Log files**: Check the log files in the `logs/` directory for detailed error messages.

## Citation

If you use this code in your research, please cite:

```
[Your citation information here]
```

## License

[Your license information here]