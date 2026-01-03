# Industrial Time Series Forecasting with Chronos-2

This repository contains code for forecasting industrial time series data using the Chronos-2 model.

## 1. Installation

### A. Set up Conda Environment
1. Install Conda if not already installed ([Instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)).

2. Create and activate a new environment (Python 3.10 recommended):
    ```bash
    conda create -n <env_name> python=3.10 -y
    conda activate <env_name>
    ```

### B. Install Dependencies
**Important:** Please install PyTorch with CUDA support first to enable GPU acceleration.

1. Install PyTorch (Visit [pytorch.org](https://pytorch.org/get-started/locally/) for your specific CUDA version):
    ```bash
    # Example for Linux/Windows with CUDA 12.6 (Adjust as needed)
    pip install torch --index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126)
    ```

2. Install other requirements:
    ```bash
    pip install -r requirements.txt
    ```

## 2. Usage

You can run the full suite of experiments (Baseline to Fine-tuning) using the provided shell script.

1. Grant execution permission (first time only):
    ```bash
    chmod +x run_experiments.sh
    ```

2. Run experiments:
    ```bash
    ./run_experiments.sh
    ```
    *Or run manually:* `bash run_experiments.sh`

### 📁 Project Structure

```
Industrial-Time-Series-Forecasting-with-Chronos-2/
├── Chronos.py              # Main Chronos forecaster class (fine-tuning, inference, evaluation)
├── chronos_run.py          # Main entry point script for running experiments
├── run_experiments.sh      # Shell script for running all experiment scenarios
├── requirements.txt        # Python dependencies
├── LICENSE                 # License file
├── README.md               # Project documentation
├── __init__.py             # Package initializer
│
├── Dataset/                # Dataset handling module
│   ├── custom_dataset.py   # Custom dataset classes and data loaders
│   └── (dataset csv file)  # Add your dataset here
│
├── Result/                 # Directory for plots of some samples at each trial 
│
└── utils/                  # Utility functions
    ├── set_seed.py         # Random seed setting for reproducibility
    └── util.py             # Visualization and plotting utilities
```
### Key Files Description

| File | Description |
|------|-------------|
| `Chronos.py` | Contains `ChronosForecaster` class that wraps the Chronos-2 pipeline for time series forecasting with covariate selection, fine-tuning, and evaluation |
| `chronos_run.py` | Entry point script with argument parsing and experiment orchestration logic |
| `run_experiments.sh` | Automated script to run baseline, covariate, cross-learning, and fine-tuning experiments |
| `Dataset/custom_dataset.py` | Implements `Dataset_Custom` for data loading, preprocessing, and `DataLoader` creation. In this repository, any dataset is not included. |
| `utils/util.py` | Generates plots for samples corresponding to the MSE quartiles (0%, 25%, 50%, 75%, and 100%) |


### Experiment Scenarios included:
- **Baseline:** Chronos (Zero-shot)
- **Covariates:** Using process data (e.g., CO2, Temp) as context (Future values of covariates are not used)
- **Cross-Learning:** Refer here [![Chronos-2-Report](https://img.shields.io/badge/Chronos--2--Report-2510.15821-red)](https://arxiv.org/abs/2510.15821) 
- **Fine-tuning:** Training Chronos on Samyang data
