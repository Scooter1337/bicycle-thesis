# Bicycle Thesis

This repository contains the codebase and LaTeX source files for the thesis. The implementation focuses on Bicycle.

## Requirements

The project uses Anaconda/Miniconda for managing Python dependencies. You can find the exact list in `environment.yml`. The primary requirements include:

- Python 3.11
- PyTorch = 2.1.*
- PyTorch Lightning
- Pandas, Numpy, Matplotlib
- Scanpy

## Installation

You can recreate the environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate bachelor-py311
```

## Usage

1. **Activate the environment**: Ensure the `bachelor-py311` conda environment is active.
2. **Download data**: Run the `notebooks/1_download_data.ipynb` Jupyter notebook to obtain the necessary datasets.
3. **Run Experiments**: Example entry points to run the experiments are in the `notebooks/` directory.

```bash
cd notebooks
python run_experiments.py
```
