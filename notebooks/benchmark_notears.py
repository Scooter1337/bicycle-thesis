import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path
import pytorch_lightning as pl
import torch
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from bicycle.model import BICYCLE
from bicycle.nodags_files.notears import NotearsClassWrapper
from bicycle.utils.metrics import compute_auprc
from bicycle.utils.data import process_data_for_llc

SEED = 0
pl.seed_everything(SEED)
device = torch.device("cpu")
MODEL_PATH = Path("/Users/luca/Developer/Universiteit/leiden-university/bachelor-project/models")
RESULTS_PATH = Path("/Users/luca/Developer/Universiteit/leiden-university/bachelor-project/results")
DATA_PATH = Path("/Users/luca/Developer/Universiteit/leiden-university/bachelor-project")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# Load real single-cell data
print("Loading real single-cell data...")
data = sc.read_h5ad(DATA_PATH / "sc_training.h5ad")
print(f"Loaded dataset with {data.n_obs} cells and {data.n_vars} genes")

# Filter for unperturbed and perturbed cells
unperturbed_cells = data[data.obs['condition'] == 'Unperturbed']
perturbed_cells = data[data.obs['condition'] != 'Unperturbed']

print(f"Unperturbed cells: {unperturbed_cells.n_obs}")
print(f"Perturbed cells: {perturbed_cells.n_obs}")
print(f"Number of different perturbations: {len(perturbed_cells.obs['condition'].unique())}")

# Select a subset of genes for analysis (to make computation feasible)
N_GENES = 50  # Increased from 10 since we have real data
selected_genes = data.var_names[:N_GENES].tolist()
print(f"Selected genes for analysis: {selected_genes[:10]}...")  # Show first 10

# Prepare data for NOTEARS analysis
unperturbed_data = unperturbed_cells[:, selected_genes].X
perturbed_data = perturbed_cells[:, selected_genes].X

# Convert sparse matrices to dense if needed
if hasattr(unperturbed_data, "toarray"):
    unperturbed_data = unperturbed_data.toarray()
if hasattr(perturbed_data, "toarray"):
    perturbed_data = perturbed_data.toarray()

# Split data into train/test sets
np.random.seed(SEED)
n_unperturbed = unperturbed_data.shape[0]
n_perturbed = perturbed_data.shape[0]

# Create train/validation/test splits
train_ratio = 0.7
valid_ratio = 0.15
test_ratio = 0.15

# Split unperturbed data
unperturbed_indices = np.random.permutation(n_unperturbed)
n_train_unperturbed = int(train_ratio * n_unperturbed)
n_valid_unperturbed = int(valid_ratio * n_unperturbed)

unperturbed_train = unperturbed_data[unperturbed_indices[:n_train_unperturbed]]
unperturbed_valid = unperturbed_data[unperturbed_indices[n_train_unperturbed:n_train_unperturbed+n_valid_unperturbed]]
unperturbed_test = unperturbed_data[unperturbed_indices[n_train_unperturbed+n_valid_unperturbed:]]

# Split perturbed data
perturbed_indices = np.random.permutation(n_perturbed)
n_train_perturbed = int(train_ratio * n_perturbed)
n_valid_perturbed = int(valid_ratio * n_perturbed)

perturbed_train = perturbed_data[perturbed_indices[:n_train_perturbed]]
perturbed_valid = perturbed_data[perturbed_indices[n_train_perturbed:n_train_perturbed+n_valid_perturbed]]
perturbed_test = perturbed_data[perturbed_indices[n_train_perturbed+n_valid_perturbed:]]

# Combine unperturbed and perturbed data for training
dataset_train = [np.vstack([unperturbed_train, perturbed_train])]
dataset_valid = [np.vstack([unperturbed_valid, perturbed_valid])]
dataset_test = [np.vstack([unperturbed_test, perturbed_test])]

# Create empty target arrays (NOTEARS doesn't need targets for unsupervised learning)
dataset_train_targets = [np.array([])]
dataset_valid_targets = [np.array([])]
dataset_test_targets = [np.array([])]

print(f"Training data shape: {dataset_train[0].shape}")
print(f"Validation data shape: {dataset_valid[0].shape}")
print(f"Test data shape: {dataset_test[0].shape}")

noise_scale = 0.5
if Path(RESULTS_PATH / "results_real_notears.parquet").exists():
    df_models = pd.read_parquet(RESULTS_PATH / "results_real_notears.parquet")
else:
    df_models = pd.DataFrame(
        columns=[
            "dataset",
            "n_genes",
            "n_train_samples",
            "l1",
            "noise_scale",
            "nll_pred_valid",
            "nll_pred_test"
        ]
    )
results = pd.DataFrame()
for loss_type in ["l2"]:
    for l1 in [1e-4, 1e-3, 1e-2, 1e-1, 1]:  # [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
        # Check if model is already in df_models
        if (
            df_models[
                (df_models["dataset"] == "real_single_cell")
                & (df_models["l1"] == l1)
                & (df_models["noise_scale"] == noise_scale)
                & (df_models["n_genes"] == N_GENES)
            ].shape[0]
            > 0
        ):
            print("Parameter set for model already exists, skipping...")
            continue

        print(f"Running NOTEARS with l1={l1}, loss_type={loss_type}")
        notears_wrapper = NotearsClassWrapper(lambda1=l1, loss_type=loss_type, noise_scale=noise_scale)
        est_beta = notears_wrapper.train(dataset_train, dataset_train_targets, return_weights=True)
        nll_pred_valid = notears_wrapper.predictLikelihood(dataset_valid, dataset_valid_targets)
        nll_pred_test = notears_wrapper.predictLikelihood(dataset_test, dataset_test_targets)

        # Append results to df_models
        df_models = pd.concat(
            [
                df_models,
                pd.DataFrame(
                    {
                        "dataset": "real_single_cell",
                        "n_genes": N_GENES,
                        "n_train_samples": dataset_train[0].shape[0],
                        "nll_pred_valid": np.mean(nll_pred_valid),
                        "nll_pred_test": np.mean(nll_pred_test),
                        "noise_scale": noise_scale,
                        "l1": l1,
                        "loss_type": loss_type,
                    },
                    index=[0],
                ),
            ],
            ignore_index=True,
        )

        df_models.to_parquet(RESULTS_PATH / "results_real_notears.parquet")
        print(f"Saved df_models with nrows: {len(df_models)}")
        
        # Visualize the learned causal graph for the best l1 parameter
        if l1 == 1e-2:  # You can adjust this to show results for different l1 values
            print(f"Learned adjacency matrix shape: {est_beta.shape}")
            print(f"Number of edges in learned graph: {np.sum(np.abs(est_beta) > 1e-3)}")
            
            # Create a simple visualization of the adjacency matrix
            plt.figure(figsize=(10, 8))
            plt.imshow(np.abs(est_beta), cmap='Blues')
            plt.colorbar(label='Edge weight magnitude')
            plt.title(f'Learned Causal Graph (l1={l1})')
            plt.xlabel('Target genes')
            plt.ylabel('Source genes')
            
            # Add gene names if not too many
            if N_GENES <= 20:
                plt.xticks(range(N_GENES), selected_genes, rotation=45, ha='right')
                plt.yticks(range(N_GENES), selected_genes)
            
            plt.tight_layout()
            plt.savefig(RESULTS_PATH / f"causal_graph_l1_{l1}.png", dpi=300, bbox_inches='tight')
            plt.show()

print("\nBenchmarking completed!")
print(f"Results saved to: {RESULTS_PATH / 'results_real_notears.parquet'}")
print(f"Final results summary:")
print(df_models.groupby(['l1']).agg({
    'nll_pred_valid': 'mean',
    'nll_pred_test': 'mean'
}).round(4))
