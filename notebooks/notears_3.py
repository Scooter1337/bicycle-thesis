import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", category=FutureWarning)

from bicycle import model
import pandas as pd
import time
import os
from pathlib import Path
from os import environ
import pytorch_lightning as pl
import torch
from bicycle.dictlogger import DictLogger
from bicycle.model import BICYCLE
from bicycle.utils.data import (
    create_data,
    create_loaders,
    # get_ring_mask,
    get_diagonal_mask,
    compute_inits,
)
from bicycle.utils.plotting import plot_training_results
from pytorch_lightning.callbacks import RichProgressBar, StochasticWeightAveraging
from bicycle.callbacks import CustomModelCheckpoint, GenerateCallback, MyLoggerCallback
import click
import numpy as np
from bicycle.nodags_files.notears import NotearsClassWrapper
from bicycle.utils.metrics import compute_auprc
from bicycle.utils.data import process_data_for_llc


MODEL_PATH = Path("/Users/luca/Developer/Universiteit/leiden-university/bachelor-project/models")
RESULTS_PATH = Path("/Users/luca/Developer/Universiteit/leiden-university/bachelor-project/results")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
noise_scale = 0.5


@click.command()
@click.option("--nlogo", default=10, type=int)
@click.option("--seed", default=1, type=int)
@click.option("--lr", default=1e-3, type=float)
@click.option("--n-genes", default=10, type=int)
@click.option("--scale-l1", default=1, type=float)
@click.option("--scale-kl", default=1, type=float)
@click.option("--scale-spectral", default=1, type=float)
@click.option("--scale-lyapunov", default=0, type=float)
@click.option("--swa", default=0, type=int)
@click.option("--n-samples-control", default=250, type=int)
@click.option("--n-samples-per-perturbation", default=250, type=int)
@click.option("--validation-size", default=0.0, type=float)
@click.option("--sem", default="linear-ou", type=str)
@click.option("--use-latents", default=False, type=bool)
@click.option("--intervention-scale", default=1.0, type=float)
@click.option("--rank-w-cov-factor", default=10, type=int)
def run_bicycle_training(
    nlogo,
    seed,
    lr,
    n_genes,
    scale_l1,
    scale_kl,
    scale_spectral,
    scale_lyapunov,
    swa,
    n_samples_control,
    n_samples_per_perturbation,
    validation_size,
    sem,
    use_latents,
    intervention_scale,
    rank_w_cov_factor,
):
    SEED = seed
    SERVER = False
    pl.seed_everything(SEED)
    torch.set_float32_matmul_precision("high")

    #
    # Paths
    #
    MODEL_PATH = Path("/Users/luca/Developer/Universiteit/leiden-university/bachelor-project/models/")
    DATA_PATH = Path("/Users/luca/Developer/Universiteit/leiden-university/bachelor-project/data/")
    PLOT_PATH = Path("/Users/luca/Developer/Universiteit/leiden-university/bachelor-project/plots/")
    
    if not MODEL_PATH.exists():
        MODEL_PATH.mkdir(parents=True, exist_ok=True)
    if not PLOT_PATH.exists():
        PLOT_PATH.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        DATA_PATH.mkdir(parents=True, exist_ok=True)

    # print("Checking CUDA availability:")
    # if not torch.cuda.is_available():
    #     raise RuntimeError("CUDA not available")

    # Convert to int if possible
    if scale_l1.is_integer():
        scale_l1 = int(scale_l1)
    if scale_kl.is_integer():
        scale_kl = int(scale_kl)
    if scale_spectral.is_integer():
        scale_spectral = int(scale_spectral)
    if scale_lyapunov.is_integer():
        scale_lyapunov = int(scale_lyapunov)

    #
    # Settings
    #
    device = torch.device("cpu")

    graph = "cycle-random"
    graph_kwargs = {"abs_weight_low": 0.25, "abs_weight_high": 0.95, "p_success": 0.4}
    graph_kwargs_str = "_".join([f"{v}" for v in graph_kwargs.values()])
    n_additional_entries = 12

    # LEARNING
    batch_size = 1024
    USE_INITS = False
    use_encoder = False
    n_epochs = 20_000
    optimizer = "adam"
    # DATA
    # nlogo REPRESENTS THE NUMBER OF GROUPS THAT SHOULD BE LEFT OUT DURING TRAINING
    LOGO = sorted(list(np.random.choice(n_genes, nlogo, replace=False)))
    train_gene_ko = [str(x) for x in set(range(0, n_genes)) - set(LOGO)]  # We start counting at 0
    # FIXME: There might be duplicates...
    ho_perturbations = sorted(list(set([tuple(sorted(np.random.choice(n_genes, 2, replace=False))) for _ in range(0, 20)])))
    test_gene_ko = [f"{x[0]},{x[1]}" for x in ho_perturbations]

    # DGP
    rank_w_cov_factor = n_genes - 1  # Same as dictys: #min(TFs, N_GENES-1)
    add_mask = False
    n_contexts = n_genes + 1  # Number of contexts
    n_samples_control = 250
    n_samples_per_perturbation = 250
    perfect_interventions = True
    make_counts = False  # True | also set x_distribution
    # LEARNING
    early_stopping = False
    early_stopping_patience = 500
    early_stopping_min_delta = 0.01
    x_distribution = "Normal"  # "Poisson"
    # DATA
    validation_size = 0.2
    # MODEL
    lyapunov_penalty = False
    GPU_DEVICE = 1
    plot_epoch_callback = 1000
    # RESULTS
    name_prefix = f"v3_inc_{graph}_{graph_kwargs_str}_{use_encoder}_{optimizer}_{batch_size}_{lyapunov_penalty}_{x_distribution}"
    SAVE_PLOT = True
    CHECKPOINTING = True
    VERBOSE_CHECKPOINTING = False
    OVERWRITE = False
    # REST
    n_samples_total = (
        n_samples_control + (len(train_gene_ko) + len(test_gene_ko)) * n_samples_per_perturbation
    )
    check_val_every_n_epoch = 1
    log_every_n_steps = 1  # We don't need more on the server

    #
    # Create Mask
    #
    # if add_mask:
    #     if graph == "cycle":
    #         mask = get_ring_mask(n_additional_entries, n_genes, device)
    #     else:
    #         raise NotImplementedError("Mask only implemented for DGP cycle")
    # else:
    #     mask = get_diagonal_mask(n_genes, device)

    #
    # Create synthetic data
    #
    _, _, samples, gt_interv, sim_regime, beta = create_data(
        n_genes,
        n_samples_control=n_samples_control,
        n_samples_per_perturbation=n_samples_per_perturbation,
        device=device,
        make_counts=make_counts,
        train_gene_ko=train_gene_ko,
        test_gene_ko=test_gene_ko,
        graph=graph,
        **graph_kwargs,
    )

    train_loader, validation_loader, test_loader = create_loaders(
        samples,
        sim_regime,
        validation_size,
        batch_size,
        SEED,
        train_gene_ko,
        test_gene_ko,
    )

    # Check if eig value of identity matrix minus beta in all contexts are < 0
    B = torch.eye(n_genes) - (1.0 - torch.eye(n_genes)) * beta.T
    for k in range(0, 11):
        B_p = B.clone()
        if k < 10:
            B_p[:, k] = 0
        eig_values = torch.real(torch.linalg.eigvals(-B_p))
        if torch.any(eig_values > 0):
            raise ValueError("Eigenvalues of identity matrix minus beta are not all negative.")

    if USE_INITS:
        init_tensors = compute_inits(train_loader.dataset, rank_w_cov_factor, n_contexts)

    print(f"Number of training samples: {len(train_loader.dataset)}")
    if validation_size > 0:
        print(f"Number of validation samples: {len(validation_loader.dataset)}")
    if LOGO:
        print(f"Number of test samples: {len(test_loader.dataset)}")

    if SERVER:
        device = torch.device("cuda")
        devices = "auto"
    else:
        device = torch.device("cpu")
        devices = 1
    print(device)
    gt_interv = gt_interv.to(device)
    n_genes = samples.shape[1]

    file_dir = (
        name_prefix
        + f"_{nlogo}_{seed}_{lr}_{n_genes}_{scale_l1}_{scale_kl}_{scale_spectral}_{scale_lyapunov}_{swa}_{n_samples_control}_{n_samples_per_perturbation}_{validation_size}_{sem}_{use_latents}_{intervention_scale}_{rank_w_cov_factor}"
    ) 

    # If final plot or final model exists: do not overwrite by default
    final_file_name = os.path.join(MODEL_PATH, file_dir, "last.ckpt")
    final_plot_name = os.path.join(PLOT_PATH, file_dir, "last.png")
    if (Path(final_file_name).exists() & SAVE_PLOT & ~OVERWRITE) | (
        Path(final_plot_name).exists() & CHECKPOINTING & ~OVERWRITE
    ):
        print("Files already exists, skipping...")
        pass
    else:
        print("Files do not exist, fitting model...")
        print("Deleting dirs")
        # Delete directories of files
        if Path(final_file_name).exists():
            print(f"Deleting {final_file_name}")
            # Delete all files in os.path.join(MODEL_PATH, file_name)
            for f in os.listdir(os.path.join(MODEL_PATH, file_dir)):
                os.remove(os.path.join(MODEL_PATH, file_dir, f))
        if Path(final_plot_name).exists():
            print(f"Deleting {final_plot_name}")
            for f in os.listdir(os.path.join(PLOT_PATH, file_dir)):
                os.remove(os.path.join(PLOT_PATH, file_dir, f))

        print("Creating dirs")
        # Create directories
        Path(os.path.join(MODEL_PATH, file_dir)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(PLOT_PATH, file_dir)).mkdir(parents=True, exist_ok=True)

        # Save pickle of train_loader into final_file_name
        torch.save(train_loader, os.path.join(MODEL_PATH, file_dir, "train_loader.pth"))
        torch.save(validation_loader, os.path.join(MODEL_PATH, file_dir, "validation_loader.pth"))
        torch.save(test_loader, os.path.join(MODEL_PATH, file_dir, "test_loader.pth"))

        for loss_type in ["l2"]:
            for l1 in [1e-4, 1e-3, 1e-2, 1e-1, 1]:  # [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
            # Check if model is already in df_models
                # if (
                #     df_models[
                #         (df_models["filename"] == str(filename))
                #         & (df_models["l1"] == l1)
                #         & (df_models["noise_scale"] == noise_scale)
                #     ].shape[0]
                #     > 0
                # ):
                #     print("Parameter set for model already exists, skipping...")
                #     continue
                filename = os.path.join(MODEL_PATH, file_dir, "test_loader.pth")

                dataset_test, dataset_test_targets = process_data_for_llc(test_loader, gt_interv, model.test_gene_ko)
                dataset_train = []
                for batch in train_loader:
                    data = batch[0][batch[1] == 0]  # Context 0 is intervention samples
                    dataset_train.append(data)
                dataset_train = [torch.cat(dataset_train, dim=0).detach().cpu().numpy()]
                dataset_train_targets = [np.array([])]

                dataset_valid = []
                dataset_valid_targets = []
                for k in range(len(dataset_train)):
                    train_idx = np.random.choice(len(dataset_train[k]), int(0.8 * len(dataset_train[k])), replace=False)
                    valid_idx = np.setdiff1d(np.arange(len(dataset_train[k])), train_idx)

                    dataset_valid.append(dataset_train[k][valid_idx])
                    dataset_valid_targets.append(dataset_train_targets[k])

                    dataset_train[k] = dataset_train[k][train_idx]

                notears_wrapper = NotearsClassWrapper(lambda1=l1, loss_type=loss_type, noise_scale=noise_scale)
                est_beta = notears_wrapper.train(dataset_train, dataset_train_targets, return_weights=True)
                nll_pred_valid = notears_wrapper.predictLikelihood(dataset_valid, dataset_valid_targets)
                est_beta = torch.from_numpy(est_beta).to(device).detach().cpu()

                # area_test = compute_auprc(gt_beta, est_beta)
                nll_pred_test = notears_wrapper.predictLikelihood(dataset_test, dataset_test_targets)

                # Append results to df_models
                df_models = pd.concat(
                    [
                        df_models,
                        pd.DataFrame(
                            {
                                "filename": str(filename),
                                "nlogo": nlogo,
                                "seed": seed,
                                "sem": sem,
                                "n_samples_control": n_samples_control,
                                # "intervention_scale": intervention_scale,
                                "nll_pred_valid": np.mean(nll_pred_valid),
                                "nll_pred_test": np.mean(nll_pred_test),
                                "noise_scale": noise_scale,
                                # "area_test": area_test,
                                "l1": l1,
                            },
                            index=[0],
                        ),
                    ],
                    ignore_index=True,
                )

                df_models.to_parquet(RESULTS_PATH / "results_synthetic_notears.parquet")
                print(f"Saved df_models with nrows: {len(df_models)}")



if __name__ == "__main__":
    run_bicycle_training()
