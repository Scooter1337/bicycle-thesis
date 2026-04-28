import warnings
import time
import os
import shutil
from pathlib import Path

import click
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import RichProgressBar, StochasticWeightAveraging

from bicycle.callbacks import (
    GenerateCallback,
    MyLoggerCallback,
    CustomModelCheckpoint,
)
from typing import Any, cast
from bicycle.dictlogger import DictLogger
from bicycle.model import BICYCLE
from bicycle.nodags_files.notears import NotearsClassWrapper
from bicycle.utils.data import (
    create_data,
    create_loaders,
    get_diagonal_mask,
    compute_inits,
    process_data_for_llc,
)
from bicycle.utils.metrics import compute_auprc
from bicycle.utils.plotting import plot_training_results

# ---------------------------------------------------------------------------
# Warning filters
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Resolve repository root (script is inside notebooks/). Adjust if layout changes.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models"
RESULTS_PATH = REPO_ROOT / "results"
MODEL_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)


@click.command()
@click.option("--nlogo", default=10, type=int, help="Number of left-out genes (LOGO) for test.")
@click.option("--seed", default=1, type=int, help="Random seed.")
@click.option("--lr", default=1e-3, type=float, help="Learning rate for BICYCLE model.")
@click.option("--n-genes", default=10, type=int, help="Number of genes in synthetic graph generation.")
@click.option("--scale-l1", default=1, type=float, help="L1 penalty scaling.")
@click.option("--scale-kl", default=1, type=float, help="KL divergence scaling.")
@click.option("--scale-spectral", default=1, type=float, help="Spectral loss scaling.")
@click.option("--scale-lyapunov", default=0.0001, type=float, help="Lyapunov loss scaling.")
@click.option("--swa", default=0, type=int, help="Start epoch for Stochastic Weight Averaging (0 to disable).")
@click.option("--n-samples-control", default=750, type=int, help="Number of control samples.")
@click.option("--n-samples-per-perturbation", default=250, type=int, help="Samples per perturbation context.")
@click.option("--validation-size", default=0.0, type=float, help="Validation split size (fraction).")
@click.option("--sem", default="linear-ou", type=str, help="Structural equation model type.")
@click.option("--use-latents", is_flag=True, default=False, help="Enable latent variable mode.")
@click.option("--intervention-scale", default=1.0, type=float, help="Intervention scale for data gen.")
@click.option("--rank-w-cov-factor", default=10, type=int, help="Rank factor for covariance matrix in Lyapunov penalty.")
@click.option("--n-epochs", default=25000, type=int, help="Max training epochs for Lightning Trainer.")
@click.option("--skip-training", is_flag=True, default=False, help="Only run preprocessing / NOTears sweep, skip BICYCLE training.")
@click.option("--skip-notears", is_flag=True, default=False, help="Skip NOTears sweep and only run BICYCLE training.")
@click.option("--use-synthetic-loaders", is_flag=True, default=False, help="Construct DataLoaders from freshly simulated synthetic data instead of loading pre-saved nodags_data loaders.")
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
    n_epochs,
    skip_training,
    skip_notears,
    use_synthetic_loaders,
):
    SEED = seed
    SERVER = False
    pl.seed_everything(SEED)
    torch.set_float32_matmul_precision("high")
    noise_scale = 0.5
    df_models = pd.DataFrame(
        columns=[
            "filename",
            "l1",
            "noise_scale"
        ]
    )


    #
    # Paths
    #
    MODEL_PATH = REPO_ROOT / "models"
    DATA_PATH = REPO_ROOT / "data"
    PLOT_PATH = REPO_ROOT / "plots"

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

    graph_type = "erdos-renyi"
    edge_assignment = "random-uniform"
    make_contractive = True
    graph_kwargs = {
        "abs_weight_low": 0.25,
        "abs_weight_high": 0.95,
        "p_success": 0.5,
        "expected_density": 2,
        "noise_scale": 0.5,
    }
    graph_kwargs_str = graph_kwargs["noise_scale"]
    graph_kwargs["intervention_scale"] = intervention_scale

    # LEARNING
    batch_size = 1024
    USE_INITS = False
    use_encoder = False
    # Overridden by CLI
    # n_epochs = 25_000
    optimizer = "adam"
    # DATA
    # nlogo REPRESENTS THE NUMBER OF GROUPS THAT SHOULD BE LEFT OUT DURING TRAINING
    LOGO = sorted(list(np.random.choice(n_genes, nlogo, replace=False)))
    train_gene_ko = [str(x) for x in set(range(0, n_genes)) - set(LOGO)]  # We start counting at 0
    # FIXME: There might be duplicates...
    ho_perturbations = sorted(
        list(set([tuple(sorted(np.random.choice(n_genes, 2, replace=False))) for _ in range(0, 20)]))
    )
    test_gene_ko = [f"{x[0]},{x[1]}" for x in ho_perturbations]

    # DGP
    # rank_w_cov_factor = n_genes - 1  # Same as dictys: #min(TFs, N_GENES-1)
    n_contexts = n_genes + 1  # Number of contexts
    perfect_interventions = True
    make_counts = False  # True | also set x_distribution
    # LEARNING
    early_stopping = True
    early_stopping_patience = 500
    early_stopping_min_delta = 0.01
    x_distribution = "Normal"  # "Poisson"
    x_distribution_kwargs = {"std": 0.1}
    # MODEL
    lyapunov_penalty = scale_lyapunov > 0
    GPU_DEVICE = 1
    plot_epoch_callback = 1000
    # RESULTS
    SAVE_PLOT = True
    CHECKPOINTING = True
    VERBOSE_CHECKPOINTING = False
    OVERWRITE = False
    # REST
    n_samples_total = (
        n_samples_control + (len(train_gene_ko) + len(test_gene_ko)) * n_samples_per_perturbation
    )
    check_val_every_n_epoch = 1
    log_every_n_steps = 1

    name_prefix = f"real_{graph_type}_{graph_kwargs_str}_{early_stopping}_{early_stopping_patience}_{early_stopping_min_delta}_{x_distribution}"

    #
    # Create Mask
    #
    mask = get_diagonal_mask(n_genes, device)

    #
    # Create synthetic data
    #
    if use_synthetic_loaders:
        _, _, samples, gt_interv, sim_regime, beta = create_data(
            n_genes,
            n_samples_control=n_samples_control,
            n_samples_per_perturbation=n_samples_per_perturbation,
            device=str(device),
            make_counts=make_counts,
            train_gene_ko=train_gene_ko,
            test_gene_ko=test_gene_ko,
            graph_type=graph_type,
            edge_assignment=edge_assignment,
            sem=sem,
            make_contractive=make_contractive,
            **graph_kwargs,
        )
    else:
        # Placeholders; real values inferred later when loading existing loaders
        samples = torch.empty(0)
        gt_interv = torch.empty(0)
        sim_regime = torch.empty(0)
        beta = torch.empty(0)

    if use_synthetic_loaders:
        print("Using synthetic data to construct DataLoaders (--use-synthetic-loaders).")
        loaders = create_loaders(
            samples,
            sim_regime,
            validation_size=validation_size,
            batch_size=batch_size,
            SEED=SEED,
            train_gene_ko=train_gene_ko,
            test_gene_ko=test_gene_ko,
            num_workers=0,
            persistent_workers=False,
        )
        # create_loaders may return (train, val, test) or (train, val, test, covariates)
        if len(loaders) == 4:
            train_loader, validation_loader, test_loader, _ = loaders
        else:
            train_loader, validation_loader, test_loader = loaders
    else:
        # -------------------------------------------------------------------
        # Load pre-saved dataloaders (expects repository root as CWD). Resolve paths robustly.
        # -------------------------------------------------------------------
        data_root = REPO_ROOT / "nodags_data" / "cocult"
        train_loader_path = data_root / "training_data" / "train_loader.pth"
        validation_loader_path = data_root / "validation_data" / "validation_loader.pth"
        test_loader_path = data_root / "test_data" / "test_loader.pth"

        for p in [train_loader_path, validation_loader_path, test_loader_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Expected dataloader file not found: {p}. If you intended to use synthetic data, add --use-synthetic-loaders."
                )

        train_loader = torch.load(train_loader_path, weights_only=False)
        validation_loader = torch.load(validation_loader_path, weights_only=False)
        test_loader = torch.load(test_loader_path, weights_only=False)

        # Derive gene and context dimensions from loaded data instead of synthetic simulation
        # Expect TensorDataset with (samples, regimes, sample_idx, data_category)
        inferred_samples = train_loader.dataset.tensors[0]
        inferred_regimes = train_loader.dataset.tensors[1]
        real_n_genes = inferred_samples.shape[1]
        # Adjust n_genes if mismatch
        if real_n_genes != n_genes:
            print(f"[INFO] Overriding n_genes CLI value {n_genes} with dataset gene count {real_n_genes}.")
            n_genes = real_n_genes
        # Build placeholder ground-truth intervention matrix (all control) sized to number of contexts
        max_context = 0
        for ld in [train_loader, validation_loader, test_loader]:
            if ld is not None:
                max_context = max(max_context, int(ld.dataset.tensors[1].max().item()))
        num_contexts = max_context + 1
        gt_interv = torch.zeros((n_genes, num_contexts), device=inferred_samples.device)
        # Placeholder beta (unknown ground truth)
        beta = torch.zeros((n_genes, n_genes), device=inferred_samples.device)
        # Recompute mask for new n_genes
        mask = get_diagonal_mask(n_genes, device)
        # Recompute n_samples_total from datasets
        n_samples_total = len(train_loader.dataset)
        if test_loader is not None:
            n_samples_total += len(test_loader.dataset)
        # Adjust train/test gene KO lists to cover gene indices (fallback if inconsistent)
        # If previous LOGO selection size mismatches, regenerate
        # Normalise KO gene lists to integers for comparison
        try:
            train_gene_ko_int = [int(x) for x in train_gene_ko]
        except Exception:
            train_gene_ko_int = list(range(n_genes))
        if len(train_gene_ko_int) == 0:
            train_gene_ko_int = list(range(n_genes))
        if max(train_gene_ko_int) >= n_genes:
            train_gene_ko_int = list(range(n_genes))
        train_gene_ko = [str(x) for x in train_gene_ko_int]
        # Auto-skip NOTears since no ground-truth adjacency
        if not skip_notears:
            print("[INFO] Auto-enabling --skip-notears (no ground truth adjacency for real loaders).")
            skip_notears = True

    init_tensors = {}
    if USE_INITS:
        init_tensors = compute_inits(train_loader.dataset, rank_w_cov_factor, n_contexts)

    print("Training data:")
    try:
        if train_loader is not None and hasattr(train_loader, 'dataset'):
            train_len = len(cast(Any, train_loader.dataset))
            print(f"- Number of training samples: {train_len}")
        if validation_loader is not None and hasattr(validation_loader, 'dataset'):
            val_len = len(cast(Any, validation_loader.dataset))
            print(f"- Number of validation samples: {val_len}")
        if test_loader is not None and hasattr(test_loader, 'dataset'):
            test_len = len(cast(Any, test_loader.dataset))
            print(f"- Number of test samples: {test_len}")
    except Exception:
        pass

    device = torch.device("cpu")
    # if SERVER:
    #     device = torch.device("cuda")
    #     devices = "auto"
    # else:
    #     device = torch.device(f"cuda:{GPU_DEVICE}")
    #     devices = [GPU_DEVICE] if str(device).startswith("cuda") else 1
    if use_synthetic_loaders:
        gt_interv = gt_interv.to(device)
        n_genes = samples.shape[1]

    file_dir = (
        name_prefix
        + f"_{nlogo}_{seed}_{lr}_{n_genes}_{scale_l1}_{scale_kl}_{scale_spectral}_{scale_lyapunov}_{swa}_{n_samples_control}_{n_samples_per_perturbation}_{validation_size}_{sem}_{use_latents}_{intervention_scale}_{rank_w_cov_factor}"
    )

    # If final plot or final model exists: do not overwrite by default
    print("Checking Model and Plot files...")
    report_file_name = os.path.join(MODEL_PATH, file_dir, "report.yaml")
    final_file_name = os.path.join(MODEL_PATH, file_dir, "last.ckpt")
    final_plot_name = os.path.join(PLOT_PATH, file_dir, "last.png")
    # Check existing artifacts (avoid deprecated bitwise ops on bool)
    if Path(report_file_name).exists() and not OVERWRITE:
        print("- Files already exists, skipping...")
        pass
    else:
        print("- Not all files exist, fitting model...")
        print("  - Deleting dirs")
        # Delete directories of files
        if Path(final_file_name).parent.exists():
            print(f"  - Removing model directory: {Path(final_file_name).parent}")
            shutil.rmtree(Path(final_file_name).parent, ignore_errors=True)
        if Path(final_plot_name).parent.exists():
            print(f"  - Removing plot directory: {Path(final_plot_name).parent}")
            shutil.rmtree(Path(final_plot_name).parent, ignore_errors=True)

        print("  - Creating dirs")
        # Create directories
        Path(os.path.join(MODEL_PATH, file_dir)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(PLOT_PATH, file_dir)).mkdir(parents=True, exist_ok=True)

        # Persist dataloaders for reproducibility (skip if None)
        torch.save(train_loader, os.path.join(MODEL_PATH, file_dir, "train_loader.pth"))
        if validation_loader is not None:
            torch.save(validation_loader, os.path.join(MODEL_PATH, file_dir, "validation_loader.pth"))
        if test_loader is not None:
            torch.save(test_loader, os.path.join(MODEL_PATH, file_dir, "test_loader.pth"))

        filename = os.path.join(MODEL_PATH, file_dir, "test_loader.pth")

        # Extract control samples from train_loader
        dataset_train = []
        for batch in train_loader:
            data = batch[0][batch[1] == 0]  # Context 0 is intervention samples
            dataset_train.append(data)
        dataset_train = [torch.cat(dataset_train, dim=0).detach().cpu().numpy()]
        dataset_train_targets = [np.array([])]

        # dataset_train, dataset_train_targets = process_data_for_llc(train_loader, gt_interv, model.train_gene_ko)
        dataset_test, dataset_test_targets = process_data_for_llc(test_loader, gt_interv, test_gene_ko)

        # Remove 20% validation data (this should actually be the same as in the original training)
        # Uncomment if you want to use all data for training
        dataset_valid = []
        dataset_valid_targets = []
        for k in range(len(dataset_train)):
            train_idx = np.random.choice(len(dataset_train[k]), int(0.8 * len(dataset_train[k])), replace=False)
            valid_idx = np.setdiff1d(np.arange(len(dataset_train[k])), train_idx)

            dataset_valid.append(dataset_train[k][valid_idx])
            dataset_valid_targets.append(dataset_train_targets[k])

            dataset_train[k] = dataset_train[k][train_idx]

        if not skip_notears:
            results = pd.DataFrame()
            for loss_type in ["l2"]:
                for l1 in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]:
                    notears_wrapper = NotearsClassWrapper(lambda1=l1, loss_type=loss_type, noise_scale=noise_scale)
                    est_beta = notears_wrapper.train(dataset_train, dataset_train_targets, return_weights=True)
                    nll_pred_valid = notears_wrapper.predictLikelihood(dataset_valid, dataset_valid_targets)
                    est_beta = torch.from_numpy(est_beta).to(device).detach().cpu()

                    if est_beta.shape != tuple(beta.shape):
                        print(
                            f"[WARN] Shape mismatch for AUPRC computation: gt beta {tuple(beta.shape)} vs est {est_beta.shape}. Skipping area_test."
                        )
                        area_test = float('nan')
                    else:
                        area_test = compute_auprc(beta.float(), est_beta)
                    nll_pred_test = notears_wrapper.predictLikelihood(dataset_test, dataset_test_targets)

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
                                    "intervention_scale": intervention_scale,
                                    "nll_pred_valid": np.mean(nll_pred_valid),
                                    "nll_pred_test": np.mean(nll_pred_test),
                                    "noise_scale": noise_scale,
                                    "area_test": area_test,
                                    "l1": l1,
                                },
                                index=[0],
                            ),
                        ],
                        ignore_index=True,
                    )

                    df_models.to_parquet(RESULTS_PATH / "results_synthetic_notears.parquet")
                    print(f"Saved df_models with nrows: {len(df_models)}")
        else:
            print("Skipping NOTears sweep (--skip-notears).")

        if skip_training:
            print("--skip-training specified: Skipping BICYCLE training after NOTears sweep.")
            return

        model = BICYCLE(
            lr,
            gt_interv,
            n_genes,
            n_samples=n_samples_total,
            lyapunov_penalty=lyapunov_penalty,
            perfect_interventions=perfect_interventions,
            rank_w_cov_factor=rank_w_cov_factor,
            init_tensors=init_tensors,
            optimizer=optimizer,
            device=str(device),
            scale_l1=scale_l1,
            scale_lyapunov=scale_lyapunov,
            scale_spectral=scale_spectral,
            scale_kl=scale_kl,
            early_stopping=early_stopping,
            early_stopping_min_delta=early_stopping_min_delta,
            early_stopping_patience=early_stopping_patience,
            early_stopping_p_mode=True,
            x_distribution=x_distribution,
            x_distribution_kwargs=x_distribution_kwargs,
            mask=mask,
            use_encoder=use_encoder,
            gt_beta=beta,
            train_gene_ko=train_gene_ko,
            test_gene_ko=test_gene_ko,
            use_latents=use_latents,
        )
        model.to(device)

        dlogger = DictLogger()
        loggers = [dlogger]

        callbacks = [
            RichProgressBar(refresh_rate=1),
            GenerateCallback(
                final_plot_name, plot_epoch_callback=plot_epoch_callback, true_beta=beta.cpu().numpy()
            ),
        ]
        if swa > 0:
            callbacks.append(StochasticWeightAveraging(0.01, swa_epoch_start=swa))
        if CHECKPOINTING:
            Path(os.path.join(MODEL_PATH, file_dir)).mkdir(parents=True, exist_ok=True)
            callbacks.append(
                CustomModelCheckpoint(
                    dirpath=os.path.join(MODEL_PATH, file_dir),
                    filename="{epoch}",
                    save_last=True,
                    save_top_k=1,
                    verbose=VERBOSE_CHECKPOINTING,
                    monitor="valid_loss",
                    mode="min",
                    save_weights_only=True,
                    start_after=0,
                    save_on_train_epoch_end=False,
                    every_n_epochs=100,
                )
            )
            callbacks.append(MyLoggerCallback(dirpath=os.path.join(MODEL_PATH, file_dir)))

        trainer = pl.Trainer(
            max_epochs=n_epochs,
            accelerator="cpu",  # Change to 'gpu' and set devices for GPU usage
            logger=loggers,
            log_every_n_steps=log_every_n_steps,
            enable_model_summary=True,
            enable_progress_bar=True,
            enable_checkpointing=CHECKPOINTING,
            check_val_every_n_epoch=check_val_every_n_epoch,
            devices=1,
            num_sanity_val_steps=0,
            callbacks=callbacks,
            gradient_clip_val=1,
            default_root_dir=str(MODEL_PATH),
            gradient_clip_algorithm="value",
            deterministic=True,
        )

        start_time = time.time()
        trainer.fit(model, train_loader, validation_loader)
        end_time = time.time()
        print(f"Training took {end_time - start_time:.2f} seconds")

        plot_training_results(
            trainer,
            model,
            model.beta.detach().cpu().numpy(),
            beta,
            scale_l1,
            scale_kl,
            scale_spectral,
            scale_lyapunov,
            final_plot_name,
            callback=False,
        )

        # except Exception as e:
        #     # Write Exception to file
        #     report_path = os.path.join(MODEL_PATH, file_dir, "report.yaml")
        #     print(f"Raised Exception, writing to file: {report_path}")
        #     # Write yaml
        #     with open(report_path, "w") as outfile:
        #         yaml.dump({"exception": str(e)}, outfile, default_flow_style=False)


if __name__ == "__main__":
    run_bicycle_training()