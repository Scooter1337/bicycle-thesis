"""
Model adapter templates for evaluate_metrics.py.

Implement `predict_for_adata` to generate per-cell predicted expression as an AnnData, aligned
to the input `adata` (same obs/var order). The evaluation script will use this to compute DES, PDS, and MAE.

Signature required by evaluate_metrics.py:

def predict_for_adata(adata, checkpoint_path: str, device: str = "cpu", batch_size: int = 128, **kwargs) -> anndata.AnnData

Expected return:
- anndata.AnnData with .obs_names and .var_names identical to `adata`
- Predicted expressions stored in .X (or in a named layer; if using a layer, pass --layer-pred to the script)

Below is a minimal skeleton; adapt it to your model's inputs/outputs.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List


def predict_for_adata(
    adata,
    checkpoint_path: str,
    device: str = "cpu",
    batch_size: int = 128,
    perturbation_strength: float = 1.0,
    perturbation_key: str = "perturbation",
    control_label: str = "control",
    target_gene_key: str = "target_gene",
    **kwargs,
):
    """Generate per-cell predictions from a linear state dict with (W, alpha) inferred from checkpoint.

    Assumptions:
    - Checkpoint is an OrderedDict with keys like 'alpha' (n,), 'beta_val' (n*(n-1),). We build W by
      filling off-diagonal entries of a (n x n) matrix from 'beta_val' in row-major order, skipping diagonals.
    - Predicted steady-state expression for control: x_c solves (I - W) x = alpha.
    - For a perturbation targeting gene g, we solve (I - W) x = alpha + s * e_g, with s = perturbation_strength.
    - We assign each cell the vector for its target gene; control cells get x_c.
    - Predictions are returned in the same gene order as adata.var_names.
    """
    import numpy as np
    import importlib
    anndata = importlib.import_module("anndata")
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required to load checkpoint state dict.") from e

    ckpt = torch.load(checkpoint_path, map_location=device)
    if not isinstance(ckpt, dict):
        raise NotImplementedError("Expected state dict checkpoint for predict_for_adata.")

    def _to_np(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    if "alpha" not in ckpt or "beta_val" not in ckpt:
        raise NotImplementedError("State dict missing required keys 'alpha' and 'beta_val'.")

    alpha = _to_np(ckpt["alpha"]).astype(float)
    beta_val = _to_np(ckpt["beta_val"]).astype(float)
    n = int(alpha.shape[0])
    expected = n * (n - 1)
    if beta_val.size != expected:
        raise ValueError(f"beta_val has size {beta_val.size}, expected {expected} for n={n} (off-diagonals).")

    # Validate gene count against adata
    n_genes_adata = adata.n_vars if hasattr(adata, "n_vars") else adata.shape[1]
    if n != n_genes_adata:
        raise ValueError(
            f"Checkpoint gene count (from alpha) is {n}, but AnnData has {n_genes_adata} genes. Ensure matching var_names and order."
        )

    # Reconstruct W: W[i, j] is effect of gene j on gene i (row i, column j), zero diagonal
    W = np.zeros((n, n), dtype=float)
    idx = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            W[i, j] = beta_val[idx]
            idx += 1

    # Solve (I - W) x = b for different b; use robust solve with fallback to pinv
    I_minus_W = np.eye(n, dtype=float) - W
    def solve_b(b: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.solve(I_minus_W, b)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(I_minus_W) @ b

    x_control = solve_b(alpha)

    # Map target gene per cell
    var_names = list(adata.var_names)
    gene_to_idx: Dict[str, int] = {g: i for i, g in enumerate(var_names)}

    obs = adata.obs
    if target_gene_key in obs:
        tg_series = obs[target_gene_key].astype(str)
    else:
        # Fallback to using the perturbation label directly
        tg_series = obs.get(perturbation_key, None)
        if tg_series is None:
            raise KeyError(
                f"Could not find '{target_gene_key}' or '{perturbation_key}' in adata.obs to infer target genes."
            )
        tg_series = tg_series.astype(str)

    # Build per-gene prediction cache
    unique_targets: List[str] = sorted(tg_series.unique())
    # Remove control label if present
    unique_targets_nocontrol = [g for g in unique_targets if g != str(control_label)]

    preds_for_gene: Dict[int, np.ndarray] = {}
    for gname in unique_targets_nocontrol:
        gidx = gene_to_idx.get(gname)
        if gidx is None:
            # Try to split multi-target labels (e.g., 'GeneA+GeneB'), take first recognized
            sep_found = None
            for sep in ["+", ",", ";", " "]:
                if sep in gname:
                    sep_found = sep
                    break
            if sep_found is not None:
                for token in gname.split(sep_found):
                    token = token.strip()
                    gidx = gene_to_idx.get(token)
                    if gidx is not None:
                        break
        if gidx is None:
            # If we still can't map, skip: will fall back to control for those cells
            continue
        if gidx in preds_for_gene:
            continue
        b = alpha.copy()
        b[gidx] += float(perturbation_strength)
        preds_for_gene[gidx] = solve_b(b)

    # Build full predictions per cell (repeat vectors for cells of same target)
    n_cells = adata.n_obs if hasattr(adata, "n_obs") else adata.shape[0]
    Y = np.empty((n_cells, n), dtype=float)
    pert_series = obs.get(perturbation_key, None)
    pert_series = pert_series.astype(str) if pert_series is not None else None

    for i in range(n_cells):
        # Control?
        if pert_series is not None and pert_series.iat[i] == str(control_label):
            Y[i, :] = x_control
            continue
        gname = tg_series.iat[i]
        gidx = gene_to_idx.get(gname)
        if gidx is None:
            # Multi-target fallback
            chosen = None
            for sep in ["+", ",", ";", " "]:
                if sep in gname:
                    for token in gname.split(sep):
                        token = token.strip()
                        if token in gene_to_idx:
                            chosen = gene_to_idx[token]
                            break
                    if chosen is not None:
                        break
            gidx = chosen
        if gidx is not None and gidx in preds_for_gene:
            Y[i, :] = preds_for_gene[gidx]
        else:
            # Fallback to control prediction if unmapped
            Y[i, :] = x_control

    out = anndata.AnnData(X=Y, obs=adata.obs.copy(), var=adata.var.copy())
    out.obs_names = adata.obs_names.copy()
    out.var_names = adata.var_names.copy()
    return out


def predict_from_tensor_checkpoint(adata, checkpoint_path: str, device: str = "cpu", batch_size: int = 128, **kwargs):
    """Adapter that treats the .pt checkpoint as predictions and wraps them in an AnnData.

    Supported checkpoint contents:
    - torch.Tensor with shape (n_cells, n_genes)
    - numpy.ndarray with shape (n_cells, n_genes)
    - dict containing one of keys in [predictions, Y_pred, y_pred, logits, outputs] with tensor/ndarray value

    Returns an AnnData aligned to `adata` with predictions in .X
    """
    import numpy as np
    import importlib
    anndata = importlib.import_module("anndata")

    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required for loading .pt checkpoints. Install torch.") from e

    ckpt = torch.load(checkpoint_path, map_location=device)

    Y = None
    if hasattr(ckpt, "detach") or str(type(ckpt)).endswith("Tensor'>"):
        # torch.Tensor
        Y = ckpt.detach().cpu().numpy()
    elif isinstance(ckpt, np.ndarray):
        Y = ckpt
    elif isinstance(ckpt, dict):
        # common keys
        for key in ["predictions", "Y_pred", "y_pred", "logits", "outputs", "yhat", "y_hat"]:
            val = ckpt.get(key)
            if val is None:
                continue
            if hasattr(val, "detach"):
                Y = val.detach().cpu().numpy()
                break
            elif isinstance(val, np.ndarray):
                Y = val
                break
        if Y is None:
            raise NotImplementedError(
                "Checkpoint appears to be a model/state_dict. Please implement `predict_for_adata` to run the model."
            )
    else:
        raise TypeError(
            f"Unsupported checkpoint content type: {type(ckpt)}. Provide predictions as tensor/ndarray or a dict with 'predictions' key."
        )

    # Validate shape
    n_cells = adata.n_obs if hasattr(adata, "n_obs") else adata.shape[0]
    n_genes = adata.n_vars if hasattr(adata, "n_vars") else adata.shape[1]
    if Y.shape != (n_cells, n_genes):
        raise ValueError(
            f"Predictions shape {Y.shape} does not match adata shape ({n_cells}, {n_genes}). Ensure same cells and genes."
        )

    out = anndata.AnnData(X=Y, obs=adata.obs.copy(), var=adata.var.copy())
    out.obs_names = adata.obs_names.copy()
    out.var_names = adata.var_names.copy()
    return out
