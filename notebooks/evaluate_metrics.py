#!/usr/bin/env python3
"""
Evaluate model predictions for single-cell perturbation datasets using:
- Differential Expression Score (DES)
- Perturbation Discrimination Score (PDS)
- Mean Absolute Error (MAE)

This script supports two input modes:
1) Provide a predictions AnnData (.h5ad) with per-cell predicted expression that aligns to the ground truth AnnData (same cells and genes order).
2) Provide a model adapter callable to generate predictions from a PyTorch model checkpoint: `--model-adapter module:function --checkpoint path`

Assumed inputs:
- Ground truth AnnData with per-cell expression and columns for perturbation labels and optionally target gene per perturbation.
- Predictions AnnData with the same cells and genes (obs/var alignment expected).

Notes/assumptions:
- DE testing uses Mann-Whitney U (Wilcoxon rank-sum) with tie-aware asymptotic p-values from SciPy.
- Multiple hypothesis correction per perturbation uses Benjamini–Hochberg (FDR at alpha=0.05 by default).
- For DES trimming when |G_pred| > |G_true|, fold change is computed as the absolute difference of mean (log1p-normalized) expression between perturbed and control cells.
- PDS excludes the target gene of the predicted perturbation p from the L1 distance; if the target gene is unknown/not found, no gene is excluded for that p.
- MAE uses pseudobulk (mean of log1p-normalized expressions) across all genes.

Outputs:
- Prints overall DES, PDS, MAE to stdout.
- Optionally writes a JSON summary and CSV of per-perturbation metrics.

Quick usage examples:
- With predictions .h5ad:
    python evaluate_metrics.py --adata path/to/ground_truth.h5ad --predictions-h5ad path/to/pred.h5ad \
        --perturbation-key perturbation --control-label control --target-gene-key target_gene
- With a tensor .pt checkpoint (e.g., reproduce2/random/cocult_model.pt) using the included adapter:
    python evaluate_metrics.py --adata path/to/ground_truth.h5ad \
        --model-adapter model_adapters:predict_from_tensor_checkpoint \
        --checkpoint reproduce2/random/cocult_model.pt \
        --perturbation-key perturbation --control-label control --target-gene-key target_gene
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import importlib

try:
    from scipy.stats import mannwhitneyu
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "scipy is required. Please install with `pip install scipy`."
    ) from e

# Implement BH FDR locally to avoid requiring statsmodels.


# -----------------------------
# Utility helpers
# -----------------------------


def _to_numpy(X) -> np.ndarray:
    """Convert AnnData.X/layer to dense numpy array (copy=False when possible)."""
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def _get_matrix(adata: Any, layer: Optional[str]) -> np.ndarray:
    if layer is None:
        return _to_numpy(adata.X)
    if layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' not found in AnnData.layers. Available: {list(adata.layers.keys())}")
    return _to_numpy(adata.layers[layer])


def _bh_significant(pvals: np.ndarray, alpha: float) -> np.ndarray:
    """Return boolean mask of BH-adjusted significance at FDR alpha (Benjamini–Hochberg).

    Implementation adapted to avoid external dependencies. Handles NaNs by treating them as 1.0.
    """
    p = np.asarray(pvals, dtype=float)
    p = np.where(np.isnan(p), 1.0, p)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, m + 1) / m)
    # Find largest k where p_(k) <= thresh_k
    comp = ranked <= thresh
    if not np.any(comp):
        return np.zeros(m, dtype=bool)
    k_max = int(np.max(np.nonzero(comp)))  # index in 0..m-1
    cutoff = ranked[k_max]
    reject = p <= cutoff
    return reject


def _mannwhitneyu_vectorized(group_a: np.ndarray, group_b: np.ndarray) -> np.ndarray:
    """Compute Mann-Whitney U test p-values (two-sided) for each column independently.

    Parameters
    ----------
    group_a : np.ndarray (n_a, g)
    group_b : np.ndarray (n_b, g)

    Returns
    -------
    pvals : np.ndarray (g,)
    """
    # Loop per gene column to keep memory/cpu in check and rely on SciPy's tie-aware asymptotic method.
    g = group_a.shape[1]
    pvals = np.empty(g, dtype=float)
    for j in range(g):
        a = group_a[:, j]
        b = group_b[:, j]
        # Drop NaNs if any
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
        if a.size == 0 or b.size == 0:
            p = 1.0
        else:
            try:
                # method='asymptotic' handles ties; two-sided by default in newer SciPy
                res = mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
                p = float(res.pvalue)
            except TypeError:
                # Older SciPy: no method kwarg
                res = mannwhitneyu(a, b, alternative="two-sided")
                p = float(res.pvalue)
        pvals[j] = p
    return pvals


def _abs_fold_change(group_a: np.ndarray, group_b: np.ndarray) -> np.ndarray:
    """Absolute fold change in log1p space, implemented as absolute difference of means per gene."""
    mean_a = np.nanmean(group_a, axis=0)
    mean_b = np.nanmean(group_b, axis=0)
    return np.abs(mean_a - mean_b)


def _pseudobulk_mean(X: np.ndarray) -> np.ndarray:
    return np.nanmean(X, axis=0)


def _ensure_alignment(adata_true: Any, adata_pred: Any) -> None:
    if not getattr(adata_true, "obs_names").equals(getattr(adata_pred, "obs_names")):
        raise ValueError(
            "obs_names (cells) do not align between true and predicted AnnData. "
            "Please ensure predictions are aligned to the same cells in the same order."
        )
    if not getattr(adata_true, "var_names").equals(getattr(adata_pred, "var_names")):
        raise ValueError(
            "var_names (genes) do not align between true and predicted AnnData. "
            "Please ensure the same genes in the same order."
        )


# -----------------------------
# Metrics
# -----------------------------


@dataclass
class MetricResults:
    des_overall: float
    pds_overall: float
    mae_overall: float
    per_perturbation: pd.DataFrame


def compute_des(
    adata_true: Any,
    adata_pred: Any,
    perturbation_key: str,
    control_label: str,
    alpha: float = 0.05,
    layer_true: Optional[str] = None,
    layer_pred: Optional[str] = None,
    min_cells: int = 3,
) -> Tuple[float, pd.Series]:
    """Compute Differential Expression Score (DES).

    Returns overall DES and per-perturbation DES as a Series.
    """
    _ensure_alignment(adata_true, adata_pred)
    X_true = _get_matrix(adata_true, layer_true)
    X_pred = _get_matrix(adata_pred, layer_pred)

    genes = getattr(adata_true, "var_names")
    obs = getattr(adata_true, "obs")
    if perturbation_key not in obs:
        raise KeyError(f"Column '{perturbation_key}' not found in adata_true.obs")
    perts = obs[perturbation_key].astype(str)

    mask_ctrl = perts == str(control_label)
    if mask_ctrl.sum() < min_cells:
        raise ValueError(
            f"Not enough control cells (found {mask_ctrl.sum()} < min_cells={min_cells})."
        )

    des_values: Dict[str, float] = {}

    for k in sorted(perts.unique()):
        if k == str(control_label):
            continue
        mask_k = perts == k
        n_k = int(mask_k.sum())
        if n_k < min_cells:
            continue  # skip low support perturbations

        # True DE set
        pvals_true = _mannwhitneyu_vectorized(X_true[mask_k], X_true[mask_ctrl])
        sig_true = _bh_significant(pvals_true, alpha=alpha)
        G_true_idx = np.flatnonzero(sig_true)

        # Pred DE set
        pvals_pred = _mannwhitneyu_vectorized(X_pred[mask_k], X_pred[mask_ctrl])
        sig_pred = _bh_significant(pvals_pred, alpha=alpha)
        G_pred_idx = np.flatnonzero(sig_pred)

        nk_true = G_true_idx.size
        nk_pred = G_pred_idx.size

        if nk_true == 0:
            # If both predict and true have no DE genes, score 1.0; else 0.0
            des_k = 1.0 if nk_pred == 0 else 0.0
            des_values[k] = des_k
            continue

        if nk_pred <= nk_true:
            inter = np.intersect1d(G_pred_idx, G_true_idx, assume_unique=False).size
            des_k = inter / float(nk_true)
        else:
            # Trim predicted set to size nk_true by absolute fold-change
            fc_abs = _abs_fold_change(X_pred[mask_k], X_pred[mask_ctrl])
            # Among significant predicted genes, select top by |FC|
            top_pred_idx = G_pred_idx[np.argsort(fc_abs[G_pred_idx])[::-1][:nk_true]]
            inter = np.intersect1d(top_pred_idx, G_true_idx, assume_unique=False).size
            des_k = inter / float(nk_true)

        des_values[k] = float(des_k)

    if not des_values:
        raise ValueError("No perturbations with sufficient cells to compute DES.")

    per_series = pd.Series(des_values).sort_index()
    overall = float(per_series.mean())
    return overall, per_series


def compute_pds(
    adata_true: Any,
    adata_pred: Any,
    perturbation_key: str,
    control_label: str,
    target_gene_key: Optional[str] = None,
    layer_true: Optional[str] = None,
    layer_pred: Optional[str] = None,
    min_cells: int = 3,
) -> Tuple[float, pd.Series]:
    """Compute Perturbation Discrimination Score (PDS)."""
    _ensure_alignment(adata_true, adata_pred)
    X_true = _get_matrix(adata_true, layer_true)
    X_pred = _get_matrix(adata_pred, layer_pred)

    obs = getattr(adata_true, "obs")
    perts = obs[perturbation_key].astype(str)
    unique_perts = sorted([p for p in perts.unique() if p != str(control_label)])
    if not unique_perts:
        raise ValueError("No non-control perturbations found for PDS.")

    # Precompute pseudobulk for all perts in true and predicted
    y_true: Dict[str, np.ndarray] = {}
    y_pred: Dict[str, np.ndarray] = {}
    for k in unique_perts:
        mask_k = (perts == k).values
        if mask_k.sum() < min_cells:
            continue
        y_true[k] = _pseudobulk_mean(X_true[mask_k])
        y_pred[k] = _pseudobulk_mean(X_pred[mask_k])

    # Ensure we have pseudobulk for the same set
    common_perts = sorted(set(y_true.keys()) & set(y_pred.keys()))
    if not common_perts:
        raise ValueError("No overlapping perturbations with sufficient cells to compute PDS.")

    gene_index_map = {g: i for i, g in enumerate(getattr(adata_true, "var_names"))}

    pds_values: Dict[str, float] = {}
    N = len(common_perts)

    for p in common_perts:
        yhat_p = y_pred[p]

        # Exclude target gene for perturbation p if specified
        exclude_idx: Optional[int] = None
        if target_gene_key is not None and target_gene_key in obs:
            # Take the most frequent target gene among cells of p
            tg_series = obs.loc[perts == p, target_gene_key].astype(str)
            if tg_series.size > 0:
                tg = tg_series.mode(dropna=True)
                if tg.size > 0:
                    tg_name = tg.iloc[0]
                    exclude_idx = gene_index_map.get(str(tg_name))

        # Compute distances to all true perts
        dists: List[Tuple[str, float]] = []
        for t in common_perts:
            yt = y_true[t]
            if exclude_idx is not None:
                # Exclude the index safely
                if 0 <= exclude_idx < yt.shape[0]:
                    # L1 distance excluding one coordinate
                    d = np.sum(np.abs(yhat_p[:exclude_idx] - yt[:exclude_idx]))
                    d += np.sum(np.abs(yhat_p[exclude_idx + 1 :] - yt[exclude_idx + 1 :]))
                else:
                    d = float(np.sum(np.abs(yhat_p - yt)))
            else:
                d = float(np.sum(np.abs(yhat_p - yt)))
            dists.append((t, d))

        # Rank distances ascending; handle ties using average rank
        d_vals = np.array([d for _, d in dists])
        # Average rank: 1 + #(< d_true) + 0.5*#(== d_true)-0.5
        # Implement by computing ranks via argsort twice
        order = np.argsort(d_vals)
        ranks_min = np.empty_like(order)
        ranks_min[order] = np.arange(1, N + 1)
        order_rev = np.argsort(-d_vals)
        ranks_max = np.empty_like(order_rev)
        ranks_max[order_rev] = np.arange(1, N + 1)
        ranks_avg = (ranks_min + (N + 1 - ranks_max)) / 2.0

        # Extract rank for true perturbation (where t == p)
        idx_p = [i for i, (t, _) in enumerate(dists) if t == p]
        if not idx_p:
            # Should not happen; continue defensively
            continue
        rp = float(ranks_avg[idx_p[0]])

        if N <= 1:
            pds_p = 1.0
        else:
            pds_p = 1.0 - (rp - 1.0) / float(N)
        pds_values[p] = float(pds_p)

    if not pds_values:
        raise ValueError("Failed to compute any PDS values.")

    per_series = pd.Series(pds_values).sort_index()
    overall = float(per_series.mean())
    return overall, per_series


def compute_mae(
    adata_true: Any,
    adata_pred: Any,
    perturbation_key: str,
    control_label: str,
    layer_true: Optional[str] = None,
    layer_pred: Optional[str] = None,
    min_cells: int = 3,
) -> Tuple[float, pd.Series]:
    """Compute mean absolute error (MAE) based on pseudobulk per perturbation."""
    _ensure_alignment(adata_true, adata_pred)
    X_true = _get_matrix(adata_true, layer_true)
    X_pred = _get_matrix(adata_pred, layer_pred)

    obs = getattr(adata_true, "obs")
    perts = obs[perturbation_key].astype(str)
    unique_perts = sorted([p for p in perts.unique() if p != str(control_label)])
    if not unique_perts:
        raise ValueError("No non-control perturbations found for MAE.")

    mae_values: Dict[str, float] = {}

    for k in unique_perts:
        mask_k = (perts == k).values
        if mask_k.sum() < min_cells:
            continue
        y_true = _pseudobulk_mean(X_true[mask_k])
        y_pred = _pseudobulk_mean(X_pred[mask_k])
        mae_k = float(np.mean(np.abs(y_pred - y_true)))
        mae_values[k] = mae_k

    if not mae_values:
        raise ValueError("No perturbations with sufficient cells to compute MAE.")

    per_series = pd.Series(mae_values).sort_index()
    overall = float(per_series.mean())
    return overall, per_series


# -----------------------------
# Prediction adapter loader
# -----------------------------


def load_adapter(adapter_spec: str) -> Callable[..., Any]:
    """Load a callable adapter given 'module:function' string."""
    if ":" not in adapter_spec:
        raise ValueError("--model-adapter must be in the form 'module_path:function_name'")
    module_name, func_name = adapter_spec.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, func_name, None)
    if fn is None or not callable(fn):
        raise AttributeError(f"Function '{func_name}' not found in module '{module_name}'")
    return fn


# -----------------------------
# CLI
# -----------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate DES, PDS, and MAE from predictions or a PyTorch model adapter.")
    p.add_argument("--adata", required=True, help="Path to ground-truth AnnData (.h5ad)")
    p.add_argument("--predictions-h5ad", help="Path to predicted AnnData (.h5ad) aligned to --adata")
    p.add_argument(
        "--model-adapter",
        help=(
            "Optional: 'module:function' to generate predictions from a model checkpoint. "
            "The callable must have signature predict_for_adata(adata, checkpoint_path, device, batch_size, **kwargs) -> AnnData"
        ),
    )
    p.add_argument("--checkpoint", help="Path to model checkpoint (used with --model-adapter)")
    p.add_argument("--adapter-kwargs", help="JSON string with extra kwargs passed to the model adapter", default=None)
    p.add_argument("--device", default="cpu", help="Device for model adapter (e.g., 'cpu', 'cuda')")
    p.add_argument("--batch-size", type=int, default=128, help="Batch size for model adapter predictions")

    p.add_argument("--perturbation-key", default="perturbation", help="Column in .obs with perturbation labels")
    p.add_argument("--control-label", default="control", help="Label in perturbation_key for control cells")
    p.add_argument(
        "--target-gene-key",
        default="target_gene",
        help="Column in .obs with target gene per cell (used by PDS to exclude target gene)",
    )
    p.add_argument("--layer-true", default=None, help="AnnData layer name to use for ground truth (default: X)")
    p.add_argument("--layer-pred", default=None, help="AnnData layer name to use for predictions (default: X)")
    p.add_argument("--alpha", type=float, default=0.05, help="FDR (BH) threshold for DE genes")
    p.add_argument("--min-cells", type=int, default=3, help="Minimum cells per perturbation/control to include")

    p.add_argument("--output-json", default=None, help="Optional path to write overall metrics as JSON")
    p.add_argument("--output-csv", default=None, help="Optional path to write per-perturbation metrics as CSV")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    # Basic guardrails: --adata should be an .h5ad file
    if not str(args.adata).lower().endswith(".h5ad"):
        raise SystemExit(
            "--adata must be a .h5ad file (AnnData). If you intended to pass a checkpoint (.pt), "
            "use --checkpoint and --model-adapter (or rely on auto-adapter)."
        )

    # Load ground truth AnnData
    # Lazy import anndata here to avoid static import issues
    anndata = importlib.import_module("anndata")
    adata_true = anndata.read_h5ad(args.adata)

    # Obtain predictions AnnData
    preds_h5ad = getattr(args, "predictions_h5ad", None)
    if preds_h5ad:
        adata_pred = anndata.read_h5ad(preds_h5ad)
    else:
        # If no adapter specified but checkpoint is provided, default to state-dict inference adapter
        if not args.model_adapter and args.checkpoint and str(args.checkpoint).lower().endswith(".pt"):
            args.model_adapter = "model_adapters:predict_for_adata"
        if not args.model_adapter or not args.checkpoint:
            raise SystemExit(
                "Provide either --predictions-h5ad OR both --model-adapter and --checkpoint."
            )
        adapter = load_adapter(args.model_adapter)
        extra_kwargs = {}
        if args.adapter_kwargs:
            try:
                extra_kwargs = json.loads(args.adapter_kwargs)
            except json.JSONDecodeError as e:
                raise SystemExit(f"Failed to parse --adapter-kwargs JSON: {e}")
        # Call adapter to generate predictions aligned to adata_true
        try:
            adata_pred = adapter(
                adata_true,
                checkpoint_path=args.checkpoint,
                device=args.device,
                batch_size=int(args.batch_size),
                **extra_kwargs,
            )
        except NotImplementedError as e:
            # Fallback: if the user picked the tensor adapter but checkpoint is a model/state_dict, try the other
            if args.model_adapter.endswith(":predict_from_tensor_checkpoint"):
                try:
                    adapter2 = load_adapter("model_adapters:predict_for_adata")
                    adata_pred = adapter2(
                        adata_true,
                        checkpoint_path=args.checkpoint,
                        device=args.device,
                        batch_size=int(args.batch_size),
                        **extra_kwargs,
                    )
                except Exception:
                    raise
            else:
                raise
        # Basic duck-typing validation
        required_attrs = ["X", "obs", "var", "obs_names", "var_names"]
        if not all(hasattr(adata_pred, a) for a in required_attrs):
            raise TypeError("Model adapter must return an AnnData-like object with .X/.obs/.var and names aligned.")

    # Align and basic validations will happen inside metric functions
    des_overall, des_per = compute_des(
        adata_true,
        adata_pred,
        perturbation_key=args.perturbation_key,
        control_label=args.control_label,
        alpha=float(args.alpha),
        layer_true=args.layer_true,
        layer_pred=args.layer_pred,
        min_cells=int(args.min_cells),
    )

    pds_overall, pds_per = compute_pds(
        adata_true,
        adata_pred,
        perturbation_key=args.perturbation_key,
        control_label=args.control_label,
        target_gene_key=args.target_gene_key,
        layer_true=args.layer_true,
        layer_pred=args.layer_pred,
        min_cells=int(args.min_cells),
    )

    mae_overall, mae_per = compute_mae(
        adata_true,
        adata_pred,
        perturbation_key=args.perturbation_key,
        control_label=args.control_label,
        layer_true=args.layer_true,
        layer_pred=args.layer_pred,
        min_cells=int(args.min_cells),
    )

    # Merge per-perturbation
    per_df = (
        pd.DataFrame({"DES": des_per, "PDS": pds_per, "MAE": mae_per})
        .sort_index()
        .reset_index()
        .rename(columns={"index": "perturbation"})
    )

    # Print summary
    print("=== Evaluation Summary ===")
    print(f"DES (overall): {des_overall:.6f}")
    print(f"PDS (overall): {pds_overall:.6f}")
    print(f"MAE (overall): {mae_overall:.6f}")
    print("==========================")

    if args.output_json:
        out = {
            "DES": float(des_overall),
            "PDS": float(pds_overall),
            "MAE": float(mae_overall),
            "n_perturbations": int(per_df.shape[0]),
        }
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)

    if args.output_csv:
        per_df.to_csv(args.output_csv, index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
