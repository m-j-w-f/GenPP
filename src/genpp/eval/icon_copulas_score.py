#!/usr/bin/env python
"""
CPU step: Generate quantile samples, apply ECC, and compute scores.

Loads the distribution parameters saved by icon_copulas_predict.py,
generates quantile samples, applies Ensemble Copula Coupling (ECC),
and computes evaluation scores. Runs entirely on CPU.

Expected input files (in model_dir):
    {split}_dist_params.pt  — saved by icon_copulas_predict.py
    {split}_ground_truth.pt — saved by icon_copulas_predict.py

Usage:
    python icon_copulas_score.py --run-path feik/genpp/abc123 --split test
    python icon_copulas_score.py --run-path feik/genpp/abc123 --split val test --skip-variogram
    python icon_copulas_score.py --run-path feik/genpp/abc123 --split test --save-predictions -v
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from einops import reduce
from tqdm import trange

from genpp import BASE_DIR
from genpp.eval.icon_copulas_eval import (
    do_ecc_icon,
    quantile_samples_icon,
)
from genpp.eval.icon_predict_eval import (
    compute_icon_scores_per_leadtime,
)
from genpp.eval.utils import (
    log_scores,
    save_scores_df,
    update_wandb_run,
)
from genpp.models.scores import EnergyScore, EnsembleCRPS, VariogramScore


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CPU step: ECC postprocessing and scoring for EMOS/DRN on ICON data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-path",
        type=str,
        nargs="+",
        required=True,
        help="WandB run path(s) (e.g., 'feik/genpp/abc123')",
    )
    parser.add_argument(
        "--split",
        type=str,
        nargs="+",
        default=["val"],
        choices=["train", "val", "test"],
        help="Dataset split(s) to evaluate",
    )
    parser.add_argument(
        "--n-quantile-samples",
        type=int,
        default=40,
        help="Number of quantile samples to generate (should match ensemble size)",
    )
    parser.add_argument(
        "--ens-dir",
        type=str,
        default=None,
        help="Path to the ICON ensemble NetCDF directory. Defaults to DATA_DIR/ens.",
    )
    parser.add_argument(
        "--skip-variogram",
        action="store_true",
        help="Skip variogram score computation (faster)",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save ECC predictions to file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    return parser.parse_args()


def log_msg(msg: str, verbose: bool) -> None:
    """Print message if verbose mode is enabled."""
    if verbose:
        print(msg)


def _rescale_y_from_info(y: torch.Tensor, reverse_info: list[dict]) -> torch.Tensor:
    """Rescale normalized y values back to original space using serialized params.

    Args:
        y: Normalized tensor with shape (..., c, x, y).
        reverse_info: List of dicts with 'mean' and 'scale' keys, one per channel.

    Returns:
        Rescaled tensor in original space.
    """
    y_rescaled = y.clone()
    for i, info in enumerate(reverse_info):
        y_rescaled[..., i, :, :] = y_rescaled[..., i, :, :] * info["scale"] + info["mean"]
    return y_rescaled


def compute_ecc_scores_cpu(
    ecc_preds: torch.Tensor,
    y_rescaled: torch.Tensor,
    timedeltas: np.ndarray,
    y_select_variables: list[str],
    *,
    score_file: Path,
    model_class: str,
    skip_variogram: bool,
) -> dict:
    """Compute CRPS, Energy Score, and Variogram Score on CPU.

    Args:
        ecc_preds: ECC predictions [N, M, n_vars, h, w].
        y_rescaled: Rescaled ground truth [N, n_vars, h, w].
        timedeltas: Array of lead times [N].
        y_select_variables: List of target variable names.
        score_file: Path to the scores CSV file.
        model_class: Name of the model class.
        skip_variogram: Whether to skip variogram score.

    Returns:
        Dict of scores per leadtime.
    """
    crps_fn = EnsembleCRPS()
    es_fn = EnergyScore(clamp=False)
    vs_fn = VariogramScore(p=0.5, chunk_size=256)

    n_times = ecc_preds.shape[0]
    crps_list, es_pv_list, es_full_list = [], [], []
    vs_pv_list, vs_full_list = [], []

    for i in trange(n_times, desc="Computing ECC scores"):
        pred_i = ecc_preds[i : i + 1]  # [1, M, n_vars, h, w]
        y_i = y_rescaled[i : i + 1]  # [1, n_vars, h, w]

        with torch.no_grad():
            crps_list.append(crps_fn(pred_i, y_i))
            es_pv_list.append(es_fn(pred_i, y_i, mode="per_var"))
            es_full_list.append(es_fn(pred_i, y_i, mode="complete"))
            if not skip_variogram:
                vs_pv_list.append(vs_fn(pred_i, y_i, mode="per_var"))
                vs_full_list.append(vs_fn(pred_i, y_i, mode="complete"))

    crps_per_margin = torch.cat(crps_list, dim=0)
    energy_score_per_var_u = torch.cat(es_pv_list, dim=0)
    energy_score_full_u = torch.cat(es_full_list, dim=0)
    variogram_pv_u = torch.cat(vs_pv_list, dim=0) if not skip_variogram else None
    variogram_full_u = torch.cat(vs_full_list, dim=0) if not skip_variogram else None

    method_label = f"{model_class}+ECC"
    crps_per_var = reduce(crps_per_margin, "t d x y -> d", "mean")
    crps_full = reduce(crps_per_margin, "t d x y -> 1", "mean")
    es_per_var = reduce(energy_score_per_var_u, "t d -> d", "mean")
    es_full = reduce(energy_score_full_u, "t -> 1", "mean")

    print(f"[{method_label}] Mean CRPS per var: {crps_per_var}")
    print(f"[{method_label}] Mean CRPS combined: {crps_full}")
    print(f"[{method_label}] Mean ES per var: {es_per_var}")
    print(f"[{method_label}] Mean ES combined: {es_full}")

    log_scores(
        file=score_file,
        model=method_label,
        metric="CRPS",
        variables=y_select_variables,
        scores=crps_per_var,
    )
    log_scores(
        file=score_file,
        model=method_label,
        metric="CRPS",
        variables=["combined"],
        scores=crps_full,
    )
    log_scores(
        file=score_file,
        model=method_label,
        metric="EnergyScore",
        variables=y_select_variables,
        scores=es_per_var,
    )
    log_scores(
        file=score_file,
        model=method_label,
        metric="EnergyScore",
        variables=["combined"],
        scores=es_full,
    )

    if not skip_variogram:
        vs_per_var = reduce(variogram_pv_u, "t d -> d", "mean")  # type: ignore
        vs_full = reduce(variogram_full_u, "t -> 1", "mean")  # type: ignore
        print(f"[{method_label}] Mean VS per var: {vs_per_var}")
        print(f"[{method_label}] Mean VS combined: {vs_full}")
        log_scores(
            file=score_file,
            model=method_label,
            metric="VariogramScore",
            variables=y_select_variables,
            scores=vs_per_var,
        )
        log_scores(
            file=score_file,
            model=method_label,
            metric="VariogramScore",
            variables=["combined"],
            scores=vs_full,
        )

    scores = compute_icon_scores_per_leadtime(
        timedeltas,
        crps_per_margin,
        energy_score_per_var_u,
        energy_score_full_u,
        y_select_variables=y_select_variables,
        vss_per_var=variogram_pv_u,
        vss_complete=variogram_full_u,
        method="ECC",
    )

    return scores


def score_split(
    split: str,
    *,
    model_dir: Path,
    ens_dir: Path,
    n_quantile_samples: int,
    skip_variogram: bool,
    save_predictions: bool,
    verbose: bool,
) -> dict:
    """Run ECC + scoring for a single split using saved distribution parameters.

    Steps:
        1. Load saved distribution parameters and ground truth
        2. Generate quantile samples from the predicted distributions
        3. Apply ECC using raw ensemble forecasts
        4. Rescale ECC predictions to original space
        5. Compute evaluation scores

    Returns:
        Dict of scores per leadtime.
    """
    params_path = model_dir / f"{split}_dist_params.pt"
    gt_path = model_dir / f"{split}_ground_truth.pt"
    predictions_path = model_dir / f"{split}_predictions_ecc.pt"

    if not params_path.exists():
        raise FileNotFoundError(
            f"Distribution params not found at {params_path}. Run icon_copulas_predict.py first."
        )
    if not gt_path.exists():
        raise FileNotFoundError(
            f"Ground truth not found at {gt_path}. Run icon_copulas_predict.py first."
        )

    # Step 1: Load saved data
    log_msg(f"Loading distribution params from {params_path}...", verbose)
    payload = torch.load(params_path, weights_only=False)

    stacked_preds = payload["params"]
    y_select_variables = payload["y_select_variables"]
    dataset_samples = payload["samples"]
    reverse_info = payload["reverse_modules"]
    model_class = payload["model_class"]
    score_file = model_dir / "scores.csv"

    log_msg(f"Loading ground truth from {gt_path}...", verbose)
    y_rescaled = torch.load(gt_path, weights_only=True)

    # Step 2: Generate quantile samples (chunked to avoid OOM)
    log_msg("Generating quantile samples from distribution parameters...", verbose)
    quantile_preds = quantile_samples_icon(stacked_preds, y_select_variables, M=n_quantile_samples)
    del stacked_preds
    log_msg(f"Quantile samples shape: {quantile_preds.shape}", verbose)

    # Step 3: Apply ECC
    log_msg("Applying ECC postprocessing...", verbose)
    ecc_preds = do_ecc_icon(quantile_preds, dataset_samples, ens_dir, verbose=verbose)
    del quantile_preds

    # Step 4: Rescale ECC predictions to original space
    log_msg("Rescaling ECC predictions to original space...", verbose)
    ecc_preds_rescaled = _rescale_y_from_info(ecc_preds, reverse_info)
    del ecc_preds

    if save_predictions:
        log_msg(f"Saving ECC predictions to {predictions_path}...", verbose)
        torch.save(ecc_preds_rescaled.cpu(), predictions_path)

    # Extract leadtimes
    timedeltas = np.array([sample[3] for sample in dataset_samples])

    # Step 5: Compute scores
    log_msg("Computing evaluation scores...", verbose)
    scores = compute_ecc_scores_cpu(
        ecc_preds_rescaled,
        y_rescaled,
        timedeltas,
        y_select_variables,
        score_file=score_file,
        model_class=model_class,
        skip_variogram=skip_variogram,
    )

    return scores


def process_run(run_path: str, args: argparse.Namespace) -> None:
    """Process a single WandB run: load saved params, ECC, score."""
    log_msg(f"\n{'#' * 60}\nProcessing run: {run_path}\n{'#' * 60}", args.verbose)

    model_id = run_path.split("/")[-1]
    output_dir = BASE_DIR.parent.parent / "outputs"

    log_msg(f"Looking for model with ID: {model_id}", args.verbose)

    model_dirs = list(output_dir.rglob(f"*{model_id}*"))
    if not model_dirs:
        raise FileNotFoundError(f"No model directory found for run ID '{model_id}' in {output_dir}")
    model_dir = model_dirs[0].parent.parent.parent

    log_msg(f"Model directory: {model_dir}", args.verbose)

    # Determine ensemble directory
    if args.ens_dir is not None:
        ens_dir = Path(args.ens_dir)
    else:
        from genpp.data.icon import DATA_DIR

        ens_dir = DATA_DIR / "ens"
    log_msg(f"Ensemble directory: {ens_dir}", args.verbose)

    # Load model_class from the first split's params for the final DataFrame
    first_split = args.split[0]
    params_path = model_dir / f"{first_split}_dist_params.pt"
    payload = torch.load(params_path, weights_only=False)
    model_class = payload["model_class"]
    del payload

    full_scores: dict = {}
    for split in args.split:
        log_msg(f"\n{'=' * 60}\nScoring split: {split}\n{'=' * 60}", args.verbose)
        scores = score_split(
            split,
            model_dir=model_dir,
            ens_dir=ens_dir,
            n_quantile_samples=args.n_quantile_samples,
            skip_variogram=args.skip_variogram,
            save_predictions=args.save_predictions,
            verbose=args.verbose,
        )
        full_scores[split] = scores

    # Update WandB
    log_msg("Updating WandB run...", args.verbose)
    update_wandb_run(run_path, full_scores)

    # Save scores DataFrame
    records = []
    for dataset, metrics in full_scores.items():
        for method_key, method_scores in metrics.items():
            for metric_name, horizons in method_scores.items():
                for horizon, value in horizons.items():
                    records.append(
                        (f"{model_class}+{method_key}", dataset, metric_name, horizon, value)
                    )
    df = pd.DataFrame(records, columns=["method", "dataset", "metric", "horizon", "value"])
    save_scores_df(df=df, run_path=run_path)

    log_msg(f"Done with run: {run_path}", args.verbose)


def main() -> None:
    """Main entry point."""
    args = parse_args()

    torch.set_float32_matmul_precision("high")

    for run_path in args.run_path:
        process_run(run_path, args)

    log_msg("All scoring completed!", args.verbose)


if __name__ == "__main__":
    main()
