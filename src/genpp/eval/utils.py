from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
from einops import reduce

try:
    import wandb
except ImportError:
    print("WandB not available")
    wandb = None


def log_scores(
    file: Path, model: str, metric: str, variables: Sequence, scores: torch.Tensor | Sequence[float]
) -> None:
    """Log evaluation scores to a CSV file.

    Creates or updates a CSV file with model evaluation scores. If the file exists,
    it removes existing entries for the same model, metric, and variables before
    appending the new scores.

    Args:
        file (Path): Path to the CSV file where scores will be logged.
        model (str): Name of the model being evaluated.
        metric (str): Name of the evaluation metric (e.g., 'CRPS', 'EnergyScore').
        variables (Sequence): Sequence of variable names corresponding to the scores.
        scores (torch.Tensor | Sequence[float]): Scores for each variable, either as
            a PyTorch tensor or a sequence of floats.
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.numpy()  # type: ignore

    # Create a list to store all score records
    records = []

    # Process each variable and its corresponding score
    for variable, score in zip(variables, scores):
        records.append({"Model": model, "Variable": variable, "Metric": metric, "Score": score})

    # Create DataFrame with new structure
    new_df = pd.DataFrame(records)

    # Check if file exists and read it, otherwise create new DataFrame
    if file.exists():
        existing_df = pd.read_csv(file)
        # Remove existing entries for this metric and variable and model
        existing_df = existing_df[
            ~(
                (existing_df["Model"] == model)
                & (existing_df["Metric"] == metric)
                & (existing_df["Variable"].isin(variables))
            )
        ]
        # Combine with new data
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(file, index=False)
    else:
        new_df.to_csv(file, index=False)


def update_wandb_run(run_path: str, updates: dict[str, Any]) -> None:
    """Update a Weights & Biases run with new summary metrics.

    Updates the summary statistics of an existing W&B run with the provided
    key-value pairs. If W&B is not available, prints a warning and returns.

    Args:
        run_path (str): The unique identifier of the W&B run to update.
        updates (dict[str, Any]): Dictionary of key-value pairs to add or update
            in the run's summary.
    """
    if wandb is None:
        print("WandB not available, skipping update.")
        return
    api = wandb.Api()
    # Locate the run
    run = api.run(run_path)

    for key, value in updates.items():
        run.summary[key] = value
    run.summary.update()


def save_scores_df(df: pd.DataFrame, run_path: str) -> None:
    if wandb is None:
        print("WandB not available, skipping update.")
        return
    short_run_id = run_path.split("/")[-1]

    # Initialize a new wandb run context to log the artifact
    api = wandb.Api()
    run = api.run(run_path)

    # Create and log artifact within a run context
    with wandb.init(
        entity=run.entity, project=run.project, id=short_run_id, resume="allow"
    ) as active_run:
        artifact = wandb.Artifact(f"scores_{short_run_id}", type="results")
        artifact.add(wandb.Table(dataframe=df), f"evaluation_scores_{short_run_id}")
        active_run.log_artifact(artifact)


def compute_scores_per_leadtime(
    prediction_timedeltas,
    crpss,
    ess_per_var,
    ess_complete,
    vss_per_var=None,
    vss_complete=None,
    method: str | None = None,
):
    """Compute scores per lead time for a given method.

    Args:
        prediction_timedeltas (array-like): Array of prediction timedeltas.
        crpss (torch.Tensor): CRPS scores with shape (time, feature, lon, lat).
        ess_per_var (torch.Tensor): Energy scores per variable with shape (time, feature).
        ess_complete (torch.Tensor): Energy scores complete with shape (time,).
        vss_per_var (torch.Tensor, optional): Variogram scores per variable with shape
            (time, feature). Defaults to None.
        vss_complete (torch.Tensor, optional): Variogram scores complete with shape (time,).
            Defaults to None.
        method (str | None, optional): Method name for the scores. Defaults to None.

    Returns:
        dict: Dictionary with scores per lead time.
    """
    td = np.unique(prediction_timedeltas)
    td_str = [f"{td / np.timedelta64(1, 'h'):.0f}h" for td in td]

    scores_delta = {
        method: {
            "CRPS_combined": {},
            "CRPS_2m_temperature": {},
            "CRPS_10m_windspeed": {},
            "EnergyScore_combined": {},
            "EnergyScore_2m_temperature": {},
            "EnergyScore_10m_windspeed": {},
        }
    }

    if vss_per_var is not None and vss_complete is not None:
        scores_delta[method].update(
            {
                "VariogramScore_combined": {},
                "VariogramScore_2m_temperature": {},
                "VariogramScore_10m_windspeed": {},
            }
        )

    for delta, delta_str in zip(td, td_str):
        mask = prediction_timedeltas == delta
        print(f"Processing leadtime {delta_str} with {np.sum(mask)} samples")
        crpss_delta = crpss[mask]
        ess_per_var_delta = ess_per_var[mask]
        ess_complete_delta = ess_complete[mask]

        scores_delta[method]["CRPS_combined"][delta_str] = reduce(
            crpss_delta, "t f lat lon -> 1", "mean"
        ).item()
        scores_delta[method]["CRPS_2m_temperature"][delta_str] = reduce(
            crpss_delta, "t f lat lon -> f", "mean"
        )[0].item()
        scores_delta[method]["CRPS_10m_windspeed"][delta_str] = reduce(
            crpss_delta, "t f lat lon -> f", "mean"
        )[1].item()
        scores_delta[method]["EnergyScore_combined"][delta_str] = ess_complete_delta.mean(
            dim=0
        ).item()
        scores_delta[method]["EnergyScore_2m_temperature"][delta_str] = ess_per_var_delta.mean(
            dim=0
        )[0].item()
        scores_delta[method]["EnergyScore_10m_windspeed"][delta_str] = ess_per_var_delta.mean(
            dim=0
        )[1].item()

        if vss_per_var is not None and vss_complete is not None:
            vss_per_var_delta = vss_per_var[mask]
            vss_complete_delta = vss_complete[mask]
            scores_delta[method]["VariogramScore_combined"][delta_str] = vss_complete_delta.mean(
                dim=0
            ).item()
            scores_delta[method]["VariogramScore_2m_temperature"][delta_str] = (
                vss_per_var_delta.mean(dim=0)[0].item()
            )
            scores_delta[method]["VariogramScore_10m_windspeed"][delta_str] = (
                vss_per_var_delta.mean(dim=0)[1].item()
            )
    if method is None:
        return scores_delta[None]
    return scores_delta


def save_predictions_dataarray(
    predictions: xr.DataArray, save_path: Path, overwrite: bool = False
) -> None:
    """Save predictions as an xarray DataArray in Zarr format.

    Args:
        predictions (xr.DataArray): Predictions with shape
            (time, prediction_timedelta, feature, lat, lon).
        save_path (Path): Path to save the Zarr file.
    """
    # Reset index, otherwise xarray will complain when saving
    predictions = predictions.reset_index("prediction")
    if not save_path.exists():
        predictions.to_zarr(save_path, consolidated=True)  # type: ignore
        print(f"Saved predictions to {save_path}.")
    else:
        if overwrite:
            predictions.to_zarr(save_path, mode="w", consolidated=True)  # type: ignore
            print(f"Overwritten existing file at {save_path}.")
        else:
            print(f"File {save_path} already exists.")


def load_predictions_dataarray(save_path: Path) -> xr.DataArray:
    """Load predictions from a Zarr file as an xarray DataArray.

    Args:
        save_path (Path): Path to the Zarr file.
    Returns:
        xr.DataArray: Loaded predictions with shape
    """
    predictions = xr.load_dataarray(save_path)
    predictions = predictions.set_index(prediction=["time", "prediction_timedelta"])
    return predictions
