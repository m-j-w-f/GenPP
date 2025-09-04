from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import torch

try:
    import wandb
except ImportError:
    print("WandB not available")
    wandb = None


def log_scores(
    file: Path, model: str, metric: str, variables: Sequence, scores: torch.Tensor | Sequence[float]
) -> None:
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


def update_wandb_run(run_id: str, updates: dict[str, Any]) -> None:
    if wandb is None:
        print("WandB not available, skipping update.")
        return
    api = wandb.Api()
    # Locate the run
    run = api.run(run_id)

    for key, value in updates.items():
        run.summary[key] = value
    run.summary.update()
