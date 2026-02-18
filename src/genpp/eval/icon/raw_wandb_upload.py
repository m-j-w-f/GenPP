"""Compute Energy Score and CRPS for raw ICON ensemble outputs and summarize them per train valid and test set.

Run raw_ensemble.py first to generate the scores.csv file
Scores should be in this folder ``<output-dir>/raw_ensemble_scores_YYYYMM.csv``.
Merge them into one file with
``awk 'FNR==1 && NR!=1 {next} {print}' $(ls *.csv | sort) > merged.csv``
for exmaple.

The script is designed to upload the score to wandb for better logging.
"""

import argparse
from pathlib import Path

import polars as pl
from omegaconf import OmegaConf

import wandb
from genpp import BASE_DIR
from genpp.data.icon import DATA_DIR


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Upload the Scores per train val and test to wandb for the raw icon ensemble.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=DATA_DIR / "scores" / "raw_ensemble" / "merged.csv",
        help="Path to the CSV file containing the scores. This should be the merged CSV file containing all scores.",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        type=str,
        default=["raw_ensemble", "icon", "final"],
        help="Tags to associate with the wandb run.",
    )
    return parser.parse_args()


def main(args):
    # Get train valid and test splits
    cfg = OmegaConf.load(BASE_DIR / "configs" / "data" / "icon_full.yaml")
    splits = cfg.splits

    splits_df = pl.DataFrame(
        [{"split": k, "start": v["start"], "end": v["end"]} for k, v in splits.items()]
    ).with_columns(pl.col("start", "end").str.to_date())

    df = pl.read_csv(args.file).with_columns(
        pl.col("valid_date").cast(pl.Utf8).str.to_date("%Y%m%d")
    )
    # Group by split
    df = (
        df.join(splits_df, how="cross")
        .with_columns(
            pl.col("valid_date").is_between(pl.col("start"), pl.col("end")).alias("in_split")
        )
        .filter(pl.col("in_split"))
        .drop("in_split", "start", "end")
    )
    # Average over lead time and metric and unpivot
    df = (
        df.group_by("split", "leadtime_hours")
        .agg(pl.col(pl.Float64).mean())
        .unpivot(index=["split", "leadtime_hours"], value_name="score", variable_name="metric")
    )
    # Fix string names
    df = df.with_columns(
        pl.col("metric").replace(
            {
                "energy_score": "EnergyScore_combined",
                "energy_score_T_2M": "EnergyScore_T_2M+height_2.0",
                "energy_score_VMAX_10M": "EnergyScore_VMAX_10M+height_2_10.0",
                "crps_mean": "CRPS_combined",
                "crps_mean_T_2M": "CRPS_T_2M+height_2.0",
                "crps_mean_VMAX_10M": "CRPS_VMAX_10M+height_2_10.0",
            }
        )
    )
    # Create Score dict
    result = {}
    for row in df.iter_rows(named=True):
        split = row["split"]
        metric = row["metric"]
        leadtime = f"{row['leadtime_hours']}h"
        score = row["score"]

        result.setdefault(split, {}).setdefault(metric, {})[leadtime] = score

    run = wandb.init(
        project="genpp",
        name="icon-raw-ensemble-baseline-scores",
        tags=["baseline", "icon", "final"],
        config={
            "method": "raw_ensemble",
            "description": "Baseline scores for raw ICON ensemble forecasts",
        },
        dir=BASE_DIR.parent.parent / "outputs" / "BASELINE",
    )
    run.summary.update(result)
    run.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
