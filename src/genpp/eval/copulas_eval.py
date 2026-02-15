#!/usr/bin/env python
"""
Copula-based postprocessing (ECC & GCA) evaluation script.

Loads EMOS/DRN models, applies Ensemble Copula Coupling (ECC) and Gaussian
Copula Approach (GCA) postprocessing, and evaluates with CRPS, Energy Score,
and Variogram Score.

Usage:
    python copulas_eval.py
    python copulas_eval.py --model-groups emos drn --split val test
    python copulas_eval.py --model-groups emos --split val --skip-variogram
    python copulas_eval.py --device 1 -v
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
import xarray as xr
from einops import rearrange, reduce
from hydra.core.global_hydra import GlobalHydra
from scipy.stats import multivariate_normal, norm, truncnorm
from tqdm import tqdm

from genpp.configs import register_resolvers
from genpp.data.weatherbench2 import (
    FC_VARS,
    FORECAST_ENS_PATH,
    OBSERVATIONS_FLAT_PATH,
)
from genpp.eval import best_models
from genpp.eval.utils import (
    compute_scores_per_leadtime,
    log_scores,
    save_predictions_dataarray,
    save_scores_df,
    update_wandb_run,
)
from genpp.models.scores import EnergyScore, EnsembleCRPS, VariogramScore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copula-based postprocessing (ECC & GCA) evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-groups",
        type=str,
        nargs="+",
        default=["emos", "drn"],
        choices=["emos", "drn"],
        help="Model groups to evaluate",
    )
    parser.add_argument(
        "--split",
        type=str,
        nargs="+",
        default=["val", "test"],
        choices=["train", "val", "test"],
        help="Dataset split(s) to evaluate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU device index (e.g., '0' or '0,1')",
    )
    parser.add_argument(
        "--skip-variogram",
        action="store_true",
        help="Skip variogram score computation",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save ECC and GCA predictions to Zarr files",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    return parser.parse_args()


def parse_device(device_str: str) -> list[int]:
    return [int(d.strip()) for d in device_str.split(",")]


def log_msg(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)


# ---------------------------------------------------------------------------
# Data helpers (ported from notebook)
# ---------------------------------------------------------------------------


def stack_predictions(predictions) -> list[dict[str, torch.Tensor]]:
    n_dicts = len(predictions[0])
    merged = [defaultdict(list) for _ in range(n_dicts)]
    for batch in predictions:
        for i, d in enumerate(batch):
            for k, v in d.items():
                merged[i][k].append(v)
    return [{k: torch.cat(vs, dim=0) for k, vs in m.items()} for m in merged]


def predictions_to_dataarray(obs_da, preds):
    features = obs_da.coords["feature"].values
    new_data = []
    new_feature_names = []
    for feat_name, pred_dict in zip(features, preds):
        for param_name, tensor in pred_dict.items():
            arr = tensor.detach().cpu().numpy()
            new_data.append(arr)
            new_feature_names.append(f"{feat_name}_{param_name}")
    data = np.stack(new_data, axis=1)
    return xr.DataArray(
        data,
        coords={
            "prediction": obs_da.coords["prediction"],
            "feature": new_feature_names,
            "longitude": obs_da.coords["longitude"],
            "latitude": obs_da.coords["latitude"],
            "prediction_time": ("prediction", obs_da.coords["prediction_time"].values),
        },
        dims=("prediction", "feature", "longitude", "latitude"),
    )


# ---------------------------------------------------------------------------
# Copula helpers
# ---------------------------------------------------------------------------


def quantile_samples(pred_da, M=50):
    prediction = pred_da.coords["prediction"]
    longitude = pred_da.coords["longitude"]
    latitude = pred_da.coords["latitude"]
    features = ["2m_temperature", "10m_wind_speed"]

    qs = np.linspace(1 / (M + 1), M / (M + 1), M)

    mu_temp = pred_da.sel(feature="2m_temperature_mu").values
    sigma_temp = pred_da.sel(feature="2m_temperature_sigma").values
    temp_samples = norm.ppf(
        qs[:, None, None, None], loc=mu_temp[None, ...], scale=sigma_temp[None, ...]
    )

    mu_wind = pred_da.sel(feature="10m_wind_speed_mu").values
    sigma_wind = pred_da.sel(feature="10m_wind_speed_sigma").values
    a = (0 - mu_wind) / sigma_wind
    b = np.inf
    wind_samples = truncnorm.ppf(
        qs[:, None, None, None],
        a[None, ...],
        b,
        loc=mu_wind[None, ...],
        scale=sigma_wind[None, ...],
    )

    data = np.stack([temp_samples, wind_samples], axis=1)
    data = rearrange(data, "sample feature prediction lon lat -> prediction sample feature lon lat")

    return xr.DataArray(
        data,
        coords={
            "prediction": prediction,
            "sample": np.arange(M),
            "feature": features,
            "longitude": longitude,
            "latitude": latitude,
        },
        dims=("prediction", "sample", "feature", "longitude", "latitude"),
    )


def transform_to_latent_gaussian(obs_da, pred_da, eps=1e-7):
    prediction = obs_da.coords["prediction"]
    longitude = obs_da.coords["longitude"]
    latitude = obs_da.coords["latitude"]
    features = ["2m_temperature", "10m_wind_speed"]

    mu_temp = pred_da.sel(feature="2m_temperature_mu").values
    sigma_temp = pred_da.sel(feature="2m_temperature_sigma").values
    obs_temp = obs_da.sel(feature="2m_temperature").values
    cdf_temp = norm.cdf(obs_temp, loc=mu_temp, scale=sigma_temp)
    cdf_temp = np.clip(cdf_temp, eps, 1 - eps)
    latent_temp = norm.ppf(cdf_temp)

    mu_wind = pred_da.sel(feature="10m_wind_speed_mu").values
    sigma_wind = pred_da.sel(feature="10m_wind_speed_sigma").values
    obs_wind = obs_da.sel(feature="10m_wind_speed").values
    a = (0 - mu_wind) / sigma_wind
    b = np.inf
    cdf_wind = truncnorm.cdf(obs_wind, a=a, b=b, loc=mu_wind, scale=sigma_wind)
    cdf_wind = np.clip(cdf_wind, eps, 1 - eps)
    latent_wind = norm.ppf(cdf_wind)

    data = np.stack([latent_temp, latent_wind], axis=1)
    return xr.DataArray(
        data,
        coords={
            "prediction": prediction,
            "feature": features,
            "longitude": longitude,
            "latitude": latitude,
        },
        dims=("prediction", "feature", "longitude", "latitude"),
    )


def inverse_transform_latent(latent_samples, pred_da, eps=1e-7):
    prediction = pred_da.coords["prediction"]
    lon = pred_da.coords["longitude"]
    lat = pred_da.coords["latitude"]
    features = ["2m_temperature", "10m_wind_speed"]

    n_time, n_samples, n_feature, n_lon, n_lat = latent_samples.shape

    uniform_samples = norm.cdf(latent_samples)
    uniform_samples = np.clip(uniform_samples, eps, 1 - eps)
    orig_samples = np.zeros_like(uniform_samples)

    mu_temp = pred_da.sel(feature="2m_temperature_mu").values
    sigma_temp = pred_da.sel(feature="2m_temperature_sigma").values
    orig_samples[:, :, 0, :, :] = norm.ppf(
        uniform_samples[:, :, 0, :, :],
        loc=mu_temp[:, None, :, :],
        scale=sigma_temp[:, None, :, :],
    )

    mu_wind = pred_da.sel(feature="10m_wind_speed_mu").values
    sigma_wind = pred_da.sel(feature="10m_wind_speed_sigma").values
    a = (0 - mu_wind) / sigma_wind
    b = np.inf
    orig_samples[:, :, 1, :, :] = truncnorm.ppf(
        uniform_samples[:, :, 1, :, :],
        a=a[:, None, :, :],
        b=b,
        loc=mu_wind[:, None, :, :],
        scale=sigma_wind[:, None, :, :],
    )

    return xr.DataArray(
        orig_samples,
        coords={
            "prediction": prediction,
            "sample": np.arange(n_samples),
            "feature": features,
            "longitude": lon,
            "latitude": lat,
        },
        dims=("prediction", "sample", "feature", "longitude", "latitude"),
    )


# ---------------------------------------------------------------------------
# Core pipeline functions
# ---------------------------------------------------------------------------

SPLIT_CONFIG = {
    "train": ("fit", "train_dataloader", "train"),
    "val": ("validate", "val_dataloader", "val"),
    "test": ("test", "test_dataloader", "test"),
}


def get_split_predictions_and_obs(split, model, trainer, datamodule, cfg, verbose):
    """Run model forward pass and load ground truth for *split*.

    Returns (predictions_xr, y_obs, prediction_index).
    """
    setup_stage, dl_method, meta_key = SPLIT_CONFIG[split]

    datamodule.setup(stage=setup_stage)
    dataloader = getattr(datamodule, dl_method)()

    log_msg(f"Running predictions on {split} split...", verbose)
    predictions = trainer.predict(model, dataloader, return_predictions=True)
    stacked = stack_predictions(predictions)

    init_times = datamodule.cache_metadata["feature_metadata"]["time"][meta_key]
    timedeltas = datamodule.cache_metadata["feature_metadata"]["prediction_timedelta"][meta_key]
    target_times = init_times + timedeltas
    prediction_index = pd.MultiIndex.from_arrays(
        [init_times, timedeltas], names=["time", "prediction_timedelta"]
    )

    y_obs = (
        xr.open_dataset(OBSERVATIONS_FLAT_PATH)
        .sel(time=target_times)
        .to_dataarray("feature")
        .transpose("time", "feature", "longitude", "latitude")
        .rename({"time": "prediction_time"})
        .assign_coords(prediction=("prediction_time", prediction_index))
        .swap_dims({"prediction_time": "prediction"})
    )
    feature_order = list(cfg.data.y_select_variables)
    y_obs = y_obs.sel(feature=feature_order)

    predictions_xr = predictions_to_dataarray(y_obs, stacked)
    return predictions_xr, y_obs, prediction_index


def do_ecc(predictions_xr, prediction_index, M=50):
    """Perform Ensemble Copula Coupling postprocessing."""
    pbar = tqdm(total=5, desc="ECC postprocessing")

    pbar.set_postfix_str("computing quantile samples")
    pred_samples = quantile_samples(predictions_xr, M=M)
    pbar.update(1)

    pbar.set_postfix_str("loading ensemble forecasts")
    ens = (
        xr.open_dataset(FORECAST_ENS_PATH)[FC_VARS]
        .stack(prediction=("time", "prediction_timedelta"))
        .sel(prediction=prediction_index)
        .to_dataarray("feature")
        .transpose("prediction", "number", "feature", "longitude", "latitude")
    )
    pbar.update(1)

    pbar.set_postfix_str("ranking ensemble members")
    rng = np.random.default_rng(seed=420)
    noise = rng.uniform(low=-1e-8, high=1e-8, size=ens.shape)
    ens_noised = ens + noise
    del noise

    ens_ranked = ens_noised.rank(dim="number") - 1
    ens_ranked = ens_ranked.astype(np.int32)
    assert (ens_ranked.sum(dim="number") == 49 * 50 / 2).all()
    del ens, ens_noised
    pbar.update(1)

    # Use np.take_along_axis instead of xr.isel to avoid xarray coordinate
    # broadcasting.  If the longitude/latitude coords of pred_samples (from
    # the model predictions) differ even slightly from those of ens_ranked
    # (from FORECAST_ENS_PATH), xarray would create a cartesian-product
    # broadcast, exploding memory.
    pbar.set_postfix_str("reordering samples")
    rank_vals = ens_ranked.values  # (prediction, number, feature, lon, lat)
    del ens_ranked
    reordered_vals = np.take_along_axis(pred_samples.values, rank_vals, axis=1)
    del rank_vals
    pbar.update(1)

    pbar.set_postfix_str("building result")
    pred_samples_reordered = xr.DataArray(
        reordered_vals,
        coords={
            "prediction": pred_samples.coords["prediction"],
            "sample": np.arange(reordered_vals.shape[1]),
            "feature": pred_samples.coords["feature"],
            "longitude": pred_samples.coords["longitude"],
            "latitude": pred_samples.coords["latitude"],
        },
        dims=("prediction", "sample", "feature", "longitude", "latitude"),
    )
    del pred_samples, reordered_vals
    pbar.update(1)
    pbar.close()
    return pred_samples_reordered


def do_gca(Sigma, predictions_xr, y_shape, n_samples=50):
    """Perform Gaussian Copula Approach postprocessing."""
    pbar = tqdm(total=2, desc="GCA postprocessing")

    pbar.set_postfix_str("sampling from Gaussian copula")
    t_steps = len(predictions_xr.prediction)
    latent_samples = multivariate_normal.rvs(
        mean=np.zeros(Sigma.shape[0]),
        cov=Sigma,
        size=(t_steps, n_samples),  # type: ignore
    )
    latent_samples = latent_samples.reshape(t_steps, n_samples, *y_shape[1:])
    pbar.update(1)

    pbar.set_postfix_str("inverse-transforming to original space")
    result = inverse_transform_latent(latent_samples, predictions_xr)
    del latent_samples
    pbar.update(1)
    pbar.close()
    return result


def compute_copula_scores(
    method_name,
    pred_samples,
    y_da,
    *,
    score_file,
    model_class,
    datamodule,
    skip_variogram,
    device,
):
    """Compute CRPS, Energy Score, (Variogram Score) and return per-leadtime dict."""
    pred_samples = pred_samples.transpose(
        "prediction", "sample", "feature", "longitude", "latitude"
    )

    crps_fn = EnsembleCRPS()
    es_fn = EnergyScore(clamp=False)
    vs_fn = VariogramScore(p=0.5)

    prediction = y_da.prediction
    crpss_list, ess_pv_list, ess_c_list = [], [], []
    vss_pv_list, vss_c_list = [], []

    for p in tqdm(prediction, desc=f"Scoring {method_name}"):
        curr_obs = y_da.sel(prediction=p).to_numpy()
        curr_pred = pred_samples.sel(prediction=p).to_numpy()

        obs_t = torch.tensor(curr_obs[None, ...], dtype=torch.float32, device=device)
        pred_t = torch.tensor(curr_pred[None, ...], dtype=torch.float32, device=device)

        with torch.no_grad():
            crpss_list.append(crps_fn(pred_t, obs_t).cpu())
            ess_pv_list.append(es_fn(pred_t, obs_t, mode="per_var").cpu())
            ess_c_list.append(es_fn(pred_t, obs_t, mode="complete").cpu())
            if not skip_variogram:
                vss_pv_list.append(vs_fn(pred_t, obs_t, mode="per_var").cpu())
                vss_c_list.append(vs_fn(pred_t, obs_t, mode="complete").cpu())

    crpss = torch.cat(crpss_list, dim=0)
    ess_per_var = torch.cat(ess_pv_list, dim=0)
    ess_complete = torch.cat(ess_c_list, dim=0)
    vss_per_var = torch.cat(vss_pv_list, dim=0) if not skip_variogram else None
    vss_complete = torch.cat(vss_c_list, dim=0) if not skip_variogram else None

    # Print summary
    print(f"[{method_name}] Mean CRPS: {reduce(crpss, 't f lat lon -> f', 'mean')}")
    print(f"[{method_name}] Mean ES per var: {ess_per_var.mean(dim=0)}")
    print(f"[{method_name}] Mean ES combined: {ess_complete.mean(dim=0)}")
    if not skip_variogram:
        print(f"[{method_name}] Mean VS per var: {vss_per_var.mean(dim=0)}")  # type: ignore
        print(f"[{method_name}] Mean VS combined: {vss_complete.mean(dim=0)}")  # type: ignore

    # Log to CSV
    log_scores(
        score_file,
        f"{model_class}+{method_name}",
        "EnsembleCRPS",
        datamodule.y_select_variables,
        reduce(crpss, "t f lat lon -> f", "mean"),
    )
    log_scores(
        score_file,
        f"{model_class}+{method_name}",
        "EnsembleCRPS",
        ["combined"],
        reduce(crpss, "t f lat lon -> 1", "mean"),
    )
    log_scores(
        score_file,
        f"{model_class}+{method_name}",
        "EnergyScore",
        datamodule.y_select_variables,
        ess_per_var.mean(dim=0),
    )
    log_scores(
        score_file,
        f"{model_class}+{method_name}",
        "EnergyScore",
        ["combined"],
        ess_complete.mean(dim=0, keepdim=True),
    )
    if not skip_variogram:
        log_scores(
            score_file,
            f"{model_class}+{method_name}",
            "VariogramScore",
            datamodule.y_select_variables,
            vss_per_var.mean(dim=0),  # type: ignore
        )
        log_scores(
            score_file,
            f"{model_class}+{method_name}",
            "VariogramScore",
            ["combined"],
            vss_complete.mean(dim=0, keepdim=True),  # type: ignore
        )

    scores_delta = compute_scores_per_leadtime(
        pred_samples.prediction_timedelta.values,
        crpss,
        ess_per_var,
        ess_complete,
        vss_per_var,
        vss_complete,
        method=method_name,
    )
    return scores_delta


# ---------------------------------------------------------------------------
# Model processing
# ---------------------------------------------------------------------------


def process_model(model_entry, splits, args):
    """Run ECC and GCA evaluation for a single model on the requested splits."""
    device = torch.device(
        f"cuda:{parse_device(args.device)[0]}" if torch.cuda.is_available() else "cpu"
    )

    print(f"\n{'#' * 60}")
    print(f"Processing model: {model_entry.id}")
    print(f"{'#' * 60}")

    model_dir = model_entry.model_dir
    score_file = model_dir / "scores.csv"

    # Load config (handles Hydra init/cleanup internally)
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=str(model_dir / ".hydra"), version_base=None):
        cfg = hydra.compose(config_name="config")

    cfg.data.module.dataloader_config.train.shuffle = False
    cfg.data.module.dataloader_config.val.shuffle = False
    cfg.data.module.dataloader_config.test.shuffle = False

    model_class = cfg.model._target_.split(".")[-1]

    datamodule = hydra.utils.instantiate(cfg.data.module)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    datamodule.setup(stage="validate")
    datamodule.setup(stage="test")

    model = model_entry.model
    devices = parse_device(args.device)
    trainer = L.Trainer(logger=False, accelerator="gpu", devices=devices)

    # --- Always obtain train predictions for GCA Sigma estimation ---
    log_msg("Getting train predictions for GCA Sigma estimation...", args.verbose)
    train_preds_xr, y_train, train_pred_idx = get_split_predictions_and_obs(
        "train", model, trainer, datamodule, cfg, args.verbose
    )
    latent = transform_to_latent_gaussian(y_train, train_preds_xr)
    flat = latent.stack(space=("feature", "longitude", "latitude"))
    Sigma = np.cov(flat.values, rowvar=False)
    del latent, flat  # free memory

    # Keep train data only if the train split is requested for evaluation
    need_train = "train" in splits
    if not need_train:
        del train_preds_xr, y_train, train_pred_idx
        train_preds_xr = train_pred_idx = y_train = None  # type: ignore[assignment]

    # --- Evaluate each requested split ---
    all_scores: dict = {}
    all_records: list = []

    for split in splits:
        print(f"\n{'=' * 60}")
        print(f"Evaluating split: {split}")
        print(f"{'=' * 60}")

        if split == "train":
            predictions_xr = train_preds_xr
            y_obs = y_train
            prediction_index = train_pred_idx
        else:
            predictions_xr, y_obs, prediction_index = get_split_predictions_and_obs(
                split, model, trainer, datamodule, cfg, args.verbose
            )

        # --- ECC ---
        log_msg("Running ECC postprocessing...", args.verbose)
        ecc_preds = do_ecc(predictions_xr, prediction_index)
        if args.save_predictions:
            save_predictions_dataarray(
                ecc_preds, model_dir / f"{split}_predictions_ecc.zarr", overwrite=True
            )

        log_msg("Computing ECC scores...", args.verbose)
        ecc_scores = compute_copula_scores(
            "ECC",
            ecc_preds,
            y_obs,
            score_file=score_file,
            model_class=model_class,
            datamodule=datamodule,
            skip_variogram=args.skip_variogram,
            device=device,
        )

        # --- GCA ---
        log_msg("Running GCA postprocessing...", args.verbose)
        gca_preds = do_gca(Sigma, predictions_xr, y_obs.shape)  # type: ignore
        if args.save_predictions:
            save_predictions_dataarray(
                gca_preds, model_dir / f"{split}_predictions_gca.zarr", overwrite=True
            )

        log_msg("Computing GCA scores...", args.verbose)
        gca_scores = compute_copula_scores(
            "GCA",
            gca_preds,
            y_obs,
            score_file=score_file,
            model_class=model_class,
            datamodule=datamodule,
            skip_variogram=args.skip_variogram,
            device=device,
        )

        split_scores = {**ecc_scores, **gca_scores}  # type: ignore
        all_scores[split] = split_scores

        for pp_model, metrics in split_scores.items():
            for metric_name, horizons in metrics.items():
                for horizon, value in horizons.items():
                    all_records.append(
                        (f"{model_class}+{pp_model}", split, metric_name, horizon, value)
                    )

    # Update WandB
    log_msg("Updating WandB run...", args.verbose)
    update_wandb_run(model_entry.run_path, all_scores)

    df = pd.DataFrame(all_records, columns=["method", "dataset", "metric", "horizon", "value"])
    save_scores_df(df=df, run_path=model_entry.run_path)

    print(f"Done with model: {model_entry.id}")


def main():
    args = parse_args()

    try:
        register_resolvers()
    except Exception:
        pass

    torch.set_float32_matmul_precision("high")

    model_groups = {
        "emos": best_models.emos,
        "drn": best_models.drn,
    }

    for group_name in args.model_groups:
        entries = model_groups[group_name]
        for model_entry in entries:
            process_model(model_entry, args.split, args)

    print("All models completed!")


if __name__ == "__main__":
    main()
