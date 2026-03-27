"""Backward-compatible import aliases for WB2 copula helpers."""


def stack_predictions(predictions):
    from genpp.eval.wb2.copulas_eval import stack_predictions as _impl

    return _impl(predictions)


def predictions_to_dataarray(obs_da, preds):
    from genpp.eval.wb2.copulas_eval import predictions_to_dataarray as _impl

    return _impl(obs_da, preds)


def transform_to_latent_gaussian(obs_da, pred_da, eps=1e-7):
    from genpp.eval.wb2.copulas_eval import transform_to_latent_gaussian as _impl

    return _impl(obs_da, pred_da, eps=eps)


def get_split_predictions_and_obs(split, model, trainer, datamodule, cfg, verbose):
    from genpp.eval.wb2.copulas_eval import get_split_predictions_and_obs as _impl

    return _impl(split, model, trainer, datamodule, cfg, verbose)


def do_ecc(predictions_xr, prediction_index, M=50):
    from genpp.eval.wb2.copulas_eval import do_ecc as _impl

    return _impl(predictions_xr, prediction_index, M=M)


def do_gca(Sigma, predictions_xr, y_shape, n_samples=50):
    from genpp.eval.wb2.copulas_eval import do_gca as _impl

    return _impl(Sigma, predictions_xr, y_shape, n_samples=n_samples)

__all__ = [
    "do_ecc",
    "do_gca",
    "get_split_predictions_and_obs",
    "predictions_to_dataarray",
    "stack_predictions",
    "transform_to_latent_gaussian",
]
