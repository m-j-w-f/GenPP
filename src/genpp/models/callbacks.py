import torch
from einops import rearrange
from lightning import Callback, LightningModule, Trainer
from tqdm import tqdm

from genpp.models.layers import CropND


class FitScaleVarianceTDCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Fit a regression of the form absolute_prediction_error ~ LeadTime for each variable.
        The regression coefficients are stored in the pl_module as a buffer called 'fit_scale_variance_td'.

        Args:
            trainer (Trainer): The trainer instance.
            pl_module (LightningModule): The Lightning module instance.
        """
        # If already fitted, skip
        if (
            hasattr(pl_module, "fit_scale_variance_td")
            and pl_module.fit_scale_variance_td is not None
        ):
            print("fit_scale_variance_td already fitted, skipping.")
            return

        # Access the training dataloader
        train_loader = trainer.train_dataloader
        if train_loader is None:
            raise ValueError("Training dataloader is not available.")

        crop_layer: CropND = pl_module.crop if hasattr(pl_module, "crop") else None  # type: ignore

        n_vars = None
        A = None
        b = None

        with torch.no_grad():
            pbar = tqdm(train_loader, desc="Fitting scale_variance~TD")
            for nwp, obs, td in pbar:
                # Figure out how many variables we have
                if n_vars is None:
                    n_vars = obs.shape[1]
                # The NWP vars we need are the first n_vars
                nwp = nwp[:, :n_vars, ...]

                # Initialize A and b
                if A is None:
                    A = torch.zeros((n_vars, 2, 2)).to(nwp)
                if b is None:
                    b = torch.zeros((n_vars, 2)).to(nwp)

                if crop_layer is not None:
                    nwp = crop_layer(nwp)
                    # If shapes don't match, crop obs as well
                    if nwp.shape != obs.shape:
                        obs = crop_layer(obs)
                # Now nwp and obs should have the same shape and no padding
                # We can run the regression now
                td = rearrange(td, "b -> b 1 1 1")
                td = td.expand_as(obs)
                td = rearrange(td, "b c h w -> c (b h w)")

                nwp = rearrange(nwp, "b c h w -> c (b h w)")
                obs = rearrange(obs, "b c h w -> c (b h w)")

                diff = (obs - nwp).abs()
                ones = torch.ones_like(diff)
                X = torch.stack([ones, td], dim=1)

                y = diff

                # The rearrange is the same as calling the transpose on the last two dims if X
                A += torch.bmm(X, rearrange(X, "c a b -> c b a"))
                b += torch.bmm(X, y.unsqueeze(-1)).squeeze(-1)
            betas = torch.linalg.solve(A, b)
            # The first dimension is the variable dimension
            # The second dimension is the intercept and slope
            pl_module.register_buffer("scale_variance_td", betas)  # Shape [n_vars, 2]
