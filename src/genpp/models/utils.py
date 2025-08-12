import torch
from omegaconf import DictConfig


def instantiate_partial_scheduler(
    partial_scheduler: DictConfig, optimizer: torch.optim.Optimizer
) -> DictConfig:
    # This is ugly but works because the lr_scheduler_partial is a DictConfig
    if (
        partial_scheduler.scheduler.func is not torch.optim.lr_scheduler.ChainedScheduler
    ):  # Just a single scheduler
        partial_scheduler.scheduler = partial_scheduler.scheduler(optimizer)
    else:  # We need to instantiate the chained scheduler with the optimizer
        # It gets even uglier
        inner_schedulers = [
            p(optimizer) for p in partial_scheduler.scheduler.keywords["schedulers"]
        ]
        # Overwrite the inner schedulers with the instantiated ones by calling the func directly
        partial_scheduler.scheduler = partial_scheduler.scheduler.func(
            *partial_scheduler.scheduler.args,
            schedulers=inner_schedulers,
            **{k: v for k, v in partial_scheduler.scheduler.keywords.items() if k != "schedulers"},
        )

    return partial_scheduler
