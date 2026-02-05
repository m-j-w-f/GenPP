import importlib
import warnings
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from pickle import UnpicklingError

import hydra

from genpp import BASE_DIR
from genpp.models.base_module import BaseModule


@dataclass(frozen=True)
class ModelEntry:
    """Represents a single model entry (id and optional tag)."""

    id: str
    tag: str | None = None

    # Derived properties computed from `id` and the repository layout
    @property
    def model_id(self) -> str:
        """The short model id (last path component of the id)."""
        return self.id.split("/")[-1]

    @property
    def run_path(self) -> str | None:
        """Return the run path if `id` contains a path-like prefix, else None."""
        return self.id if "/" in self.id else None

    @property
    def output_dir(self) -> Path:
        """Path to the global outputs directory used to locate runs."""
        return BASE_DIR.parent.parent / "outputs"

    @property
    def model_dir(self) -> Path:
        """Locate the model run directory in `output_dir` (or return None if not found)."""
        try:
            model_dir = list(self.output_dir.rglob(f"*{self.model_id}*"))[0].parent.parent.parent
            return model_dir
        except IndexError:
            raise ValueError(f"Model directory for model id {self.model_id} not found.")

    @property
    def model_checkpoint(self) -> Path:
        """Return the first checkpoint file found under `model_dir`, or None if not found."""
        md = self.model_dir
        try:
            return list(md.rglob("*.ckpt"))[0]
        except IndexError:
            raise ValueError(f"Model checkpoint for model id {self.model_id} not found.")

    @cached_property
    def config(self):
        with hydra.initialize_config_dir(
            config_dir=str(self.model_dir / ".hydra"), version_base=None
        ):
            cfg = hydra.compose(
                config_name="config",
            )
        return cfg

    @cached_property
    def model(self) -> BaseModule:
        """Load and return the model from the checkpoint."""
        cfg = self.config
        class_path = cfg.model._target_

        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        ModelClass = getattr(module, class_name)
        try:
            encoder = ModelClass.load_from_checkpoint(self.model_checkpoint)
            return encoder
        except UnpicklingError:
            warnings.warn(
                f"UnpicklingError encountered when loading model {self.model_id}: Trying without weights_only."
            )
            encoder = ModelClass.load_from_checkpoint(self.model_checkpoint, weights_only=False)
            return encoder


@dataclass(frozen=True)
class BestModels:
    """Container for named best models used in evaluation."""

    # All model groups are lists of ModelEntry to keep representation uniform
    emos: list[ModelEntry] = field(default_factory=list)
    drn: list[ModelEntry] = field(default_factory=list)
    chen: list[ModelEntry] = field(default_factory=list)
    fm: list[ModelEntry] = field(default_factory=list)
    engression: list[ModelEntry] = field(default_factory=list)

    def __iter__(self):
        """Iterate over all model lists in the dataclass."""
        for field_value in self.__dataclass_fields__.values():
            yield field_value.name, getattr(self, field_value.name)


@dataclass(frozen=True)
class BestEncoders:
    """Container for named best encoders used in evaluation."""

    autoencoder: ModelEntry
    classifierEncoder: ModelEntry


best_models: BestModels = BestModels(
    emos=[ModelEntry(id="k32mygar")],
    drn=[ModelEntry(id="hn0gdrqm")],
    chen=[
        # ModelEntry(id="2f1vpjz0", tag="standard"),
        # ModelEntry(id="23phjuuc", tag="chen_spatial_3"),
        # ModelEntry(id="ynl8hbdr", tag="chen_spatial_5"),
        # ModelEntry(id="eu94vgqa", tag="chen_spatial_7"),
        ModelEntry(id="e4oxnxiy", tag="es"),
        ModelEntry(id="f327mrxm", tag="pes"),
        ModelEntry(id="k3i9kcxd", tag="mes"),
        ModelEntry(id="upfya4wp", tag="pmes"),
    ],
    fm=[
        ModelEntry(id="pwb8kh5a", tag="unet_std"),
        ModelEntry(id="qiso22uq", tag="unet_abs"),
        ModelEntry(id="ftmjxjq9", tag="uvit_std"),
        ModelEntry(id="s19rsj2i", tag="uvit_abs"),
    ],
    engression=[ModelEntry(id="hbuy7eio", tag="standard")],
)

best_encoders: BestEncoders = BestEncoders(
    autoencoder=ModelEntry(id="60kge09d"), classifierEncoder=ModelEntry(id="feik/genpp/zln61d2q")
)


__all__ = ["best_models", "best_encoders"]
