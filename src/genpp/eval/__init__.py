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
    emos=[ModelEntry(id="co74yk64")],
    drn=[ModelEntry(id="m5y9kwlh")],
    chen=[
        ModelEntry(id="057uzdg4", tag="ind_es"),
        ModelEntry(id="4g2v39ob", tag="ind_pes"),
        ModelEntry(id="1wmdbxm1", tag="ind_mses"),
        ModelEntry(id="5b2jan4d", tag="ind_mspes"),
        ModelEntry(id="unt6oe9w", tag="dir_es"),
        ModelEntry(id="y2to8vmf", tag="dir_pes"),
        ModelEntry(id="hrf26g7y", tag="dir_mses"),
        ModelEntry(id="yfigjk3e", tag="dir_mspes"),
    ],
    engression=[
        ModelEntry(id="3j5g7ils", tag="ind_es"),
        ModelEntry(id="2ajwxmir", tag="ind_pes"),
        ModelEntry(id="euak9uee", tag="ind_mses"),
        ModelEntry(id="3eevjkfj", tag="ind_mspes"),
        ModelEntry(id="iet9dund", tag="dir_es"),
        ModelEntry(id="7urden2d", tag="dir_pes"),
        ModelEntry(id="ku0pbqp1", tag="dir_mses"),
        ModelEntry(id="1vnjy1mj", tag="dir_mspes"),
    ],
    fm=[
        ModelEntry(id="f5yyzzxf", tag="ind_unet"),
        ModelEntry(id="fmz08y1j", tag="dir_unet"),
        ModelEntry(id="2t98jag4", tag="ind_uvit"),
        ModelEntry(id="oddm8ydj", tag="dir_uvit"),
    ],
)

best_encoders: BestEncoders = BestEncoders(
    autoencoder=ModelEntry(id="60kge09d"), classifierEncoder=ModelEntry(id="feik/genpp/zln61d2q")
)


__all__ = ["best_models", "best_encoders"]
