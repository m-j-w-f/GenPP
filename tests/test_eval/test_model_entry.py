from pathlib import Path

from genpp.eval import best_encoders, best_models


def test_encoder_model_entry_properties():
    ce = best_encoders.classifierEncoder
    assert ce.model_id == "zln61d2q"
    assert ce.run_path == "feik/genpp/zln61d2q"
    assert isinstance(ce.output_dir, Path)
    # model_dir and model_checkpoint should be discoverable in the test environment
    assert ce.model_dir is not None and ce.model_dir.exists()
    assert ce.model_checkpoint is not None and ce.model_checkpoint.suffix == ".ckpt"


def test_basic_model_entry_properties():
    emos = best_models.emos[0]
    assert emos.model_id == "k32mygar"
    assert emos.run_path is None
    assert isinstance(emos.output_dir, Path)
