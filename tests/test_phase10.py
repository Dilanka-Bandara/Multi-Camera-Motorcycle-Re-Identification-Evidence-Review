"""Dataset transforms and archive safety with independent expected geometry."""

import zipfile
from pathlib import Path

import pytest

from app.config import InputConfig
from app.evaluation.split import tuning_identity, tuning_pair
from scripts.download_data import extract
from scripts.make_synthetic_pair import transform_box


def test_partial_camera_crop_normalization() -> None:
    assert transform_box((40, 10, 30, 20), 50, 50, 100) == (0, 0.1, 0.4, 0.3)
    assert transform_box((0, 0, 10, 10), 50, 50, 100) is None
    assert transform_box((80, 90, 40, 30), 50, 50, 100) == (0.6, 0.9, 1, 1)


def test_zip_preserves_existing_and_rejects_escape(tmp_path: Path) -> None:
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("nested/data.txt", "known content")
    destination = tmp_path / "extracted"
    extract(archive, destination)
    extract(archive, destination)
    (destination / "nested/data.txt").write_text("user edits")
    with pytest.raises(FileExistsError):
        extract(archive, destination)
    assert (destination / "nested/data.txt").read_text() == "user edits"
    malicious = tmp_path / "escape.zip"
    with zipfile.ZipFile(malicious, "w") as stream:
        stream.writestr("../outside.txt", "outside")
    with pytest.raises(ValueError):
        extract(malicious, destination)
    assert not (tmp_path / "outside.txt").exists()


@pytest.mark.parametrize("directory", ["../secret", "/tmp/secret", "C:\\secret", "a/../../b"])
def test_preset_paths_are_relative(directory: str) -> None:
    with pytest.raises(ValueError):
        InputConfig(directory=directory)


def test_identity_split_is_stable_and_nontrivial() -> None:
    first = {str(i): tuning_identity(str(i)) for i in range(40)}
    assert set(first.values()) == {True, False}
    assert all(tuning_identity(k) == v for k, v in first.items())


def test_training_excludes_held_out_identities_in_either_role() -> None:
    training = next(str(i) for i in range(40) if tuning_identity(str(i)))
    held_out = next(str(i) for i in range(40) if not tuning_identity(str(i)))
    assert tuning_pair(training, training)
    assert not tuning_pair(training, held_out)
    assert not tuning_pair(held_out, training)
    assert not tuning_pair(held_out, held_out)
