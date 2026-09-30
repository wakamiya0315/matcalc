from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pytest
from monty.serialization import dumpfn

from matcalc.datasets import load_benchmark_data, sample_subset

if TYPE_CHECKING:
    from pathlib import Path


def test_sample_subset_matches_upstream_draw() -> None:
    items = [f"mp-{i}" for i in range(500)]
    random.seed(2025)
    upstream = random.sample(items, 10)  # what upstream matcalc does
    state = random.getstate()
    assert sample_subset(items, 10, 2025) == upstream
    assert random.getstate() == state  # the global generator is left alone


def test_sample_subset_keeps_everything_without_n() -> None:
    items = list(range(7))
    assert sample_subset(items, None, 1) == items
    assert sample_subset(items, 0, 1) == items


def test_load_local_dataset(tmp_path: Path) -> None:
    path = tmp_path / "tiny.json.gz"
    dumpfn([{"a": 1}], path)
    assert load_benchmark_data(path) == [{"a": 1}]


def test_download_file_checks_the_checksum_and_keeps_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib

    from matcalc import datasets
    from matcalc.datasets import RemoteFile, download_file

    source = tmp_path / "published.txt"
    source.write_text("data")
    md5 = hashlib.md5(b"data").hexdigest()  # noqa: S324
    monkeypatch.setattr(datasets, "BENCHMARK_DATA_DIR", tmp_path / "cache")
    with pytest.raises(OSError, match="MD5"):
        download_file(RemoteFile(source.as_uri(), "bad.txt", "0" * 32), "remote")
    assert not list((tmp_path / "cache" / "remote").iterdir())  # nothing half-written left behind
    path = download_file(RemoteFile(source.as_uri(), "good.txt", md5), "remote")
    assert path.read_text() == "data"
    source.unlink()
    assert download_file(RemoteFile(source.as_uri(), "good.txt", md5), "remote") == path  # cached
