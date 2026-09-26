from __future__ import annotations

import random
from typing import TYPE_CHECKING

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
