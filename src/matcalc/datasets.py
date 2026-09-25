"""Download the benchmark datasets and draw reproducible subsets of them."""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

from huggingface_hub import HfApi, hf_hub_download
from monty.json import MontyDecoder
from monty.serialization import loadfn

from .config import BENCHMARK_DATA_DIR, BENCHMARK_HF_REPO_ID

if TYPE_CHECKING:
    from collections.abc import Sequence


def list_benchmark_files() -> list[str]:
    """List the benchmark files available in the ``materialyze/matcalc-bench`` Hugging Face dataset.

    Returns:
        File names ending in ``.json.gz``.
    """
    files = HfApi().list_repo_files(BENCHMARK_HF_REPO_ID, repo_type="dataset")
    return [f for f in files if f.endswith(".json.gz")]


def load_benchmark_data(name: str | Path) -> Any:
    """Load a benchmark dataset.

    Args:
        name: A file name in the ``materialyze/matcalc-bench`` Hugging Face dataset (downloaded
            once and cached in ``BENCHMARK_DATA_DIR``), or a ``pathlib.Path`` to a local JSON file.

    Returns:
        The decoded dataset: a list of entries, or (Softening) a dict of material id to frames.
        pymatgen objects such as ``Structure`` are decoded.
    """
    if isinstance(name, Path):
        return loadfn(name)
    local_path = hf_hub_download(
        repo_id=BENCHMARK_HF_REPO_ID,
        filename=name,
        repo_type="dataset",
        cache_dir=str(BENCHMARK_DATA_DIR),
    )
    with gzip.open(local_path, "rt", encoding="utf-8") as f:
        return json.load(f, cls=MontyDecoder)


def sample_subset[T](items: Sequence[T], n_samples: int | None, seed: int) -> list[T]:
    """Draw ``n_samples`` items at random, reproducibly.

    The draw is identical to upstream matcalc's ``random.seed(seed); random.sample(items, n)``,
    so a given ``seed`` selects the same materials as before, but the global ``random`` state is
    left untouched.

    Args:
        items: All items, in dataset order.
        n_samples: How many to draw; ``None`` or 0 keeps everything (in dataset order).
        seed: Seed of the random generator.

    Returns:
        The selected items, in the order they were drawn.
    """
    if not n_samples:
        return list(items)
    return random.Random(seed).sample(list(items), n_samples)  # noqa: S311 - reproducible sampling, not security
