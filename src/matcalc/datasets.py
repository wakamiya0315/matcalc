"""Download the benchmark datasets and draw reproducible subsets of them."""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from huggingface_hub import HfApi, hf_hub_download
from monty.json import MontyDecoder
from monty.serialization import loadfn

from .config import BENCHMARK_DATA_DIR, BENCHMARK_HF_REPO_ID

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

FIGSHARE_DOWNLOAD_URL = "https://ndownloader.figshare.com/files/{file_id}"
"""Direct download address of a Figshare file (``figshare.com/ndownloader`` answers scripts with an empty page)."""


@dataclass(frozen=True)
class RemoteFile:
    """A published file, identified by its address and checked by its MD5 checksum.

    Attributes:
        url: Download address (https).
        name: File name in the local cache.
        md5: MD5 checksum of the published file.
    """

    url: str
    name: str
    md5: str


@dataclass(frozen=True)
class FigshareFile:
    """A file published on Figshare, identified by its file id and checked by its MD5 checksum.

    Attributes:
        file_id: Figshare file id (the number in ``figshare.com/files/<id>``).
        name: File name in the local cache.
        md5: MD5 checksum of the published file.
    """

    file_id: int
    name: str
    md5: str

    @property
    def url(self) -> str:
        """Direct download address."""
        return FIGSHARE_DOWNLOAD_URL.format(file_id=self.file_id)


def download_file(file: RemoteFile | FigshareFile, subdirectory: str) -> Path:
    """Download a published file once into the cache (``BENCHMARK_DATA_DIR/<subdirectory>``).

    Args:
        file: The file.
        subdirectory: Folder of the cache that holds the file.

    Returns:
        Path of the cached file.

    Raises:
        OSError: If the downloaded file does not have the published MD5 checksum.
    """
    path = BENCHMARK_DATA_DIR / subdirectory / file.name
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"~{path.name}")
    logger.info("Downloading %s from %s", file.name, file.url)
    request = urllib.request.Request(file.url, headers={"User-Agent": "matcalc"})  # noqa: S310 - fixed https address
    digest = hashlib.md5()  # noqa: S324 - checksum of the published file, not a security measure
    with urllib.request.urlopen(request) as response, partial.open("wb") as out:  # noqa: S310
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            out.write(chunk)
    if digest.hexdigest() != file.md5:
        partial.unlink()
        raise OSError(f"{file.name}: MD5 {digest.hexdigest()} differs from the published {file.md5}; try again")
    partial.replace(path)
    return path


def download_figshare_file(file: FigshareFile, subdirectory: str = "figshare") -> Path:
    """``download_file`` for a Figshare file (kept for the benchmarks that use it).

    Args:
        file: The file.
        subdirectory: Folder of the cache that holds the file.

    Returns:
        Path of the cached file.
    """
    return download_file(file, subdirectory)


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
