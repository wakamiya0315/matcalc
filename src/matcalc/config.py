"""Where the benchmark data comes from and where it is cached."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

BENCHMARK_HF_REPO_ID: str = "materialyze/matcalc-bench"
"""Hugging Face dataset repository that hosts the benchmark files."""

BENCHMARK_DATA_DIR: Path = Path(os.environ.get("MATCALC_CACHE_DIR", Path.home() / ".cache" / "matcalc"))
"""Local cache of downloaded benchmark files. Override it with the ``MATCALC_CACHE_DIR`` environment variable."""


def clear_cache(*, confirm: bool = True) -> None:
    """Delete every downloaded benchmark file in ``BENCHMARK_DATA_DIR``.

    Args:
        confirm: Ask on the terminal before deleting.
    """
    answer = "" if confirm else "y"
    while answer not in ("y", "n"):
        answer = input(f"Do you really want to delete everything in {BENCHMARK_DATA_DIR} (y|n)? ").lower().strip()
    if answer == "y":
        try:
            shutil.rmtree(BENCHMARK_DATA_DIR)
        except FileNotFoundError:
            logger.info("matcalc cache dir %s not found", BENCHMARK_DATA_DIR)
