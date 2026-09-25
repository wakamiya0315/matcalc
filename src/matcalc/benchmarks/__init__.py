"""The four MatCalc benchmarks and a helper that runs several of them for several models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._common import Benchmark, Material
from .elasticity import ElasticityBenchmark
from .equilibrium import EquilibriumBenchmark
from .phonon import PhononBenchmark
from .softening import SofteningBenchmark

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd

BENCHMARKS: dict[str, type[Benchmark]] = {
    "equilibrium": EquilibriumBenchmark,
    "elasticity": ElasticityBenchmark,
    "phonon": PhononBenchmark,
    "softening": SofteningBenchmark,
}
"""Benchmark classes by name."""


def run_benchmarks(
    benchmarks: Sequence[Benchmark],
    models: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    chunk_size: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Run every benchmark for every model and merge the models' predictions per benchmark.

    Args:
        benchmarks: Benchmarks to run.
        models: Model label → ASE calculator, simulator or MACE model name.
        output_dir: If given, each (benchmark, model) run keeps its checkpoint here
            (``<benchmark>_<model>.json.gz``) and can be resumed.
        chunk_size: Materials per chunk (default: each benchmark's own).

    Returns:
        Benchmark name → table with one row per material and the predictions of every model.
    """
    tables: dict[str, pd.DataFrame] = {}
    for benchmark in benchmarks:
        merged: pd.DataFrame | None = None
        for model_name, model in models.items():
            checkpoint = Path(output_dir) / f"{benchmark.name}_{model_name}.json.gz" if output_dir else None
            table = benchmark.run(model, model_name, checkpoint_file=checkpoint, chunk_size=chunk_size)
            if merged is None:
                merged = table
            else:
                own = [benchmark.id_column, *(c for c in table.columns if c.endswith(f"_{model_name}"))]
                merged = merged.merge(table[own], on=benchmark.id_column, how="outer", validate="one_to_one")
        if merged is not None:
            tables[benchmark.name] = merged
    return tables


__all__ = [
    "BENCHMARKS",
    "Benchmark",
    "ElasticityBenchmark",
    "EquilibriumBenchmark",
    "Material",
    "PhononBenchmark",
    "SofteningBenchmark",
    "run_benchmarks",
]
