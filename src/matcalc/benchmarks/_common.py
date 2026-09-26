"""What the four benchmarks share: dataset rows, resumable runs, result tables and summaries.

Each benchmark subclass states its science in two methods:

- ``read_entries`` turns the raw dataset into ``Material`` records;
- ``evaluate`` computes the MLIP's predictions for a list of materials, stage by stage.

``Benchmark.run`` does the bookkeeping around them: it splits the materials into chunks, saves the
finished rows to a checkpoint after every chunk (so an interrupted run resumes where it stopped),
and returns a table with the DFT reference values next to the predictions.
"""

from __future__ import annotations

import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd
from monty.serialization import dumpfn, loadfn

from matcalc.datasets import load_benchmark_data, sample_subset
from matcalc.simulation import as_simulator

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from pymatgen.core import Structure

    from matcalc.simulation import Simulator

logger = logging.getLogger(__name__)

OK = "ok"
"""``status`` of a material whose prediction succeeded."""


@dataclass
class Material:
    """One benchmark entry.

    Attributes:
        material_id: Identifier used by the dataset (e.g. ``mp-149``).
        formula: Chemical formula.
        structure: Input structure (DFT-relaxed; for Softening, the first high-energy frame).
        reference: DFT reference data of this material; keys depend on the benchmark.
        settings: Settings of the DFT calculation that the benchmark reuses (Phonon: supercell and
            displacements).
    """

    material_id: str
    formula: str
    structure: Structure
    reference: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)


class Benchmark:
    """Common driver of the four benchmarks.

    Attributes:
        dataset: Dataset file name on Hugging Face, or a local ``Path``.
        n_samples: Number of materials drawn at random (``None`` = all).
        seed: Seed of the random draw (and of the Equilibrium perturbation).
        workers: Processes for the CPU-heavy post-processing.
        materials: The materials of this run, in the order they were drawn.
        timings: Wall time spent in each stage so far (s).
    """

    name: ClassVar[str]
    """Short name, used for file names (e.g. ``"elasticity"``)."""
    id_column: ClassVar[str]
    """Name of the material-id column of the result table (as in upstream matcalc)."""
    default_dataset: ClassVar[str | Path]
    """Dataset used when none is given: a file name on Hugging Face or a local ``Path``."""
    reference_columns: ClassVar[tuple[str, ...]] = ()
    """DFT quantities copied into the table as ``<quantity>_DFT`` columns."""
    summary_metrics: ClassVar[dict[str, str]] = {}
    """Quantities summarized by ``summarize``: ``"error"`` (absolute error vs DFT) or ``"value"``."""
    default_chunk_size: ClassVar[int] = 100
    """Materials per chunk (a checkpoint is written after each chunk)."""
    batched_chunk_size: ClassVar[int] = 1_000_000
    """Materials per chunk for batched simulators (TorchSim): large, so the GPU stays full."""

    def __init__(
        self,
        dataset: str | Path | None = None,
        *,
        n_samples: int | None = None,
        seed: int = 42,
        workers: int = 1,
    ) -> None:
        """
        Args:
            dataset: Dataset file name on Hugging Face, or a local ``Path``; default:
                ``default_dataset``.
            n_samples: Draw this many materials at random (``None`` = all). The draw is the same as
                upstream matcalc's for the same seed.
            seed: Seed of the random draw.
            workers: Processes for the CPU-heavy post-processing (phonopy, structural fingerprints).
                With more than one, a script that runs a benchmark must be guarded by
                ``if __name__ == "__main__":`` (the workers are started with "spawn").
        """
        self.dataset = dataset or self.default_dataset
        self.n_samples = n_samples
        self.seed = seed
        self.workers = workers
        self.materials = sample_subset(self.read_entries(load_benchmark_data(self.dataset)), n_samples, seed)
        self.timings: dict[str, float] = {}

    def read_entries(self, raw: Any) -> list[Material]:
        """Turn the raw dataset into ``Material`` records (implemented by each benchmark).

        Args:
            raw: The decoded dataset.

        Returns:
            All materials, in dataset order.
        """
        raise NotImplementedError

    def prepare(self, simulator: Simulator, cache: dict[str, Any]) -> None:
        """Work shared by all materials, done once before the first chunk (default: nothing).

        Args:
            simulator: The simulator of this run.
            cache: Values saved in the checkpoint; store results here so a resumed run can reuse them.
        """

    def evaluate(self, materials: Sequence[Material], simulator: Simulator) -> list[dict[str, Any]]:
        """Predict the benchmark quantities for some materials (implemented by each benchmark).

        Args:
            materials: Materials to evaluate.
            simulator: The simulator that evaluates the MLIP.

        Returns:
            One dict per material, in the same order: quantity name → predicted value, plus a
            ``status`` entry (``"ok"`` or the reason the prediction is missing).
        """
        raise NotImplementedError

    def run(
        self,
        model: Any,
        model_name: str,
        *,
        checkpoint_file: str | Path | None = None,
        chunk_size: int | None = None,
    ) -> pd.DataFrame:
        """Run the benchmark for one model.

        Args:
            model: An ASE calculator, a TorchSim model, or a simulator.
            model_name: Label of the model; predicted columns are named ``<quantity>_<model_name>``.
            checkpoint_file: JSON file (``.json`` or ``.json.gz``) with the rows finished so far.
                It is written after every chunk; if it exists, finished materials are skipped.
            chunk_size: Materials per chunk (default: ``default_chunk_size``, or ``batched_chunk_size``
                for batched simulators).

        Returns:
            One row per material: id, formula, ``<quantity>_DFT`` references and
            ``<quantity>_<model_name>`` predictions (NaN where ``status_<model_name>`` is not "ok").
        """
        simulator = as_simulator(model)
        dataset = self.dataset.name if isinstance(self.dataset, Path) else str(self.dataset)
        checkpoint = _Checkpoint(checkpoint_file, benchmark=self.name, dataset=dataset, model=model_name)
        finished = {row[self.id_column] for row in checkpoint.rows}
        todo = [material for material in self.materials if material.material_id not in finished]
        if todo:
            self.prepare(simulator, checkpoint.cache)
        size = chunk_size or (
            self.batched_chunk_size if getattr(simulator, "batched", False) else self.default_chunk_size
        )
        for start in range(0, len(todo), size):
            chunk = todo[start : start + size]
            predictions = self.evaluate(chunk, simulator)
            checkpoint.rows.extend(
                self._row(material, prediction, model_name)
                for material, prediction in zip(chunk, predictions, strict=True)
            )
            checkpoint.save()
            logger.info("%s: %d/%d materials done", self.name, len(finished) + start + len(chunk), len(self.materials))
        return self._table(checkpoint.rows)

    def summarize(self, table: pd.DataFrame, model_name: str) -> dict[str, Any]:
        """Summary statistics of one model's predictions.

        For ``"error"`` quantities: mean absolute error (MAE) and standard deviation of the absolute
        errors (STDAE, population) against DFT. For ``"value"`` quantities: mean and standard deviation.
        Materials without a prediction are left out.

        Args:
            table: Result of ``run``.
            model_name: The label used in ``run``.

        Returns:
            Counts, one entry per summarized quantity, and the wall time per stage (s).
        """
        status = table[f"status_{model_name}"].astype(str)
        summary: dict[str, Any] = {"n_materials": len(table), "n_ok": int(status.str.startswith(OK).sum())}
        for quantity, kind in self.summary_metrics.items():
            predicted = pd.to_numeric(table[f"{quantity}_{model_name}"], errors="coerce")
            if kind == "error":
                values = (predicted - pd.to_numeric(table[f"{quantity}_DFT"], errors="coerce")).abs().dropna()
                summary[quantity] = {"MAE": float(values.mean()), "STDAE": float(values.std(ddof=0)), "n": len(values)}
            else:
                values = predicted.dropna()
                summary[quantity] = {"mean": float(values.mean()), "std": float(values.std(ddof=0)), "n": len(values)}
        summary["timings_s"] = dict(self.timings)
        return summary

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a stage of ``evaluate``; the wall time accumulates in ``timings[name]``.

        Args:
            name: Name of the stage.

        Yields:
            Nothing; the block inside ``with`` is timed.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = self.timings.get(name, 0.0) + time.perf_counter() - start

    def _row(self, material: Material, prediction: dict[str, Any], model_name: str) -> dict[str, Any]:
        row: dict[str, Any] = {self.id_column: material.material_id, "formula": material.formula}
        row |= {f"{quantity}_DFT": material.reference[quantity] for quantity in self.reference_columns}
        row |= {f"{quantity}_{model_name}": value for quantity, value in prediction.items()}
        return row

    def _table(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        by_id = {row[self.id_column]: row for row in rows}
        return pd.DataFrame([by_id[m.material_id] for m in self.materials if m.material_id in by_id])


def parallel_map[T, R](function: Callable[[T], R], items: Sequence[T], *, workers: int) -> list[R]:
    """``[function(item) for item in items]``, computed in ``workers`` processes when that is more than one.

    Used for CPU-heavy post-processing while the GPU has nothing to do. The processes are started with
    "spawn" (fork is not safe after CUDA and OpenMP have started threads), so ``function`` must be
    importable, i.e. defined at module level.

    Args:
        function: Function applied to every item.
        items: The items.
        workers: Number of processes; 1 computes everything in this process.

    Returns:
        The results, in the order of ``items``.
    """
    if workers <= 1 or len(items) <= 1:
        return [function(item) for item in items]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, len(items)), mp_context=context) as pool:
        return list(pool.map(function, items))


def failed(reason: str, quantities: Sequence[str]) -> dict[str, Any]:
    """Prediction of a material that could not be computed.

    Args:
        reason: Why (stored in the ``status`` column).
        quantities: Quantities to set to NaN.

    Returns:
        ``{quantity: NaN, ..., "status": reason}``.
    """
    return {quantity: float("nan") for quantity in quantities} | {"status": reason}


class _Checkpoint:
    """Finished rows of a run, kept in a JSON file so that an interrupted run can resume."""

    def __init__(self, path: str | Path | None, **meta: str) -> None:
        self.path = Path(path) if path is not None else None
        self.meta = meta
        self.rows: list[dict[str, Any]] = []
        self.cache: dict[str, Any] = {}
        if self.path is not None and self.path.exists():
            saved = loadfn(self.path)
            if saved["meta"] != meta:
                raise ValueError(f"{self.path} belongs to another run ({saved['meta']}), not to {meta}.")
            self.rows, self.cache = saved["rows"], saved["cache"]
            logger.info("Resuming from %s: %d materials already done", self.path, len(self.rows))

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Write next to the target, then rename, so a crash never leaves a half-written file. The
        # temporary name keeps the extension because monty picks the format from it.
        partial = self.path.with_name(f"~{self.path.name}")
        dumpfn({"meta": self.meta, "rows": self.rows, "cache": self.cache}, partial)
        partial.replace(self.path)
