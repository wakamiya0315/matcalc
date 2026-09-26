"""Elasticity benchmark: bulk and shear moduli of binary compounds vs DFT (Materials Project, PBE).

Recipe (settings as in upstream matcalc):

1. Relax atoms and cell (FIRE, fmax = 0.05 eV/Å, at most 500 steps). Compounds whose relaxation does
   not converge get no prediction.
2. Strain the relaxed cell by ±0.5 % and ±1 % along xx, yy, zz and by ±3 % and ±6 % along yz, xz, xy
   (24 strained cells). Atoms are not relaxed in the strained cells.
3. Stress of every strained cell and of the relaxed cell (single points).
4. Linear stress-strain fits give the elastic tensor C_ij; K_vrh and G_vrh are its
   Voigt-Reuss-Hill averages (GPa).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from matcalc.properties.elasticity import (
    NORMAL_STRAINS,
    SHEAR_STRAINS,
    ElasticFit,
    fit_elastic_tensor,
    strain_states,
    strained_structures,
)

from ._common import OK, Benchmark, Material, failed, pool_map, split_into_parts, worker_pool

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy as np

    from matcalc.simulation import RelaxResult, Simulator

logger = logging.getLogger(__name__)

QUANTITIES = ("K_vrh", "G_vrh")
MIN_MEAN_R2 = 0.95
"""Below this mean R² of the stress-strain fits a warning is logged, as upstream matcalc does."""


class ElasticityBenchmark(Benchmark):
    """Bulk modulus ``K_vrh`` and shear modulus ``G_vrh`` (GPa) of binary compounds.

    Attributes:
        fmax: Force threshold of the relaxation (eV/Å).
        max_steps: Maximum number of FIRE steps.
        normal_strains: Strains along xx, yy, zz.
        shear_strains: Strains along yz, xz, xy.
    """

    name = "elasticity"
    id_column = "mp_id"
    default_dataset = "mp-binary-pbe-elasticity-2025.1.json.gz"
    reference_columns = QUANTITIES
    summary_metrics: ClassVar[dict[str, str]] = {"K_vrh": "error", "G_vrh": "error"}
    default_chunk_size = 200

    def __init__(
        self,
        dataset: str | Path | None = None,
        *,
        n_samples: int | None = None,
        seed: int = 42,
        fmax: float = 0.05,
        max_steps: int = 500,
        normal_strains: Sequence[float] = NORMAL_STRAINS,
        shear_strains: Sequence[float] = SHEAR_STRAINS,
        workers: int = 1,
    ) -> None:
        """
        Args:
            dataset: Dataset file name on Hugging Face, or a local ``Path``.
            n_samples: Draw this many compounds at random (``None`` = all).
            seed: Seed of the random draw.
            fmax: Force threshold of the relaxation (eV/Å).
            max_steps: Maximum number of FIRE steps.
            normal_strains: Strains along xx, yy, zz.
            shear_strains: Strains along yz, xz, xy.
            workers: Processes for the stress-strain fits, which run next to the GPU (see ``Benchmark``).
        """
        super().__init__(dataset, n_samples=n_samples, seed=seed, workers=workers)
        self.fmax = fmax
        self.max_steps = max_steps
        self.normal_strains = tuple(normal_strains)
        self.shear_strains = tuple(shear_strains)

    def read_entries(self, raw: Any) -> list[Material]:
        """Read the Materials Project entries.

        Args:
            raw: List of entries with ``mp_id``, ``formula``, ``structure``, ``bulk_modulus_vrh`` and
                ``shear_modulus_vrh`` (GPa).

        Returns:
            One ``Material`` per compound.
        """
        return [
            Material(
                entry["mp_id"],
                entry["formula"],
                entry["structure"],
                {"K_vrh": entry["bulk_modulus_vrh"], "G_vrh": entry["shear_modulus_vrh"]},
            )
            for entry in raw
        ]

    def evaluate(self, materials: Sequence[Material], simulator: Simulator) -> list[dict[str, Any]]:
        """Steps 1-4 for some compounds.

        Args:
            materials: Compounds to evaluate.
            simulator: The simulator of this run.

        Returns:
            Per compound: ``K_vrh`` and ``G_vrh`` (GPa), ``status`` and ``relax_steps``.
        """
        with self.stage("relax"):
            relaxed = simulator.relax([m.structure for m in materials], fmax=self.fmax, max_steps=self.max_steps)

        predictions: list[dict[str, Any]] = [{} for _ in materials]
        with self.stage("strain"):
            # Per compound: the strained cells, then the relaxed cell (whose stress is the zero-strain point).
            cells = {}
            for i, result in enumerate(relaxed):
                if result.converged and result.structure is not None:
                    strained, _ = strained_structures(result.structure, self.normal_strains, self.shear_strains)
                    cells[i] = [*strained, result.structure]
                else:
                    predictions[i] = _relaxation_failure(result)
        todo = sorted(cells)
        # The stresses are computed part by part; the fits of a part (CPU only) run in the worker
        # processes while the GPU computes the next part.
        with worker_pool(self.workers) as pool:
            pending = []
            for part in split_into_parts(todo, [sum(len(c) for c in cells[i]) for i in todo]):
                with self.stage("single points"):
                    stresses = iter(simulator.single_point([c for i in part for c in cells[i]]))
                jobs = []
                for i in part:
                    own = [next(stresses) for _ in cells[i]]
                    errors = [r.error for r in own if r.error is not None]
                    if errors:
                        predictions[i] = failed(f"single point failed: {errors[0]}", QUANTITIES)
                    else:
                        stress = [r.stress for r in own]
                        jobs.append((i, FitJob(self.normal_strains, self.shear_strains, stress[:-1], stress[-1])))
                with self.stage("fit"):  # only the time the GPU waits for the CPU
                    pending.append(([i for i, _ in jobs], pool_map(pool, moduli_of, [job for _, job in jobs])))
            with self.stage("fit"):
                for indices, fits in pending:
                    for i, fit in zip(indices, fits, strict=True):
                        if fit.mean_r2 < MIN_MEAN_R2:
                            logger.warning(
                                "%s: stress-strain fits have mean R^2 = %.4f < %.2f; the elastic tensor may be "
                                "unreliable",
                                materials[i].material_id,
                                fit.mean_r2,
                                MIN_MEAN_R2,
                            )
                        predictions[i] = {"K_vrh": fit.bulk_modulus_vrh, "G_vrh": fit.shear_modulus_vrh, "status": OK}
        return [
            prediction | {"relax_steps": result.n_steps}
            for prediction, result in zip(predictions, relaxed, strict=True)
        ]


@dataclass
class FitJob:
    """Stress-strain data of one compound, passed to a worker process.

    Attributes:
        normal_strains: Strains along xx, yy, zz.
        shear_strains: Strains along yz, xz, xy.
        stresses: Stress of every strained cell (3x3, eV/Å^3), in the order of ``strain_states``.
        equilibrium_stress: Stress of the relaxed cell (eV/Å^3).
    """

    normal_strains: tuple[float, ...]
    shear_strains: tuple[float, ...]
    stresses: list[np.ndarray]
    equilibrium_stress: np.ndarray


def moduli_of(job: FitJob) -> ElasticFit:
    """Fit the elastic tensor of one compound.

    Args:
        job: Its stress-strain data.

    Returns:
        The fit (moduli in GPa).
    """
    _, strains = strain_states(job.normal_strains, job.shear_strains)
    return fit_elastic_tensor(strains, job.stresses, job.equilibrium_stress)


def _relaxation_failure(result: RelaxResult) -> dict[str, Any]:
    if result.structure is None:
        reason = result.error or "relaxation failed"
    else:
        reason = f"relaxation not converged (max force {result.max_force:.3g} eV/A after {result.n_steps} steps)"
    return failed(reason, QUANTITIES)
