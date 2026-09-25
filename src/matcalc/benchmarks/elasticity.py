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
from typing import TYPE_CHECKING, Any, ClassVar

from matcalc.properties.elasticity import NORMAL_STRAINS, SHEAR_STRAINS, fit_elastic_tensor, strained_structures

from ._common import OK, Benchmark, Material, failed

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from pymatgen.core.elasticity import Strain

    from matcalc.simulation import RelaxResult, Simulator, SinglePointResult

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
        """
        super().__init__(dataset, n_samples=n_samples, seed=seed)
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

        with self.stage("strain"):
            strained = [
                strained_structures(result.structure, self.normal_strains, self.shear_strains)
                if result.converged and result.structure is not None
                else None
                for result in relaxed
            ]

        # The stresses of all compounds are computed in one call: first the strained cells of a
        # compound, then its relaxed cell.
        cells = []
        for result, deformed in zip(relaxed, strained, strict=True):
            if deformed is not None:
                cells.extend([*deformed[0], result.structure])
        with self.stage("single points"):
            stresses = iter(simulator.single_point(cells))

        predictions = []
        with self.stage("fit"):
            for material, result, deformed in zip(materials, relaxed, strained, strict=True):
                if deformed is None:
                    predictions.append(_relaxation_failure(result))
                    continue
                strains = deformed[1]
                own = [next(stresses) for _ in range(len(strains) + 1)]
                predictions.append(_moduli(material, strains, own[:-1], own[-1]) | {"relax_steps": result.n_steps})
        return predictions


def _relaxation_failure(result: RelaxResult) -> dict[str, Any]:
    if result.structure is None:
        reason = result.error or "relaxation failed"
    else:
        reason = f"relaxation not converged (max force {result.max_force:.3g} eV/A after {result.n_steps} steps)"
    return failed(reason, QUANTITIES) | {"relax_steps": result.n_steps}


def _moduli(
    material: Material,
    strains: Sequence[Strain],
    strained: Sequence[SinglePointResult],
    relaxed: SinglePointResult,
) -> dict[str, Any]:
    errors = [r.error for r in (*strained, relaxed) if r.error is not None]
    if errors:
        return failed(f"single point failed: {errors[0]}", QUANTITIES)
    fit = fit_elastic_tensor(strains, [r.stress for r in strained], relaxed.stress)
    if fit.mean_r2 < MIN_MEAN_R2:
        logger.warning(
            "%s: stress-strain fits have mean R^2 = %.4f < %.2f; the elastic tensor may be unreliable",
            material.material_id,
            fit.mean_r2,
            MIN_MEAN_R2,
        )
    return {"K_vrh": fit.bulk_modulus_vrh, "G_vrh": fit.shear_modulus_vrh, "status": OK}
