"""Diatomics benchmark: potential-energy curves of homonuclear dimers (Matbench Discovery).

The diatomics task of Matbench Discovery (J. Riebesell et al., Nat. Mach. Intell. 7, 836 (2025)): the energy
and forces of the dimers X2 of the 87 elements from H to U that the Materials Project covers (all but Po, At,
Rn, Fr and Ra) at 119 separations from 0.1 to 6 Å, compared with PBE curves and checked for smoothness. The
PBE curves (VASP with Materials Project settings in a 15 Å box, at each separation the lowest of several
constrained spin states; Figshare file 68541277, CC BY 4.0) are downloaded on first use.

Recipe:

1. Single points of every dimer in a 50 Å box; no relaxation.
2. Per element, the metrics of ``matcalc.properties.diatomics``. A curve with a non-finite energy or force
   between 0.8 r_cov and the end of the element's window gets none (as in Matbench Discovery, which also
   drops non-finite points outside that range, e.g. at 0.1 Å).
3. The PBE curves too rough to score against (8 lanthanides; found from the data, see
   ``properties.diatomics.is_rough``) give their elements the smoothness metrics only.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from ase.data import chemical_symbols

from matcalc.datasets import FigshareFile, download_figshare_file, sample_subset
from matcalc.properties.diatomics import (
    DIMER_DISTANCES,
    METRICS,
    WALL_LOWER,
    curve_metrics,
    dimers,
    evaluation_window,
    is_rough,
    vib_freq_eligible,
)

from ._common import OK, Benchmark, Material, failed

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matcalc.simulation import Simulator

PBE_CURVES = FigshareFile(68541277, "diatomics-dft.json.gz", "d090dcf5d208ad94c188e017d0e0b149")
"""Matbench Discovery's DFT curves (PBE and r2SCAN; the benchmark uses PBE)."""

NON_MP_ELEMENTS = frozenset({"Po", "At", "Rn", "Fr", "Ra"})
"""Elements up to U missing from the Materials Project, left out for every model (as in Matbench Discovery)."""


class DiatomicsBenchmark(Benchmark):
    """Smoothness and PBE errors of the potential-energy curves of homonuclear dimers.

    Attributes:
        rough_references: Elements whose PBE curve is too rough to score against.
        vib_freq_eligible: Elements whose PBE curve has a vibrational frequency to compare with.
    """

    name = "diatomics"
    id_column = "element"
    default_dataset = "matbench-discovery-diatomics"
    batched_chunk_size = 1_000

    def __init__(
        self, dataset: str | Path | None = None, *, n_samples: int | None = None, seed: int = 42, workers: int = 1
    ) -> None:
        """
        Args:
            dataset: ``"matbench-discovery-diatomics"`` (downloaded from Figshare on first use), or a
                ``pathlib.Path`` to a file of the same format.
            n_samples: Draw this many elements at random (``None`` = all 87).
            seed: Seed of the random draw.
            workers: Not used (the metrics are cheap); kept for the common interface.
        """
        self.rough_references: set[str] = set()
        self.vib_freq_eligible: set[str] = set()
        super().__init__(dataset, n_samples=n_samples, seed=seed, workers=workers)

    def load_materials(self) -> list[Material]:
        """Read the PBE curves of the elements H-U of the Materials Project.

        Returns:
            One ``Material`` per element (``structure``: the dimer at the PBE minimum), in the order drawn.
        """
        if isinstance(self.dataset, Path):
            path = self.dataset
        elif self.dataset == self.default_dataset:
            path = download_figshare_file(PBE_CURVES, "matbench-discovery")
        else:
            raise ValueError(f"Unknown dataset {self.dataset!r}: use {self.default_dataset!r} or a Path to a file")
        with gzip.open(path, "rt") as f:
            curves = {formula.split("-")[0]: curve for formula, curve in json.load(f)["PBE"].items()}
        self.rough_references = {el for el, c in curves.items() if is_rough(el, c["distances"], c["energies"])}
        excluded = NON_MP_ELEMENTS | self.rough_references
        self.vib_freq_eligible = {
            el
            for el, c in curves.items()
            if el not in excluded and vib_freq_eligible(el, c["distances"], c["energies"])
        }
        materials = []
        for element in chemical_symbols[1:93]:
            if element in NON_MP_ELEMENTS or element not in curves:
                continue
            curve = curves[element]
            closest = float(curve["distances"][int(np.argmin(curve["energies"]))])
            materials.append(
                Material(
                    element,
                    f"{element}2",
                    dimers(element, [closest])[0],
                    reference={key: np.asarray(curve[key], dtype=float) for key in ("distances", "energies", "forces")},
                )
            )
        return sample_subset(materials, self.n_samples, self.seed)

    def evaluate(self, materials: Sequence[Material], simulator: Simulator) -> list[dict[str, Any]]:
        """Single points of the dimers of some elements, then their metrics.

        Args:
            materials: Elements to evaluate.
            simulator: The simulator of this run.

        Returns:
            Per element: the metrics of ``properties.diatomics.METRICS`` (NaN where not defined) and ``status``.
        """
        n = len(DIMER_DISTANCES)
        structures = [dimer for material in materials for dimer in dimers(material.material_id)]
        with self.stage("single points"):
            results = simulator.single_point(structures, compute_stress=False)
        predictions = []
        for k, material in enumerate(materials):
            own = results[k * n : (k + 1) * n]
            energies = np.array([r.energy if r.error is None else np.nan for r in own], dtype=float)
            forces = np.array(
                [r.forces if r.error is None and r.forces is not None else np.full((2, 3), np.nan) for r in own],
                dtype=float,
            )
            predictions.append(self._metrics(material, energies, forces))
        return predictions

    def summarize(self, table: pd.DataFrame, model_name: str) -> dict[str, Any]:
        """The mean of every metric over the elements that have it, as Matbench Discovery reports them.

        Args:
            table: Result of ``run``.
            model_name: The label used in ``run``.

        Returns:
            Counts, the mean of each metric, the coverage of the vibrational frequency (elements with a
            frequency error among those whose PBE curve has one) and the wall time per stage (s).
        """
        status = table[f"status_{model_name}"].astype(str)
        summary: dict[str, Any] = {"n_materials": len(table), "n_ok": int((status == OK).sum())}
        for metric in METRICS:
            values = pd.to_numeric(table[f"{metric}_{model_name}"], errors="coerce")
            finite = values[np.isfinite(values)]
            summary[metric] = float(finite.mean()) if len(finite) else float("nan")
        frequency = pd.to_numeric(table[f"pbe_vib_freq_error_{model_name}"], errors="coerce")
        eligible = set(table[self.id_column]) & self.vib_freq_eligible
        summary["pbe_vib_freq_coverage"] = {
            "n_valid": int(np.isfinite(frequency[table[self.id_column].isin(eligible)]).sum()),
            "n_eligible": len(eligible),
        }
        summary["timings_s"] = dict(self.timings)
        return summary

    def _metrics(self, material: Material, energies: np.ndarray, forces: np.ndarray) -> dict[str, Any]:
        element = material.material_id
        distances = DIMER_DISTANCES
        finite = np.isfinite(energies) & np.isfinite(forces).all(axis=(1, 2))
        wall_min, r_max = evaluation_window(element, float(distances.max()), WALL_LOWER)
        if not finite[(distances >= wall_min - 1e-12) & (distances <= r_max)].all():
            return failed(f"non-finite energy or forces between {wall_min:.2f} and {r_max:.2f} Å", METRICS)
        ref = material.reference
        reference = None if element in self.rough_references else (ref["distances"], ref["energies"], ref["forces"])
        metrics = curve_metrics(element, distances[finite], energies[finite], forces[finite], reference)
        if not metrics:
            return failed("fewer than 5 separations in the scored window", METRICS)
        return dict.fromkeys(METRICS, float("nan")) | metrics | {"status": OK}
