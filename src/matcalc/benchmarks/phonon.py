"""Phonon benchmark: heat capacity C_V at 300 K of binary compounds vs DFT (Alexandria, PBE).

The benchmark repeats, with the MLIP, the phonon calculation of its DFT reference (A. Loew et al., npj
Comput. Mater. 2025, doi:10.1038/s41524-025-01650-1; data from Alexandria, CC BY 4.0):

1. Relax atoms and cell from the PBE unit cell, keeping its space group (FIRE on a Frechet cell filter
   with a symmetry constraint), until every force is below 0.005 eV/Å. A compound whose relaxation does
   not converge within ``max_steps`` gets no prediction (``status`` says so).
2. Build the supercells of the DFT calculation: the same supercell and primitive matrices, and the same
   displaced atoms and displacements (0.01 Å).
3. Forces on every displaced supercell (single points).
4. Force constants → phonon frequencies on a 20 x 20 x 20 q-point mesh → C_V at 300 K in the harmonic
   approximation, in J/(K·mol) per mole of primitive cells. The compound counts as dynamically stable
   when no frequency at the q-points commensurate with the supercell is below -50 K (-1.04 THz), the
   criterion of the reference.

174 of the 1,170 compounds are dynamically unstable in DFT itself (``stable_DFT``); ``summarize`` also
reports the errors over the DFT-stable compounds only.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd
from pymatgen.core import Structure

from matcalc.properties.phonon import HarmonicProperties, displaced_supercells, harmonic_properties, make_phonopy

from ._common import OK, Benchmark, Material, failed, parallel_map

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from ase import Atoms

    from matcalc.simulation import Simulator

QUANTITIES = ("CV", "stable", "min_frequency")

DATASET = Path(__file__).parent / "data" / "alexandria-pbe-phonon.json.gz"
"""The 1,170 compounds with the settings of their DFT phonon calculations (built by
``scripts/build_phonon_dataset.py`` from the Alexandria files)."""


class PhononBenchmark(Benchmark):
    """Heat capacity ``CV`` (J/(K·mol)) at 300 K and dynamical stability of binary compounds.

    Columns besides ``CV``: ``stable`` (no imaginary mode below -50 K at the commensurate q-points),
    ``min_frequency`` (the lowest frequency there, THz) and ``relax_steps``.

    Attributes:
        fmax: Force threshold of the relaxation (eV/Å).
        max_steps: Maximum number of FIRE steps; a relaxation that has not converged by then gives NaN.
        symprec: Symmetry tolerance of the symmetry constraint during the relaxation (Å). phonopy uses
            the tolerance of the DFT calculation (1e-5 Å).
        temperature: Temperature of the heat capacity (K).
        mesh: q-point mesh of the heat capacity.
    """

    name = "phonon"
    id_column = "mp_id"
    default_dataset = DATASET
    reference_columns = ("CV", "stable")
    summary_metrics: ClassVar[dict[str, str]] = {"CV": "error"}
    default_chunk_size = 20

    def __init__(
        self,
        dataset: str | Path | None = None,
        *,
        n_samples: int | None = None,
        seed: int = 42,
        fmax: float = 0.005,
        max_steps: int = 5000,
        symprec: float = 0.01,
        temperature: float = 300.0,
        mesh: tuple[int, int, int] = (20, 20, 20),
        workers: int = 1,
    ) -> None:
        """
        Args:
            dataset: Local ``Path`` to a dataset in the format of the default one (default: the
                packaged Alexandria dataset).
            n_samples: Draw this many compounds at random (``None`` = all).
            seed: Seed of the random draw.
            fmax: Force threshold of the relaxation (eV/Å).
            max_steps: Maximum number of FIRE steps.
            symprec: Symmetry tolerance of the symmetry constraint during the relaxation (Å).
            temperature: Temperature of the heat capacity (K).
            mesh: q-point mesh of the heat capacity.
            workers: Processes for the phonopy steps (see ``Benchmark``).
        """
        super().__init__(dataset, n_samples=n_samples, seed=seed, workers=workers)
        self.fmax = fmax
        self.max_steps = max_steps
        self.symprec = symprec
        self.temperature = temperature
        self.mesh = mesh

    def read_entries(self, raw: Any) -> list[Material]:
        """Read the compounds and the settings of their DFT phonon calculations.

        Args:
            raw: ``{"entries": [...]}``; each entry has ``mp_id``, ``formula``, the unit cell
                (``lattice``, ``species``, ``frac_coords``), ``supercell_matrix``, ``primitive_matrix``,
                ``displacements``, ``symprec``, ``heat_capacity`` (J/(K·mol) at 300 K) and ``stable``.

        Returns:
            One ``Material`` per compound; the phonopy settings are in ``Material.settings``.
        """
        return [
            Material(
                entry["mp_id"],
                entry["formula"],
                Structure(entry["lattice"], entry["species"], entry["frac_coords"]),
                reference={"CV": entry["heat_capacity"], "stable": entry["stable"]},
                settings={
                    key: entry[key] for key in ("supercell_matrix", "primitive_matrix", "displacements", "symprec")
                },
            )
            for entry in raw["entries"]
        ]

    def evaluate(self, materials: Sequence[Material], simulator: Simulator) -> list[dict[str, Any]]:
        """Steps 1-4 for some compounds.

        Args:
            materials: Compounds to evaluate.
            simulator: The simulator of this run.

        Returns:
            Per compound: ``CV`` (J/(K·mol)), ``stable``, ``min_frequency`` (THz), ``status`` and
            ``relax_steps``.
        """
        with self.stage("relax"):
            relaxed = simulator.relax(
                [m.structure for m in materials],
                fmax=self.fmax,
                max_steps=self.max_steps,
                fix_symmetry=True,
                symprec=self.symprec,
            )

        jobs: list[PhononJob | None] = [
            PhononJob(result.structure, [], **material.settings, temperature=self.temperature, mesh=self.mesh)
            if result.optimizer_converged
            else None
            for material, result in zip(materials, relaxed, strict=True)
        ]
        # Setting up phonopy (symmetry of the supercell) needs only the CPU; it runs in parallel processes.
        with self.stage("displacements"):
            generated = iter(
                parallel_map(displaced_supercells_of, [j for j in jobs if j is not None], workers=self.workers)
            )
            supercells: list[list[Atoms]] = [[] if job is None else next(generated) for job in jobs]

        # The forces of all displaced supercells of all compounds are computed in one call.
        with self.stage("single points"):
            forces = iter(
                simulator.single_point([cell for cells in supercells for cell in cells], compute_stress=False)
            )

        predictions: list[dict[str, Any]] = []
        todo: list[tuple[int, PhononJob]] = []
        for result, job, cells in zip(relaxed, jobs, supercells, strict=True):
            own = [next(forces) for _ in cells]
            errors = [r.error for r in own if r.error is not None]
            if job is None:
                predictions.append(failed(result.error or "relaxation not converged", QUANTITIES))
            elif errors:
                predictions.append(failed(f"single point failed: {errors[0]}", QUANTITIES))
            else:
                predictions.append({})
                todo.append((len(predictions) - 1, replace(job, forces=[r.forces for r in own])))

        # Force constants and frequencies need only the CPU; they run in parallel processes.
        with self.stage("phonopy"):
            results = parallel_map(harmonic_properties_of, [job for _, job in todo], workers=self.workers)
        for (i, _), harmonic in zip(todo, results, strict=True):
            predictions[i] = {
                "CV": harmonic.heat_capacity,
                "stable": harmonic.dynamically_stable,
                "min_frequency": harmonic.min_frequency,
                "status": OK,
            }
        return [
            prediction | {"relax_steps": result.n_steps}
            for prediction, result in zip(predictions, relaxed, strict=True)
        ]

    def summarize(self, table: pd.DataFrame, model_name: str) -> dict[str, Any]:
        """Errors over all compounds and over the DFT-stable ones, and the agreement on stability.

        Args:
            table: Result of ``run``.
            model_name: The label used in ``run``.

        Returns:
            ``Benchmark.summarize`` plus ``"CV (DFT-stable)"`` (MAE and STDAE over the compounds that are
            dynamically stable in DFT) and ``"stability"``: counts of compounds stable in both (TS),
            unstable in both (TU), and stable in only DFT (FU) or only the MLIP (FS).
        """
        summary = super().summarize(table, model_name)
        dft_stable = table[table["stable_DFT"].astype(bool)]
        errors = (
            (pd.to_numeric(dft_stable[f"CV_{model_name}"], errors="coerce") - pd.to_numeric(dft_stable["CV_DFT"]))
            .abs()
            .dropna()
        )
        summary["CV (DFT-stable)"] = {"MAE": float(errors.mean()), "STDAE": float(errors.std(ddof=0)), "n": len(errors)}
        predicted = table[f"stable_{model_name}"]
        has = predicted.notna()
        dft, mlip = table.loc[has, "stable_DFT"].astype(bool), predicted[has].astype(bool)
        summary["stability"] = {
            "TS": int((dft & mlip).sum()),
            "TU": int((~dft & ~mlip).sum()),
            "FU": int((dft & ~mlip).sum()),
            "FS": int((~dft & mlip).sum()),
        }
        return summary


@dataclass
class PhononJob:
    """Everything phonopy needs for one compound, passed to another process.

    Attributes:
        structure: Relaxed unit cell.
        forces: Forces on every displaced supercell (eV/Å), in the order of ``displacements`` (empty
            before the single points).
        supercell_matrix: Supercell matrix of the DFT calculation.
        primitive_matrix: Primitive matrix of the DFT calculation (``None``: the unit cell is primitive).
        displacements: Displaced atoms and displacements of the DFT calculation.
        symprec: Symmetry tolerance of phonopy in the DFT calculation (Å).
        temperature: Temperature of the heat capacity (K).
        mesh: q-point mesh of the heat capacity.
    """

    structure: Structure
    forces: list[np.ndarray]
    supercell_matrix: list[list[int]]
    primitive_matrix: list[list[float]] | None
    displacements: list[list[float]]
    symprec: float
    temperature: float
    mesh: tuple[int, int, int]


def displaced_supercells_of(job: PhononJob) -> list[Atoms]:
    """Set up phonopy for one compound and build its displaced supercells.

    Args:
        job: Relaxed cell and phonopy settings of one compound (``forces`` is not used).

    Returns:
        The displaced supercells, in the order of ``job.displacements``.
    """
    phonon = make_phonopy(job.structure, job.supercell_matrix, job.primitive_matrix, symprec=job.symprec)
    return displaced_supercells(phonon, job.displacements)


def harmonic_properties_of(job: PhononJob) -> HarmonicProperties:
    """Set up phonopy for one compound again and compute its heat capacity and stability.

    Args:
        job: Relaxed cell, phonopy settings and forces of one compound.

    Returns:
        Its harmonic properties.
    """
    phonon = make_phonopy(job.structure, job.supercell_matrix, job.primitive_matrix, symprec=job.symprec)
    displaced_supercells(phonon, job.displacements)
    return harmonic_properties(phonon, job.forces, temperature=job.temperature, mesh=job.mesh)
