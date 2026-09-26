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

import numpy as np
import pandas as pd
from pymatgen.core import Structure

from matcalc.properties.phonon import HarmonicProperties, displaced_supercells, harmonic_properties, make_phonopy

from ._common import OK, Benchmark, Material, failed, pool_map, split_into_parts, worker_pool

if TYPE_CHECKING:
    from collections.abc import Sequence

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
        symprec: Symmetry tolerance of the symmetry constraint during the relaxation (Å): that of the DFT
            calculation (1e-5 Å), so that the constraint keeps the space group of the DFT structure (29
            structures would get a higher one at ASE's default of 0.01 Å).
        temperature: Temperature of the heat capacity (K).
        mesh: q-point mesh of the heat capacity.
    """

    name = "phonon"
    id_column = "mp_id"
    default_dataset = DATASET
    reference_columns = ("CV", "stable", "min_frequency")
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
        symprec: float = 1e-5,
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
                reference={
                    "CV": entry["heat_capacity"],
                    "stable": entry["stable"],
                    "min_frequency": entry["min_frequency"],
                },
                settings={
                    key: entry[key]
                    for key in ("supercell_matrix", "primitive_matrix", "displacements", "symprec", "space_group")
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

        jobs = {
            i: PhononJob(result.structure, [], **material.settings, temperature=self.temperature, mesh=self.mesh)
            for i, (material, result) in enumerate(zip(materials, relaxed, strict=True))
            if result.optimizer_converged
        }
        predictions: list[dict[str, Any]] = [
            {} if i in jobs else failed(result.error or "relaxation not converged", QUANTITIES)
            for i, result in enumerate(relaxed)
        ]
        todo = list(jobs)
        with worker_pool(self.workers) as pool:
            # Setting up phonopy (symmetry of the supercell) needs only the CPU.
            with self.stage("displacements"):
                generated = dict(
                    zip(todo, pool_map(pool, displaced_supercells_of, [jobs[i] for i in todo]), strict=True)
                )
            supercells: dict[int, list[Atoms]] = {}
            notes: dict[int, str] = {}
            for i, outcome in generated.items():
                if isinstance(outcome, str):
                    predictions[i] = failed(outcome, QUANTITIES)
                else:
                    supercells[i], displacements, notes[i] = outcome
                    jobs[i] = replace(jobs[i], displacements=displacements)
            todo = [i for i in todo if i in supercells]
            # The forces are computed part by part; the phonopy step of a part (force constants and
            # frequencies, CPU only) runs in the worker processes while the GPU computes the next part.
            pending = []
            for part in split_into_parts(todo, [sum(len(c) for c in supercells[i]) for i in todo]):
                with self.stage("single points"):
                    forces = iter(
                        simulator.single_point([c for i in part for c in supercells[i]], compute_stress=False)
                    )
                ready = []
                for i in part:
                    own = [next(forces) for _ in supercells[i]]
                    errors = [r.error for r in own if r.error is not None]
                    if errors:
                        predictions[i] = failed(f"single point failed: {errors[0]}", QUANTITIES)
                    else:
                        ready.append((i, replace(jobs[i], forces=[r.forces for r in own])))
                with self.stage("phonopy"):  # only the time the GPU waits for the CPU
                    pending.append(
                        ([i for i, _ in ready], pool_map(pool, harmonic_properties_of, [j for _, j in ready]))
                    )
            with self.stage("phonopy"):
                for indices, results in pending:
                    for i, harmonic in zip(indices, results, strict=True):
                        if isinstance(harmonic, str):
                            predictions[i] = failed(harmonic, QUANTITIES)
                            continue
                        predictions[i] = {
                            "CV": harmonic.heat_capacity,
                            "stable": harmonic.dynamically_stable,
                            "min_frequency": harmonic.min_frequency,
                            "status": OK + notes[i],
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
        space_group: Space group number of the DFT structure.
        temperature: Temperature of the heat capacity (K).
        mesh: q-point mesh of the heat capacity.
    """

    structure: Structure
    forces: list[np.ndarray]
    supercell_matrix: list[list[int]]
    primitive_matrix: list[list[float]] | None
    displacements: list[list[float]]
    symprec: float
    space_group: int
    temperature: float
    mesh: tuple[int, int, int]


def displaced_supercells_of(job: PhononJob) -> tuple[list[Atoms], list[list[float]], str] | str:
    """Set up phonopy for one compound and build its displaced supercells.

    The displacements of the DFT calculation match the symmetry of the DFT structure. The relaxation keeps
    that space group (symmetry constraint with the DFT's tolerance); should phonopy find another one in the
    relaxed structure (it can only become higher), phonopy generates displacements for it anew, with the
    same amplitude.

    Args:
        job: Relaxed cell and phonopy settings of one compound (``forces`` is not used).

    Returns:
        The displaced supercells, the displacements used, and a note for ``status`` ("" normally); or why
        phonopy failed.
    """
    try:
        phonon = make_phonopy(job.structure, job.supercell_matrix, job.primitive_matrix, symprec=job.symprec)
        displacements, note = job.displacements, ""
        found = phonon.symmetry.dataset.number
        if found != job.space_group:
            amplitude = float(np.linalg.norm(job.displacements[0][1:4]))
            phonon.generate_displacements(distance=amplitude)
            displacements = [[d["number"], *d["displacement"]] for d in phonon.dataset["first_atoms"]]
            note = f" (space group {job.space_group} -> {found}; displacements generated by phonopy)"
        return displaced_supercells(phonon, displacements), displacements, note
    except Exception as exc:  # noqa: BLE001 - one compound must not stop the benchmark
        return f"phonopy failed: {type(exc).__name__}: {exc}"


def harmonic_properties_of(job: PhononJob) -> HarmonicProperties | str:
    """Set up phonopy for one compound again and compute its heat capacity and stability.

    Args:
        job: Relaxed cell, phonopy settings and forces of one compound.

    Returns:
        Its harmonic properties, or why phonopy failed.
    """
    try:
        phonon = make_phonopy(job.structure, job.supercell_matrix, job.primitive_matrix, symprec=job.symprec)
        displaced_supercells(phonon, job.displacements)
        return harmonic_properties(phonon, job.forces, temperature=job.temperature, mesh=job.mesh)
    except Exception as exc:  # noqa: BLE001 - one compound must not stop the benchmark
        return f"phonopy failed: {type(exc).__name__}: {exc}"
