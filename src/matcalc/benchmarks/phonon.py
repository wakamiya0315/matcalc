"""Phonon benchmark: heat capacity C_V at 300 K of binary compounds vs DFT (Alexandria, PBE).

Recipe (settings as in upstream matcalc, except the supercell size):

1. Relax atoms and cell (FIRE, fmax = 0.05 eV/Å, at most 5000 steps). As upstream, the relaxed
   structure is used even if the relaxation did not converge (``status`` says so).
2. Build a phonopy supercell at least 15 Å long along each lattice vector (upstream: 20 Å) and
   displace each symmetry-distinct atom by 0.015 Å. For dynamically stable compounds C_V is converged
   at 15 Å; see docs/validation.md.
3. Forces on every displaced supercell (single points).
4. Force constants → phonon frequencies on a q-point mesh → C_V(T) in the harmonic approximation.
   C_V at 300 K is compared, in J/(K·mol) per mole of primitive cells.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar

from matcalc.properties.phonon import ThermalProperties, displaced_supercells, make_phonopy, thermal_properties

from ._common import OK, Benchmark, Material, failed, parallel_map

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy as np
    from ase import Atoms
    from pymatgen.core import Structure

    from matcalc.simulation import Simulator

QUANTITIES = ("CV", "min_frequency")


class PhononBenchmark(Benchmark):
    """Heat capacity ``CV`` (J/(K·mol)) at ``temperature`` of binary compounds.

    The table also has ``min_frequency`` (THz): the lowest phonon frequency on the mesh, negative
    when the relaxed structure has imaginary modes (dynamically unstable with this potential).

    Attributes:
        fmax: Force threshold of the relaxation (eV/Å).
        max_steps: Maximum number of FIRE steps.
        displacement: Finite displacement of phonopy (Å).
        min_supercell_length: Minimum supercell length along each lattice vector (Å).
        symprec: Symmetry tolerance of phonopy/spglib (Å).
        temperature: Temperature at which C_V is compared (K).
    """

    name = "phonon"
    id_column = "mp_id"
    default_dataset = "alexandria-binary-pbe-phonon-2025.1.json.gz"
    reference_columns = ("CV",)
    summary_metrics: ClassVar[dict[str, str]] = {"CV": "error"}
    default_chunk_size = 20
    batched_chunk_size = 100  # bounded: displaced supercells and force constants of a chunk stay in memory

    def __init__(
        self,
        dataset: str | Path | None = None,
        *,
        n_samples: int | None = None,
        seed: int = 42,
        fmax: float = 0.05,
        max_steps: int = 5000,
        displacement: float = 0.015,
        min_supercell_length: float = 15.0,
        symprec: float = 1e-5,
        temperature: float = 300.0,
        workers: int = 1,
    ) -> None:
        """
        Args:
            dataset: Dataset file name on Hugging Face, or a local ``Path``.
            n_samples: Draw this many compounds at random (``None`` = all).
            seed: Seed of the random draw.
            fmax: Force threshold of the relaxation (eV/Å).
            max_steps: Maximum number of FIRE steps.
            displacement: Finite displacement of phonopy (Å).
            min_supercell_length: Minimum supercell length along each lattice vector (Å); 20 reproduces
                upstream matcalc.
            symprec: Symmetry tolerance of phonopy/spglib (Å).
            temperature: Temperature at which C_V is compared (K); must be a multiple of 10 K.
            workers: Processes for the phonopy step (see ``Benchmark``).
        """
        super().__init__(dataset, n_samples=n_samples, seed=seed, workers=workers)
        self.fmax = fmax
        self.max_steps = max_steps
        self.displacement = displacement
        self.min_supercell_length = min_supercell_length
        self.symprec = symprec
        self.temperature = temperature

    def read_entries(self, raw: Any) -> list[Material]:
        """Read the Alexandria entries.

        Args:
            raw: List of entries with ``mp_id``, ``formula``, ``structure`` (primitive cell) and
                ``heat_capacity`` (J/(K·mol) at 300 K).

        Returns:
            One ``Material`` per compound.
        """
        return [
            Material(entry["mp_id"], entry["formula"], entry["structure"], {"CV": entry["heat_capacity"]})
            for entry in raw
        ]

    def evaluate(self, materials: Sequence[Material], simulator: Simulator) -> list[dict[str, Any]]:
        """Steps 1-4 for some compounds.

        Args:
            materials: Compounds to evaluate.
            simulator: The simulator of this run.

        Returns:
            Per compound: ``CV`` (J/(K·mol)), ``min_frequency`` (THz), ``status`` and ``relax_steps``.
        """
        with self.stage("relax"):
            relaxed = simulator.relax([m.structure for m in materials], fmax=self.fmax, max_steps=self.max_steps)

        # Setting up phonopy (symmetry of the supercell) needs only the CPU; it runs in parallel processes.
        with self.stage("displacements"):
            settings = [
                None
                if result.structure is None
                else PhononJob(result.structure, [], self.min_supercell_length, self.symprec, self.displacement)
                for result in relaxed
            ]
            generated = iter(
                parallel_map(
                    displaced_supercells_of, [job for job in settings if job is not None], workers=self.workers
                )
            )
            supercells: list[list[Atoms]] = [[] if job is None else next(generated) for job in settings]

        # The forces of all displaced supercells of all compounds are computed in one call.
        with self.stage("single points"):
            forces = iter(
                simulator.single_point([cell for cells in supercells for cell in cells], compute_stress=False)
            )

        predictions: list[dict[str, Any]] = []
        jobs: list[tuple[int, PhononJob]] = []
        for result, setting, cells in zip(relaxed, settings, supercells, strict=True):
            own = [next(forces) for _ in cells]
            errors = [r.error for r in own if r.error is not None]
            if setting is None:
                predictions.append(failed(result.error or "relaxation failed", QUANTITIES))
            elif errors:
                predictions.append(failed(f"single point failed: {errors[0]}", QUANTITIES))
            else:
                predictions.append({})
                jobs.append((len(predictions) - 1, replace(setting, forces=[r.forces for r in own])))

        # Force constants and thermal properties need only the CPU; they run in parallel processes.
        with self.stage("phonopy"):
            thermals = parallel_map(thermal_properties_of, [job for _, job in jobs], workers=self.workers)
        for (i, _), thermal in zip(jobs, thermals, strict=True):
            predictions[i] = {
                "CV": thermal.heat_capacity_at(self.temperature),
                "min_frequency": thermal.min_frequency,
                "status": OK if relaxed[i].converged else f"{OK} (relaxation not converged)",
            }
        return [
            prediction | {"relax_steps": result.n_steps}
            for prediction, result in zip(predictions, relaxed, strict=True)
        ]


@dataclass
class PhononJob:
    """Everything needed to rebuild phonopy for one compound in another process.

    Attributes:
        structure: Relaxed primitive cell.
        forces: Forces on every displaced supercell (eV/Å), in phonopy's order (empty before the single
            points).
        min_supercell_length: Minimum supercell length (Å).
        symprec: Symmetry tolerance (Å).
        displacement: Finite displacement (Å).
    """

    structure: Structure
    forces: list[np.ndarray]
    min_supercell_length: float
    symprec: float
    displacement: float


def displaced_supercells_of(job: PhononJob) -> list[Atoms]:
    """Set up phonopy for one compound and generate its displaced supercells.

    Args:
        job: Relaxed cell and phonopy settings of one compound (``forces`` is not used).

    Returns:
        The displaced supercells, in phonopy's order.
    """
    phonon = make_phonopy(job.structure, min_supercell_length=job.min_supercell_length, symprec=job.symprec)
    return displaced_supercells(phonon, displacement=job.displacement)


def thermal_properties_of(job: PhononJob) -> ThermalProperties:
    """Rebuild phonopy (the displacements are deterministic) and compute the thermal properties.

    Args:
        job: Relaxed cell, forces and phonopy settings of one compound.

    Returns:
        Its harmonic thermal properties.
    """
    phonon = make_phonopy(job.structure, min_supercell_length=job.min_supercell_length, symprec=job.symprec)
    phonon.generate_displacements(distance=job.displacement)
    return thermal_properties(phonon, job.forces)
