"""Kappa benchmark: lattice thermal conductivity of 103 PhononDB crystals (Matbench Discovery κ_SRME).

The κ_SRME task of Matbench Discovery (B. Póta, P. Ahlawat, G. Csányi, M. Simoncelli, arXiv:2408.00755;
J. Riebesell et al., Nat. Mach. Intell. 7, 836 (2025)): rock-salt, zinc-blende and wurtzite crystals
whose PBE phono3py calculations (A. Togo, PhononDB; Phys. Rev. B 91, 094306 (2015)) are the reference.
The packaged dataset (``data/phonondb-pbe-kappa.json.gz``, built by ``scripts/build_kappa_dataset.py``;
CC BY 4.0) holds the task's crystals and the PBE conductivities recomputed from PhononDB's force sets with
the phono3py of this package, so reference and prediction use the same solver (Matbench Discovery pins
phono3py 3.30, whose Wigner solver phono3py 4 replaced).

Recipe (Matbench Discovery's protocol "phonondb-v1"):

1. Relax the PBE unit cell keeping its space group (FIRE on a Frechet cell filter with ASE's
   ``FixSymmetry`` at 0.01 Å, fmax = 1e-4 eV/Å, at most 300 steps). For these cubic and hexagonal cells
   the symmetrized cell step has no shear, as with the reference's no-tilt cell filter.
2. Harmonic force constants from the displaced supercells of the reference (0.01 Å), and the phonon
   frequencies on the q-point mesh. A crystal with imaginary modes, or whose space group changed in the
   relaxation, gets no conductivity (its error counts as the maximum, 2).
3. Third-order force constants and the lattice thermal conductivity at 300 K from the Wigner transport
   equation in the relaxation-time approximation (phono3py, isotope scattering).
4. Errors against DFT: symmetric relative difference of κ (SRD, and its absolute value SRE) and the
   mode-resolved symmetric relative mean error (SRME); their means over the crystals are κ_SRD, κ_SRE and
   κ_SRME.

Needs phono3py (extra ``kappa``). phono3py parallelizes its C code with OpenMP: set ``OMP_NUM_THREADS``
so that ``workers`` x threads fits the CPU cores.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from matcalc.properties.thermal_conductivity import (
    KAPPA_ERROR_MAX,
    fc2_supercells,
    fc3_supercells,
    harmonic_frequencies,
    has_imaginary_modes,
    make_phono3py,
    symmetric_relative_difference,
    symmetric_relative_mean_error,
    thermal_conductivity,
)
from matcalc.structures import to_ase_atoms

from ._common import OK, Benchmark, Material, failed, pool_map, split_into_parts

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms

    from matcalc.simulation import RelaxResult, Simulator

DATASET = Path(__file__).parent / "data" / "phonondb-pbe-kappa.json.gz"
"""The 103 PhononDB crystals, their supercells and meshes, and their PBE conductivity at 300 K."""

QUANTITIES = ("kappa", "srd", "sre", "srme")


@dataclass
class HarmonicJob:
    """What a worker process needs for the harmonic step of one crystal."""

    unit_cell: Atoms
    fc2_supercell: list[list[int]]
    fc3_supercell: list[list[int]]
    mesh: list[int]
    displacement: float
    symprec: float
    fc2_forces: list[np.ndarray]


@dataclass
class ConductivityJob(HarmonicJob):
    """What a worker process needs for the conductivity of one crystal."""

    fc3_forces: list[np.ndarray | None]
    temperature: float
    reference_mode_kappa: np.ndarray
    reference_kappa: float


class KappaBenchmark(Benchmark):
    """Lattice thermal conductivity ``kappa`` (W/(m K)) at 300 K and its errors against DFT.

    Attributes:
        fmax: Force threshold of the relaxations (eV/Å).
        max_steps: Maximum number of FIRE steps per relaxation.
        relax_symprec: Tolerance of the symmetry constraint of the relaxation (Å).
        symprec: Tolerance of phono3py and of the space-group check (Å).
        displacement: Displacement amplitude (Å).
        temperature: Temperature of the conductivity (K).
    """

    name = "kappa"
    id_column = "mp_id"
    default_dataset = DATASET
    reference_columns = ("kappa",)
    batched_chunk_size = 1_000

    def __init__(
        self,
        dataset: str | Path | None = None,
        *,
        n_samples: int | None = None,
        seed: int = 42,
        fmax: float = 1e-4,
        max_steps: int = 300,
        relax_symprec: float = 0.01,
        symprec: float = 1e-5,
        displacement: float = 0.01,
        temperature: float = 300.0,
        workers: int = 1,
    ) -> None:
        """
        Args:
            dataset: Dataset file (default: the packaged ``DATASET``).
            n_samples: Draw this many crystals at random (``None`` = all 103).
            seed: Seed of the random draw.
            fmax: Force threshold of the relaxations (eV/Å).
            max_steps: Maximum number of FIRE steps per relaxation.
            relax_symprec: Tolerance of the symmetry constraint of the relaxation (Å).
            symprec: Tolerance of phono3py and of the space-group check (Å).
            displacement: Displacement amplitude (Å).
            temperature: Temperature of the conductivity (K).
            workers: Processes for phono3py (see ``Benchmark``).
        """
        self.fmax = fmax
        self.max_steps = max_steps
        self.relax_symprec = relax_symprec
        self.symprec = symprec
        self.displacement = displacement
        self.temperature = temperature
        super().__init__(dataset, n_samples=n_samples, seed=seed, workers=workers)

    def read_entries(self, raw: Any) -> list[Material]:
        """Read the crystals and their DFT conductivities.

        Args:
            raw: The dataset (see ``scripts/build_kappa_dataset.py``).

        Returns:
            One ``Material`` per crystal.
        """
        from ase import Atoms

        return [
            Material(
                entry["mp_id"],
                entry["formula"],
                Atoms(entry["species"], cell=entry["lattice"], positions=entry["positions"], pbc=True),
                reference={"kappa": entry["kappa"], "mode_kappa": np.asarray(entry["mode_kappa"])},
                settings={
                    "fc2_supercell": entry["fc2_supercell"],
                    "fc3_supercell": entry["fc3_supercell"],
                    "mesh": entry["q_point_mesh"],
                    "space_group": entry["space_group"],
                },
            )
            for entry in raw["entries"]
        ]

    def evaluate(self, materials: Sequence[Material], simulator: Simulator) -> list[dict[str, Any]]:
        """Steps 1-4 for some crystals.

        Args:
            materials: Crystals to evaluate.
            simulator: The simulator of this run.

        Returns:
            Per crystal: ``kappa`` (W/(m K)), ``srd``, ``sre``, ``srme``, ``status`` (with the reason of a
            censored prediction) and ``relax_steps``.
        """
        with self.stage("relax"):
            relaxed = simulator.relax(
                [m.structure for m in materials],
                fmax=self.fmax,
                max_steps=self.max_steps,
                fix_symmetry=True,
                symprec=self.relax_symprec,
            )
        predictions: list[dict[str, Any]] = [{} for _ in materials]
        cells = self._symmetric_cells(materials, relaxed, predictions)
        with self.worker_pool() as pool:
            harmonic = self._harmonic_step(materials, cells, simulator, pool, predictions)
            self._conductivity_step(materials, cells, harmonic, simulator, pool, predictions)
        return [p | {"relax_steps": r.n_steps} for p, r in zip(predictions, relaxed, strict=True)]

    def summarize(self, table: pd.DataFrame, model_name: str) -> dict[str, Any]:
        """κ_SRME, κ_SRE and κ_SRD (means over the crystals; censored ones count as 2) and failure rates.

        Args:
            table: Result of ``run``.
            model_name: The label used in ``run``.

        Returns:
            Counts, ``kappa_SRME``, ``kappa_SRE``, ``kappa_SRD``, ``failure_rate`` (censored or failed),
            ``imaginary_mode_rate`` and the wall time per stage (s).
        """
        status = table[f"status_{model_name}"].astype(str)
        return {
            "n_materials": len(table),
            "n_ok": int((status == OK).sum()),
            "kappa_SRME": float(
                pd.to_numeric(table[f"srme_{model_name}"], errors="coerce").fillna(KAPPA_ERROR_MAX).mean()
            ),
            "kappa_SRE": float(
                pd.to_numeric(table[f"sre_{model_name}"], errors="coerce").fillna(KAPPA_ERROR_MAX).mean()
            ),
            "kappa_SRD": float(
                pd.to_numeric(table[f"srd_{model_name}"], errors="coerce").fillna(-KAPPA_ERROR_MAX).mean()
            ),
            "failure_rate": float((status != OK).mean()),
            "imaginary_mode_rate": float((status == "censored: imaginary modes").mean()),
            "timings_s": dict(self.timings),
        }

    def _phono3py(self, material: Material, cell: Atoms) -> Any:
        return make_phono3py(
            cell,
            material.settings["fc2_supercell"],
            material.settings["fc3_supercell"],
            material.settings["mesh"],
            displacement=self.displacement,
            symprec=self.symprec,
        )

    def _symmetric_cells(
        self, materials: Sequence[Material], relaxed: Sequence[RelaxResult], predictions: list[dict[str, Any]]
    ) -> dict[int, Atoms]:
        """Relaxed unit cells that kept their space group; the others are censored in ``predictions``."""
        cells: dict[int, Atoms] = {}
        for i, (material, result) in enumerate(zip(materials, relaxed, strict=True)):
            if result.structure is None:
                predictions[i] = _censored(result.error or "relaxation failed")
                continue
            cell = to_ase_atoms(result.structure)
            found = _space_group(cell, self.symprec)
            if found == material.settings["space_group"]:
                cells[i] = cell
            else:
                predictions[i] = _censored(f"space group {material.settings['space_group']} -> {found}")
        return cells

    def _harmonic_step(
        self,
        materials: Sequence[Material],
        cells: dict[int, Atoms],
        simulator: Simulator,
        pool: Any,
        predictions: list[dict[str, Any]],
    ) -> dict[int, HarmonicJob]:
        """Step 2: forces on the FC2 supercells, then the frequencies in the worker processes.

        Returns:
            The harmonic jobs of the crystals without imaginary modes; the others are censored.
        """
        supercells = {i: fc2_supercells(self._phono3py(materials[i], cell)) for i, cell in cells.items()}
        with self.stage("single points"):
            forces = iter(simulator.single_point([c for i in cells for c in supercells[i]], compute_stress=False))
        jobs: dict[int, HarmonicJob] = {}
        for i, cell in cells.items():
            own = [next(forces) for _ in supercells[i]]
            if any(r.error is not None or r.forces is None for r in own):
                predictions[i] = _censored("single point of an FC2 supercell failed")
                continue
            jobs[i] = HarmonicJob(
                unit_cell=cell,
                fc2_supercell=materials[i].settings["fc2_supercell"],
                fc3_supercell=materials[i].settings["fc3_supercell"],
                mesh=materials[i].settings["mesh"],
                displacement=self.displacement,
                symprec=self.symprec,
                fc2_forces=[r.forces for r in own if r.forces is not None],
            )
        with self.stage("phonons"):
            outcomes = list(pool_map(pool, _imaginary_modes_or_error, list(jobs.values())))
        conductive = {}
        for (i, job), outcome in zip(jobs.items(), outcomes, strict=True):
            if isinstance(outcome, str):
                predictions[i] = _censored(f"phono3py failed: {outcome}")
            elif outcome:
                predictions[i] = _censored("imaginary modes")
            else:
                conductive[i] = job
        return conductive

    def _conductivity_step(
        self,
        materials: Sequence[Material],
        cells: dict[int, Atoms],
        harmonic: dict[int, HarmonicJob],
        simulator: Simulator,
        pool: Any,
        predictions: list[dict[str, Any]],
    ) -> None:
        """Steps 3-4, part by part.

        The conductivities of one part are computed in the worker processes while the GPU evaluates the
        FC3 supercells of the next part.
        """
        phono3py = {i: self._phono3py(materials[i], cells[i]) for i in harmonic}
        supercells = {i: fc3_supercells(p) for i, p in phono3py.items()}
        # The costliest conductivities first (bands squared times q-points), so that no worker is left with
        # a long one at the end.
        order = sorted(
            harmonic, key=lambda i: -(len(phono3py[i].primitive) ** 2) * int(np.prod(materials[i].settings["mesh"]))
        )
        sizes = [sum(len(c) for c in supercells[i] if c is not None) for i in order]
        pending = []
        for part in split_into_parts(order, sizes):
            with self.stage("single points"):
                todo = [c for i in part for c in supercells[i] if c is not None]
                forces = iter(simulator.single_point(todo, compute_stress=False))
            jobs = []
            for i in part:
                own = [None if c is None else next(forces) for c in supercells[i]]
                if any(r is not None and (r.error is not None or r.forces is None) for r in own):
                    predictions[i] = _censored("single point of an FC3 supercell failed")
                    continue
                job = ConductivityJob(
                    **vars(harmonic[i]),
                    fc3_forces=[None if r is None else r.forces for r in own],
                    temperature=self.temperature,
                    reference_mode_kappa=materials[i].reference["mode_kappa"],
                    reference_kappa=materials[i].reference["kappa"],
                )
                jobs.append((i, job))
            with self.stage("conductivity"):  # only the time the GPU waits for the CPU
                pending.append(([i for i, _ in jobs], pool_map(pool, _conductivity_or_error, [j for _, j in jobs])))
        with self.stage("conductivity"):
            for indices, results in pending:
                for i, outcome in zip(indices, results, strict=True):
                    if isinstance(outcome, str):
                        predictions[i] = _censored(f"phono3py failed: {outcome}")
                    else:
                        predictions[i] = outcome | {"status": OK}


def _censored(reason: str) -> dict[str, Any]:
    """No conductivity: the error counts as the maximum (SRME = SRE = 2, SRD = -2)."""
    return failed(f"censored: {reason}", QUANTITIES) | {
        "srme": KAPPA_ERROR_MAX,
        "sre": KAPPA_ERROR_MAX,
        "srd": -KAPPA_ERROR_MAX,
    }


def _space_group(cell: Atoms, symprec: float) -> int:
    import moyopy
    from moyopy.interface import MoyoAdapter

    return int(moyopy.MoyoDataset(MoyoAdapter.from_py_obj(cell), symprec=symprec).number)


def _setup(job: HarmonicJob) -> Any:
    phono3py = make_phono3py(
        job.unit_cell,
        job.fc2_supercell,
        job.fc3_supercell,
        job.mesh,
        displacement=job.displacement,
        symprec=job.symprec,
    )
    frequencies = harmonic_frequencies(phono3py, job.fc2_forces)
    return phono3py, frequencies


def _imaginary_modes_or_error(job: HarmonicJob) -> bool | str:
    """Whether the harmonic phonons have imaginary modes (in a worker process); the error message if it fails."""
    try:
        return has_imaginary_modes(_setup(job)[1])
    except Exception as exc:  # noqa: BLE001 - one crystal must not stop the benchmark
        return f"{type(exc).__name__}: {exc}"


def _conductivity_or_error(job: ConductivityJob) -> dict[str, float] | str:
    """κ and its errors against DFT for one crystal (in a worker process); the error message if it fails."""
    try:
        phono3py, _ = _setup(job)
        conductivity = thermal_conductivity(phono3py, job.fc3_forces, temperature=job.temperature)
    except Exception as exc:  # noqa: BLE001 - one crystal must not stop the benchmark
        return f"{type(exc).__name__}: {exc}"
    kappa = conductivity.kappa_average
    srd = symmetric_relative_difference(kappa, job.reference_kappa)
    srme = symmetric_relative_mean_error(conductivity, job.reference_mode_kappa, job.reference_kappa)
    return {"kappa": kappa, "srd": srd, "sre": abs(srd), "srme": srme}
