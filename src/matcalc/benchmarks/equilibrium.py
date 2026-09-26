"""Equilibrium benchmark: relaxed structures and formation energies of WBM compounds vs DFT (PBE).

Recipe (settings as in upstream matcalc):

1. Relax the candidate ground-state structures of every element in the dataset (Materials Project
   PBE references); the lowest energy per atom of each element is its chemical potential μ_i.
2. Move every atom of each DFT-relaxed compound in a random direction by a random distance of at most
   0.1 Å, then relax atoms and cell (FIRE, fmax = 0.05 eV/Å, at most 500 steps). Compounds whose
   relaxation does not converge get no prediction.
3. Formation energy E_form = (E - Σ_i n_i μ_i) / N (eV/atom).
4. Distance d between the local-environment fingerprints of the MLIP- and the DFT-relaxed structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from matcalc.properties.energetics import elemental_reference_structures, formation_energy_per_atom
from matcalc.properties.similarity import fingerprint_distance, structure_fingerprint_or_error

from ._common import OK, Benchmark, Material, failed, pool_map, worker_pool

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy as np
    from pymatgen.core import Structure

    from matcalc.simulation import RelaxResult, Simulator

QUANTITIES = ("structure", "Eform", "d")


class EquilibriumBenchmark(Benchmark):
    """Formation energy ``Eform`` (eV/atom) and structural distance ``d`` of WBM compounds.

    Attributes:
        fmax: Force threshold of the relaxations (eV/Å).
        max_steps: Maximum number of FIRE steps per relaxation.
        perturb_distance: Largest distance an atom is moved before relaxing (Å); 0 or None to skip.
        reference_energies: Chemical potential of each element (eV/atom), filled by ``prepare``.
    """

    name = "equilibrium"
    id_column = "material_id"
    default_dataset = "wbm-random-pbe52-equilibrium-2025.1.json.gz"
    reference_columns = ("structure", "Eform")
    summary_metrics: ClassVar[dict[str, str]] = {"Eform": "error", "d": "value"}
    default_chunk_size = 100

    def __init__(
        self,
        dataset: str | Path | None = None,
        *,
        n_samples: int | None = None,
        seed: int = 42,
        fmax: float = 0.05,
        max_steps: int = 500,
        perturb_distance: float | None = 0.1,
        workers: int = 1,
    ) -> None:
        """
        Args:
            dataset: Dataset file name on Hugging Face, or a local ``Path``.
            n_samples: Draw this many compounds at random (``None`` = all).
            seed: Seed of the random draw and of the random displacements (every structure is
                displaced with a generator seeded by ``seed``, so its start does not depend on
                which other compounds are in the run).
            fmax: Force threshold of the relaxations (eV/Å).
            max_steps: Maximum number of FIRE steps per relaxation.
            perturb_distance: Largest distance an atom is moved before relaxing (Å); 0 or None to skip.
            workers: Processes for the structural fingerprints (see ``Benchmark``).
        """
        super().__init__(dataset, n_samples=n_samples, seed=seed, workers=workers)
        self.fmax = fmax
        self.max_steps = max_steps
        self.perturb_distance = perturb_distance
        self.reference_energies: dict[str, float] = {}
        self._dft_fingerprints: dict[str, np.ndarray | str] = {}

    def read_entries(self, raw: Any) -> list[Material]:
        """Read the WBM entries.

        Args:
            raw: List of entries with ``material_id``, ``formula``, ``structure`` (DFT-relaxed) and
                ``formation_energy_per_atom`` (eV/atom).

        Returns:
            One ``Material`` per compound.
        """
        return [
            Material(
                entry["material_id"],
                entry["formula"],
                entry["structure"],
                {"structure": entry["structure"], "Eform": entry["formation_energy_per_atom"]},
            )
            for entry in raw
        ]

    def prepare(self, simulator: Simulator, cache: dict[str, Any]) -> None:
        """Step 1: chemical potential of every element (computed once, kept in the checkpoint).

        Args:
            simulator: The simulator of this run.
            cache: Checkpoint cache; holds ``reference_energies`` after the first run.
        """
        if "reference_energies" in cache:
            self.reference_energies = cache["reference_energies"]
            return
        elements = {element.symbol for m in self.materials for element in m.structure.composition.elements}
        candidates = [
            (element, structure)
            for element, structures in elemental_reference_structures(elements).items()
            for structure in structures
        ]
        with self.stage("relax elemental references"):
            relaxed = simulator.relax([s for _, s in candidates], fmax=self.fmax, max_steps=self.max_steps)
        energies_per_atom: dict[str, list[float]] = {}
        for (element, _), result in zip(candidates, relaxed, strict=True):
            if result.structure is not None:  # as upstream, unconverged references are still used
                energies_per_atom.setdefault(element, []).append(result.energy / len(result.structure))
        self.reference_energies = {element: min(values) for element, values in energies_per_atom.items()}
        cache["reference_energies"] = self.reference_energies

    def evaluate(self, materials: Sequence[Material], simulator: Simulator) -> list[dict[str, Any]]:
        """Steps 2-4 for some compounds.

        Args:
            materials: Compounds to evaluate.
            simulator: The simulator of this run.

        Returns:
            Per compound: relaxed ``structure``, ``Eform`` (eV/atom), ``d``, ``status`` and
            ``relax_steps``.
        """
        with worker_pool(self.workers) as pool:
            # The fingerprints of the DFT structures do not depend on the model; they are computed once per
            # benchmark, in the worker processes while the GPU relaxes.
            new_dft = sorted({m.material_id for m in materials} - set(self._dft_fingerprints))
            by_id = {m.material_id: m for m in materials}
            dft_fingerprints = pool_map(
                pool,
                structure_fingerprint_or_error,
                [by_id[material_id].reference["structure"] for material_id in new_dft],
            )
            with self.stage("relax"):
                starts = [self._displaced(material.structure) for material in materials]
                relaxed = simulator.relax(starts, fmax=self.fmax, max_steps=self.max_steps)
            predictions = [self._predict(result) for result in relaxed]

            # Fingerprints need only the CPU; they are computed in parallel processes.
            with self.stage("fingerprints"):
                done = [i for i, prediction in enumerate(predictions) if prediction["status"] == OK]
                fingerprints = pool_map(
                    pool, structure_fingerprint_or_error, [predictions[i]["structure"] for i in done]
                )
                self._dft_fingerprints.update(zip(new_dft, dft_fingerprints, strict=True))
                for i, relaxed_fingerprint in zip(done, fingerprints, strict=True):
                    dft_fingerprint = self._dft_fingerprints[materials[i].material_id]
                    for fingerprint in (relaxed_fingerprint, dft_fingerprint):
                        if isinstance(fingerprint, str):  # the error message
                            predictions[i]["status"] = f"fingerprint failed: {fingerprint}"
                    if predictions[i]["status"] == OK:
                        predictions[i]["d"] = fingerprint_distance(relaxed_fingerprint, dft_fingerprint)
        return predictions

    def _displaced(self, structure: Structure) -> Structure:
        if not self.perturb_distance:
            return structure
        # copy() first: perturb() works in place and must not change the dataset structure. Each atom moves
        # in a random direction by a distance drawn uniformly from [0, perturb_distance]; min_distance=0.0
        # is pymatgen's current default (the one upstream matcalc gets) and is passed explicitly so the
        # benchmark does not change if that default changes.
        return structure.copy().perturb(distance=self.perturb_distance, min_distance=0.0, seed=self.seed)

    def _predict(self, result: RelaxResult) -> dict[str, Any]:
        steps = {"relax_steps": result.n_steps}
        if result.structure is None:
            return failed(result.error or "relaxation failed", QUANTITIES) | steps
        if not result.converged:
            reason = f"relaxation not converged (max force {result.max_force:.3g} eV/A after {result.n_steps} steps)"
            return failed(reason, QUANTITIES) | steps
        composition = result.structure.composition
        missing = sorted(el.symbol for el in composition.elements if el.symbol not in self.reference_energies)
        if missing:
            return failed(f"no elemental reference energy for {', '.join(missing)}", QUANTITIES) | steps
        return {
            "structure": result.structure,
            "Eform": formation_energy_per_atom(result.energy, composition, self.reference_energies),
            "d": float("nan"),  # filled in by evaluate() once the fingerprints are known
            "status": OK,
        } | steps
