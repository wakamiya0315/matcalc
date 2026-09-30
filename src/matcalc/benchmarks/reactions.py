"""Reactions benchmark: barrier heights and reaction energies of elementary organic reactions (RDB7).

RDB7 (K. A. Spiekermann, L. Pattanaik, W. H. Green, Sci. Data 9, 417 (2022); Zenodo 10.5281/zenodo.6618262,
CC BY 4.0) refines the reactions of C. A. Grambow, L. Pattanaik, W. H. Green, Sci. Data 7, 137 (2020): 11,926
elementary reactions of closed-shell molecules with up to 7 heavy atoms (H, C, N, O), found by single-ended
transition-state searches from GDB-7 molecules. Reactant, transition state and products are optimized with
ωB97X-D3/def2-TZVP (a reaction that breaks the reactant apart has 2 or 3 product molecules, each optimized on
its own), and their energies computed with CCSD(T)-F12a/cc-pVDZ-F12. The Molpro outputs of these
calculations (a 143 MB archive) are downloaded on first use; geometries and energies are read from them.

Recipe (as MLIPAudit's reactivity benchmark, which uses the same reactants and transition states with the
ωB97X-D3 energies of Grambow et al. as reference; see ``docs/validation.md``):

1. Single points of the reactant, the transition state and every product molecule (each in a large periodic
   box, neutral singlets; no relaxation).
2. Barrier height (transition state minus reactant) and reaction energy (products minus reactant) of the MLIP
   and of the reference, electronic energies in kcal/mol (``properties.molecules``).

Reactions with elements outside ``elements`` (the ones the model supports, if given) are skipped. Evaluate
the MLIP in float64 when its energies include the atomic energies of all-electron quantum chemistry (MLIPs
trained on molecular data), as for the other molecular benchmarks.
"""

from __future__ import annotations

import re
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from ase import units

from matcalc.datasets import RemoteFile, download_file, sample_subset
from matcalc.properties.molecules import KCAL_PER_MOL, error_statistics, reaction_energetics
from matcalc.structures import molecule_in_box

from ._common import OK, Benchmark, Material, failed

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from ase import Atoms

    from matcalc.simulation import Simulator

RDB7_LOGS = RemoteFile(
    "https://zenodo.org/records/6618262/files/ccsdtf12_dz.tar.gz?download=1",
    "ccsdtf12_dz.tar.gz",
    "0440556b4c9d79ab202efdae6e2b3364",
)
"""The Molpro outputs of RDB7's CCSD(T)-F12a/cc-pVDZ-F12 single points (Zenodo record 6618262, v1.0.1)."""

HARTREE_PER_KCAL_MOL = units.Hartree / KCAL_PER_MOL
"""1 Hartree in kcal/mol."""

QUANTITIES = ("barrier", "reaction_energy")

_ENERGY = re.compile(r"!CCSD\(T\)-F12a total energy\s+(-?\d+\.\d+)")
_WAVEFUNCTION = re.compile(r"wf,spin=(\d+),charge=(-?\d+)")


class ReactionBenchmark(Benchmark):
    """Barrier heights and reaction energies (kcal/mol) against CCSD(T)-F12a.

    Attributes:
        elements: Elements the model supports (``None``: all); reactions with others are skipped.
    """

    name = "reactions"
    id_column = "reaction"
    default_dataset = "rdb7"
    reference_columns = QUANTITIES
    reference_label = "ref"
    batched_chunk_size = 20_000

    def __init__(
        self,
        dataset: str | Path | None = None,
        *,
        elements: Collection[str] | None = None,
        n_samples: int | None = None,
        seed: int = 42,
        workers: int = 1,
    ) -> None:
        """
        Args:
            dataset: ``"rdb7"`` (downloaded from Zenodo on first use), or a ``pathlib.Path`` to an archive of
                the same layout.
            elements: Elements the model supports; reactions with other elements are skipped.
            n_samples: Draw this many reactions at random (``None`` = all 11,926).
            seed: Seed of the random draw.
            workers: Not used (the analysis is cheap); kept for the common interface.
        """
        self.elements = frozenset(elements) if elements is not None else None
        super().__init__(dataset, n_samples=n_samples, seed=seed, workers=workers)

    def run_settings(self) -> dict[str, str]:
        """The element filter, when one is given.

        Returns:
            Setting name → value, kept in the checkpoint so that a run is not resumed with other settings.
        """
        return {} if self.elements is None else {"elements": ",".join(sorted(self.elements))}

    def load_materials(self) -> list[Material]:
        """Read the reactants, transition states and products and their CCSD(T)-F12a energies.

        Returns:
            One ``Material`` per reaction (``structure``: the reactant), in the order drawn.
        """
        if isinstance(self.dataset, Path):
            path = self.dataset
        elif self.dataset == self.default_dataset:
            path = download_file(RDB7_LOGS, "rdb7")
        else:
            raise ValueError(f"Unknown dataset {self.dataset!r}: use {self.default_dataset!r} or a Path")
        reactions: dict[str, dict[str, tuple[Atoms, float]]] = defaultdict(dict)
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                folder, _, name = member.name.rpartition("/")
                # ._ files are macOS metadata that the archive also holds
                if not member.isfile() or not name.endswith(".log") or name.startswith("._"):
                    continue
                log = archive.extractfile(member)
                if log is not None:
                    reactions[folder.rsplit("/", 1)[-1].removeprefix("rxn")][name] = _read_log(log.read().decode())
        materials = []
        for reaction, logs in sorted(reactions.items()):
            reactant, energy_r = logs[f"r{reaction}.log"]
            transition_state, energy_ts = logs[f"ts{reaction}.log"]
            products = [logs[name] for name in sorted(logs) if name.startswith("p")]
            reference = reaction_energetics(energy_r, energy_ts, [energy for _, energy in products])
            materials.append(
                Material(
                    reaction,
                    reactant.get_chemical_formula(),
                    reactant,
                    reference=reference,
                    settings={
                        "transition_state": transition_state,
                        "products": [atoms for atoms, _ in products],
                        "elements": sorted(set(reactant.get_chemical_symbols())),
                    },
                )
            )
        return sample_subset(materials, self.n_samples, self.seed)

    def evaluate(self, materials: Sequence[Material], simulator: Simulator) -> list[dict[str, Any]]:
        """Single points of the reactants, transition states and products, and the reaction energetics.

        Args:
            materials: Reactions to evaluate.
            simulator: The simulator of this run.

        Returns:
            Per reaction: ``barrier`` and ``reaction_energy`` (kcal/mol) and ``status``.
        """
        predictions: list[dict[str, Any]] = [{} for _ in materials]
        todo = []
        for i, material in enumerate(materials):
            missing = self._unsupported(material)
            if missing:
                predictions[i] = failed(f"skipped: {', '.join(missing)} not in the elements", QUANTITIES)
            else:
                todo.append(i)
        with self.stage("single points"):
            results = iter(
                simulator.single_point(
                    [structure for i in todo for structure in _structures(materials[i])], compute_stress=False
                )
            )
        for i in todo:
            own = [next(results) for _ in _structures(materials[i])]
            error = next((r.error for r in own if r.error is not None), None)
            if error is not None:
                predictions[i] = failed(error, QUANTITIES)
                continue
            energies = np.array([r.energy for r in own]) / KCAL_PER_MOL
            predictions[i] = reaction_energetics(energies[0], energies[1], energies[2:]) | {"status": OK}
        return predictions

    def summarize(self, table: pd.DataFrame, model_name: str) -> dict[str, Any]:
        """Errors of the barrier heights and reaction energies.

        Args:
            table: Result of ``run``.
            model_name: The label used in ``run``.

        Returns:
            Counts (``n_skipped``: unsupported elements), ``barrier`` and ``reaction_energy`` (MAE, RMSE, mean
            signed error ME and number of reactions, kcal/mol), and the wall time per stage (s).
        """
        status = table[f"status_{model_name}"].astype(str)
        ok = status == OK
        summary: dict[str, Any] = {
            "n_reactions": len(table),
            "n_ok": int(ok.sum()),
            "n_skipped": int(status.str.startswith("skipped").sum()),
        }
        for quantity in QUANTITIES:
            predicted = pd.to_numeric(table[f"{quantity}_{model_name}"], errors="coerce")
            reference = pd.to_numeric(table[f"{quantity}_ref"], errors="coerce")
            summary[quantity] = error_statistics((predicted - reference)[ok])
        summary["timings_s"] = dict(self.timings)
        return summary

    def _row(self, material: Material, prediction: dict[str, Any], model_name: str) -> dict[str, Any]:
        row = super()._row(material, prediction, model_name)
        row["n_products"] = len(material.settings["products"])
        return row

    def _unsupported(self, material: Material) -> list[str]:
        if self.elements is None:
            return []
        return sorted(set(material.settings["elements"]) - self.elements)


def _structures(material: Material) -> list[Any]:
    """Reactant, transition state and products of a reaction, in this order."""
    return [material.structure, material.settings["transition_state"], *material.settings["products"]]


def _read_log(text: str) -> tuple[Atoms, float]:
    """The geometry (in a box) and the CCSD(T)-F12a energy (kcal/mol) of a Molpro output."""
    start = text.index("geometry={angstrom;") + len("geometry={angstrom;")
    rows = [line.split() for line in text[start : text.index("}", start)].split("\n") if line.strip()]
    wavefunction = _WAVEFUNCTION.search(text)
    spin, charge = (int(wavefunction[1]), int(wavefunction[2])) if wavefunction else (0, 0)
    atoms = molecule_in_box(
        [row[0] for row in rows],
        [[float(x) for x in row[1:4]] for row in rows],
        charge=charge,
        multiplicity=spin + 1,  # Molpro's spin is 2S
    )
    energies = _ENERGY.findall(text)
    if not energies:
        raise ValueError("no CCSD(T)-F12a energy in the output")
    return atoms, float(energies[-1]) * HARTREE_PER_KCAL_MOL
