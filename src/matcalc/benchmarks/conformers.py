"""Conformers benchmark: relative energies of the conformers of drug-like molecules (Folmsbee and Hutchison).

The conformer benchmark of D. L. Folmsbee and G. R. Hutchison (Int. J. Quantum Chem. 121, e26381 (2021);
github.com/hutchisonlab/conformer-benchmark, MIT license): up to 10 conformers of each of 702 drug-like
molecules (86 of them ions; elements H, C, N, O, F, P, S, Cl, Br), optimized with B3LYP-D3BJ, with
DLPNO-CCSD(T) single-point energies. The repository at a fixed commit (a 41 MB archive) is downloaded on
first use.

Recipe (as MLIPAudit's conformer-selection benchmark, whose published results it reproduces):

1. The molecules with at least ``MIN_CONFORMERS`` conformers (693).
2. Single points of every conformer (the molecule in a large periodic box, its charge in ``info``; no
   relaxation).
3. Per molecule, the energies relative to the conformer lowest in the reference: their mean absolute and
   root-mean-square errors (kcal/mol) and the Spearman rank correlation with the reference
   (``properties.molecules``). The summary averages them over the molecules.

Molecules containing elements outside ``elements`` (the ones the model supports, if given) are skipped.
"""

from __future__ import annotations

import zipfile
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from ase import units

from matcalc.datasets import RemoteFile, download_file, sample_subset
from matcalc.properties.molecules import KCAL_PER_MOL, conformer_errors
from matcalc.structures import molecule_in_box

from ._common import OK, Benchmark, Material, failed

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from ase import Atoms

    from matcalc.simulation import Simulator

CONFORMER_COMMIT = "0109c8ec7cff3c10d3617ed65392fffc81492755"
"""Commit of github.com/hutchisonlab/conformer-benchmark whose data are used."""

CONFORMER_ARCHIVE = RemoteFile(
    f"https://codeload.github.com/hutchisonlab/conformer-benchmark/zip/{CONFORMER_COMMIT}",
    "conformer-benchmark.zip",
    "1e339d869c408a8eeed5dc3b837ba24d",
)
"""The repository as a zip archive."""

MIN_CONFORMERS = 3
"""Molecules with fewer conformers are left out (as in MLIPAudit)."""

HARTREE_PER_KCAL_MOL = units.Hartree / KCAL_PER_MOL
"""1 Hartree in kcal/mol."""

QUANTITIES = ("energies", "mae", "rmse", "spearman")


class ConformerBenchmark(Benchmark):
    """Relative energies of the conformers of drug-like molecules (kcal/mol) against DLPNO-CCSD(T).

    Attributes:
        elements: Elements the model supports (``None``: all); molecules with others are skipped.
    """

    name = "conformers"
    id_column = "molecule"
    default_dataset = "folmsbee-hutchison"
    reference_columns = ("energies",)
    reference_label = "ref"
    batched_chunk_size = 10_000

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
            dataset: ``"folmsbee-hutchison"`` (downloaded from GitHub on first use), or a ``pathlib.Path`` to
                a zip archive of the repository.
            elements: Elements the model supports; molecules with other elements are skipped.
            n_samples: Draw this many molecules at random (``None`` = all 693).
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
        """Read the conformers and their reference energies.

        Returns:
            One ``Material`` per molecule (``structure``: its conformer lowest in the reference), in the
            order drawn.
        """
        if isinstance(self.dataset, Path):
            path = self.dataset
        elif self.dataset == self.default_dataset:
            path = download_file(CONFORMER_ARCHIVE, "conformer-benchmark")
        else:
            raise ValueError(f"Unknown dataset {self.dataset!r}: use {self.default_dataset!r} or a Path")
        with zipfile.ZipFile(path) as archive:
            root = archive.namelist()[0].split("/")[0]
            conformers: dict[str, list[tuple[str, float]]] = defaultdict(list)
            for line in archive.read(f"{root}/energies/ccsdt.txt").decode().splitlines():
                if line.strip():
                    fields = line.split()
                    conformers[fields[0]].append((fields[1].removesuffix(".out.bz2"), float(fields[-1])))
            charges: dict[str, int] = {}
            for line in archive.read(f"{root}/geometries/CHG-charges.txt").decode().splitlines():
                if line.strip():
                    name, value = line.split()[:2]
                    charges[name] = int(value.removeprefix("CHARGE="))
            materials = []
            for molecule, entries in conformers.items():
                if len(entries) < MIN_CONFORMERS:
                    continue
                folder = f"{root}/geometries/{'CHG_jobs' if molecule in charges else 'Neutral_jobs'}/{molecule}"
                charge = charges.get(molecule, 0)
                structures = [_read_xyz(archive.read(f"{folder}/{name}.xyz").decode(), charge) for name, _ in entries]
                energies = np.array([energy for _, energy in entries]) * HARTREE_PER_KCAL_MOL
                lowest = int(np.argmin(energies))
                materials.append(
                    Material(
                        molecule,
                        structures[lowest].get_chemical_formula(),
                        structures[lowest],
                        reference={"energies": (energies - energies[lowest]).tolist()},
                        settings={
                            "conformers": [name for name, _ in entries],
                            "structures": structures,
                            "charge": charge,
                            "elements": sorted(set(structures[lowest].get_chemical_symbols())),
                        },
                    )
                )
        return sample_subset(materials, self.n_samples, self.seed)

    def evaluate(self, materials: Sequence[Material], simulator: Simulator) -> list[dict[str, Any]]:
        """Single points of the conformers and the errors of their relative energies.

        Args:
            materials: Molecules to evaluate.
            simulator: The simulator of this run.

        Returns:
            Per molecule: ``energies`` (relative to the conformer lowest in the reference, kcal/mol), ``mae``,
            ``rmse`` (kcal/mol), ``spearman`` and ``status``.
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
                    [structure for i in todo for structure in materials[i].settings["structures"]],
                    compute_stress=False,
                )
            )
        for i in todo:
            reference = np.asarray(materials[i].reference["energies"])
            own = [next(results) for _ in reference]
            error = next((r.error for r in own if r.error is not None), None)
            if error is not None:
                predictions[i] = failed(error, QUANTITIES)
                continue
            energies = np.array([r.energy for r in own]) / KCAL_PER_MOL
            relative = energies - energies[int(np.argmin(reference))]
            predictions[i] = {"energies": relative.tolist()} | conformer_errors(reference, energies) | {"status": OK}
        return predictions

    def summarize(self, table: pd.DataFrame, model_name: str) -> dict[str, Any]:
        """The errors averaged over the molecules, as MLIPAudit reports them.

        Args:
            table: Result of ``run``.
            model_name: The label used in ``run``.

        Returns:
            Counts (``n_skipped``: unsupported elements), the means over the molecules of ``mae``, ``rmse``
            (kcal/mol) and ``spearman``, and the wall time per stage (s).
        """
        status = table[f"status_{model_name}"].astype(str)
        ok = status == OK
        summary: dict[str, Any] = {
            "n_molecules": len(table),
            "n_ok": int(ok.sum()),
            "n_skipped": int(status.str.startswith("skipped").sum()),
        }
        for metric in ("mae", "rmse", "spearman"):
            values = pd.to_numeric(table[f"{metric}_{model_name}"], errors="coerce")[ok]
            summary[metric] = float(values.mean()) if len(values) else float("nan")
        summary["timings_s"] = dict(self.timings)
        return summary

    def _row(self, material: Material, prediction: dict[str, Any], model_name: str) -> dict[str, Any]:
        row = super()._row(material, prediction, model_name)
        row["charge"] = material.settings["charge"]
        row["n_conformers"] = len(material.settings["conformers"])
        return row

    def _unsupported(self, material: Material) -> list[str]:
        if self.elements is None:
            return []
        return sorted(set(material.settings["elements"]) - self.elements)


def _read_xyz(text: str, charge: int) -> Atoms:
    """A conformer from its xyz file, in its box."""
    lines = text.splitlines()
    rows = [line.split() for line in lines[2 : 2 + int(lines[0])]]
    return molecule_in_box([r[0] for r in rows], [[float(x) for x in r[1:4]] for r in rows], charge=charge)
