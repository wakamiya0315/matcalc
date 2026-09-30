"""Noncovalent benchmark: interaction energies of molecular complexes (Non-Covalent Interactions Atlas).

The dissociation curves of the NCI Atlas (J. Řezáč and co-workers, www.nciatlas.org; data CC BY 4.0,
github.com/Honza-R/NCIAtlas): 2,206 complexes of two molecules at 10 separations (the closest intermolecular
contact scaled from 0.8 to 2.0 times its equilibrium length) or, for the repulsive contacts of R739x5, at 5
separations (scaled from 1.0 to 1.25), with CCSD(T)/CBS interaction energies at every point. Six data sets
(downloaded on first use):

- D442x10: London dispersion (J. Řezáč, Phys. Chem. Chem. Phys. 24, 14780 (2022));
- HB375x10 and IHB100x10: hydrogen bonds and ionic hydrogen bonds in organic molecules (J. Řezáč, J. Chem.
  Theory Comput. 16, 2355 (2020));
- HB300SPXx10: hydrogen bonds of S, P and halogens (J. Řezáč, J. Chem. Theory Comput. 16, 6305 (2020));
- R739x5: repulsive contacts (K. Kříž, M. Nováček, J. Řezáč, J. Chem. Theory Comput. 17, 1548 (2021));
- SH250x10: sigma-hole interactions (K. Kříž, J. Řezáč, Phys. Chem. Chem. Phys. 24, 14794 (2022)).

Recipe (as MLIPAudit's noncovalent-interactions benchmark, whose published results it reproduces up to the
rounding of MLIPAudit's float32 energies; see ``docs/validation.md``):

1. Single points of every point of every curve (the complex in a large periodic box, its total charge in
   ``info``; no relaxation).
2. Interaction energy of the MLIP and of the reference: the lowest energy of the curve (the highest for
   repulsive contacts) minus the energy at the largest separation (``properties.molecules``), in kcal/mol.

Complexes containing elements outside ``elements`` (the ones the model supports, if given) are skipped.
Evaluate the MLIP in float64 when its energies include the atomic energies of all-electron quantum chemistry
(MLIPs trained on molecular data): a float32 total energy of these complexes is then a multiple of up to
0.7 kcal/mol (with MACE-OFF23), a sizeable part of many interaction energies.
"""

from __future__ import annotations

import zipfile
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from matcalc.datasets import RemoteFile, download_file, sample_subset
from matcalc.properties.molecules import KCAL_PER_MOL, error_statistics, interaction_energy
from matcalc.structures import molecule_in_box

from ._common import OK, Benchmark, Material, failed

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from ase import Atoms

    from matcalc.simulation import Simulator

NCIA_COMMIT = "1816bfc72609d7deb1d4f93ab9e27eb13bb44bec"
"""Commit of github.com/Honza-R/NCIAtlas whose packaged data sets are used."""

NCIA_SETS: dict[str, tuple[str, str]] = {
    "D442x10": ("Dispersion", "a0757b3301b5a7481d4c4adb9123e311"),
    "HB375x10": ("Hydrogen bonds", "0759cfd7956b5e89a15bc60565eb0f9e"),
    "HB300SPXx10": ("Hydrogen bonds", "b1b5a1d8aac3e0341faf1861d6a0a3a6"),
    "IHB100x10": ("Ionic hydrogen bonds", "e48a1783f8453383faa798c7da2a8902"),
    "R739x5": ("Repulsive contacts", "02f4dc94898cc01e55d2d6effd51678a"),
    "SH250x10": ("Sigma hole", "60c1e331dcdb1e6c564f47063ea82e8b"),
}
"""The data sets: descriptive name (as MLIPAudit reports them) and MD5 of the packaged zip."""

REPULSIVE_SETS = frozenset({"R739x5"})
"""Data sets of repulsive contacts: their interaction energy is the highest point of the curve."""

GROUP_NAMES = {
    "CH-Oa": "CH-O(-)",
    "CH-Na": "CH-N(-)",
    "CH-Ca": "CH-C(-)",
    "NH-Oa": "NH-O(-)",
    "NH-Na": "NH-N(-)",
    "NH-Ca": "NH-C(-)",
    "OH-Oa": "OH-O(-)",
    "OH-Na": "OH-N(-)",
    "OH-Ca": "OH-C(-)",
    "NHk-O": "NH(+)-O",
    "NHk-C": "NH(+)-C",
    "NHk-N": "NH(+)-N",
    "OHk-O": "OH(+)-O",
    "B": "Boron",
}
"""Readable names of the groups of the data sets (anions 'a', cations 'k'), as MLIPAudit reports them."""


def group_of(set_name: str, group: str, symbols: Collection[str]) -> str:
    """The group of a complex as MLIPAudit reports it.

    Args:
        set_name: Its data set.
        group: Its group in the NCI Atlas files.
        symbols: Its chemical symbols.

    Returns:
        The readable name of ``GROUP_NAMES``; the group HBCNO of D442x10 is split into HCNO and Boron
        (the complexes containing boron).
    """
    if set_name == "D442x10" and group == "HBCNO":
        group = "B" if "B" in symbols else "HCNO"
    return GROUP_NAMES.get(group, group)


def ncia_file(set_name: str) -> RemoteFile:
    """The packaged zip of an NCI Atlas data set."""
    name = f"NCIA_{set_name}_github_package.zip"
    url = f"https://raw.githubusercontent.com/Honza-R/NCIAtlas/{NCIA_COMMIT}/packaged_sets/{name}"
    return RemoteFile(url, name, NCIA_SETS[set_name][1])


class NoncovalentBenchmark(Benchmark):
    """Interaction energies of noncovalent complexes (kcal/mol) against CCSD(T)/CBS.

    Attributes:
        sets: The NCI Atlas data sets of the run.
        elements: Elements the model supports (``None``: all); complexes with others are skipped.
    """

    name = "noncovalent"
    id_column = "system_id"
    default_dataset = "nci-atlas"
    reference_columns = ("interaction_energy",)
    reference_label = "ref"
    batched_chunk_size = 10_000

    def __init__(
        self,
        dataset: str | Path | None = None,
        *,
        sets: Sequence[str] = tuple(NCIA_SETS),
        elements: Collection[str] | None = None,
        n_samples: int | None = None,
        seed: int = 42,
        workers: int = 1,
    ) -> None:
        """
        Args:
            dataset: ``"nci-atlas"`` (downloaded from GitHub on first use), or a ``pathlib.Path`` to a
                directory holding the packaged zips of ``ncia_file``.
            sets: Data sets to use (keys of ``NCIA_SETS``).
            elements: Elements the model supports; complexes with other elements are skipped.
            n_samples: Draw this many complexes at random (``None`` = all).
            seed: Seed of the random draw.
            workers: Not used (the analysis is cheap); kept for the common interface.

        Raises:
            ValueError: For a data set that is not one of ``NCIA_SETS``.
        """
        self.sets = (sets,) if isinstance(sets, str) else tuple(sets)
        if unknown := [name for name in self.sets if name not in NCIA_SETS]:
            raise ValueError(f"Unknown NCI Atlas data sets {unknown}; choose from {list(NCIA_SETS)}")
        self.elements = frozenset(elements) if elements is not None else None
        super().__init__(dataset, n_samples=n_samples, seed=seed, workers=workers)

    def run_settings(self) -> dict[str, str]:
        """The data sets and element filter, when not the defaults.

        Returns:
            Setting name → value, kept in the checkpoint so that a run is not resumed with other settings.
        """
        settings = {} if self.sets == tuple(NCIA_SETS) else {"sets": ",".join(self.sets)}
        return settings | ({} if self.elements is None else {"elements": ",".join(sorted(self.elements))})

    def load_materials(self) -> list[Material]:
        """Read the dissociation curves of the data sets.

        Returns:
            One ``Material`` per complex (``structure``: the complex at scaling 1.0), in the order drawn.
        """
        materials = []
        for set_name in self.sets:
            path = self.dataset / ncia_file(set_name).name if isinstance(self.dataset, Path) else None
            if path is None:
                if self.dataset != self.default_dataset:
                    raise ValueError(f"Unknown dataset {self.dataset!r}: use {self.default_dataset!r} or a Path")
                path = download_file(ncia_file(set_name), "nci-atlas")
            materials.extend(self._read_set(set_name, path))
        return sample_subset(materials, self.n_samples, self.seed)

    def evaluate(self, materials: Sequence[Material], simulator: Simulator) -> list[dict[str, Any]]:
        """Single points along the curves and the interaction energies.

        Args:
            materials: Complexes to evaluate.
            simulator: The simulator of this run.

        Returns:
            Per complex: ``interaction_energy`` (kcal/mol) and ``status``.
        """
        predictions: list[dict[str, Any]] = [{} for _ in materials]
        todo = []
        for i, material in enumerate(materials):
            missing = self._unsupported(material)
            if missing:
                predictions[i] = failed(f"skipped: {', '.join(missing)} not in the elements", ["interaction_energy"])
            else:
                todo.append(i)
        with self.stage("single points"):
            results = iter(
                simulator.single_point(
                    [point for i in todo for point in materials[i].settings["points"]], compute_stress=False
                )
            )
        for i in todo:
            material = materials[i]
            own = [next(results) for _ in material.settings["points"]]
            if any(r.error is not None for r in own):
                predictions[i] = failed(next(r.error for r in own if r.error is not None), ["interaction_energy"])
                continue
            energies = np.array([r.energy for r in own]) / KCAL_PER_MOL
            repulsive = material.settings["set"] in REPULSIVE_SETS
            value = interaction_energy(material.settings["scalings"], energies, repulsive=repulsive)
            predictions[i] = {"interaction_energy": value, "status": OK}
        return predictions

    def summarize(self, table: pd.DataFrame, model_name: str) -> dict[str, Any]:
        """Errors of the interaction energies, overall, per data set and per group (MLIPAudit's subsets).

        Args:
            table: Result of ``run``.
            model_name: The label used in ``run``.

        Returns:
            Counts (``n_skipped``: unsupported elements), ``interaction_energy`` (MAE, RMSE, mean signed
            error ME and number of complexes, kcal/mol), ``datasets`` and ``subsets``
            (``"<data set>: <group>"``) with the same, and the wall time per stage (s).
        """
        status = table[f"status_{model_name}"].astype(str)
        ok = (status == OK).to_numpy()
        deviations = (
            pd.to_numeric(table[f"interaction_energy_{model_name}"], errors="coerce")
            - pd.to_numeric(table["interaction_energy_ref"], errors="coerce")
        )[ok]
        datasets, subsets = defaultdict(list), defaultdict(list)
        for deviation, dataset, group in zip(deviations, table["dataset"][ok], table["group"][ok], strict=True):
            datasets[dataset].append(deviation)
            subsets[f"{dataset}: {group}"].append(deviation)
        return {
            "n_systems": len(table),
            "n_ok": int(ok.sum()),
            "n_skipped": int(status.str.startswith("skipped").sum()),
            "interaction_energy": error_statistics(deviations),
            "datasets": {name: error_statistics(values) for name, values in datasets.items()},
            "subsets": {name: error_statistics(values) for name, values in subsets.items()},
            "timings_s": dict(self.timings),
        }

    def _row(self, material: Material, prediction: dict[str, Any], model_name: str) -> dict[str, Any]:
        row = super()._row(material, prediction, model_name)
        settings = material.settings
        row["dataset"] = NCIA_SETS[settings["set"]][0]
        row["group"] = group_of(settings["set"], settings["group"], settings["elements"])
        row["name"] = settings["name"]
        return row

    def _unsupported(self, material: Material) -> list[str]:
        if self.elements is None:
            return []
        return sorted(set(material.settings["elements"]) - self.elements)

    def _read_set(self, set_name: str, path: Path) -> list[Material]:
        """The curves of one data set from its packaged zip."""
        curves: dict[str, list[tuple[float, float, Atoms, dict[str, str]]]] = defaultdict(list)
        with zipfile.ZipFile(path) as archive:
            names = {}
            names_file = f"NCIA_{set_name}/NCIA_{set_name}_system_names.txt"
            for line in archive.read(names_file).decode().splitlines():
                if line and not line.startswith("#"):
                    point, name = line.split("\t", 1)
                    names[point.rsplit("_", 1)[0]] = name.strip()
            for entry in archive.namelist():
                if not entry.startswith(f"NCIA_{set_name}/geometries/") or not entry.endswith(".xyz"):
                    continue
                lines = archive.read(entry).decode().splitlines()
                header = dict(field.split("=", 1) for field in lines[1].split() if "=" in field)
                symbols = [line.split()[0] for line in lines[2 : 2 + int(lines[0])]]
                positions = [[float(x) for x in line.split()[1:4]] for line in lines[2 : 2 + int(lines[0])]]
                atoms = molecule_in_box(symbols, positions, charge=int(header.get("charge", 0)))
                curve = entry.rsplit("/", 1)[1].removesuffix(".xyz").rsplit("_", 1)[0]
                curves[curve].append((float(header["scaling"]), float(header["benchmark_Eint"]), atoms, header))
        materials = []
        repulsive = set_name in REPULSIVE_SETS
        for curve, points in sorted(curves.items()):
            points.sort(key=lambda point: point[0])
            scalings = [p[0] for p in points]
            reference = interaction_energy(scalings, [p[1] for p in points], repulsive=repulsive)
            equilibrium = min(points, key=lambda point: abs(point[0] - 1.0))[2]
            materials.append(
                Material(
                    f"{set_name}:{curve}",
                    equilibrium.get_chemical_formula(),
                    equilibrium,
                    reference={"interaction_energy": reference, "profile": [p[1] for p in points]},
                    settings={
                        "set": set_name,
                        "group": points[0][3].get("group", ""),
                        "name": names.get(curve, ""),
                        "elements": sorted(set(equilibrium.get_chemical_symbols())),
                        "scalings": scalings,
                        "points": [p[2] for p in points],
                    },
                )
            )
        return materials
