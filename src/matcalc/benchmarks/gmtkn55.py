"""GMTKN55 benchmark: main-group thermochemistry, kinetics and noncovalent interactions (Goerigk et al.).

GMTKN55 (L. Goerigk, A. Hansen, C. Bauer, S. Ehrlich, A. Najibi, S. Grimme, Phys. Chem. Chem. Phys. 19, 32184
(2017); github.com/grimme-lab/GMTKN55, CC BY 4.0): 1,505 relative energies (reaction energies, barrier
heights, conformer and noncovalent interaction energies) of 2,442 molecules and complexes of main-group
elements up to Bi (neutral and ionic, closed- and open-shell; 1 to 81 atoms) in 55 subsets, with high-level
reference values (mostly CCSD(T)/CBS or W-n). The repository's ``v1`` branch, the original publication, is
downloaded at a fixed commit (a 39 MB archive) on first use.

Recipe (as the evaluator of the repository, whose results for PBEh-3c the benchmark reproduces; see
``docs/validation.md``):

1. Single points of the molecules of the reactions, each once however many reactions share it (in a large
   periodic box, its charge and spin multiplicity in ``info``; no relaxation).
2. Reaction energies Σ_i c_i E_i (kcal/mol) with the stoichiometric coefficients of the subsets' ``.res``
   files.
3. Per subset, the mean absolute deviation MAD_i from the reference, and WTMAD-2 = Σ_i N_i (⟨|ΔE|⟩ / |ΔE|_i)
   MAD_i / Σ_i N_i, overall and per category, where N_i is the number of reactions of subset i, |ΔE|_i the
   mean absolute reference energy of subset i and ⟨|ΔE|⟩ the mean of the |ΔE|_i (57.82 kcal/mol for the whole
   set): all taken over the reactions evaluated.

Reactions with a molecule outside ``elements``, ``charges`` or ``max_unpaired_electrons`` are skipped, as the
evaluator's filters do; WTMAD-2 then covers the other reactions and is not comparable with that of the whole
set, which is why the summary lists every subset with its number of reactions. Evaluate the MLIP in float64
when its energies include the atomic energies of all-electron quantum chemistry, as for the other molecular
benchmarks.
"""

from __future__ import annotations

import re
import shlex
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from matcalc.datasets import RemoteFile, download_file, sample_subset
from matcalc.properties.molecules import KCAL_PER_MOL, error_statistics
from matcalc.structures import molecule_in_box

from ._common import OK, Benchmark, Material, failed

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from ase import Atoms

    from matcalc.simulation import Simulator

GMTKN55_COMMIT = "8d485b37a1ca8837e395042671ca5ba4e0714691"
"""Commit of the ``v1`` branch of github.com/grimme-lab/GMTKN55 (the original publication; the ``v2`` branch
with updated references is still changing and, at its commit ccabc16, names WATER27 molecules that it lacks)."""

GMTKN55_ARCHIVE = RemoteFile(
    f"https://codeload.github.com/grimme-lab/GMTKN55/zip/{GMTKN55_COMMIT}",
    "gmtkn55-v1.zip",
    "1c83333b9009045e83c7cf3a6ccdf5cc",
)
"""The repository as a zip archive."""

CATEGORIES: dict[str, tuple[str, ...]] = {
    "small reactions": (
        "W4-11",
        "G21EA",
        "G21IP",
        "DIPCS10",
        "PA26",
        "SIE4x4",
        "ALKBDE10",
        "YBDE18",
        "AL2X6",
        "HEAVYSB11",
        "NBPRC",
        "ALK8",
        "RC21",
        "G2RC",
        "BH76RC",
        "FH51",
        "TAUT15",
        "DC13",
    ),
    "large reactions": ("MB16-43", "DARC", "RSE43", "BSR36", "CDIE20", "ISO34", "ISOL24", "C60ISO", "PArel"),
    "barrier heights": ("BH76", "BHPERI", "BHDIV10", "INV24", "BHROT27", "PX13", "WCPT18"),
    "intermolecular NCI": (
        "RG18",
        "ADIM6",
        "S22",
        "S66",
        "HEAVY28",
        "WATER27",
        "CARBHB12",
        "PNICO23",
        "HAL59",
        "AHB21",
        "CHB6",
        "IL16",
    ),
    "intramolecular NCI": (
        "IDISP",
        "ICONF",
        "ACONF",
        "Amino20x4",
        "PCONF21",
        "MCONF",
        "SCONF",
        "UPU23",
        "BUT14DIOL",
    ),
}
"""The categories of the 55 subsets (as the GMTKN55 evaluator groups them; BH76RC is BH76's ``.resRC``)."""

_TMER = ("$tmer", "tmer", "tmer2++")
_NUMBER = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eEdD][-+]?\d+)?$")


class GMTKN55Benchmark(Benchmark):
    """Reaction energies of GMTKN55 (kcal/mol) against its references, summarized by WTMAD-2.

    Attributes:
        elements: Elements the model supports (``None``: all).
        charges: ``(lowest, highest)`` total charge of the molecules to evaluate (``None``: all).
        max_unpaired_electrons: Most unpaired electrons of the molecules to evaluate (``None``: all).
        molecules: The molecules, by ``"<subset>/<name>"``, in their boxes.
    """

    name = "gmtkn55"
    id_column = "reaction"
    default_dataset = "gmtkn55-v1"
    reference_columns = ("energy",)
    reference_label = "ref"
    # One chunk: every molecule is evaluated once, however many reactions (of any subset) share it.
    default_chunk_size = 10_000
    batched_chunk_size = 10_000

    def __init__(
        self,
        dataset: str | Path | None = None,
        *,
        elements: Collection[str] | None = None,
        charges: tuple[int, int] | None = None,
        max_unpaired_electrons: int | None = None,
        n_samples: int | None = None,
        seed: int = 42,
        workers: int = 1,
    ) -> None:
        """
        Args:
            dataset: ``"gmtkn55-v1"`` (downloaded from GitHub on first use), or a ``pathlib.Path`` to a zip
                archive of the repository.
            elements: Elements the model supports; reactions with a molecule of other elements are skipped.
            charges: ``(lowest, highest)`` total charge; reactions with a molecule outside are skipped
                (``(0, 0)``: neutral molecules only).
            max_unpaired_electrons: Reactions with a molecule with more unpaired electrons are skipped
                (``0``: closed-shell molecules only).
            n_samples: Draw this many reactions at random (``None`` = all 1,505).
            seed: Seed of the random draw.
            workers: Not used (the analysis is cheap); kept for the common interface.

        Raises:
            ValueError: If ``charges`` is not ``(lowest, highest)`` with lowest <= highest.
        """
        if charges is not None and not (len(charges) == 2 and charges[0] <= charges[1]):  # noqa: PLR2004
            raise ValueError(f"charges must be (lowest, highest) with lowest <= highest, not {charges!r}")
        self.elements = frozenset(elements) if elements is not None else None
        self.charges = None if charges is None else (int(charges[0]), int(charges[1]))
        self.max_unpaired_electrons = max_unpaired_electrons
        self.molecules: dict[str, Atoms] = {}
        super().__init__(dataset, n_samples=n_samples, seed=seed, workers=workers)

    def run_settings(self) -> dict[str, str]:
        """The filters that are set.

        Returns:
            Setting name → value, kept in the checkpoint so that a run is not resumed with other settings.
        """
        settings = {} if self.elements is None else {"elements": ",".join(sorted(self.elements))}
        if self.charges is not None:
            settings["charges"] = f"{self.charges[0]},{self.charges[1]}"
        if self.max_unpaired_electrons is not None:
            settings["max_unpaired_electrons"] = str(self.max_unpaired_electrons)
        return settings

    def load_materials(self) -> list[Material]:
        """Read the reactions and their molecules (only the geometries, charges and spins of the molecules).

        Returns:
            One ``Material`` per reaction (id ``"<subset>:<number>"``, counted from 1 in the order of the
            subset's file; ``structure``: its first molecule), in the order drawn.
        """
        if isinstance(self.dataset, Path):
            path = self.dataset
        elif self.dataset == self.default_dataset:
            path = download_file(GMTKN55_ARCHIVE, "gmtkn55")
        else:
            raise ValueError(f"Unknown dataset {self.dataset!r}: use {self.default_dataset!r} or a Path")
        category = {subset: name for name, subsets in CATEGORIES.items() for subset in subsets}
        materials = []
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            root = archive.namelist()[0].split("/")[0]
            definitions = sorted(name for name in names if re.fullmatch(rf"{re.escape(root)}/[^/]+/\.res(RC)?", name))
            for definition in definitions:
                folder = definition.split("/")[1]
                subset = folder + ("RC" if definition.endswith("RC") else "")
                for number, (systems, coefficients, reference) in enumerate(
                    reactions_of(archive.read(definition).decode()), start=1
                ):
                    keys = [f"{folder}/{system}" for system in systems]
                    for key in keys:
                        if key not in self.molecules:
                            self.molecules[key] = _read_molecule(archive, f"{root}/{key}", names)
                    materials.append(
                        Material(
                            f"{subset}:{number}",
                            reaction_formula(systems, coefficients),
                            self.molecules[keys[0]],
                            reference={"energy": reference},
                            settings={
                                "subset": subset,
                                "category": category.get(subset, ""),
                                "molecules": keys,
                                "coefficients": coefficients,
                            },
                        )
                    )
        return sample_subset(materials, self.n_samples, self.seed)

    def evaluate(self, materials: Sequence[Material], simulator: Simulator) -> list[dict[str, Any]]:
        """Single points of the molecules (each once) and the reaction energies.

        Args:
            materials: Reactions to evaluate.
            simulator: The simulator of this run.

        Returns:
            Per reaction: ``energy`` (kcal/mol) and ``status``.
        """
        predictions: list[dict[str, Any]] = [{} for _ in materials]
        needed: dict[str, None] = {}  # ordered set of the molecules of the reactions evaluated
        for i, material in enumerate(materials):
            reason = self._skip_reason(material)
            if reason is not None:
                predictions[i] = failed(f"skipped: {reason}", ["energy"])
                continue
            needed |= dict.fromkeys(material.settings["molecules"])
        with self.stage("single points"):
            results = simulator.single_point([self.molecules[key] for key in needed], compute_stress=False)
        outcome = dict(zip(needed, results, strict=True))
        for i, material in enumerate(materials):
            if predictions[i]:
                continue
            own = [outcome[key] for key in material.settings["molecules"]]
            error = next((r.error for r in own if r.error is not None), None)
            if error is not None:
                predictions[i] = failed(error, ["energy"])
                continue
            energy = sum(c * r.energy for c, r in zip(material.settings["coefficients"], own, strict=True))
            predictions[i] = {"energy": energy / KCAL_PER_MOL, "status": OK}
        return predictions

    def summarize(self, table: pd.DataFrame, model_name: str) -> dict[str, Any]:
        """WTMAD-2 overall and per category, and the errors of every subset, over the reactions evaluated.

        Args:
            table: Result of ``run``.
            model_name: The label used in ``run``.

        Returns:
            Counts (``n_skipped``: filtered out), ``WTMAD-2`` (``total`` and per category, NaN for a category
            without reactions), ``mean_abs_reference`` (⟨|ΔE|⟩, kcal/mol), ``subsets`` (per subset: ``N``,
            ``mean_abs_reference``, ``MAE``, ``RMSE`` and mean signed error ``ME``, kcal/mol) and the wall time
            per stage (s).
        """
        status = table[f"status_{model_name}"].astype(str)
        ok = status == OK
        predicted = pd.to_numeric(table[f"energy_{model_name}"], errors="coerce")[ok]
        reference = pd.to_numeric(table["energy_ref"], errors="coerce")[ok]
        subsets: dict[str, dict[str, float]] = {}
        for subset, rows in table[ok].groupby("subset", sort=False):
            errors = error_statistics(predicted[rows.index] - reference[rows.index])
            subsets[str(subset)] = {
                "N": errors["n"],
                "mean_abs_reference": float(np.mean(np.abs(reference[rows.index]))),
            } | {key: errors[key] for key in ("MAE", "RMSE", "ME")}
        total, per_category, mean_abs = wtmad2(subsets)
        return {
            "n_reactions": len(table),
            "n_ok": int(ok.sum()),
            "n_skipped": int(status.str.startswith("skipped").sum()),
            "WTMAD-2": {"total": total} | per_category,
            "mean_abs_reference": mean_abs,
            "subsets": subsets,
            "timings_s": dict(self.timings),
        }

    def _row(self, material: Material, prediction: dict[str, Any], model_name: str) -> dict[str, Any]:
        row = super()._row(material, prediction, model_name)
        row["subset"] = material.settings["subset"]
        row["category"] = material.settings["category"]
        return row

    def _skip_reason(self, material: Material) -> str | None:
        for key in material.settings["molecules"]:
            atoms = self.molecules[key]
            if self.elements is not None and (missing := sorted(set(atoms.get_chemical_symbols()) - self.elements)):
                return f"{', '.join(missing)} not in the elements"
            if self.charges is not None and not self.charges[0] <= atoms.info["charge"] <= self.charges[1]:
                return f"charge {atoms.info['charge']} of {key}"
            unpaired = atoms.info["spin"] - 1
            if self.max_unpaired_electrons is not None and unpaired > self.max_unpaired_electrons:
                return f"{unpaired} unpaired electrons in {key}"
        return None


def reactions_of(text: str) -> list[tuple[list[str], list[float], float]]:
    """The reactions of a GMTKN55 ``.res`` (or ``.resRC``) file.

    Its lines ``$tmer a/$f {b,c}/$f x -1 -2 1 $w 12.3`` are read as the shell and the evaluator read them
    (comments dropped, braces expanded; coefficients after ``x``, the reference after ``$w``).

    Args:
        text: The file.

    Returns:
        Per reaction: the names of its molecules, their coefficients and the reference energy (kcal/mol).

    Raises:
        ValueError: For a reaction with an energy added to it or an error multiplier (two optional numbers
            after the reference, which the evaluator applies; GMTKN55 v1 has none).
    """
    reactions = []
    for line in text.splitlines():
        try:
            tokens = shlex.split(line, comments=True, posix=True)
        except ValueError:
            continue
        if not tokens or not (tokens[0] in _TMER or tokens[0].endswith("tmer2++")) or "x" not in tokens:
            continue
        x = tokens.index("x")
        systems = [path.split("/")[0] for token in tokens[1:x] for path in _expand_braces(token)]
        tail = tokens[x + 1 :]
        coefficients = [float(token) for token in tail[: len(systems)]]
        after = [float(t.replace("D", "E").replace("d", "e")) for t in tail[len(systems) + 1 :] if _NUMBER.match(t)]
        if len(after) > 1 and (after[1] > 1e-8 or (len(after) > 2 and after[2] > 1e-8 and after[2] != 1.0)):  # noqa: PLR2004
            raise ValueError(f"reaction with an added energy or an error multiplier: {line.strip()}")
        reactions.append((systems, coefficients, after[0] if after else 0.0))
    return reactions


def reaction_formula(systems: Sequence[str], coefficients: Sequence[float]) -> str:
    """``"a + 2 b -> c"`` for coefficients -1, -2, 1."""

    def side(sign: float) -> str:
        return " + ".join(
            (name if abs(c) == 1 else f"{abs(c):g} {name}")
            for name, c in zip(systems, coefficients, strict=True)
            if c * sign > 0
        )

    return f"{side(-1)} -> {side(1)}"


def wtmad2(subsets: Mapping[str, Mapping[str, float]]) -> tuple[float, dict[str, float], float]:
    """WTMAD-2 of GMTKN55 from the statistics of its subsets, as the repository's evaluator computes it.

    Args:
        subsets: Per subset evaluated: ``N`` (reactions), ``mean_abs_reference`` (|ΔE|_i) and ``MAE`` (kcal/mol).

    Returns:
        WTMAD-2 of all subsets, WTMAD-2 per category (with ``all NCI``; NaN without reactions) and ⟨|ΔE|⟩, the
        mean of the |ΔE|_i (kcal/mol).
    """
    if not subsets:
        return float("nan"), dict.fromkeys([*CATEGORIES, "all NCI"], float("nan")), float("nan")
    mean_abs = float(np.mean([s["mean_abs_reference"] for s in subsets.values()]))

    def weighted(names: Collection[str]) -> float:
        chosen = [subsets[name] for name in names if name in subsets]
        count = sum(s["N"] for s in chosen)
        if not count:
            return float("nan")
        return float(sum(s["N"] * mean_abs * s["MAE"] / s["mean_abs_reference"] for s in chosen) / count)

    per_category = {name: weighted(names) for name, names in CATEGORIES.items()}
    per_category["all NCI"] = weighted(CATEGORIES["intermolecular NCI"] + CATEGORIES["intramolecular NCI"])
    return weighted(list(subsets)), per_category, mean_abs


def _expand_braces(token: str) -> list[str]:
    """Shell brace expansion of one token: ``a{b,c}d`` → ``abd``, ``acd`` (``1{,A}`` → ``1``, ``1A``)."""
    start = token.find("{")
    end = token.find("}", start)
    if start < 0 or end < 0:
        return [token]
    return [
        token[:start] + option + tail
        for option in token[start + 1 : end].split(",")
        for tail in _expand_braces(token[end + 1 :])
    ]


def _read_molecule(archive: zipfile.ZipFile, folder: str, names: set[str]) -> Atoms:
    """A molecule from its ``struc.xyz`` (Å), ``.CHRG`` and ``.UHF`` (unpaired electrons), in its box."""

    def number(name: str) -> int:
        return int(archive.read(f"{folder}/{name}").split()[0]) if f"{folder}/{name}" in names else 0

    lines = archive.read(f"{folder}/struc.xyz").decode().splitlines()
    rows = [line.split() for line in lines[2 : 2 + int(lines[0])]]
    return molecule_in_box(
        [row[0].capitalize() for row in rows],
        [[float(x) for x in row[1:4]] for row in rows],
        charge=number(".CHRG"),
        multiplicity=number(".UHF") + 1,
    )
