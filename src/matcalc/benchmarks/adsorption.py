"""Adsorption benchmark: energies of molecules adsorbing on metal and oxide surfaces, against experiment.

Two sets of experimental adsorption energies, 54 reactions in all (``benchmarks/data/adsorption.json``):

- ADS41 (S. Mallikarjun Sharada, R. K. B. Karlsson, Y. Maimaiti, J. Voss, T. Bligaard, Phys. Rev. B 100,
  035439 (2019), Table I): 41 molecular and dissociative adsorption energies on late transition-metal
  surfaces, 26 dominated by covalent bonds (``chemisorption``) and 15 by dispersion (``dispersion``; 7 of these
  with covalent contributions, ``mixed``). 39 come from the CE39 database (J. Wellendorff et al., Surf. Sci. 640,
  36 (2015)) of calorimetry, temperature-programmed desorption and equilibrium-adsorption measurements. The
  reference is the experimental reaction energy minus the zero-point energy change computed with PBE: a static
  energy, directly comparable with the MLIP's.
- Surf13 (B. X. Shi et al., Nat. Chem. 17, 1688 (2025)): the 13 molecules of the Surf13 set on MgO(001),
  rutile TiO2(110) and anatase TiO2(101) (Surf13 is a set of CCSD(T) interaction energies; only its systems are
  used here), with the experimental adsorption enthalpies collected and re-analysed in that paper (SI
  Table 32) minus the zero-point, thermal and -RT contributions of its DFT ensemble (SI Table 30). CO2 on
  MgO(001) is the chemisorbed (carbonate) state that the paper assigns to the experiment.

The configurations are fixed by the dataset (no site search): sites, orientations, coverages and slabs of the
reference studies, with the later corrections of Araujo et al., Nat. Commun. 13, 6853 (2022) for ADS41 and the
published DFT structures of Shi et al. for Surf13 (``scripts/build_adsorption_dataset.py`` documents every
choice).

Recipe:

1. Relax the bulk crystals (atoms and cell, keeping the space group), starting from experimental lattice
   constants: every MLIP works with its own lattice constants, as each functional of the references did.
2. Cut the slabs from the relaxed crystals (``matcalc.surfaces``: metals four layers with the bottom two
   fixed; MgO(001), rutile(110) and anatase(101) as in Shi et al.) and relax them in their fixed cells.
3. Place the adsorbates on the relaxed slabs and relax the adsorbed slabs, and the gas-phase molecules in a
   50 Å box, in fixed cells (FIRE, ``fmax``, at most ``max_steps`` steps).
4. Reaction energy ΔE = Σ_i c_i E_i of the reaction as written (eV); dissociated molecules are computed as
   separate adlayers, as in CE39.

A relaxation that reaches the step limit still gives a prediction; the status names it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from matcalc.datasets import sample_subset
from matcalc.properties.adsorption import adsorbate_displacement, adsorbate_rearranged, reaction_energy
from matcalc.properties.molecules import error_statistics
from matcalc.structures import molecule_in_box, to_ase_atoms
from matcalc.surfaces import add_adsorbates, build_slab, bulk_crystal

from ._common import OK, Benchmark, Material, failed

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from ase import Atoms
    from pymatgen.core import Structure

    from matcalc.simulation import RelaxResult, Simulator

DATASET = Path(__file__).parent / "data" / "adsorption.json"
"""The reactions, their reference energies and the fixed configurations (built by
``scripts/build_adsorption_dataset.py``)."""

SUBSETS = ("ADS41", "Surf13")
"""The two sets of reactions."""

QUANTITIES = ("energy", "displacement", "rearranged")

MOVED = 1.0
"""Displacement (Å) of a heavy adsorbate atom above which the summary counts the adsorbate as having left its
starting configuration."""


class AdsorptionBenchmark(Benchmark):
    """Adsorption energies (eV) on metal and oxide surfaces against experiment (ADS41 and Surf13).

    Attributes:
        subsets: The sets of reactions evaluated (``"ADS41"``, ``"Surf13"``).
        elements: Elements the model supports (``None``: all); reactions with others are skipped.
        fmax: Force threshold of all relaxations (eV/Å).
        max_steps: Maximum number of FIRE steps per relaxation.
        definitions: The crystals, slabs, molecules and adsorbed structures of the dataset.
    """

    name = "adsorption"
    id_column = "reaction"
    default_dataset = DATASET
    reference_columns = ("energy",)
    reference_label = "exp"
    # One chunk: the crystals, slabs and molecules are shared by many reactions and computed once.
    default_chunk_size = 1000
    batched_chunk_size = 1000

    def __init__(
        self,
        dataset: str | Path | None = None,
        *,
        subsets: Sequence[str] = SUBSETS,
        elements: Collection[str] | None = None,
        fmax: float = 0.02,
        max_steps: int = 1000,
        n_samples: int | None = None,
        seed: int = 42,
        workers: int = 1,
    ) -> None:
        """
        Args:
            dataset: A ``pathlib.Path`` to a dataset file of the same format (default: the packaged one).
            subsets: The sets of reactions to evaluate.
            elements: Elements the model supports; reactions with other elements are skipped.
            fmax: Force threshold of the relaxations (eV/Å), the 0.02 eV/Å of Sharada et al.; the bulk
                relaxations also need the forces on the cell below it.
            max_steps: Maximum number of FIRE steps per relaxation.
            n_samples: Draw this many reactions at random (``None`` = all).
            seed: Seed of the random draw.
            workers: Not used (the analysis is cheap); kept for the common interface.

        Raises:
            ValueError: For an unknown subset.
        """
        unknown = set(subsets) - set(SUBSETS)
        if unknown:
            raise ValueError(f"Unknown subsets {sorted(unknown)}: use some of {SUBSETS}")
        self.subsets = tuple(s for s in SUBSETS if s in subsets)
        self.elements = frozenset(elements) if elements is not None else None
        self.fmax = fmax
        self.max_steps = max_steps
        self.definitions: dict[str, Any] = {}
        super().__init__(dataset, n_samples=n_samples, seed=seed, workers=workers)

    def run_settings(self) -> dict[str, str]:
        """The settings that differ from the defaults.

        Returns:
            Setting name → value, kept in the checkpoint so that a run is not resumed with other settings.
        """
        settings = self._changed_settings("fmax", "max_steps")
        if self.subsets != SUBSETS:
            settings["subsets"] = ",".join(self.subsets)
        if self.elements is not None:
            settings["elements"] = ",".join(sorted(self.elements))
        return settings

    def load_materials(self) -> list[Material]:
        """Read the reactions of the selected subsets.

        Returns:
            One ``Material`` per reaction (``formula``: the reaction as written; ``structure``: its gas-phase
            molecule in a box), in the order drawn.

        Raises:
            TypeError: If ``dataset`` is not a ``pathlib.Path``.
        """
        if not isinstance(self.dataset, Path):
            raise TypeError(f"Unknown dataset {self.dataset!r}: use a pathlib.Path (default: the packaged file)")
        self.definitions = json.loads(self.dataset.read_text())
        molecules = self.definitions["molecules"]
        materials = []
        for reaction in self.definitions["reactions"]:
            if reaction["subset"] not in self.subsets:
                continue
            gas = next(name for name in reaction["terms"] if name in molecules)
            materials.append(
                Material(
                    reaction["id"],
                    reaction["equation"],
                    self._molecule(gas),
                    reference={"energy": reaction["reference"]["energy"]},
                    settings={
                        "terms": reaction["terms"],
                        "subset": reaction["subset"],
                        "category": reaction["category"],
                        "mixed": reaction["mixed"],
                        "adsorbates": reaction["adsorbates_per_reaction"],
                        "elements": sorted({e for name in reaction["terms"] for e in self._elements(name)}),
                    },
                )
            )
        return sample_subset(materials, self.n_samples, self.seed)

    def evaluate(self, materials: Sequence[Material], simulator: Simulator) -> list[dict[str, Any]]:
        """Relax the crystals, the slabs, and the adsorbed slabs and molecules; the reaction energies.

        Args:
            materials: Reactions to evaluate.
            simulator: The simulator of this run.

        Returns:
            Per reaction: ``energy`` (eV), ``displacement`` (largest distance a heavy adsorbate atom moved in
            the relaxations, Å; H atoms count only for adsorbates of H alone), ``rearranged`` (whether a bond
            between adsorbate atoms broke or formed) and ``status``.
        """
        predictions: list[dict[str, Any]] = [{} for _ in materials]
        needed: dict[str, None] = {}  # ordered set of the structures of the reactions evaluated
        for i, material in enumerate(materials):
            missing = self._unsupported(material)
            if missing:
                predictions[i] = failed(f"skipped: {', '.join(missing)} not in the elements", QUANTITIES)
            else:
                needed |= dict.fromkeys(material.settings["terms"])
        outcome = self._relax_everything(list(needed), simulator)
        for i, material in enumerate(materials):
            if not predictions[i]:
                terms = material.settings["terms"]
                relaxed = list(dict.fromkeys(d for name in terms for d in (*self._built_from(name), name)))
                predictions[i] = _prediction(terms, relaxed, outcome)
        return predictions

    def summarize(self, table: pd.DataFrame, model_name: str) -> dict[str, Any]:
        """Errors of the reaction energies, overall, per subset and per category.

        Args:
            table: Result of ``run``.
            model_name: The label used in ``run``.

        Returns:
            Counts (``n_not_converged``: with a relaxation that reached the step limit; ``n_moved``: with a
            heavy adsorbate atom that moved more than ``MOVED`` Å; ``n_rearranged``: with a bond between
            adsorbate atoms broken or formed), the errors (MAE, RMSE, mean signed error ME and
            number of reactions, eV) of all reactions and per subset and category, the ADS41 errors per
            adsorbed fragment (the error divided by the number of adsorbates the reaction forms, as Sharada et
            al. report them), and the wall time per stage (s).
        """
        status = table[f"status_{model_name}"].astype(str)
        ok = status.str.startswith(OK)
        error = pd.to_numeric(table[f"energy_{model_name}"], errors="coerce") - pd.to_numeric(table["energy_exp"])
        moved = pd.to_numeric(table[f"displacement_{model_name}"], errors="coerce") > MOVED
        # True/False, NaN for reactions without a prediction, or the strings "True"/"False" of a CSV file.
        rearranged = table[f"rearranged_{model_name}"].astype(str).str.lower().eq("true")
        summary: dict[str, Any] = {
            "n_reactions": len(table),
            "n_ok": int(ok.sum()),
            "n_skipped": int(status.str.startswith("skipped").sum()),
            "n_not_converged": int(status.str.contains("not converged").sum()),
            "n_moved": int((moved & ok).sum()),
            "n_rearranged": int((rearranged & ok).sum()),
            "all": error_statistics(error[ok]),
        }
        for subset, rows in table.groupby("subset", sort=False):
            part = {"all": error_statistics(error[rows.index][ok[rows.index]])}
            for category, members in rows.groupby("category", sort=False):
                part[str(category)] = error_statistics(error[members.index][ok[members.index]])
            if subset == "ADS41":
                per_adsorbate = error[rows.index] / rows["adsorbates"]
                part["per_adsorbate"] = error_statistics(per_adsorbate[ok[rows.index]])
            summary[str(subset)] = part
        summary["timings_s"] = dict(self.timings)
        return summary

    def _row(self, material: Material, prediction: dict[str, Any], model_name: str) -> dict[str, Any]:
        row = super()._row(material, prediction, model_name)
        for key in ("subset", "category", "mixed", "adsorbates"):
            row[key] = material.settings[key]
        return row

    def _relax_everything(self, names: Sequence[str], simulator: Simulator) -> dict[str, _Outcome]:
        """Relax the crystals, slabs, adsorbed slabs and molecules that the named structures need."""
        adsorbed, molecules = self.definitions["adsorbed"], self.definitions["molecules"]
        slab_names = list(
            dict.fromkeys(adsorbed[n]["slab"] if n in adsorbed else n for n in names if n not in molecules)
        )
        crystal_names = list(dict.fromkeys(self.definitions["slabs"][s]["crystal"] for s in slab_names))
        outcome: dict[str, _Outcome] = {}
        with self.stage("bulk relaxation"):
            crystals = self._relax_crystals(crystal_names, simulator, outcome)
        with self.stage("slab relaxation"):
            slabs = self._relax_slabs(slab_names, crystals, simulator, outcome)
        with self.stage("adsorbate relaxation"):
            self._relax_adsorbed(names, slabs, simulator, outcome)
        return outcome

    def _relax_crystals(
        self, names: Sequence[str], simulator: Simulator, outcome: dict[str, _Outcome]
    ) -> dict[str, Atoms]:
        """Relax the crystals (atoms and cell, keeping the space group); the relaxed ones by name."""
        starts = [self._crystal(name) for name in names]
        results = simulator.relax(starts, fmax=self.fmax, max_steps=self.max_steps, fix_symmetry=True)
        crystals = {}
        for name, result in zip(names, results, strict=True):
            outcome[name] = _Outcome.of(result)
            structure = outcome[name].structure
            if structure is not None:
                crystals[name] = to_ase_atoms(structure)
        return crystals

    def _relax_slabs(
        self,
        names: Sequence[str],
        crystals: Mapping[str, Atoms],
        simulator: Simulator,
        outcome: dict[str, _Outcome],
    ) -> dict[str, Atoms]:
        """Cut the slabs from the relaxed crystals and relax them.

        Returns:
            The relaxed slabs (with their tags, site positions and fixed atoms) by name.
        """
        ideal = {}
        for name in names:
            crystal = self.definitions["slabs"][name]["crystal"]
            if crystal in crystals:
                ideal[name] = build_slab(crystals[crystal], self.definitions["slabs"][name])
            else:
                outcome[name] = _Outcome.failed(f"relaxation of the {crystal} crystal: {outcome[crystal].error}")
        results = simulator.relax(list(ideal.values()), fmax=self.fmax, max_steps=self.max_steps, relax_cell=False)
        relaxed = {}
        for name, result in zip(ideal, results, strict=True):
            outcome[name] = _Outcome.of(result)
            structure = outcome[name].structure
            if structure is not None:
                relaxed[name] = ideal[name].copy()
                relaxed[name].positions = structure.cart_coords
        return relaxed

    def _relax_adsorbed(
        self, names: Sequence[str], slabs: Mapping[str, Atoms], simulator: Simulator, outcome: dict[str, _Outcome]
    ) -> None:
        """Place the adsorbates on the relaxed slabs; relax the adsorbed slabs and the molecules."""
        adsorbed, molecules = self.definitions["adsorbed"], self.definitions["molecules"]
        starts: dict[str, Atoms] = {}
        for name in names:
            if name in molecules:
                starts[name] = self._molecule(name)
            elif name in adsorbed:
                slab = adsorbed[name]["slab"]
                if slab in slabs:
                    starts[name] = add_adsorbates(slabs[slab], adsorbed[name]["adsorbates"])
                else:
                    outcome[name] = _Outcome.failed(f"relaxation of {slab}: {outcome[slab].error}")
        results = simulator.relax(list(starts.values()), fmax=self.fmax, max_steps=self.max_steps, relax_cell=False)
        for (name, start), result in zip(starts.items(), results, strict=True):
            outcome[name] = _Outcome.of(result)
            structure = outcome[name].structure
            if name in adsorbed and structure is not None:
                outcome[name].displacement = adsorbate_displacement(start, structure.cart_coords)
                outcome[name].rearranged = adsorbate_rearranged(start, structure.cart_coords)

    def _built_from(self, name: str) -> list[str]:
        """The relaxed structures a structure is built from: crystal (and slab) of slabs and adsorbed slabs."""
        if name in self.definitions["adsorbed"]:
            slab = self.definitions["adsorbed"][name]["slab"]
            return [*self._built_from(slab), slab]
        if name in self.definitions["slabs"]:
            return [self.definitions["slabs"][name]["crystal"]]
        return []

    def _crystal(self, name: str) -> Atoms:
        spec = self.definitions["crystals"][name]
        return bulk_crystal(spec["lattice"], spec["symbols"], spec["a"], spec.get("c"), spec.get("u"))

    def _molecule(self, name: str) -> Atoms:
        spec = self.definitions["molecules"][name]
        return molecule_in_box(spec["symbols"], spec["positions"], multiplicity=spec["multiplicity"])

    def _elements(self, name: str) -> set[str]:
        """The elements of a structure of the dataset."""
        if name in self.definitions["molecules"]:
            return set(self.definitions["molecules"][name]["symbols"])
        if name in self.definitions["adsorbed"]:
            entry = self.definitions["adsorbed"][name]
            return self._elements(entry["slab"]) | {s for group in entry["adsorbates"] for s in group["symbols"]}
        crystal = self.definitions["slabs"][name]["crystal"]
        return set(self.definitions["crystals"][crystal]["symbols"])

    def _unsupported(self, material: Material) -> list[str]:
        if self.elements is None:
            return []
        return sorted(set(material.settings["elements"]) - self.elements)


class _Outcome:
    """A relaxed structure and its energy, whether the relaxation converged, or the error that stopped it."""

    def __init__(
        self, structure: Structure | None, energy: float, *, converged: bool, error: str | None = None
    ) -> None:
        self.structure = structure
        self.energy = energy
        self.converged = converged
        self.error = error
        self.displacement = 0.0
        self.rearranged = False

    @classmethod
    def of(cls, result: RelaxResult) -> _Outcome:
        if result.error is not None or result.structure is None or not np.isfinite(result.energy):
            return cls.failed(result.error or "relaxation failed")
        return cls(result.structure, result.energy, converged=result.optimizer_converged)

    @classmethod
    def failed(cls, error: str) -> _Outcome:
        return cls(None, float("nan"), converged=False, error=error)


def _prediction(terms: Mapping[str, float], relaxed: Sequence[str], outcome: Mapping[str, _Outcome]) -> dict[str, Any]:
    """Reaction energy, largest adsorbate displacement, rearrangement and status of one reaction.

    ``relaxed`` names every structure the reaction depends on, crystals and slabs included.
    """
    for name in terms:
        if outcome[name].error is not None:
            return failed(f"{name}: {outcome[name].error}", QUANTITIES)
    energy = reaction_energy(terms, {name: outcome[name].energy for name in terms})
    displacement = max(outcome[name].displacement for name in terms)
    rearranged = any(outcome[name].rearranged for name in terms)
    unconverged = [name for name in relaxed if not outcome[name].converged]
    status = OK if not unconverged else f"{OK} (not converged: {', '.join(unconverged)})"
    return {"energy": energy, "displacement": displacement, "rearranged": rearranged, "status": status}
