"""Shared fixtures: tiny benchmark datasets that ASE's EMT potential can handle.

EMT (effective medium theory) is fast and deterministic and supports Al, Cu, Ag, Au, Ni, Pd and Pt,
so the benchmark pipelines can be tested offline on a CPU. The DFT values in these datasets are
made up; the tests check the machinery, not the physics of EMT.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from monty.serialization import dumpfn
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor

from matcalc import ASESimulator

from .helpers import FCC_PRIMITIVE, SOFTENING_FACTOR, phonon_entry, structure

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def emt_simulator() -> ASESimulator:
    return ASESimulator(EMT(), show_progress=False)


@pytest.fixture
def elasticity_dataset(tmp_path: Path) -> Path:
    entries = [
        {
            "mp_id": "t-Cu",
            "formula": "Cu",
            "structure": structure("Cu"),
            "bulk_modulus_vrh": 140.0,
            "shear_modulus_vrh": 48.0,
        },
        {
            "mp_id": "t-NiAl",
            "formula": "NiAl",
            "structure": structure("NiAl"),
            "bulk_modulus_vrh": 160.0,
            "shear_modulus_vrh": 70.0,
        },
    ]
    path = tmp_path / "elasticity.json.gz"
    dumpfn(entries, path)
    return path


@pytest.fixture
def phonon_dataset(tmp_path: Path) -> Path:
    """fcc Cu (conventional cell, 2 x 2 x 2 supercell) and B2 NiAl (3 x 3 x 3), the latter marked unstable."""
    entries = [
        phonon_entry("t-Cu", "Cu", [[2, 0, 0], [0, 2, 0], [0, 0, 2]], FCC_PRIMITIVE, 24.4, stable=True),
        phonon_entry("t-NiAl", "NiAl", [[3, 0, 0], [0, 3, 0], [0, 0, 3]], None, 45.0, stable=False),
    ]
    path = tmp_path / "phonon.json.gz"
    dumpfn({"entries": entries}, path)
    return path


@pytest.fixture
def equilibrium_dataset(tmp_path: Path) -> Path:
    entries = [
        {
            "material_id": "t-Cu3Au",
            "formula": "Cu3Au",
            "structure": structure("Cu3Au"),
            "formation_energy_per_atom": -0.05,
        },
        {
            "material_id": "t-CuAu",
            "formula": "CuAu",
            "structure": structure("CuAu"),
            "formation_energy_per_atom": -0.06,
        },
    ]
    path = tmp_path / "equilibrium.json.gz"
    dumpfn(entries, path)
    return path


DISCOVERY_ENTRIES = (
    # id, formula, e_form DFT, e_above_hull DFT, MP2020 correction per atom, unique prototype
    ("wbm-t-1", "Cu", 0.0, 0.0, 0.0, True),
    ("wbm-t-2", "Cu3Au", -0.07, -0.01, 0.0, True),
    ("wbm-t-3", "CuAu", -0.02, 0.04, -0.01, True),
    ("wbm-t-4", "NiAl", -0.60, 0.02, 0.0, False),
)
"""Made-up DFT values of the tiny Discovery dataset."""


@pytest.fixture
def discovery_dataset(tmp_path: Path) -> Path:
    """The Matbench Discovery files for four EMT crystals: rattled and strained starts, EMT-relaxed "DFT"."""
    import gzip
    import json

    import pandas as pd

    moyopy = pytest.importorskip("moyopy")
    from moyopy.interface import MoyoAdapter

    from matcalc.benchmarks.discovery import WBM_FILES

    rows, initial, relaxed = [], [], []
    for k, (material_id, formula, e_form, e_hull, correction, unique) in enumerate(DISCOVERY_ENTRIES):
        dft = structure(formula)
        start = dft.copy()
        start.perturb(0.1, seed=k)
        start.scale_lattice(start.volume * 1.04)
        initial.append({"material_id": material_id, "initial_structure": start.as_dict()})
        entry = ComputedStructureEntry(dft, 0.0).as_dict()
        relaxed.append({"material_id": material_id, "computed_structure_entry": entry})
        rows.append(
            {
                "material_id": material_id,
                "formula": formula,
                "n_sites": len(dft),
                "e_correction_per_atom_mp2020": correction,
                "e_form_per_atom_mp2020_corrected": e_form,
                "e_above_hull_mp2020_corrected_ppd_mp": e_hull,
                "unique_prototype": unique,
            }
        )
    pd.DataFrame(rows).to_csv(tmp_path / WBM_FILES["summary"].name, index=False)
    for key, entries in (("initial_structures", initial), ("dft_entries", relaxed)):
        with gzip.open(tmp_path / WBM_FILES[key].name, "wt") as f:
            f.writelines(json.dumps(entry) + "\n" for entry in entries)
    references = {}
    for element in ("Cu", "Au", "Ni", "Al"):
        atoms = bulk(element, "fcc", a={"Cu": 3.61, "Au": 4.08, "Ni": 3.52, "Al": 4.05}[element])
        atoms.calc = EMT()
        references[element] = {"energy": atoms.get_potential_energy(), "composition": {element: 1.0}}
    with gzip.open(tmp_path / WBM_FILES["elemental_references"].name, "wt") as f:
        json.dump(references, f)
    for key, symprec in (("dft_symmetry_1e-5", 1e-5), ("dft_symmetry_1e-2", 1e-2)):
        symmetry = []
        for row in rows:
            data = moyopy.MoyoDataset(MoyoAdapter.from_py_obj(structure(row["formula"])), symprec=symprec)
            symmetry.append(
                {"material_id": row["material_id"], "spg_num": data.number, "n_sym_ops": data.operations.num_operations}
            )
        pd.DataFrame(symmetry).to_csv(tmp_path / WBM_FILES[key].name, index=False)
    return tmp_path


@pytest.fixture
def kappa_dataset(tmp_path: Path) -> Path:
    """fcc Cu relaxed with EMT, whose EMT conductivity is the "DFT" reference.

    A second copy claims the wrong space group, so the benchmark must censor it.
    """
    pytest.importorskip("phono3py")
    from ase.constraints import FixSymmetry
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE

    from matcalc.properties.thermal_conductivity import (
        fc2_supercells,
        fc3_supercells,
        harmonic_frequencies,
        make_phono3py,
        thermal_conductivity,
    )

    cell = bulk("Cu", "fcc", a=3.6, cubic=True)
    cell.calc = EMT()
    cell.set_constraint(FixSymmetry(cell))
    FIRE(FrechetCellFilter(cell), logfile=None).run(fmax=1e-6, steps=500)
    cell.set_constraint()
    settings = {
        "fc2_supercell": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        "fc3_supercell": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        "q_point_mesh": [9, 9, 9],
        "primitive_matrix": FCC_PRIMITIVE,
    }
    phono3py = make_phono3py(
        cell,
        settings["fc2_supercell"],
        settings["fc3_supercell"],
        settings["q_point_mesh"],
        primitive_matrix=FCC_PRIMITIVE,
    )

    def forces(supercells: list) -> list:
        out = []
        for atoms in supercells:
            atoms.calc = EMT()
            out.append(atoms.get_forces())
        return out

    harmonic_frequencies(phono3py, forces(fc2_supercells(phono3py)))
    conductivity = thermal_conductivity(phono3py, forces(fc3_supercells(phono3py)))
    entries = [
        {
            "mp_id": material_id,
            "formula": "Cu",
            "space_group": space_group,
            "lattice": cell.cell[:].tolist(),
            "species": cell.get_chemical_symbols(),
            "positions": cell.positions.tolist(),
            **settings,
            "kappa": conductivity.kappa_average,
            "mode_kappa": conductivity.mode_kappa.tolist(),
        }
        for material_id, space_group in (("mp-t-1", 225), ("mp-t-2", 221))
    ]
    path = tmp_path / "kappa.json.gz"
    dumpfn({"entries": entries}, path)
    return path


@pytest.fixture
def softening_dataset(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    frames = {}
    for k in range(3):
        atoms = bulk("Cu", "fcc", a=3.62, cubic=True).repeat(2)
        atoms.positions += rng.normal(scale=0.08, size=atoms.positions.shape)
        atoms.calc = EMT()
        frames[f"t-Cu-{k}"] = {
            "structure": AseAtomsAdaptor.get_structure(atoms),
            "vasp_f": (atoms.get_forces() * SOFTENING_FACTOR).tolist(),
        }
    path = tmp_path / "softening.json.gz"
    dumpfn({"t-Cu": frames}, path)
    return path


@pytest.fixture
def diatomics_dataset(tmp_path: Path) -> Path:
    """EMT curves of Al2 and Cu2 as the "PBE" reference (40 separations from 0.8 r_cov to 6 Å, as in Matbench
    Discovery's grid), and a Ni2 curve with ±1 eV steps that is too rough to score against."""
    import gzip
    import json

    from ase.data import atomic_numbers, covalent_radii

    from matcalc.properties.diatomics import dimers

    curves = {}
    for element in ("Al", "Cu", "Ni"):
        distances = np.geomspace(0.8 * covalent_radii[atomic_numbers[element]], 6.0, 40)
        energies, forces = [], []
        for atoms in dimers(element, distances):
            atoms.calc = EMT()
            energies.append(atoms.get_potential_energy())
            forces.append(atoms.get_forces().tolist())
        if element == "Ni":
            energies = [e + (1.0 if k % 2 else 0.0) for k, e in enumerate(energies)]
        curves[f"{element}-{element}"] = {"distances": distances.tolist(), "energies": energies, "forces": forces}
    path = tmp_path / "diatomics-dft.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump({"PBE": curves, "r2SCAN": {}}, f)
    return path
