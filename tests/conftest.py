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

    from ase import Atoms


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


def _emt_energies(structures: list) -> np.ndarray:
    energies = []
    for original in structures:
        atoms = original.copy()
        atoms.calc = EMT()
        energies.append(atoms.get_potential_energy())
    return np.array(energies)


def _xyz(atoms: Atoms, comment: str) -> str:
    rows = [f"{s} {x:.8f} {y:.8f} {z:.8f}" for s, (x, y, z) in zip(atoms.symbols, atoms.positions, strict=True)]
    return "\n".join([str(len(atoms)), comment, *rows])


@pytest.fixture
def ncia_dataset(tmp_path: Path) -> Path:
    """Packaged NCI Atlas sets in the published layout, with EMT curves of metal "complexes" as the reference.

    D442x10 has a Cu2 curve (group HBCNO without boron, reported as HCNO) and a BCu curve (HBCNO with boron,
    reported as Boron; EMT has no boron); IHB100x10 a charged Cu3 curve; R739x5 a repulsive Cu2 curve.
    """
    import zipfile

    from matcalc.properties.molecules import KCAL_PER_MOL
    from matcalc.structures import molecule_in_box

    long_scalings = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 1.5, 2.0]
    curves = {
        "D442x10": [
            ("1.01.01", "HBCNO", ["Cu", "Cu"], 2.4, long_scalings, 0),
            ("1.02.01", "HBCNO", ["B", "Cu"], 2.4, long_scalings, 0),
        ],
        "IHB100x10": [("01.001", "OHk-O", ["Cu", "Cu", "Cu"], 2.5, long_scalings, 1)],
        "R739x5": [("001.01", "HCNO", ["Cu", "Cu"], 1.8, [1.0, 1.05, 1.1, 1.15, 1.25], 0)],
    }
    for set_name, set_curves in curves.items():
        folder = f"NCIA_{set_name}"
        with zipfile.ZipFile(tmp_path / f"NCIA_{set_name}_github_package.zip", "w") as archive:
            names = ["# system names"]
            for curve, group, symbols, contact, scalings, charge in set_curves:
                # monomer A: the first atom; monomer B: the others, moved along x with the contact
                others = range(len(symbols) - 1)
                geometries = [
                    molecule_in_box(symbols, [[0.0, 0.0, 0.0]] + [[contact * scaling, 2.4 * k, 0.0] for k in others])
                    for scaling in scalings
                ]
                if "B" in symbols:
                    energies = np.arange(len(scalings), dtype=float)
                else:
                    energies = _emt_energies(geometries) / KCAL_PER_MOL - 0.1  # any zero will do
                for scaling, atoms, energy in zip(scalings, geometries, energies, strict=True):
                    point = f"{curve}_{round(scaling * 100):03d}"
                    names.append(f"{point}\t{symbols[0]} ... {''.join(symbols[1:])}")
                    header = (
                        f"charge={charge} charge_a={charge} charge_b=0 selection_a=1-1 selection_b=2-{len(symbols)} "
                        f"scaling={scaling:.2f} benchmark_Eint={energy:.10f} benchmark_unit=kcal/mol group={group}"
                    )
                    archive.writestr(f"{folder}/geometries/{point}.xyz", _xyz(atoms, header))
            archive.writestr(f"{folder}/{folder}_system_names.txt", "\n".join(names) + "\n")
    return tmp_path


@pytest.fixture
def conformer_dataset(tmp_path: Path) -> Path:
    """A zip archive in the layout of hutchisonlab/conformer-benchmark, with metal clusters as "molecules".

    ``neutral`` (3 conformers of Cu3): the reference is EMT itself. ``cation`` (charge 1, 4 conformers of
    Cu4): the reference relative energies are twice EMT's. ``pair`` (2 conformers): left out. ``gold`` (3
    conformers of Au3): the reference is EMT.
    """
    import zipfile

    from ase import units

    from matcalc.structures import molecule_in_box

    def chain(symbol: str, n: int, bond: float, stretch: float) -> list:
        return molecule_in_box([symbol] * n, [[bond * k * (1 + stretch * k), 0.3 * k * k, 0.0] for k in range(n)])

    molecules = {
        "neutral": ("Neutral_jobs", [chain("Cu", 3, 2.4, s) for s in (0.0, 0.05, 0.1)], 1.0, 0),
        "cation": ("CHG_jobs", [chain("Cu", 4, 2.5, s) for s in (0.08, 0.0, 0.03, 0.12)], 2.0, 1),
        "pair": ("Neutral_jobs", [chain("Cu", 2, 2.4, s) for s in (0.0, 0.1)], 1.0, 0),
        "gold": ("Neutral_jobs", [chain("Au", 3, 2.7, s) for s in (0.0, 0.05, 0.1)], 1.0, 0),
    }
    root = "conformer-benchmark-0123abc"
    path = tmp_path / "conformer-benchmark.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{root}/", "")
        lines, charged = [], []
        for name, (folder, conformers, factor, charge) in molecules.items():
            relative = _emt_energies(conformers) / units.Hartree
            energies = factor * (relative - relative[0]) - 1000.0  # Hartree
            for k, (atoms, energy) in enumerate(zip(conformers, energies, strict=True)):
                lines.append(f"{name} rmsd{k:03d}-opt.out.bz2 FINAL SINGLE POINT ENERGY     {energy:.12f}")
                archive.writestr(f"{root}/geometries/{folder}/{name}/rmsd{k:03d}-opt.xyz", _xyz(atoms, "x.gzmat"))
            if charge:
                charged.append(f"{name} CHARGE={charge} ")
        archive.writestr(f"{root}/energies/ccsdt.txt", "\n".join(lines) + "\n")
        archive.writestr(f"{root}/geometries/CHG-charges.txt", "\n".join(charged) + "\n")
    return path


@pytest.fixture
def rdb7_dataset(tmp_path: Path) -> Path:
    """An archive of Molpro outputs in RDB7's layout, with Cu and Au chains as "reactions" and EMT as reference.

    ``000000``: Cu3 to one product; ``000001``: Cu3 to Cu2 + Cu; ``000002``: an Au3 reaction. The archive also
    holds a macOS metadata file, as the published one does.
    """
    import io
    import tarfile

    from ase import units

    from matcalc.structures import molecule_in_box

    def chain(symbol: str, n: int, bonds: list[float]) -> list:
        return molecule_in_box([symbol] * n, [[sum(bonds[:k]), 0.2 * k * k, 0.0] for k in range(n)])

    def log(atoms: Atoms, energy: float) -> str:
        rows = [
            f" {s}  {x:14.8f}{y:14.8f}{z:14.8f}" for s, (x, y, z) in zip(atoms.symbols, atoms.positions, strict=True)
        ]
        return "\n".join(
            [
                " ***,name",
                " memory,1152,m;",
                " geometry={angstrom;",
                *rows[:-1],
                rows[-1] + "}",
                " ",
                " basis=cc-pvdz-f12",
                " {hf;",
                " maxit,300;",
                " wf,spin=0,charge=0;}",
                " ccsd(t)-f12;",
                "",
                f" !CCSD(T)-F12a total energy          {energy:.12f}",
                f" !CCSD(T)-F12b total energy          {energy + 0.01:.12f}",
                "",
            ]
        )

    reactions = {
        "000000": {
            "r": chain("Cu", 3, [2.3, 2.3]),
            "ts": chain("Cu", 3, [2.3, 3.0]),
            "p": [chain("Cu", 3, [2.2, 2.6])],
        },
        "000001": {
            "r": chain("Cu", 3, [2.4, 2.35]),
            "ts": chain("Cu", 3, [2.4, 3.4]),
            "p": [chain("Cu", 2, [2.3]), molecule_in_box(["Cu"], [[0.0, 0.0, 0.0]])],
        },
        "000002": {
            "r": chain("Au", 3, [2.6, 2.6]),
            "ts": chain("Au", 3, [2.6, 3.3]),
            "p": [chain("Au", 3, [2.5, 2.8])],
        },
    }
    path = tmp_path / "ccsdtf12_dz.tar.gz"
    with tarfile.open(path, "w:gz") as archive:

        def add(name: str, text: str) -> None:
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))

        for reaction, states in reactions.items():
            folder = f"qm_logs/rxn{reaction}"
            products = states["p"]
            names = (
                [f"p{reaction}.log"] if len(products) == 1 else [f"p{reaction}_{k}.log" for k in range(len(products))]
            )
            for name, atoms in [(f"r{reaction}.log", states["r"]), (f"ts{reaction}.log", states["ts"])]:
                add(f"{folder}/{name}", log(atoms, _emt_energies([atoms])[0] / units.Hartree - 100.0))
            for name, atoms in zip(names, products, strict=True):
                add(f"{folder}/{name}", log(atoms, _emt_energies([atoms])[0] / units.Hartree - 100.0 / len(products)))
        add("qm_logs/rxn000000/._r000000.log", "Mac OS X metadata")
    return path
