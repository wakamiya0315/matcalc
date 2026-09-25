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
from pymatgen.io.ase import AseAtomsAdaptor

from matcalc import ASESimulator

from .helpers import SOFTENING_FACTOR, structure

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
    entries = [{"mp_id": "t-Cu", "formula": "Cu", "structure": structure("Cu1"), "heat_capacity": 24.4}]
    path = tmp_path / "phonon.json.gz"
    dumpfn(entries, path)
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
