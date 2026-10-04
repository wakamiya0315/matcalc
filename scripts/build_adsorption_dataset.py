"""Build the dataset of the Adsorption benchmark (``src/matcalc/benchmarks/data/adsorption.json``).

The benchmark compares the reaction energies of molecules adsorbing on surfaces with experiment:

- ADS41: 41 adsorption energies on transition-metal surfaces (S. Mallikarjun Sharada, R. K. B. Karlsson,
  Y. Maimaiti, J. Voss, T. Bligaard, Phys. Rev. B 100, 035439 (2019), Table I), 39 of them from the CE39
  database (J. Wellendorff et al., Surf. Sci. 640, 36 (2015)): experimental energies minus the zero-point
  energy change computed with PBE.
- Surf13: the 13 molecules on MgO(001), rutile TiO2(110) and anatase TiO2(101) of the Surf13 set (B. X. Shi
  et al., Nat. Chem. 17, 1688 (2025)), with the experimental adsorption enthalpies of the paper (SI Table 32)
  minus the zero-point, thermal and -RT contributions of its DFT ensemble (SI Table 30).

Every adsorbed configuration is fixed here (the benchmark searches no sites). The configurations were
chosen from the structures of the reference studies:

- ADS41: sites, coverages and slabs of Wellendorff et al. (SI Fig. S1, Table 3) and Sharada et al.
  (Appendix A), with the later corrections of R. B. Araujo, G. L. S. Rodrigues, E. C. dos Santos,
  L. G. M. Pettersson, Nat. Commun. 13, 6853 (2022): water on Pt(111) as the 2/3 ML hexagonal network, CH
  and CH3 on Pt(111) at 1/16 ML, benzene and cyclohexene on five-layer Pt(111) slabs with four relaxed layers,
  benzene in the chemisorbed geometry of their SI (Supplementary Note 1). Sites on (111) surfaces follow
  Araujo et al. Table 3 where Wellendorff's figures do not tell fcc from hcp hollows. The geometries are
  built here from these choices (heights from typical DFT bond lengths) and were checked against the
  published figures.
- Surf13: the relaxed revPBE-D4 structures that Shi et al. published (GitHub benshi97/Data_autoSKZCAM, CC BY
  4.0), downloaded once into ``temp/``; the adsorbate atoms keep their positions relative to the surface
  atom they bind to. CO2 on MgO(001) is the chemisorbed (carbonate) state that Shi et al. assign to the
  experiment (Surf13 itself holds the physisorbed state, whose measured enthalpy they question).

Run from the repository root::

    python scripts/build_adsorption_dataset.py src/matcalc/benchmarks/data/adsorption.json --images temp/adsorption

``--images`` draws every adsorbed structure (top and side view) for checking the configurations by eye.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.geometry import find_mic
from ase.io import read
from ase.neighborlist import neighbor_list

from matcalc.datasets import RemoteFile, download_file
from matcalc.surfaces import BOND_LENGTH, add_adsorbates, build_slab, bulk_crystal

# ----------------------------------------------------------------------------------------------------------
# Crystals: experimental lattice constants (room temperature). They are only the starting point of the
# MLIP's own bulk relaxation.
# ----------------------------------------------------------------------------------------------------------

CRYSTALS: dict[str, dict[str, Any]] = {
    "Ag": {"lattice": "fcc", "symbols": ["Ag"], "a": 4.086},
    "Au": {"lattice": "fcc", "symbols": ["Au"], "a": 4.078},
    "Cu": {"lattice": "fcc", "symbols": ["Cu"], "a": 3.615},
    "Ir": {"lattice": "fcc", "symbols": ["Ir"], "a": 3.839},
    "Ni": {"lattice": "fcc", "symbols": ["Ni"], "a": 3.524},
    "Pd": {"lattice": "fcc", "symbols": ["Pd"], "a": 3.891},
    "Pt": {"lattice": "fcc", "symbols": ["Pt"], "a": 3.924},
    "Rh": {"lattice": "fcc", "symbols": ["Rh"], "a": 3.803},
    "Co": {"lattice": "hcp", "symbols": ["Co"], "a": 2.507, "c": 4.069},
    "Ru": {"lattice": "hcp", "symbols": ["Ru"], "a": 2.706, "c": 4.282},
    # Oxides: the experimental references of Shi et al. SI Tables 20-22; internal coordinates u of
    # rutile (O at (u, u, 0)) and anatase (O at (0, 0, u)) from neutron diffraction.
    "MgO": {"lattice": "rocksalt", "symbols": ["Mg", "O"], "a": 4.214},
    "TiO2 (rutile)": {"lattice": "rutile", "symbols": ["Ti", "O"], "a": 4.587, "c": 2.954, "u": 0.3048},
    "TiO2 (anatase)": {"lattice": "anatase", "symbols": ["Ti", "O"], "a": 3.782, "c": 9.502, "u": 0.2081},
}

# ----------------------------------------------------------------------------------------------------------
# Slabs. Metals: four layers with the bottom two fixed (Wellendorff et al., Sharada et al.); benzene and
# cyclohexene on Pt(111): five layers with the bottom one fixed (Araujo et al.). Oxides as in Shi et al.:
# MgO(001) four layers (two stacked conventional cells) with the bottom two fixed; rutile TiO2(110) p(4x2),
# five O-Ti-O trilayers with the bottom three fixed; anatase TiO2(101) 3x1, four O-Ti-O blocks with the bottom
# one fixed.
# ----------------------------------------------------------------------------------------------------------


def _metal(crystal: str, facet: str, size: int | tuple[int, int], layers: int = 4, fixed: int = 2) -> dict[str, Any]:
    n1, n2 = (size, size) if isinstance(size, int) else size
    return {"crystal": crystal, "facet": facet, "size": [n1, n2], "layers": layers, "fixed_layers": fixed}


SLABS: dict[str, dict[str, Any]] = {
    "Pt(111) 2x2": _metal("Pt", "111", 2),
    "Pt(111) 3x2": _metal("Pt", "111", (3, 2)),
    "Pt(111) 3x3": _metal("Pt", "111", 3),
    "Pt(111) 4x4": _metal("Pt", "111", 4),
    "Pt(111) 3x3 5 layers": _metal("Pt", "111", 3, layers=5, fixed=1),
    "Pd(111) 2x2": _metal("Pd", "111", 2),
    "Pd(100) 2x2": _metal("Pd", "100", 2),
    "Ni(111) 2x2": _metal("Ni", "111", 2),
    "Ni(100) 2x2": _metal("Ni", "100", 2),
    "Rh(111) 2x2": _metal("Rh", "111", 2),
    "Rh(100) 2x2": _metal("Rh", "100", 2),
    "Ir(111) 2x2": _metal("Ir", "111", 2),
    "Cu(111) 2x2": _metal("Cu", "111", 2),
    "Cu(111) 3x3": _metal("Cu", "111", 3),
    "Cu(100) 2x2": _metal("Cu", "100", 2),
    "Ag(111) 3x3": _metal("Ag", "111", 3),
    "Au(111) 3x3": _metal("Au", "111", 3),
    "Co(0001) 2x2": _metal("Co", "0001", 2),
    "Ru(0001) 2x2": _metal("Ru", "0001", 2),
    "MgO(001) 2x2": {
        "crystal": "MgO",
        "in_plane": [[1, 0, 0], [0, 1, 0]],
        "stacking": [0, 0, 1],
        "size": [2, 2],
        "layers": 2,
        "fixed_layers": 1,
        "note": "a layer is a conventional cell (two atomic layers): four atomic layers, the bottom two fixed",
    },
    "TiO2 rutile(110) p(4x2)": {
        "crystal": "TiO2 (rutile)",
        "in_plane": [[-1, 1, 0], [0, 0, 1]],
        "stacking": [0, 1, 0],
        "size": [2, 4],
        "layers": 5,
        "fixed_layers": 3,
        "note": "a layer is an O-Ti-O trilayer; x along [-110], y along [001] (the rows of five-fold Ti)",
    },
    "TiO2 anatase(101) 3x1": {
        "crystal": "TiO2 (anatase)",
        "in_plane": [[1, 0, -1], [0, 1, 0]],
        "stacking": [0, 0, 1],
        "size": [1, 3],
        "layers": 4,
        "fixed_layers": 1,
        "note": "a layer is an O-Ti-O block (12 atoms per surface cell); x along [10-1], y along [010]",
    },
}

# ----------------------------------------------------------------------------------------------------------
# Molecules (starting geometries of the gas-phase relaxations; the MLIP relaxes them)
# ----------------------------------------------------------------------------------------------------------

# Cyclohexene (half-chair): RDKit 2026.03 ETKDG embedding + MMFF94 (seed 42); C1=C2 are atoms 0 and 1.
CYCLOHEXENE = [
    ("C", (0.8351, 1.1725, 0.6078)),
    ("C", (-0.4394, 1.4668, 0.3111)),
    ("C", (-1.4000, 0.4666, -0.2634)),
    ("C", (-0.8922, -0.9747, -0.1712)),
    ("C", (0.5860, -1.0679, -0.5399)),
    ("C", (1.4433, -0.1780, 0.3639)),
    ("H", (1.4759, 1.9356, 1.0424)),
    ("H", (-0.8154, 2.4708, 0.4916)),
    ("H", (-2.3599, 0.5407, 0.2602)),
    ("H", (-1.5846, 0.7279, -1.3122)),
    ("H", (-1.0343, -1.3487, 0.8509)),
    ("H", (-1.4865, -1.6179, -0.8300)),
    ("H", (0.7178, -0.7630, -1.5860)),
    ("H", (0.9283, -2.1064, -0.4665)),
    ("H", (2.4326, -0.0553, -0.0913)),
    ("H", (1.5934, -0.6692, 1.3327)),
]


def _naphthalene() -> Atoms:
    """Planar naphthalene: two regular hexagons (C-C 1.40 Å) sharing the bond along y, long axis along x."""
    side, ch = 1.40, 1.08
    carbons, hydrogens = [(0.0, side / 2), (0.0, -side / 2)], []
    for sign in (-1, 1):
        centre = np.array([sign * side * math.sqrt(3) / 2, 0.0])
        for angle in (90, 150, 210, 270):
            direction = np.array([-sign * math.cos(math.radians(angle)), math.sin(math.radians(angle))])
            carbons.append(tuple(centre + side * direction))
            hydrogens.append(tuple(centre + (side + ch) * direction))
    return Atoms("C10H8", positions=[(*xy, 0.0) for xy in carbons + hydrogens])


def _methyl_iodide() -> Atoms:
    """CH3I with I below C (r(C-I) 2.136 Å, r(C-H) 1.085 Å, angle H-C-I 107.7°; experiment)."""
    return Atoms("CI", positions=[(0, 0, 0), (0, 0, -2.136)]) + _umbrella("H", (0, 0, 0), 1.085, 107.7)


def _diiodomethane() -> Atoms:
    """CH2I2 (r(C-I) 2.123 Å, angle I-C-I 113.97°, r(C-H) 1.078 Å, angle H-C-H 111.8°; experiment)."""
    half_i, half_h = math.radians(113.97 / 2), math.radians(111.8 / 2)
    return Atoms(
        "CI2H2",
        positions=[
            (0, 0, 0),
            (2.123 * math.sin(half_i), 0, -2.123 * math.cos(half_i)),
            (-2.123 * math.sin(half_i), 0, -2.123 * math.cos(half_i)),
            (0, 1.078 * math.sin(half_h), 1.078 * math.cos(half_h)),
            (0, -1.078 * math.sin(half_h), 1.078 * math.cos(half_h)),
        ],
    )


def _umbrella(
    symbol: str, centre: tuple, bond: float, angle_to_down: float, azimuths: tuple[float, ...] = (-90, 30, 150)
) -> Atoms:
    """Three atoms bonded to ``centre``, at ``angle_to_down`` degrees from the downward direction (-z)."""
    theta = math.radians(180.0 - angle_to_down)  # angle from +z
    return Atoms(
        symbol * 3,
        positions=[
            np.asarray(centre, dtype=float)
            + bond
            * np.array(
                [
                    math.sin(theta) * math.cos(math.radians(p)),
                    math.sin(theta) * math.sin(math.radians(p)),
                    math.cos(theta),
                ]
            )
            for p in azimuths
        ],
    )


MOLECULES: dict[str, tuple[Atoms, int]] = {
    "H2": (molecule("H2"), 1),
    "O2": (molecule("O2"), 3),
    "I2": (Atoms("I2", positions=[(0, 0, 0), (0, 0, 2.666)]), 1),
    "CO": (molecule("CO"), 1),
    "NO": (molecule("NO"), 2),
    "CO2": (molecule("CO2"), 1),
    "N2O": (molecule("N2O"), 1),
    "H2O": (molecule("H2O"), 1),
    "NH3": (molecule("NH3"), 1),
    "CH4": (molecule("CH4"), 1),
    "CH3OH": (molecule("CH3OH"), 1),
    "CH3I": (_methyl_iodide(), 1),
    "CH2I2": (_diiodomethane(), 1),
    "C2H4": (molecule("C2H4"), 1),
    "C2H6": (molecule("C2H6"), 1),
    "C3H8": (molecule("C3H8"), 1),
    "C4H10": (molecule("trans-butane"), 1),
    "C6H6": (molecule("C6H6"), 1),
    "C6H10": (Atoms([s for s, _ in CYCLOHEXENE], positions=[p for _, p in CYCLOHEXENE]), 1),
    "C10H8": (_naphthalene(), 1),
}

# ----------------------------------------------------------------------------------------------------------
# ADS41 adsorbates
# ----------------------------------------------------------------------------------------------------------

# Adsorbate-metal bond lengths (Å), typical of DFT (PBE/RPBE) geometries; they set the starting heights.
BONDS = {
    ("C", "Co", "ontop"): 1.75,
    ("C", "Cu", "ontop"): 1.85,
    ("C", "Ir", "ontop"): 1.86,
    ("C", "Ni", "fcc"): 1.95,
    ("C", "Pd", "bridge"): 1.98,
    ("C", "Pd", "fcc"): 2.07,
    ("C", "Pt", "ontop"): 1.85,
    ("C", "Rh", "ontop"): 1.83,
    ("C", "Ru", "ontop"): 1.92,
    ("N", "Pd", "hollow"): 2.15,
    ("N", "Pd", "fcc"): 2.05,
    ("N", "Pt", "fcc"): 2.07,
    ("N", "Ni", "hollow"): 1.85,
    ("O", "Ni", "hollow"): 1.96,
    ("O", "Ni", "fcc"): 1.86,
    ("O", "Pt", "fcc"): 2.05,
    ("O", "Rh", "hollow"): 2.08,
    ("H", "Ni", "hollow"): 1.85,
    ("H", "Ni", "fcc"): 1.71,
    ("H", "Pd", "fcc"): 1.80,
    ("H", "Pt", "fcc"): 1.86,
    ("H", "Rh", "fcc"): 1.85,
    ("I", "Pt", "fcc"): 2.75,
}


def nearest_neighbour(crystal: str) -> float:
    """Nearest-neighbour distance of a metal at its experimental lattice constant (Å)."""
    spec = CRYSTALS[crystal]
    return spec["a"] / math.sqrt(2) if spec["lattice"] == "fcc" else spec["a"]


def height(symbol: str, crystal: str, site: str, bond: float | None = None) -> float:
    """Height above the top layer at which an atom on ``site`` is ``bond`` Å from its nearest metal atoms."""
    bond = bond if bond is not None else BONDS[symbol, crystal, site]
    a_nn = nearest_neighbour(crystal)
    lateral = {"ontop": 0.0, "bridge": a_nn / 2, "fcc": a_nn / math.sqrt(3), "hcp": a_nn / math.sqrt(3)}
    lateral["hollow"] = a_nn / math.sqrt(2)
    return math.sqrt(bond**2 - lateral[site] ** 2)


def group(site: str | dict, atoms: Atoms) -> dict[str, Any]:
    """An adsorbate group: atoms at offsets (Å) from an anchor."""
    anchor = {"site": site} if isinstance(site, str) else site
    return {
        "anchor": anchor,
        "symbols": atoms.get_chemical_symbols(),
        "positions": np.round(atoms.positions, 4).tolist(),
    }


def point(symbol: str, z: float) -> Atoms:
    """One atom ``z`` Å above the anchor."""
    return Atoms(symbol, positions=[(0.0, 0.0, z)])


def upright(symbols: str, z: float, bond: float) -> Atoms:
    """A diatomic standing on its first atom."""
    return Atoms(symbols, positions=[(0.0, 0.0, z), (0.0, 0.0, z + bond)])


def frame(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Orthonormal frame (columns): ``first``, the part of ``second`` normal to it, and their cross product."""
    e1 = first / np.linalg.norm(first)
    e2 = second - np.dot(second, e1) * e1
    e2 /= np.linalg.norm(e2)
    return np.column_stack([e1, e2, np.cross(e1, e2)])


def rotated(atoms: Atoms, first: np.ndarray, second: np.ndarray, to_first: np.ndarray, to_second: np.ndarray) -> Atoms:
    """``atoms`` rotated rigidly: ``first`` along ``to_first``, and ``second`` towards ``to_second``."""
    rotation = frame(np.asarray(to_first, float), np.asarray(to_second, float)) @ frame(first, second).T
    turned = atoms.copy()
    turned.positions = atoms.positions @ rotation.T
    return turned


def raised(
    atoms: Atoms, *, lowest: float | None = None, reference: np.ndarray | None = None, z: float | None = None
) -> Atoms:
    """``atoms`` translated: in-plane so that ``reference`` (default: centroid) is on the anchor, and in height so
    that the lowest atom is at ``lowest`` or ``reference`` at ``z``.
    """
    moved = atoms.copy()
    ref = atoms.positions.mean(axis=0) if reference is None else np.asarray(reference, float)
    moved.positions[:, :2] -= ref[:2]
    moved.positions[:, 2] += (lowest - atoms.positions[:, 2].min()) if lowest is not None else (z - ref[2])
    return moved


def ethylidyne(z: float) -> Atoms:
    """CCH3 standing on its first carbon (C-C 1.50 Å, C-H 1.09 Å, tetrahedral methyl)."""
    return Atoms("CC", positions=[(0, 0, z), (0, 0, z + 1.50)]) + _umbrella("H", (0, 0, z + 1.50), 1.09, 109.5)


def methyl(z: float) -> Atoms:
    """CH3 bound through C, hydrogens pointing up (tetrahedral)."""
    return Atoms("C", positions=[(0, 0, z)]) + _umbrella("H", (0, 0, z), 1.09, 109.5)


def methylidyne(z: float) -> Atoms:
    """CH bound through C, H pointing up."""
    return Atoms("CH", positions=[(0, 0, z), (0, 0, z + 1.09)])


def ammonia(z: float) -> Atoms:
    """NH3 bound through N (N-H 1.02 Å, angle H-N-surface 112°), hydrogens pointing up."""
    return Atoms("N", positions=[(0, 0, z)]) + _umbrella("H", (0, 0, z), 1.02, 112.0, azimuths=(0, 120, 240))


def methyl_iodide(z: float) -> Atoms:
    """CH3I standing on I (the molecular state of Wellendorff et al. Fig. S1, system 25)."""
    gas = _methyl_iodide()
    return raised(gas, reference=gas.positions[1], z=z)


def methanol(z: float, tilt: float = 45.0) -> Atoms:
    """CH3OH bound through O: C-O tilted by ``tilt`` degrees from the normal towards -x, the hydroxyl H in the
    same vertical plane on the +x side (Wellendorff et al. Fig. S1, system 26).
    """
    gas = molecule("CH3OH")  # C, O, then the H atoms (the hydroxyl H is the one nearest O)
    o, c = gas.positions[1], gas.positions[0]
    h = min(gas.positions[2:], key=lambda p: np.linalg.norm(p - o))
    t = math.radians(tilt)
    turned = rotated(gas, c - o, h - o, (-math.sin(t), 0.0, math.cos(t)), (1.0, 0.0, 0.0))
    return raised(turned, reference=turned.positions[1], z=z)


def methane_tripod(crystal: str, lowest: float) -> Atoms:
    """CH4 with three H down (C3v), its lower H pointing at the three metal atoms around an fcc site
    (Wellendorff et al. Fig. S1, system 28: one H up).
    """
    gas = molecule("CH4")
    up = gas.positions[1] - gas.positions[0]
    a_nn = nearest_neighbour(crystal)
    # From the fcc site, the top-layer atoms are at azimuths 90, 210 and 330 degrees (ASE's fcc111 frame).
    towards_atom = np.array([math.cos(math.radians(90)), math.sin(math.radians(90)), 0.0]) * a_nn
    turned = rotated(gas, up, gas.positions[2] - gas.positions[0], (0, 0, 1), towards_atom)
    return raised(turned, reference=turned.positions[0], lowest=lowest)


def flat_alkane(name: str, direction_deg: float, lowest: float) -> Atoms:
    """An n-alkane lying flat: its carbon plane parallel to the surface (for ethane, the C-C axis horizontal with
    one methyl H pointing down), the end-to-end carbon axis along ``direction_deg`` from x.
    """
    gas = molecule(name)
    carbons = [i for i, s in enumerate(gas.get_chemical_symbols()) if s == "C"]
    bonded = [[j for j in carbons if j != i and gas.get_distance(i, j) < 1.7] for i in carbons]
    ends = [i for i, partners in zip(carbons, bonded, strict=True) if len(partners) == 1]
    axis = gas.positions[ends[-1]] - gas.positions[ends[0]]
    if len(carbons) == 2:  # ethane: rotate about C-C so that one H of the first carbon points down
        h = next(i for i in range(len(gas)) if gas[i].symbol == "H" and gas.get_distance(i, ends[0]) < 1.2)
        second, target_second = gas.positions[h] - gas.positions[ends[0]], np.array([0.0, 0.0, -1.0])
    else:  # the carbon plane horizontal: an inner carbon, seen from the centroid, stays in the plane
        centre = gas.positions[carbons].mean(axis=0)
        inner = next(i for i, partners in zip(carbons, bonded, strict=True) if len(partners) == 2)
        second = gas.positions[inner] - centre
        target_second = np.array([-math.sin(math.radians(direction_deg)), math.cos(math.radians(direction_deg)), 0.0])
    target_axis = np.array([math.cos(math.radians(direction_deg)), math.sin(math.radians(direction_deg)), 0.0])
    turned = rotated(gas, axis, second, target_axis, target_second)
    centroid = turned.positions[[i for i in range(len(turned)) if turned[i].symbol == "C"]].mean(axis=0)
    return raised(turned, reference=centroid, lowest=lowest)


def flat_benzene(z: float) -> Atoms:
    """Benzene lying flat, two C-H bonds along y (Wellendorff et al. Fig. S1, systems 33-35)."""
    gas = molecule("C6H6")
    carbons = gas.positions[:6]
    normal = np.cross(carbons[1] - carbons[0], carbons[2] - carbons[0])
    turned = rotated(gas, normal, carbons[0] - carbons.mean(axis=0), (0, 0, 1), (0, 1, 0))
    return raised(turned, reference=turned.positions[:6].mean(axis=0), z=z)


def naphthalene(z: float, bend: float = 20.0) -> Atoms:
    """Naphthalene lying flat, its long axis along x, the C-H bonds bent ``bend`` degrees away from the surface
    (as the H atoms of benzene chemisorbed on Pt(111) are, Araujo et al.).
    """
    flat = _naphthalene()
    flat = raised(flat, reference=flat.positions[:10].mean(axis=0), z=z)
    carbons = flat.positions[:10]
    for h in range(10, 18):
        c = carbons[np.argmin(np.linalg.norm(carbons - flat.positions[h], axis=1))]
        outward = (flat.positions[h] - c) / np.linalg.norm(flat.positions[h] - c)
        angle = math.radians(bend)
        flat.positions[h] = c + 1.08 * (math.cos(angle) * outward + math.sin(angle) * np.array([0.0, 0.0, 1.0]))
    return flat


def cyclohexene_di_sigma(z: float, ring_rise: float = 50.0) -> Atoms:
    """Cyclohexene di-sigma bonded through C1 and C2 over a bridge site along x, the ring rising towards +y at
    ``ring_rise`` degrees above the surface plane (Wellendorff et al. Fig. S1, system 36); the H atoms of C1
    and C2 moved to sp3 positions pointing away from the surface.
    """
    gas = Atoms([s for s, _ in CYCLOHEXENE], positions=[p for _, p in CYCLOHEXENE])
    c1, c2 = gas.positions[0], gas.positions[1]
    middle = (c1 + c2) / 2
    ring = gas.positions[:6].mean(axis=0)
    rise = math.radians(ring_rise)
    turned = rotated(gas, c2 - c1, ring - middle, (1, 0, 0), (0, math.cos(rise), math.sin(rise)))
    turned = raised(turned, reference=(turned.positions[0] + turned.positions[1]) / 2, z=z)
    bonded = {0: (1, 5, 6), 1: (0, 2, 7)}  # carbon: (other olefinic C, ring neighbour, its H)
    for c, (other, neighbour, h) in bonded.items():
        directions = [np.array([0.0, 0.0, -1.0])]
        for j in (other, neighbour):
            v = turned.positions[j] - turned.positions[c]
            directions.append(v / np.linalg.norm(v))
        away = -np.sum(directions, axis=0)
        turned.positions[h] = turned.positions[c] + 1.09 * away / np.linalg.norm(away)
    return turned


# Benzene on Pt(111) as relaxed by Araujo et al. (PBE+D3, five-layer 3x3 slab, the bottom layer fixed; SI
# Supplementary Note 1): chemisorbed over a bridge site (a = 3.930 Å), the C atoms 2.06-2.10 Å above the mean
# height of the top layer, two C-C bonds of 1.435 Å and four of 1.474 Å, the H atoms bent up. Offsets (Å) of
# C1-C6, then H1-H6, from the bridge site along x of the ideal lattice that the fixed bottom layer of their
# slab defines (their slab maps onto ASE's fcc111 by a translation); z from the mean height of their top layer.
ARAUJO_BENZENE = [
    (-0.1011, 1.4872, 2.0639),
    (1.1379, 0.6903, 2.1001),
    (1.1433, -0.7448, 2.0978),
    (-0.0925, -1.5490, 2.0613),
    (-1.3317, -0.7513, 2.0940),
    (-1.3354, 0.6836, 2.0976),
    (-2.2531, 1.1910, 2.4018),
    (-0.0915, -2.4482, 2.6876),
    (-0.1051, 2.3892, 2.6864),
    (2.0503, 1.2041, 2.4082),
    (2.0585, -1.2517, 2.4095),
    (-2.2459, -1.2642, 2.3991),
]


def benzene_araujo() -> Atoms:
    """Benzene chemisorbed on the bridge site of Pt(111), as relaxed by Araujo et al."""
    return Atoms("C6H6", positions=ARAUJO_BENZENE)


# ----------------------------------------------------------------------------------------------------------
# Water layers on Pt(111), 3x3 cell: O on the top sites of a honeycomb (sites (i, j) with (i - j) mod 3 != 0),
# sublattice A ((i - j) mod 3 = 1) and B ((i - j) mod 3 = 2). Each B donates one H bond to its A neighbour at
# (-1, 0); each A donates H bonds to its B neighbours at (0, -1) and (-1, +1).
# ----------------------------------------------------------------------------------------------------------

A_SITES, B_SITES = [(1, 0), (2, 1), (0, 2)], [(2, 0), (0, 1), (1, 2)]


def _cell_vector(crystal: str, di: int, dj: int) -> np.ndarray:
    a_nn = nearest_neighbour(crystal)
    return di * np.array([a_nn, 0.0, 0.0]) + dj * np.array([a_nn / 2, a_nn * math.sqrt(3) / 2, 0.0])


def _water(o_height: float, donor_directions: list[np.ndarray]) -> Atoms:
    """H2O with O at ``o_height`` above its top site; H atoms (0.99 Å) along the H-bond directions (one or two
    in-plane unit vectors; with one, the second H points down, 104.5° from the first).
    """
    if len(donor_directions) == 2:
        d1, d2 = donor_directions
        bisector = (d1 + d2) / np.linalg.norm(d1 + d2)
        normal = np.cross(np.cross(d1, d2), bisector)
        normal /= np.linalg.norm(normal)
        half = math.radians(104.5 / 2)
        h1 = math.cos(half) * bisector + math.sin(half) * normal
        h2 = math.cos(half) * bisector - math.sin(half) * normal
        if np.dot(h1, d1) < np.dot(h2, d1):
            h1, h2 = h2, h1
    else:
        (h1,) = donor_directions
        h2 = math.cos(math.radians(104.5)) * h1 + math.sin(math.radians(104.5)) * np.array([0.0, 0.0, -1.0])
    o = np.array([0.0, 0.0, o_height])
    return Atoms("OH2", positions=[o, o + 0.99 * h1, o + 0.99 * h2])


def water_bilayer(crystal: str = "Pt") -> list[dict[str, Any]]:
    """The 'H-down' water bilayer at 2/3 ML (Araujo et al.; 6 H2O per 3x3 cell): A waters flat-lying (O 2.30 Å
    above Pt) donating two H bonds, B waters 0.8 Å higher donating one H bond and pointing the other H down.
    """
    groups = []
    for i, j in A_SITES:
        directions = [_cell_vector(crystal, 0, -1), _cell_vector(crystal, -1, 1)]
        directions = [d / np.linalg.norm(d) for d in directions]
        groups.append(group({"site": "ontop", "shift": [i, j]}, _water(2.30, directions)))
    for i, j in B_SITES:
        d = _cell_vector(crystal, -1, 0)
        groups.append(group({"site": "ontop", "shift": [i, j]}, _water(3.10, [d / np.linalg.norm(d)])))
    return groups


def water_hydroxyl_network(crystal: str = "Pt") -> list[dict[str, Any]]:
    """The flat (H2O...OH) honeycomb (Wellendorff et al. Fig. S1, system 39): 3 H2O (A) and 3 OH (B) per 3x3
    cell, every H in an H bond; O of OH 2.05 Å and of H2O 2.20 Å above Pt.
    """
    groups = []
    for i, j in A_SITES:
        directions = [_cell_vector(crystal, 0, -1), _cell_vector(crystal, -1, 1)]
        directions = [d / np.linalg.norm(d) for d in directions]
        groups.append(group({"site": "ontop", "shift": [i, j]}, _water(2.20, directions)))
    for i, j in B_SITES:
        d = _cell_vector(crystal, -1, 0)
        o = np.array([0.0, 0.0, 2.05])
        hydroxyl = Atoms("OH", positions=[o, o + 0.98 * d / np.linalg.norm(d)])
        groups.append(group({"site": "ontop", "shift": [i, j]}, hydroxyl))
    return groups


ADSORBED: dict[str, dict[str, Any]] = {}


def adsorbed(name: str, slab: str, groups: list[dict[str, Any]], configuration: str) -> None:
    """Register an adsorbed structure: the slab, its adsorbate groups and how the configuration was chosen."""
    if name in ADSORBED:
        raise ValueError(f"duplicate adsorbed structure {name!r}")
    ADSORBED[name] = {"slab": slab, "adsorbates": groups, "configuration": configuration}


CE39_FIGURE = "Wellendorff et al. (2015) SI Fig. S1"


def ads41_structures() -> None:
    """The adsorbed structures of ADS41 (separated adlayers for dissociated molecules, as in CE39)."""
    # CO, 1/4 ML (2x2), upright through C (C-O 1.17 Å); sites of Fig. S1 (systems 1-9) and Araujo Table 3.
    for metal, facet, site in [
        ("Co", "0001", "ontop"),
        ("Cu", "111", "ontop"),
        ("Ir", "111", "ontop"),
        ("Ni", "111", "fcc"),
        ("Pd", "100", "bridge"),
        ("Pd", "111", "fcc"),
        ("Pt", "111", "ontop"),
        ("Rh", "111", "ontop"),
        ("Ru", "0001", "ontop"),
    ]:
        z = height("C", metal, site)
        adsorbed(
            f"CO/{metal}({facet}) {site}",
            f"{metal}({facet}) 2x2",
            [group(site, upright("CO", z, 1.17))],
            f"CO upright on the {site} site at 1/4 ML ({CE39_FIGURE}; Araujo et al. (2022) Table 3); "
            f"C-{metal} {BONDS['C', metal, site]:.2f} Å.",
        )
    # NO molecular, 1/4 ML, upright through N (N-O 1.20 Å).
    for metal, facet, site in [("Pd", "100", "hollow"), ("Pd", "111", "fcc"), ("Pt", "111", "fcc")]:
        z = height("N", metal, site)
        adsorbed(
            f"NO/{metal}({facet}) {site}",
            f"{metal}({facet}) 2x2",
            [group(site, upright("NO", z, 1.20))],
            f"NO upright on the {site} site at 1/4 ML ({CE39_FIGURE}, systems 11-13; Araujo et al. Table 3); "
            f"N-{metal} {BONDS['N', metal, site]:.2f} Å.",
        )
    # Atoms at 1/4 ML (2x2): H, N, O, I; O on Pt(111) at 1/9 ML (3x3).
    atoms = [
        ("H", "Ni", "100", "hollow", "2x2"),
        ("H", "Ni", "111", "fcc", "2x2"),
        ("H", "Pd", "111", "fcc", "2x2"),
        ("H", "Pt", "111", "fcc", "2x2"),
        ("H", "Rh", "111", "fcc", "2x2"),
        ("I", "Pt", "111", "fcc", "2x2"),
        ("N", "Ni", "100", "hollow", "2x2"),
        ("O", "Ni", "100", "hollow", "2x2"),
        ("O", "Ni", "111", "fcc", "2x2"),
        ("O", "Pt", "111", "fcc", "3x3"),
        ("O", "Rh", "100", "hollow", "2x2"),
    ]
    for symbol, metal, facet, site, size in atoms:
        coverage = "1/9" if size == "3x3" else "1/4"
        note = f"{symbol} on the {site} site at {coverage} ML ({CE39_FIGURE}; Araujo et al. Table 3)"
        if symbol == "H" and metal == "Pt":
            note += "; Araujo et al. list the top site, but Fig. S1 (system 18) shows H in a three-fold hollow"
        adsorbed(
            f"{symbol}/{metal}({facet}) {site} {size}",
            f"{metal}({facet}) {size}",
            [group(site, point(symbol, height(symbol, metal, site)))],
            f"{note}; {symbol}-{metal} {BONDS[symbol, metal, site]:.2f} Å.",
        )
    # Fragments of the dissociation of ethylene, CH3I and CH2I2 on Pt(111).
    adsorbed(
        "H/Pt(111) fcc 3x3",
        "Pt(111) 3x3",
        [group("fcc", point("H", height("H", "Pt", "fcc")))],
        "H on the fcc site at 1/9 ML, the 3x3 slab of the ethylene dissociation (Sharada et al. (2019) Appendix A).",
    )
    adsorbed(
        "CCH3/Pt(111) fcc 3x3",
        "Pt(111) 3x3",
        [group("fcc", ethylidyne(height("C", "Pt", "fcc", 2.00)))],
        (
            "Ethylidyne on the fcc site at 1/9 ML, C-C along the normal (Sharada et al. Appendix A: 3x3 slab); "
            "C-Pt 2.00 Å, C-C 1.50 Å."
        ),
    )
    adsorbed(
        "CH/Pt(111) fcc 4x4",
        "Pt(111) 4x4",
        [group("fcc", methylidyne(height("C", "Pt", "fcc", 2.02)))],
        (
            "Methylidyne on the fcc site at 1/16 ML (Araujo et al., to mimic the low coverage of the experiment; "
            "CE39 used 1/4 ML); C-Pt 2.02 Å."
        ),
    )
    adsorbed(
        "CH3/Pt(111) ontop 4x4",
        "Pt(111) 4x4",
        [group("ontop", methyl(2.08))],
        f"Methyl on the top site ({CE39_FIGURE}, system 27) at 1/16 ML (Araujo et al.; CE39 used 1/4 ML); C-Pt 2.08 Å.",
    )
    # Molecules physisorbed or weakly chemisorbed on Pt(111) and Cu(100).
    adsorbed(
        "CH3I/Pt(111) ontop",
        "Pt(111) 2x2",
        [group("ontop", methyl_iodide(2.80))],
        f"CH3I standing on I on the top site at 1/4 ML ({CE39_FIGURE}, system 25); I-Pt 2.80 Å.",
    )
    adsorbed(
        "CH3OH/Pt(111) ontop",
        "Pt(111) 2x2",
        [group("ontop", methanol(2.40))],
        f"CH3OH bound through O on the top site at 1/4 ML, C-O tilted 45° from the normal and the hydroxyl H "
        f"near the surface plane ({CE39_FIGURE}, system 26); O-Pt 2.40 Å.",
    )
    adsorbed(
        "CH4/Pt(111) fcc",
        "Pt(111) 2x2",
        [group("fcc", methane_tripod("Pt", lowest=2.60))],
        f"CH4 over the fcc site at 1/4 ML with three H down (tripod, {CE39_FIGURE} system 28; site: Araujo et al. "
        "Table 3); the lower H 2.60 Å above the top layer.",
    )
    adsorbed(
        "NH3/Cu(100) ontop",
        "Cu(100) 2x2",
        [group("ontop", ammonia(2.10))],
        f"NH3 bound through N on the top site at 1/4 ML ({CE39_FIGURE}, system 24); N-Cu 2.10 Å.",
    )
    adsorbed(
        "C2H6/Pt(111) fcc",
        "Pt(111) 3x3",
        [group("fcc", flat_alkane("C2H6", -30.0, lowest=2.50))],
        f"Ethane at 1/9 ML with its C-C axis parallel to the surface along [2a1 - a2] (as in {CE39_FIGURE}, "
        "system 29), the C-C midpoint over the fcc site (Araujo et al. Table 3); the lowest H 2.50 Å above "
        "the top layer.",
    )
    for name, formula in [("C3H8", "C3H8"), ("C4H10", "trans-butane")]:
        adsorbed(
            f"{name}/Pt(111) ontop",
            "Pt(111) 3x3",
            [group("ontop", flat_alkane(formula, 0.0, lowest=2.50))],
            f"All-trans {name} at 1/9 ML lying flat, its carbon plane parallel to the surface and the chain along "
            f"a close-packed row (as in {CE39_FIGURE}, systems 30-31), the carbon centroid over a top site "
            "(Araujo et al. Table 3); the lowest H 2.50 Å above the top layer.",
        )
    for metal, z in [("Cu", 3.0), ("Ag", 3.1), ("Au", 3.1)]:
        adsorbed(
            f"C6H6/{metal}(111) fcc",
            f"{metal}(111) 3x3",
            [group("fcc", flat_benzene(z))],
            f"Benzene at 1/9 ML lying flat over the fcc site, two C-H bonds along y ({CE39_FIGURE}, systems 33-35; "
            f"site: Araujo et al. Table 3); the ring {z:.1f} Å above the top layer.",
        )
    adsorbed(
        "C6H6/Pt(111) bridge",
        "Pt(111) 3x3 5 layers",
        [group("bridge", benzene_araujo())],
        "Benzene at 1/9 ML chemisorbed over the bridge site, in the geometry relaxed by Araujo et al. (2022) "
        "on a five-layer slab with four relaxed layers (their SI Supplementary Note 1; CE39 used a flat "
        "benzene on a four-layer slab).",
    )
    adsorbed(
        "C6H10/Pt(111) bridge",
        "Pt(111) 3x3 5 layers",
        [group("bridge", cyclohexene_di_sigma(2.00))],
        f"Cyclohexene at 1/9 ML di-sigma bonded through C1 and C2 over a bridge site, the ring rising at 50° "
        f"from the surface ({CE39_FIGURE}, system 36), on the five-layer slab with four relaxed layers of "
        "Araujo et al.; C1 and C2 2.00 Å above the top layer, their H atoms in sp3 positions.",
    )
    adsorbed(
        "C10H8/Pt(111) ontop",
        "Pt(111) 4x4",
        [group("ontop", naphthalene(2.10))],
        "Naphthalene at 1/16 ML (4x4 slab, Sharada et al. Appendix A) lying flat, the molecule centred over a "
        "top site with the long axis along a close-packed row, so that both rings sit next to bridge sites "
        "(the di-bridge geometry, the most stable of Morin et al., J. Phys. Chem. B 108, 12084 (2004)); "
        "the carbons 2.10 Å above the top layer, the C-H bonds bent 20° up.",
    )
    adsorbed(
        "(H2O)6/Pt(111)",
        "Pt(111) 3x3",
        water_bilayer(),
        "Water at 2/3 ML as the hexagonal H-down bilayer (Araujo et al. (2022), as in the experiment's "
        "islands; CE39 used a single molecule at 1/4 ML): O atoms on the top sites of a honeycomb, three "
        "flat-lying molecules (O 2.30 Å above Pt) donating two H bonds and three molecules 0.8 Å higher "
        "donating one and pointing the other H down; 6 H2O per 3x3 cell.",
    )
    adsorbed(
        "(H2O...OH)3/Pt(111)",
        "Pt(111) 3x3",
        water_hydroxyl_network(),
        f"The flat (H2O...OH) honeycomb of {CE39_FIGURE}, system 39: O atoms on the top sites, H2O and OH "
        "alternating, every H in an H bond (3 H2O + 3 OH per 3x3 cell, 2/3 ML).",
    )
    adsorbed(
        "O/Pt(111) fcc 3x2",
        "Pt(111) 3x2",
        [group("fcc", point("O", height("O", "Pt", "fcc")))],
        (
            "O on the fcc site at 1/6 ML: the O coverage consumed by the (H2O...OH) network of the 3x3 cell "
            "(1.5 O per 9 Pt, Wellendorff et al. Table 3); O-Pt 2.05 Å."
        ),
    )


# ----------------------------------------------------------------------------------------------------------
# Surf13: the revPBE-D4 structures of Shi et al. (GitHub benshi97/Data_autoSKZCAM, CC BY 4.0)
# ----------------------------------------------------------------------------------------------------------

SHI_COMMIT = "b12d5017b4126b08d5541a62db6880120341bf82"
"""Commit of github.com/benshi97/Data_autoSKZCAM (2025-06-12; the data of Zenodo 10.5281/zenodo.15651018)."""

SHI_FILES = {
    "02-Surface/MgO": "6af3e02906c9f4653e1e0bfff48e2b63",
    "02-Surface/r-TiO2": "c1176e683526c800815238a9175113e7",
    "02-Surface/a-TiO2": "27362690583d3d5511b9b35a1d96f392",
    "04-Molecule_Surface/MgO/CH4": "9ca6172ecb02dbce2b913c1c8d57df23",
    "04-Molecule_Surface/MgO/C2H6": "34dea7100e25cebfa88d790676a074b7",
    "04-Molecule_Surface/MgO/CO": "f05aa334e06aadee53243684efc81405",
    "04-Molecule_Surface/MgO/CO2_Chemisorbed": "8611afc8e8167af3a11611cf98c01b1f",
    "04-Molecule_Surface/MgO/H2O_Monomer": "06e89ea817b4dc1c40ab81e76f88dafe",
    "04-Molecule_Surface/MgO/N2O_Parallel": "6b4bbbf48ccd407b7468224cc5fd3d31",
    "04-Molecule_Surface/MgO/NH3": "e14a6f9ae12983eec2d9d22404dfc488",
    "04-Molecule_Surface/r-TiO2/CH4": "618f7feac830e6cde76055277f9fd267",
    "04-Molecule_Surface/r-TiO2/CO2_Tilted": "5c56eb1e830b9030e68d1e6b1ed5c99b",
    "04-Molecule_Surface/r-TiO2/H2O": "032ac1075799ae299e5483375f1b0add",
    "04-Molecule_Surface/r-TiO2/CH3OH": "aec62586b95fa73355aaecf151fa6f54",
    "04-Molecule_Surface/a-TiO2/H2O": "0fb439f6d734db2203a20bcbcfc34bcd",
    "04-Molecule_Surface/a-TiO2/NH3": "adacc10f190056f5b82d92912fb658cd",
}
"""The relaxed revPBE-D4 structures used (``<path>/02_revPBE-D4/OUTCAR``) and their MD5 checksums."""

SHI_LATTICES = {  # revPBE-D4 bulk lattices of Shi et al. (01-Unit_Cell/*/02_revPBE-D4)
    "MgO": {"lattice": "rocksalt", "symbols": ["Mg", "O"], "a": 4.22049},
    "TiO2 (rutile)": {"lattice": "rutile", "symbols": ["Ti", "O"], "a": 4.5976, "c": 2.95758, "u": 0.304},
    "TiO2 (anatase)": {"lattice": "anatase", "symbols": ["Ti", "O"], "a": 3.7896, "c": 9.54811, "u": 0.20882},
}

TI5C = {"atom": "Ti", "neighbours": 5}

# name, source path, slab, anchor, binding element, configuration
SURF13_STRUCTURES = [
    ("CH4/MgO(001)", "MgO/CH4", "MgO(001) 2x2", {"atom": "Mg"}, "C", "CH4 over Mg with two H down (dipod)"),
    ("C2H6/MgO(001)", "MgO/C2H6", "MgO(001) 2x2", {"atom": "Mg"}, "C", "ethane lying nearly flat over Mg"),
    ("CO/MgO(001)", "MgO/CO", "MgO(001) 2x2", {"atom": "Mg"}, "C", "CO upright on Mg, C down"),
    (
        "CO2/MgO(001) chemisorbed",
        "MgO/CO2_Chemisorbed",
        "MgO(001) 2x2",
        {"atom": "O"},
        "C",
        (
            "CO2 chemisorbed as a carbonate on a surface O, its O atoms towards two Mg (the state Shi et al. assign "
            "to the experiment of Chakradhar and Burghaus; Surf13 holds the physisorbed state)"
        ),
    ),
    (
        "H2O/MgO(001)",
        "MgO/H2O_Monomer",
        "MgO(001) 2x2",
        {"atom": "Mg"},
        "O",
        "H2O monomer on Mg, one H bond to surface O",
    ),
    ("N2O/MgO(001)", "MgO/N2O_Parallel", "MgO(001) 2x2", {"atom": "Mg"}, "O", "N2O lying parallel to the surface"),
    ("NH3/MgO(001)", "MgO/NH3", "MgO(001) 2x2", {"atom": "Mg"}, "N", "NH3 on Mg, N down"),
    ("CH4/TiO2 rutile(110)", "r-TiO2/CH4", "TiO2 rutile(110) p(4x2)", TI5C, "C", "CH4 over a five-fold Ti, two H down"),
    (
        "CO2/TiO2 rutile(110) tilted",
        "r-TiO2/CO2_Tilted",
        "TiO2 rutile(110) p(4x2)",
        TI5C,
        "O",
        "CO2 tilted, one O on a five-fold Ti",
    ),
    (
        "H2O/TiO2 rutile(110)",
        "r-TiO2/H2O",
        "TiO2 rutile(110) p(4x2)",
        TI5C,
        "O",
        "H2O on a five-fold Ti, one H bond to a bridging O",
    ),
    (
        "CH3OH/TiO2 rutile(110)",
        "r-TiO2/CH3OH",
        "TiO2 rutile(110) p(4x2)",
        TI5C,
        "O",
        "CH3OH bound through O to a five-fold Ti, H bond to a bridging O",
    ),
    (
        "H2O/TiO2 anatase(101)",
        "a-TiO2/H2O",
        "TiO2 anatase(101) 3x1",
        TI5C,
        "O",
        "H2O on a five-fold Ti, H atoms towards two-fold O",
    ),
    ("NH3/TiO2 anatase(101)", "a-TiO2/NH3", "TiO2 anatase(101) 3x1", TI5C, "N", "NH3 on a five-fold Ti, N down"),
]


def shi_structure(path: str) -> Atoms:
    """A relaxed revPBE-D4 structure of Shi et al. (downloaded once, MD5-checked)."""
    url = f"https://raw.githubusercontent.com/benshi97/Data_autoSKZCAM/{SHI_COMMIT}/Data/{path}/02_revPBE-D4/OUTCAR"
    file = RemoteFile(url, path.replace("/", "__") + ".OUTCAR", SHI_FILES[path])
    return read(download_file(file, "shi2025-autoskzcam"), format="vasp-out", index=-1)


def check_shi_slabs() -> None:
    """The oxide slabs built from Shi et al.'s bulk lattices must reproduce the fixed layers of their slabs."""
    for name, path in [
        ("MgO(001) 2x2", "02-Surface/MgO"),
        ("TiO2 rutile(110) p(4x2)", "02-Surface/r-TiO2"),
        ("TiO2 anatase(101) 3x1", "02-Surface/a-TiO2"),
    ]:
        definition = SLABS[name]
        spec = SHI_LATTICES[definition["crystal"]]
        mine = build_slab(bulk_crystal(**spec), definition)
        theirs = shi_structure(path)
        if len(mine) != len(theirs) or mine.get_chemical_formula() != theirs.get_chemical_formula():
            raise AssertionError(f"{name}: {mine.get_chemical_formula()} vs {theirs.get_chemical_formula()}")
        fixed_theirs = [
            i for i in range(len(theirs)) if theirs.positions[i, 2] < _fixed_height(mine) + theirs.positions[:, 2].min()
        ]
        deviation = _largest_deviation(mine, theirs, fixed_theirs)
        if deviation > 1e-3:
            raise AssertionError(f"{name}: the fixed layers differ from Shi et al.'s by {deviation:.4f} Å")
        print(f"{name}: fixed layers as in Shi et al. (largest deviation {deviation:.1e} Å)")


def _fixed_height(slab: Atoms) -> float:
    """Height of the fixed layers of a slab above its lowest atom, plus 0.1 Å."""
    fixed = slab.constraints[0].get_indices()
    return float(slab.positions[fixed, 2].max() - slab.positions[:, 2].min() + 0.1)


def _largest_deviation(mine: Atoms, theirs: Atoms, selected: list[int]) -> float:
    """Largest distance from the selected atoms of ``theirs`` to the same element in ``mine``, after the in-plane
    translation that best overlays the two (their slabs are cut and oriented as ours).
    """
    symbols = np.array(mine.get_chemical_symbols())
    mine_z = mine.positions[:, 2] - mine.positions[:, 2].min()
    their_z = theirs.positions[:, 2] - theirs.positions[:, 2].min()
    first = min(selected, key=lambda i: their_z[i])
    best = math.inf
    for candidate in np.where((symbols == theirs[first].symbol) & (mine_z < 0.01))[0]:
        shift = theirs.positions[first, :2] - mine.positions[candidate, :2]
        worst = 0.0
        for i in selected:
            d = np.zeros((len(mine), 3))
            d[:, :2] = mine.positions[:, :2] + shift - theirs.positions[i, :2]
            d, _ = find_mic(d, theirs.cell, pbc=[True, True, False])
            d[:, 2] = mine_z - their_z[i]
            worst = max(worst, float(np.linalg.norm(d[symbols == theirs[i].symbol], axis=1).min()))
        best = min(best, worst)
    return best


def surf13_structures() -> None:
    """The adsorbed structures of Surf13, from the relaxed structures of Shi et al."""
    clean = {
        "MgO": shi_structure("02-Surface/MgO"),
        "r-TiO2": shi_structure("02-Surface/r-TiO2"),
        "a-TiO2": shi_structure("02-Surface/a-TiO2"),
    }
    for name, path, slab, anchor, binding, configuration in SURF13_STRUCTURES:
        complex_ = shi_structure(f"04-Molecule_Surface/{path}")
        n_molecule = len(complex_) - len(clean[path.split("/")[0]])
        molecule_atoms = complex_[:n_molecule]
        substrate = complex_[n_molecule:]
        if substrate.get_chemical_formula() != clean[path.split("/")[0]].get_chemical_formula():
            raise AssertionError(f"{name}: the first {n_molecule} atoms are not the molecule")
        bind = min(
            (i for i in range(n_molecule) if molecule_atoms[i].symbol == binding),
            key=lambda i: molecule_atoms.positions[i, 2],
        )
        anchor_index = _anchor_in_substrate(substrate, anchor, molecule_atoms.positions[bind])
        offsets, _ = find_mic(
            molecule_atoms.positions - substrate.positions[anchor_index], complex_.cell, pbc=[True, True, False]
        )
        anchor_label = "a five-fold Ti" if anchor.get("neighbours") == 5 else f"a surface {anchor['atom']}"
        adsorbed(
            name,
            slab,
            [
                {
                    "anchor": dict(anchor),
                    "symbols": molecule_atoms.get_chemical_symbols(),
                    "positions": np.round(offsets, 4).tolist(),
                }
            ],
            f"{configuration}: the relaxed revPBE-D4 geometry of Shi et al. (2025) "
            f"(Data_autoSKZCAM, Data/04-Molecule_Surface/{path}), placed relative to {anchor_label}.",
        )


def _anchor_in_substrate(substrate: Atoms, anchor: dict[str, Any], near: np.ndarray) -> int:
    """Index of the anchor atom of a relaxed substrate that is nearest (in plane) to ``near``."""
    symbols = np.array(substrate.get_chemical_symbols())
    heights = substrate.positions[:, 2]
    candidates = (symbols == anchor["atom"]) & (heights > heights[symbols == anchor["atom"]].max() - 0.5)
    if "neighbours" in anchor:
        i, j = neighbor_list("ij", substrate, BOND_LENGTH)
        counted = (symbols[j] == "O") != (symbols[i] == "O")
        candidates &= np.bincount(i[counted], minlength=len(substrate)) == anchor["neighbours"]
    indices = np.where(candidates)[0]
    vectors = np.zeros((len(indices), 3))
    vectors[:, :2] = substrate.positions[indices, :2] - near[:2]
    vectors, _ = find_mic(vectors, substrate.cell, pbc=[True, True, False])
    return int(indices[np.argmin(np.linalg.norm(vectors, axis=1))])


# ----------------------------------------------------------------------------------------------------------
# Reactions and reference energies
# ----------------------------------------------------------------------------------------------------------

SHARADA = "Sharada et al., Phys. Rev. B 100, 035439 (2019), Table I"

# category, mixed (Sharada's asterisk: dispersion with covalent contributions), equation, terms (structure:
# coefficient), adsorbates per reaction (Sharada et al. Table III scale the errors by it), reference (eV),
# number in CE39 (Wellendorff et al. Table 2; None for the two systems Sharada et al. added from Gautier et
# al., Phys. Chem. Chem. Phys. 17, 28921 (2015)). D2O is computed as H2O.
ADS41_REACTIONS: list[tuple[str, bool, str, dict[str, Any], int, float, int | None]] = [
    (
        "chemisorption",
        False,
        "C2H4 + Pt(111) -> CCH3/Pt(111) + H/Pt(111)",
        {"CCH3/Pt(111) fcc 3x3": 1, "H/Pt(111) fcc 3x3": 1, "Pt(111) 3x3": -2, "C2H4": -1},
        2,
        -1.36,
        None,
    ),
    (
        "chemisorption",
        False,
        "CH2I2 + Pt(111) -> CH/Pt(111) + H/Pt(111) + 2 I/Pt(111)",
        {
            "CH/Pt(111) fcc 4x4": 1,
            "Pt(111) 4x4": -1,
            "H/Pt(111) fcc 2x2": 1,
            "I/Pt(111) fcc 2x2": 2,
            "Pt(111) 2x2": -3,
            "CH2I2": -1,
        },
        4,
        -4.72,
        24,
    ),
    (
        "chemisorption",
        False,
        "CH3I + Pt(111) -> CH3/Pt(111) + I/Pt(111)",
        {"CH3/Pt(111) ontop 4x4": 1, "Pt(111) 4x4": -1, "I/Pt(111) fcc 2x2": 1, "Pt(111) 2x2": -1, "CH3I": -1},
        2,
        -2.17,
        25,
    ),
    (
        "chemisorption",
        False,
        "CO + Co(0001) -> CO/Co(0001)",
        {"CO/Co(0001) ontop": 1, "Co(0001) 2x2": -1, "CO": -1},
        1,
        -1.23,
        9,
    ),
    (
        "chemisorption",
        False,
        "CO + Cu(111) -> CO/Cu(111)",
        {"CO/Cu(111) ontop": 1, "Cu(111) 2x2": -1, "CO": -1},
        1,
        -0.59,
        7,
    ),
    (
        "chemisorption",
        False,
        "CO + Ir(111) -> CO/Ir(111)",
        {"CO/Ir(111) ontop": 1, "Ir(111) 2x2": -1, "CO": -1},
        1,
        -1.70,
        6,
    ),
    (
        "chemisorption",
        False,
        "CO + Ni(111) -> CO/Ni(111)",
        {"CO/Ni(111) fcc": 1, "Ni(111) 2x2": -1, "CO": -1},
        1,
        -1.29,
        1,
    ),
    (
        "chemisorption",
        False,
        "CO + Pd(100) -> CO/Pd(100)",
        {"CO/Pd(100) bridge": 1, "Pd(100) 2x2": -1, "CO": -1},
        1,
        -1.63,
        4,
    ),
    (
        "chemisorption",
        False,
        "CO + Pd(111) -> CO/Pd(111)",
        {"CO/Pd(111) fcc": 1, "Pd(111) 2x2": -1, "CO": -1},
        1,
        -1.49,
        3,
    ),
    (
        "chemisorption",
        False,
        "CO + Pt(111) -> CO/Pt(111)",
        {"CO/Pt(111) ontop": 1, "Pt(111) 2x2": -1, "CO": -1},
        1,
        -1.29,
        2,
    ),
    (
        "chemisorption",
        False,
        "CO + Rh(111) -> CO/Rh(111)",
        {"CO/Rh(111) ontop": 1, "Rh(111) 2x2": -1, "CO": -1},
        1,
        -1.47,
        5,
    ),
    (
        "chemisorption",
        False,
        "CO + Ru(0001) -> CO/Ru(0001)",
        {"CO/Ru(0001) ontop": 1, "Ru(0001) 2x2": -1, "CO": -1},
        1,
        -1.67,
        8,
    ),
    (
        "chemisorption",
        False,
        "H2 + Ni(100) -> 2 H/Ni(100)",
        {"H/Ni(100) hollow 2x2": 2, "Ni(100) 2x2": -2, "H2": -1},
        2,
        -0.90,
        20,
    ),
    (
        "chemisorption",
        False,
        "H2 + Ni(111) -> 2 H/Ni(111)",
        {"H/Ni(111) fcc 2x2": 2, "Ni(111) 2x2": -2, "H2": -1},
        2,
        -1.04,
        19,
    ),
    (
        "chemisorption",
        False,
        "H2 + Pd(111) -> 2 H/Pd(111)",
        {"H/Pd(111) fcc 2x2": 2, "Pd(111) 2x2": -2, "H2": -1},
        2,
        -0.93,
        22,
    ),
    (
        "chemisorption",
        False,
        "H2 + Pt(111) -> 2 H/Pt(111)",
        {"H/Pt(111) fcc 2x2": 2, "Pt(111) 2x2": -2, "H2": -1},
        2,
        -0.75,
        18,
    ),
    (
        "chemisorption",
        False,
        "H2 + Rh(111) -> 2 H/Rh(111)",
        {"H/Rh(111) fcc 2x2": 2, "Rh(111) 2x2": -2, "H2": -1},
        2,
        -0.75,
        21,
    ),
    (
        "chemisorption",
        False,
        "I2 + Pt(111) -> 2 I/Pt(111)",
        {"I/Pt(111) fcc 2x2": 2, "Pt(111) 2x2": -2, "I2": -1},
        2,
        -3.24,
        23,
    ),
    (
        "chemisorption",
        False,
        "NO + Ni(100) -> N/Ni(100) + O/Ni(100)",
        {"N/Ni(100) hollow 2x2": 1, "O/Ni(100) hollow 2x2": 1, "Ni(100) 2x2": -2, "NO": -1},
        2,
        -3.10,
        10,
    ),
    (
        "chemisorption",
        False,
        "NO + Pd(100) -> NO/Pd(100)",
        {"NO/Pd(100) hollow": 1, "Pd(100) 2x2": -1, "NO": -1},
        1,
        -1.69,
        13,
    ),
    (
        "chemisorption",
        False,
        "NO + Pd(111) -> NO/Pd(111)",
        {"NO/Pd(111) fcc": 1, "Pd(111) 2x2": -1, "NO": -1},
        1,
        -1.89,
        12,
    ),
    (
        "chemisorption",
        False,
        "NO + Pt(111) -> NO/Pt(111)",
        {"NO/Pt(111) fcc": 1, "Pt(111) 2x2": -1, "NO": -1},
        1,
        -1.23,
        11,
    ),
    (
        "chemisorption",
        False,
        "O2 + Ni(100) -> 2 O/Ni(100)",
        {"O/Ni(100) hollow 2x2": 2, "Ni(100) 2x2": -2, "O2": -1},
        2,
        -5.49,
        15,
    ),
    (
        "chemisorption",
        False,
        "O2 + Ni(111) -> 2 O/Ni(111)",
        {"O/Ni(111) fcc 2x2": 2, "Ni(111) 2x2": -2, "O2": -1},
        2,
        -5.03,
        14,
    ),
    (
        "chemisorption",
        False,
        "O2 + Pt(111) -> 2 O/Pt(111)",
        {"O/Pt(111) fcc 3x3": 2, "Pt(111) 3x3": -2, "O2": -1},
        2,
        -2.16,
        16,
    ),
    (
        "chemisorption",
        False,
        "O2 + Rh(100) -> 2 O/Rh(100)",
        {"O/Rh(100) hollow 2x2": 2, "Rh(100) 2x2": -2, "O2": -1},
        2,
        -3.68,
        17,
    ),
    (
        "dispersion",
        False,
        "C2H6 + Pt(111) -> C2H6/Pt(111)",
        {"C2H6/Pt(111) fcc": 1, "Pt(111) 3x3": -1, "C2H6": -1},
        1,
        -0.28,
        30,
    ),
    (
        "dispersion",
        False,
        "C3H8 + Pt(111) -> C3H8/Pt(111)",
        {"C3H8/Pt(111) ontop": 1, "Pt(111) 3x3": -1, "C3H8": -1},
        1,
        -0.40,
        31,
    ),
    (
        "dispersion",
        False,
        "C4H10 + Pt(111) -> C4H10/Pt(111)",
        {"C4H10/Pt(111) ontop": 1, "Pt(111) 3x3": -1, "C4H10": -1},
        1,
        -0.50,
        32,
    ),
    (
        "dispersion",
        True,
        "C6H6 + Ag(111) -> C6H6/Ag(111)",
        {"C6H6/Ag(111) fcc": 1, "Ag(111) 3x3": -1, "C6H6": -1},
        1,
        -0.63,
        35,
    ),
    (
        "dispersion",
        True,
        "C6H6 + Au(111) -> C6H6/Au(111)",
        {"C6H6/Au(111) fcc": 1, "Au(111) 3x3": -1, "C6H6": -1},
        1,
        -0.73,
        36,
    ),
    (
        "dispersion",
        True,
        "C6H6 + Cu(111) -> C6H6/Cu(111)",
        {"C6H6/Cu(111) fcc": 1, "Cu(111) 3x3": -1, "C6H6": -1},
        1,
        -0.68,
        34,
    ),
    (
        "dispersion",
        True,
        "C6H6 + Pt(111) -> C6H6/Pt(111)",
        {"C6H6/Pt(111) bridge": 1, "Pt(111) 3x3 5 layers": -1, "C6H6": -1},
        1,
        -1.68,
        33,
    ),
    (
        "dispersion",
        True,
        "C6H10 + Pt(111) -> C6H10/Pt(111)",
        {"C6H10/Pt(111) bridge": 1, "Pt(111) 3x3 5 layers": -1, "C6H10": -1},
        1,
        -1.27,
        37,
    ),
    (
        "dispersion",
        True,
        "C10H8 + Pt(111) -> C10H8/Pt(111)",
        {"C10H8/Pt(111) ontop": 1, "Pt(111) 4x4": -1, "C10H8": -1},
        1,
        -2.76,
        None,
    ),
    (
        "dispersion",
        True,
        "CH3I + Pt(111) -> CH3I/Pt(111)",
        {"CH3I/Pt(111) ontop": 1, "Pt(111) 2x2": -1, "CH3I": -1},
        1,
        -0.87,
        27,
    ),
    (
        "dispersion",
        False,
        "CH3OH + Pt(111) -> CH3OH/Pt(111)",
        {"CH3OH/Pt(111) ontop": 1, "Pt(111) 2x2": -1, "CH3OH": -1},
        1,
        -0.57,
        28,
    ),
    (
        "dispersion",
        False,
        "CH4 + Pt(111) -> CH4/Pt(111)",
        {"CH4/Pt(111) fcc": 1, "Pt(111) 2x2": -1, "CH4": -1},
        1,
        -0.15,
        29,
    ),
    (
        "dispersion",
        False,
        "D2O + 1/3 O/Pt(111) -> 2/3 (D2O...OD)/Pt(111)",
        {
            "(H2O...OH)3/Pt(111)": "2/9",
            "Pt(111) 3x3": "-2/9",
            "O/Pt(111) fcc 3x2": "-1/3",
            "Pt(111) 3x2": "1/3",
            "H2O": -1,
        },
        2,
        -0.68,
        39,
    ),
    (
        "dispersion",
        False,
        "D2O + Pt(111) -> D2O/Pt(111)",
        {"(H2O)6/Pt(111)": "1/6", "Pt(111) 3x3": "-1/6", "H2O": -1},
        1,
        -0.57,
        38,
    ),
    (
        "dispersion",
        False,
        "NH3 + Cu(100) -> NH3/Cu(100)",
        {"NH3/Cu(100) ontop": 1, "Cu(100) 2x2": -1, "NH3": -1},
        1,
        -0.62,
        26,
    ),
]

SHI = "Shi et al., Nat. Chem. 17, 1688 (2025)"

SURFACES = {
    "MgO(001) 2x2": "MgO(001)",
    "TiO2 rutile(110) p(4x2)": "rutile TiO2(110)",
    "TiO2 anatase(101) 3x1": "anatase TiO2(101)",
}

# adsorbed structure, gas molecule, slab, category, H_ads and its uncertainty (2 sigma, meV), temperature (K),
# Delta H = E_ZPV + E_T - RT and its uncertainty (meV), experiment (SI Tables 30, 32 and 33).
SURF13_REACTIONS = [
    (
        "CH4/MgO(001)",
        "CH4",
        "MgO(001) 2x2",
        "MgO",
        -115,
        19,
        47,
        11,
        13,
        "TPD, dilute limit: Tait, Dohnálek, Campbell, Kay, J. Chem. Phys. 122, 164707 and 164708 (2005)",
    ),
    (
        "C2H6/MgO(001)",
        "C2H6",
        "MgO(001) 2x2",
        "MgO",
        -221,
        30,
        75,
        -1,
        10,
        "TPD, dilute limit: Tait, Dohnálek, Campbell, Kay, J. Chem. Phys. 122, 164707 and 164708 (2005)",
    ),
    (
        "CO/MgO(001)",
        "CO",
        "MgO(001) 2x2",
        "MgO",
        -176,
        21,
        61,
        19,
        3,
        (
            "TPD, low coverage: Dohnálek et al., J. Phys. Chem. B 105, 3747 (2001); Wichtendahl et al., Phys. Status "
            "Solidi A 173, 93 (1999); re-analysed by Campbell and Sellers, Chem. Rev. 113, 4106 (2013)"
        ),
    ),
    (
        "CO2/MgO(001) chemisorbed",
        "CO2",
        "MgO(001) 2x2",
        "MgO",
        -664,
        125,
        230,
        18,
        2,
        "TPD of the chemisorbed state: Chakradhar and Burghaus, Surf. Sci. 616, 171 (2013)",
    ),
    (
        "H2O/MgO(001)",
        "H2O",
        "MgO(001) 2x2",
        "MgO",
        -520,
        121,
        203,
        47,
        3,
        (
            "LEED isotherms, monolayer minus lateral interactions: Ferry et al., Surf. Sci. 377-379, 634 (1997) and "
            "409, 101 (1998)"
        ),
    ),
    (
        "N2O/MgO(001)",
        "N2O",
        "MgO(001) 2x2",
        "MgO",
        -239,
        31,
        77,
        -6,
        1,
        "TPD: Lian et al., J. Phys. Chem. C 114, 3148 (2010)",
    ),
    (
        "NH3/MgO(001)",
        "NH3",
        "MgO(001) 2x2",
        "MgO",
        -613,
        65,
        160,
        44,
        3,
        "TPD: Arthur et al., J. Chem. Phys. 95, 8521 (1991)",
    ),
    (
        "CH4/TiO2 rutile(110)",
        "CH4",
        "TiO2 rutile(110) p(4x2)",
        "TiO2",
        -249,
        34,
        85,
        6,
        11,
        "TPD, low coverage: Chen, Smith, Kay, Dohnálek, Surf. Sci. 650, 83 (2016)",
    ),
    (
        "CO2/TiO2 rutile(110) tilted",
        "CO2",
        "TiO2 rutile(110) p(4x2)",
        "TiO2",
        -493,
        62,
        177,
        -2,
        4,
        "TPD: Thompson, Diwald, Yates, J. Phys. Chem. B 107, 11700 (2003); re-analysed by Campbell and Sellers",
    ),
    (
        "H2O/TiO2 rutile(110)",
        "H2O",
        "TiO2 rutile(110) p(4x2)",
        "TiO2",
        -917,
        111,
        303,
        65,
        5,
        (
            "TPD, low coverage: Hugenschmidt, Gamble, Campbell, Surf. Sci. 302, 329 (1994); Dohnálek et al., J. Phys. "
            "Chem. B 110, 6229 (2006); re-analysed by Campbell and Sellers"
        ),
    ),
    (
        "CH3OH/TiO2 rutile(110)",
        "CH3OH",
        "TiO2 rutile(110) p(4x2)",
        "TiO2",
        -1197,
        130,
        370,
        48,
        7,
        "TPD: Li, Smith, Kay, Dohnálek, J. Phys. Chem. C 115, 22534 (2011); re-analysed by Campbell and Sellers",
    ),
    (
        "H2O/TiO2 anatase(101)",
        "H2O",
        "TiO2 anatase(101) 3x1",
        "TiO2",
        -786,
        90,
        257,
        69,
        13,
        (
            "TPD: Herman, Dohnálek, Ruzycki, Diebold, J. Phys. Chem. B 107, 2788 (2003); re-analysed by Campbell and "
            "Sellers"
        ),
    ),
    (
        "NH3/TiO2 anatase(101)",
        "NH3",
        "TiO2 anatase(101) 3x1",
        "TiO2",
        -1180,
        182,
        410,
        75,
        7,
        "TPD, lowest coverage: Koust et al., J. Chem. Phys. 148, 124704 (2018)",
    ),
]


def _coefficient(value: Any) -> float:
    from fractions import Fraction

    return float(Fraction(value)) if isinstance(value, str) else float(value)


def reactions() -> list[dict[str, Any]]:
    """The 54 reactions with their reference energies."""
    entries = []
    for number, (category, mixed, equation, terms, n_ads, energy, ce39) in enumerate(ADS41_REACTIONS, start=1):
        names = [name for name in terms if name in ADSORBED]
        entries.append(
            {
                "id": f"ADS41-{number:02d}",
                "subset": "ADS41",
                "category": category,
                "mixed": mixed,
                "equation": equation,
                "terms": {name: _coefficient(c) for name, c in terms.items()},
                "adsorbates_per_reaction": n_ads,
                "reference": {
                    "energy": energy,
                    "quantity": "experimental reaction energy minus the PBE zero-point energy change",
                    "source": SHARADA
                    + ("" if ce39 is None else f"; CE39 reaction {ce39} (Wellendorff et al. Table 4)"),
                },
                "configuration": " ".join(ADSORBED[name]["configuration"] for name in names),
            }
        )
    for number, (name, gas, slab, category, h_ads, h_error, temperature, d_h, d_h_error, experiment) in enumerate(
        SURF13_REACTIONS, start=1
    ):
        entries.append(
            {
                "id": f"Surf13-{number:02d}",
                "subset": "Surf13",
                "category": category,
                "mixed": False,
                "equation": f"{gas} + {SURFACES[slab]} -> {gas}/{SURFACES[slab]}",
                "terms": {name: 1.0, slab: -1.0, gas: -1.0},
                "adsorbates_per_reaction": 1,
                "reference": {
                    "energy": round((h_ads - d_h) / 1000, 3),
                    "quantity": "experimental adsorption enthalpy minus the DFT zero-point, thermal and -RT terms",
                    "enthalpy": h_ads / 1000,
                    "enthalpy_uncertainty": h_error / 1000,
                    "temperature": temperature,
                    "vibrational_enthalpy": d_h / 1000,
                    "vibrational_enthalpy_uncertainty": d_h_error / 1000,
                    "source": f"{SHI}, SI Table 32 (H_ads) and Table 30 (Delta H)",
                    "experiment": experiment,
                },
                "configuration": ADSORBED[name]["configuration"],
            }
        )
    return entries


# ----------------------------------------------------------------------------------------------------------
# Checks, output and drawings
# ----------------------------------------------------------------------------------------------------------


def build(name: str) -> Atoms:
    """A slab, an adsorbed structure or a molecule of the dataset, at the experimental lattice constants."""
    if name in MOLECULES:
        return MOLECULES[name][0].copy()
    if name in SLABS:
        definition = SLABS[name]
        return build_slab(bulk_crystal(**CRYSTALS[definition["crystal"]]), definition)
    entry = ADSORBED[name]
    return add_adsorbates(build(entry["slab"]), entry["adsorbates"])


def check(entries: list[dict[str, Any]]) -> None:
    """Element balance of every reaction, and no adsorbate atom closer than 1 Å to another atom."""
    from collections import Counter

    for entry in entries:
        balance: Counter[str] = Counter()
        for name, coefficient in entry["terms"].items():
            for symbol, count in Counter(build(name).get_chemical_symbols()).items():
                balance[symbol] += coefficient * count
        if any(abs(v) > 1e-9 for v in balance.values()):
            raise AssertionError(f"{entry['id']} is not balanced: {dict(balance)}")
    for name in ADSORBED:
        atoms = build(name)
        adsorbate = atoms.get_tags() == 0
        i, j, d = neighbor_list("ijd", atoms, 4.0)
        to_surface = adsorbate[i] & ~adsorbate[j]
        k = int(np.argmin(np.where(to_surface, d, np.inf)))
        within = adsorbate[i] & adsorbate[j] & (d < 0.8)  # adsorbate atoms on top of each other
        pair = f"{atoms[i[k]].symbol}-{atoms[j[k]].symbol}"
        print(f"{name:32s} {len(atoms):4d} atoms, nearest surface atom {pair} {d[k]:.2f} Å")
        if d[k] < 1.0 or within.any():
            raise AssertionError(f"{name}: atoms too close")


def draw(directory: Path) -> None:
    """Top and side views of every adsorbed structure (PNG), and an index page."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from ase.visualize.plot import plot_atoms

    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, entry in ADSORBED.items():
        atoms = build(name)
        figure, axes = plt.subplots(1, 2, figsize=(9, 4.5))
        plot_atoms(atoms, axes[0], rotation="0x,0y,0z", radii=0.6)
        plot_atoms(atoms, axes[1], rotation="-90x,0y,0z", radii=0.6)
        for axis, title in zip(axes, ("top", "side"), strict=True):
            axis.set_title(title)
            axis.set_axis_off()
        figure.suptitle(name)
        file = directory / (name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "") + ".png")
        figure.savefig(file, dpi=110, bbox_inches="tight")
        plt.close(figure)
        rows.append(f"<h3>{name}</h3><p>{entry['configuration']}</p><img src='{file.name}' width='800'>")
    (directory / "index.html").write_text("<html><body>" + "\n".join(rows) + "</body></html>\n")
    print(f"drew {len(rows)} structures into {directory}")


def main() -> None:
    """Build the dataset, check it and write it (and the drawings)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path, help="JSON file to write")
    parser.add_argument("--images", type=Path, help="directory for drawings of the adsorbed structures")
    args = parser.parse_args()

    check_shi_slabs()
    ads41_structures()
    surf13_structures()
    entries = reactions()
    check(entries)
    used = {name for entry in entries for name in entry["terms"]}
    dataset = {
        "description": (
            "Reaction energies of molecules adsorbing on metal and oxide surfaces, against experiment: ADS41 "
            f"({SHARADA}) and Surf13 ({SHI}). Built by scripts/build_adsorption_dataset.py; energies in eV, "
            "lengths in Å."
        ),
        "crystals": CRYSTALS,
        "slabs": {name: SLABS[name] for name in SLABS if name in used},
        "molecules": {
            name: {
                "symbols": atoms.get_chemical_symbols(),
                "positions": np.round(atoms.positions - atoms.positions.mean(axis=0), 4).tolist(),
                "multiplicity": multiplicity,
            }
            for name, (atoms, multiplicity) in MOLECULES.items()
            if name in used
        },
        "adsorbed": ADSORBED,
        "reactions": entries,
    }
    unused = set(ADSORBED) - used
    if unused:
        raise AssertionError(f"adsorbed structures used by no reaction: {sorted(unused)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {len(entries)} reactions to {args.output}")
    if args.images:
        draw(args.images)


if __name__ == "__main__":
    main()
