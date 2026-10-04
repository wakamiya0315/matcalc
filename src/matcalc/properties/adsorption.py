"""Reaction energies of adsorption, from the total energies of slabs, adsorbed slabs and molecules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase.data import covalent_radii
from ase.geometry import find_mic

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ase import Atoms
    from ase.cell import Cell

BOND_FACTOR = 1.2
"""Two adsorbate atoms are bonded when closer than this times the sum of their covalent radii (ASE's): 1.16 Å
for O-H, so that hydrogen bonds (about 1.8 Å) are not bonds."""


def reaction_energy(terms: Mapping[str, float], energies: Mapping[str, float]) -> float:
    """Energy of a reaction written as a linear combination of structures, ΔE = Σ_i c_i E_i.

    For CO + Pt(111) -> CO/Pt(111) the terms are ``{"CO/Pt(111)": 1, "Pt(111)": -1, "CO": -1}``: products
    count positive, reactants negative. Dissociated molecules are computed as separate adlayers, each with its
    clean slab subtracted (for NO + Ni(100) -> N/Ni(100) + O/Ni(100): ``{"N/Ni(100)": 1, "O/Ni(100)": 1,
    "Ni(100)": -2, "NO": -1}``).

    Args:
        terms: Structure name → stoichiometric coefficient.
        energies: Structure name → total energy (eV).

    Returns:
        The reaction energy (eV); negative when adsorption releases energy.
    """
    return float(sum(coefficient * energies[name] for name, coefficient in terms.items()))


def adsorbate_displacement(initial: Atoms, relaxed_positions: np.ndarray) -> float:
    """Largest distance a heavy atom (not H) of the adsorbate (tag 0) moved in a relaxation, through the periodic
    boundaries; for an adsorbate of H atoms alone, the largest distance an H atom moved.

    H atoms that rotate about their heavy atom (a methyl group, the H of water) move by up to about 1.5 Å while
    the adsorbate stays where it is, so they do not count. A heavy atom moving by more than about 1 Å signals that
    the adsorbate left the configuration it started from (another site or orientation, desorption);
    ``adsorbate_rearranged`` detects dissociation and H atoms changing partners.

    Args:
        initial: The structure before the relaxation, adsorbate atoms tagged 0.
        relaxed_positions: Cartesian positions after the relaxation, in the same order (Å).

    Returns:
        The largest displacement (Å); 0 without adsorbate atoms.
    """
    adsorbate = initial.get_tags() == 0
    if not adsorbate.any():
        return 0.0
    heavy = adsorbate & (initial.numbers != 1)
    moving = heavy if heavy.any() else adsorbate
    _, lengths = find_mic(np.asarray(relaxed_positions)[moving] - initial.positions[moving], initial.cell, initial.pbc)
    return float(np.max(lengths))


def adsorbate_rearranged(initial: Atoms, relaxed_positions: np.ndarray) -> bool:
    """Whether a relaxation broke or formed a bond between atoms of the adsorbate (tag 0).

    This catches dissociation, an H atom moving from one adsorbed molecule to another (a proton hopping within a
    water network counts) and new bonds within the adsorbate. Bonds to the surface do not count: an H atom that
    leaves its molecule for a surface atom shows as the broken bond within the molecule.

    Args:
        initial: The structure before the relaxation, adsorbate atoms tagged 0.
        relaxed_positions: Cartesian positions after the relaxation, in the same order (Å).

    Returns:
        ``True`` if the bonds between adsorbate atoms changed (bonded: closer than ``BOND_FACTOR`` times the sum
        of the covalent radii, through the periodic boundaries).
    """
    adsorbate = np.flatnonzero(initial.get_tags() == 0)
    if len(adsorbate) < 2:  # noqa: PLR2004 - a bond needs two atoms
        return False
    numbers = initial.numbers[adsorbate]
    before = _bonds(initial.positions[adsorbate], numbers, initial.cell, initial.pbc)
    after = _bonds(np.asarray(relaxed_positions)[adsorbate], numbers, initial.cell, initial.pbc)
    return bool((before != after).any())


def _bonds(positions: np.ndarray, numbers: np.ndarray, cell: Cell, pbc: np.ndarray) -> np.ndarray:
    """Which pairs of atoms are bonded (minimum-image distance below the bond threshold), as a boolean matrix."""
    n = len(positions)
    _, lengths = find_mic((positions[:, None, :] - positions[None, :, :]).reshape(-1, 3), cell, pbc)
    radii = covalent_radii[numbers]
    bonded = lengths.reshape(n, n) < BOND_FACTOR * (radii[:, None] + radii[None, :])
    np.fill_diagonal(bonded, val=False)
    return bonded
