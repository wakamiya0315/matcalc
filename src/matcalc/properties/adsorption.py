"""Reaction energies of adsorption, from the total energies of slabs, adsorbed slabs and molecules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase.geometry import find_mic

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ase import Atoms


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
    """Largest distance an adsorbate atom (tag 0) moved in a relaxation, through the periodic boundaries.

    A value above about 1 Å signals that the adsorbate left the configuration it started from (another site,
    another orientation, desorption or dissociation).

    Args:
        initial: The structure before the relaxation, adsorbate atoms tagged 0.
        relaxed_positions: Cartesian positions after the relaxation, in the same order (Å).

    Returns:
        The largest displacement (Å); 0 without adsorbate atoms.
    """
    adsorbate = initial.get_tags() == 0
    if not adsorbate.any():
        return 0.0
    moves, lengths = find_mic(np.asarray(relaxed_positions)[adsorbate] - initial.positions[adsorbate], initial.cell)
    return float(np.max(lengths)) if len(moves) else 0.0
