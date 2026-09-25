"""Formation energies relative to the elemental ground states."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from monty.serialization import loadfn

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from pymatgen.core import Composition, Structure

MP_PBE_ELEMENT_REFS = Path(__file__).parents[1] / "elemental_refs" / "MP-PBE-Element-Refs.json.gz"
"""Materials Project PBE structures of every element (several polymorphs for some elements)."""


def elemental_reference_structures(elements: Iterable[str]) -> dict[str, list[Structure]]:
    """Candidate ground-state structures of each element.

    Args:
        elements: Element symbols.

    Returns:
        Element symbol → its candidate structures (all polymorphs in the MP-PBE reference set).
    """
    references = loadfn(MP_PBE_ELEMENT_REFS)
    candidates = {}
    for element in sorted(set(elements)):
        structures = references[element]["structure"]
        candidates[element] = structures if isinstance(structures, list) else [structures]
    return candidates


def formation_energy_per_atom(
    energy: float, composition: Composition, reference_energies: Mapping[str, float]
) -> float:
    """Formation energy per atom, E_form = (E - Σ_i n_i μ_i) / N.

    Args:
        energy: Total energy of the compound cell (eV).
        composition: Composition of that cell (n_i atoms of element i, N atoms in total).
        reference_energies: Energy per atom μ_i of each element in its ground state (eV/atom).

    Returns:
        Formation energy (eV/atom).
    """
    reference = sum(reference_energies[element.symbol] * amount for element, amount in composition.items())
    return (energy - reference) / composition.num_atoms
