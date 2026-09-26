"""What a simulator returns, and the two operations every simulator provides.

A *simulator* evaluates the potential energy surface (PES) of a machine-learning interatomic
potential (MLIP). The benchmarks only ever ask it for two things:

- ``relax``: relax the atomic positions and the cell of many structures;
- ``single_point``: energy, forces and stress of many structures at fixed geometry.

Units follow ASE: energies in eV, forces in eV/Å, stresses in eV/Å^3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from ase import Atoms
    from pymatgen.core import Structure


@dataclass
class SinglePointResult:
    """Energy, forces and stress of one structure at fixed geometry.

    Attributes:
        energy: Potential energy of the whole cell (eV). NaN if the calculation failed.
        forces: Force on every atom, shape ``(n_atoms, 3)`` (eV/Å).
        stress: Stress tensor, shape ``(3, 3)`` (eV/Å^3), with ASE's sign convention (positive
            under tension). ``None`` when stress was not requested.
        error: Why the calculation failed, or ``None`` if it succeeded.
    """

    energy: float
    forces: np.ndarray | None
    stress: np.ndarray | None = None
    error: str | None = None

    @classmethod
    def failed(cls, error: str) -> SinglePointResult:
        """Result for a calculation that raised an exception.

        Args:
            error: Description of what went wrong.

        Returns:
            A result with NaN energy carrying ``error``.
        """
        return cls(math.nan, None, None, error)


@dataclass
class RelaxResult:
    """Outcome of relaxing the atoms and the cell of one structure.

    Attributes:
        structure: The relaxed structure (``None`` if the relaxation failed).
        energy: Potential energy of the relaxed cell (eV).
        forces: Forces at the relaxed geometry, shape ``(n_atoms, 3)`` (eV/Å).
        stress: Stress at the relaxed geometry, shape ``(3, 3)`` (eV/Å^3).
        max_force: Largest force on any atom at the end (eV/Å).
        converged: ``True`` when ``max_force <= fmax``. As in upstream matcalc, only the atomic
            forces are checked here, not the residual stress on the cell.
        n_steps: Number of optimizer steps taken.
        optimizer_converged: The optimizer's own stopping criterion was met: every force on the atoms
            and on the cell of the Frechet cell filter is below ``fmax`` (the Phonon benchmark's test).
        error: Why the relaxation failed, or ``None`` if it ran.
    """

    structure: Structure | None
    energy: float
    forces: np.ndarray | None
    stress: np.ndarray | None
    max_force: float
    converged: bool
    n_steps: int
    optimizer_converged: bool = False
    error: str | None = None

    @classmethod
    def failed(cls, error: str) -> RelaxResult:
        """Result for a relaxation that raised an exception.

        Args:
            error: Description of what went wrong.

        Returns:
            A non-converged result carrying ``error``.
        """
        return cls(None, math.nan, None, None, math.nan, converged=False, n_steps=0, error=error)


class Simulator(Protocol):
    """The two operations the benchmarks need from a simulator."""

    def relax(
        self,
        structures: Sequence[Structure],
        *,
        fmax: float,
        max_steps: int,
        fix_symmetry: bool = False,
        symprec: float = 0.01,
    ) -> list[RelaxResult]:
        """Relax atomic positions and cell of every structure (FIRE with a Frechet cell filter).

        Args:
            structures: Structures to relax.
            fmax: The optimizer stops when every force on atoms and cell is below this (eV/Å).
            max_steps: The optimizer gives up after this many steps.
            fix_symmetry: Keep the space group of each structure (forces, stress and steps are
                symmetrized, as ASE's ``FixSymmetry`` constraint does).
            symprec: Symmetry tolerance used to find the space group (Å).

        Returns:
            One ``RelaxResult`` per structure, in input order.
        """

    def single_point(
        self, structures: Sequence[Structure | Atoms], *, compute_stress: bool = True
    ) -> list[SinglePointResult]:
        """Energy, forces and (optionally) stress of every structure at fixed geometry.

        Args:
            structures: Structures to evaluate.
            compute_stress: Also compute the stress tensor.

        Returns:
            One ``SinglePointResult`` per structure, in input order.
        """
