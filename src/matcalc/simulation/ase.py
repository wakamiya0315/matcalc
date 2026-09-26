"""Relaxations and single points with an ASE calculator, one structure at a time.

This reproduces what upstream matcalc does (``RelaxCalc`` + ``backend._ase.run_ase``): ASE's FIRE
optimizer acting on a ``FrechetCellFilter``, so that atoms and cell relax together.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from tqdm import tqdm

from matcalc.structures import to_ase_atoms, to_pmg_structure

from .base import RelaxResult, SinglePointResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


class ASESimulator:
    """Evaluate structures with an ASE calculator, one structure at a time.

    This is the reference implementation: it follows upstream matcalc step by step. It works with
    any ASE calculator but keeps a GPU mostly idle, because every call holds a single small cell.

    Attributes:
        calculator: ASE calculator of the MLIP.
        show_progress: Show a progress bar over structures.
    """

    def __init__(self, calculator: Calculator, *, show_progress: bool = True) -> None:
        """
        Args:
            calculator: ASE calculator of the MLIP.
            show_progress: Show a progress bar over structures.
        """
        self.calculator = calculator
        self.show_progress = show_progress

    def relax(
        self,
        structures: Sequence[Structure],
        *,
        fmax: float,
        max_steps: int,
        fix_symmetry: bool = False,
        symprec: float = 0.01,
    ) -> list[RelaxResult]:
        """Relax atoms and cell of every structure with FIRE on a ``FrechetCellFilter``.

        Args:
            structures: Structures to relax.
            fmax: FIRE stops when every force on atoms and cell is below this (eV/Å).
            max_steps: FIRE gives up after this many steps.
            fix_symmetry: Keep the space group of each structure with ASE's ``FixSymmetry`` constraint.
            symprec: Symmetry tolerance used to find the space group (Å).

        Returns:
            One ``RelaxResult`` per structure, in input order.
        """
        return [
            self._relax_one(structure, fmax=fmax, max_steps=max_steps, fix_symmetry=fix_symmetry, symprec=symprec)
            for structure in tqdm(structures, desc="relax", disable=not self.show_progress)
        ]

    def single_point(
        self, structures: Sequence[Structure | Atoms], *, compute_stress: bool = True
    ) -> list[SinglePointResult]:
        """Energy, forces and (optionally) stress of every structure.

        Args:
            structures: Structures to evaluate.
            compute_stress: Also compute the stress tensor.

        Returns:
            One ``SinglePointResult`` per structure, in input order.
        """
        return [
            self._single_point_one(structure, compute_stress=compute_stress)
            for structure in tqdm(structures, desc="single point", disable=not self.show_progress)
        ]

    def _relax_one(
        self, structure: Structure | Atoms, *, fmax: float, max_steps: int, fix_symmetry: bool, symprec: float
    ) -> RelaxResult:
        try:
            atoms = to_ase_atoms(structure)
            atoms.calc = self.calculator
            if fix_symmetry:
                atoms.set_constraint(FixSymmetry(atoms, symprec=symprec))
            cell_filter = FrechetCellFilter(atoms)
            optimizer = FIRE(cell_filter, logfile=None)
            optimizer.run(fmax=fmax, steps=max_steps)
            # FIRE's stopping test on the forces of the filter (atoms and cell)
            optimizer_converged = bool((cell_filter.get_forces() ** 2).sum(axis=1).max() < fmax**2)
            forces = atoms.get_forces()
            energy = float(atoms.get_potential_energy())
            stress = atoms.get_stress(voigt=False)
        except Exception as exc:  # noqa: BLE001 - one bad structure must not stop a whole benchmark
            logger.warning("Relaxation failed: %s: %s", type(exc).__name__, exc)
            return RelaxResult.failed(f"{type(exc).__name__}: {exc}")
        max_force = float(np.linalg.norm(forces, axis=1).max())
        # pymatgen would keep a reference to the calculator and the constraint on the structure.
        atoms.calc = None
        atoms.set_constraint()
        return RelaxResult(
            structure=to_pmg_structure(atoms),
            energy=energy,
            forces=forces,
            stress=stress,
            max_force=max_force,
            converged=max_force <= fmax,
            n_steps=optimizer.nsteps,
            optimizer_converged=optimizer_converged,
        )

    def _single_point_one(self, structure: Structure | Atoms, *, compute_stress: bool) -> SinglePointResult:
        try:
            atoms = to_ase_atoms(structure)
            atoms.calc = self.calculator
            energy = float(atoms.get_potential_energy())
            forces = atoms.get_forces()
            stress = atoms.get_stress(voigt=False) if compute_stress else None
        except Exception as exc:  # noqa: BLE001 - one bad structure must not stop a whole benchmark
            logger.warning("Single point failed: %s: %s", type(exc).__name__, exc)
            return SinglePointResult.failed(f"{type(exc).__name__}: {exc}")
        return SinglePointResult(energy=energy, forces=forces, stress=stress)
