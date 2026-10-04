"""Relaxations and single points with an ASE calculator, one structure at a time.

This reproduces what upstream matcalc does (``RelaxCalc`` + ``backend._ase.run_ase``): ASE's FIRE
optimizer acting on a ``FrechetCellFilter``, so that atoms and cell relax together. Relaxations in a
fixed cell (slabs, molecules in a box) run FIRE on the atoms alone.
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
        structures: Sequence[Structure | Atoms],
        *,
        fmax: float,
        max_steps: int,
        fix_symmetry: bool = False,
        symprec: float = 0.01,
        relax_cell: bool = True,
    ) -> list[RelaxResult]:
        """Relax atoms and cell of every structure with FIRE on a ``FrechetCellFilter``.

        Args:
            structures: Structures to relax. Constraints of ASE ``Atoms`` (such as ``FixAtoms``) are kept.
            fmax: FIRE stops when every force on atoms and cell is below this (eV/Å).
            max_steps: FIRE gives up after this many steps.
            fix_symmetry: Keep the space group of each structure with ASE's ``FixSymmetry`` constraint.
            symprec: Symmetry tolerance used to find the space group (Å).
            relax_cell: Relax the cell as well; with ``False`` FIRE moves only the atoms, in the fixed cell.

        Returns:
            One ``RelaxResult`` per structure, in input order.
        """
        return [
            self._relax_one(
                structure,
                fmax=fmax,
                max_steps=max_steps,
                fix_symmetry=fix_symmetry,
                symprec=symprec,
                relax_cell=relax_cell,
            )
            for structure in tqdm(structures, desc="relax", disable=not self.show_progress)
        ]

    def single_point(
        self, structures: Sequence[Structure | Atoms], *, compute_stress: bool = True
    ) -> list[SinglePointResult]:
        """Energy, forces and (optionally) stress of every structure.

        Args:
            structures: Structures to evaluate (constraints are ignored: the forces are those of the
                potential).
            compute_stress: Also compute the stress tensor.

        Returns:
            One ``SinglePointResult`` per structure, in input order.
        """
        return [
            self._single_point_one(structure, compute_stress=compute_stress)
            for structure in tqdm(structures, desc="single point", disable=not self.show_progress)
        ]

    def _relax_one(
        self,
        structure: Structure | Atoms,
        *,
        fmax: float,
        max_steps: int,
        fix_symmetry: bool,
        symprec: float,
        relax_cell: bool,
    ) -> RelaxResult:
        try:
            atoms = to_ase_atoms(structure)
            atoms.calc = self.calculator
            if fix_symmetry:
                atoms.set_constraint([*atoms.constraints, FixSymmetry(atoms, symprec=symprec)])
            optimizable = FrechetCellFilter(atoms) if relax_cell else atoms
            optimizer = FIRE(optimizable, logfile=None)
            optimizer.run(fmax=fmax, steps=max_steps)
            # FIRE's stopping test on the forces it moves along (atoms, and the cell with the filter)
            optimizer_converged = bool((optimizable.get_forces() ** 2).sum(axis=1).max() < fmax**2)
            forces = atoms.get_forces()  # zero on fixed atoms, as FIRE saw them
            energy = float(atoms.get_potential_energy())
            stress = atoms.get_stress(voigt=False) if relax_cell else None
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
            forces = atoms.get_forces(apply_constraint=False)  # the potential's forces, as TorchSim returns them
            stress = atoms.get_stress(voigt=False) if compute_stress else None
        except Exception as exc:  # noqa: BLE001 - one bad structure must not stop a whole benchmark
            logger.warning("Single point failed: %s: %s", type(exc).__name__, exc)
            return SinglePointResult.failed(f"{type(exc).__name__}: {exc}")
        return SinglePointResult(energy=energy, forces=forces, stress=stress)
