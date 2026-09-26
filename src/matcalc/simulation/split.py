"""A simulator that relaxes with one simulator and evaluates single points with another."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms
    from pymatgen.core import Structure

    from .base import RelaxResult, Simulator, SinglePointResult


class SplitSimulator:
    """Relaxations with one simulator, single points with another.

    Typical use: relax in float64, so that the relaxed structures keep their symmetry exactly and match
    a float64 reference, and compute the many single points of a benchmark (phonon supercells, strained
    cells, softening frames) with a faster float32 model.

    Attributes:
        relaxer: Simulator used for ``relax``.
        evaluator: Simulator used for ``single_point``.
    """

    def __init__(self, relaxer: Simulator, evaluator: Simulator) -> None:
        """
        Args:
            relaxer: Simulator used for ``relax``.
            evaluator: Simulator used for ``single_point``.
        """
        self.relaxer = relaxer
        self.evaluator = evaluator

    @property
    def batched(self) -> bool:
        """Whether either simulator evaluates many structures per call (benchmarks then use large chunks)."""
        return bool(getattr(self.relaxer, "batched", False) or getattr(self.evaluator, "batched", False))

    def relax(self, structures: Sequence[Structure], *, fmax: float, max_steps: int) -> list[RelaxResult]:
        """Relax with ``relaxer`` (see ``Simulator.relax``)."""
        return self.relaxer.relax(structures, fmax=fmax, max_steps=max_steps)

    def single_point(
        self, structures: Sequence[Structure | Atoms], *, compute_stress: bool = True
    ) -> list[SinglePointResult]:
        """Evaluate with ``evaluator`` (see ``Simulator.single_point``)."""
        return self.evaluator.single_point(structures, compute_stress=compute_stress)
