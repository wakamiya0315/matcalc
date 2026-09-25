"""Simulators: how the benchmarks evaluate the potential energy surface of an MLIP."""

from __future__ import annotations

from typing import Any

from ase.calculators.calculator import Calculator

from .ase import ASESimulator
from .base import RelaxResult, Simulator, SinglePointResult


def as_simulator(model: Any) -> Simulator:
    """Turn what the user passed to ``Benchmark.run`` into a simulator.

    Args:
        model: A simulator (anything with ``relax`` and ``single_point``), an ASE calculator, or
            the name of a MACE model understood by ``matcalc.load_mace``.

    Returns:
        A simulator ready to use.

    Raises:
        TypeError: If ``model`` is none of the above.
    """
    if isinstance(model, str):
        from matcalc.models import load_mace

        model = load_mace(model)
    if isinstance(model, Calculator):
        return ASESimulator(model)
    if callable(getattr(model, "relax", None)) and callable(getattr(model, "single_point", None)):
        return model
    raise TypeError(
        f"Cannot use {type(model).__name__} as a model: pass an ASE calculator, a simulator, or a MACE model name."
    )


__all__ = ["ASESimulator", "RelaxResult", "Simulator", "SinglePointResult", "as_simulator"]
