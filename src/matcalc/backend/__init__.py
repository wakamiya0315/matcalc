"""Provides the backend used to run simulations (ASE)."""

from __future__ import annotations

from ._ase import run_ase
from ._base import SimulationResult  # noqa: TC001


def run_pes_calc(*arg, **kwargs) -> SimulationResult:  # noqa:ANN002,ANN003
    """
    Executes the potential energy surface (PES) calculation with the ASE backend.

    Args:
        *arg: Positional arguments forwarded to ``run_ase``.
        **kwargs: Keyword arguments forwarded to ``run_ase``.

    Returns:
        SimulationResult from the ASE backend.
    """
    return run_ase(*arg, **kwargs)
