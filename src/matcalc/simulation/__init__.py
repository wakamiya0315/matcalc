"""Simulators: how the benchmarks evaluate the potential energy surface of an MLIP.

- ``ASESimulator``: any ASE calculator, one structure at a time (the reference implementation).
- ``TorchSimSimulator``: a TorchSim model, many structures per GPU forward pass (needs
  ``torch-sim-atomistic``; imported on first use).
"""

from __future__ import annotations

from typing import Any

from ase.calculators.calculator import Calculator

from .ase import ASESimulator
from .base import RelaxResult, Simulator, SinglePointResult


def as_simulator(model: Any) -> Simulator:
    """Turn what the user passed to ``Benchmark.run`` into a simulator.

    Args:
        model: A simulator (anything with ``relax`` and ``single_point``), an ASE calculator, or a
            TorchSim model.

    Returns:
        A simulator ready to use.

    Raises:
        TypeError: If ``model`` is none of the above.
    """
    if isinstance(model, Calculator):
        return ASESimulator(model)
    if callable(getattr(model, "relax", None)) and callable(getattr(model, "single_point", None)):
        return model
    if _is_torchsim_model(model):
        from .torchsim import TorchSimSimulator

        return TorchSimSimulator(model)
    raise TypeError(
        f"Cannot use {type(model).__name__} as a model: pass an ASE calculator, a TorchSim model or a simulator."
    )


def _is_torchsim_model(model: Any) -> bool:
    try:
        from torch_sim.models.interface import ModelInterface
    except ImportError:
        return False
    return isinstance(model, ModelInterface)


def __getattr__(name: str) -> Any:
    # TorchSimSimulator is imported lazily so that matcalc works without torch-sim installed.
    if name == "TorchSimSimulator":
        from .torchsim import TorchSimSimulator

        return TorchSimSimulator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ASESimulator",
    "RelaxResult",
    "Simulator",
    "SinglePointResult",
    "TorchSimSimulator",
    "as_simulator",
]
