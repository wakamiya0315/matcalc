"""MatCalc benchmarks: compare a machine-learning interatomic potential (MLIP) with DFT.

Four benchmarks are provided (see ``docs/benchmarks.md``):

- ``EquilibriumBenchmark``: relaxed structures and formation energies (WBM, PBE);
- ``ElasticityBenchmark``: bulk and shear moduli (Materials Project, PBE);
- ``PhononBenchmark``: heat capacity at 300 K from harmonic phonons (Alexandria, PBE);
- ``SofteningBenchmark``: systematic softening of forces on high-energy configurations (WBM).

Each benchmark asks a *simulator* for relaxations and single points; ``ASESimulator`` works with
any ASE calculator.

Example:
    >>> import matcalc
    >>> calculator = matcalc.load_mace("MACE-MatPES-PBE-0")
    >>> table = matcalc.ElasticityBenchmark(n_samples=10).run(calculator, "MACE")
"""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version

from .benchmarks import (
    BENCHMARKS,
    Benchmark,
    ElasticityBenchmark,
    EquilibriumBenchmark,
    PhononBenchmark,
    SofteningBenchmark,
    run_benchmarks,
)
from .config import clear_cache
from .models import load_mace
from .simulation import ASESimulator, RelaxResult, Simulator, SinglePointResult

# Library convention: matcalc.* loggers stay silent unless the application configures logging.
logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    __version__ = version("matcalc")
except PackageNotFoundError:
    pass  # package not installed (e.g. used from a source checkout via PYTHONPATH)

__all__ = [
    "BENCHMARKS",
    "ASESimulator",
    "Benchmark",
    "ElasticityBenchmark",
    "EquilibriumBenchmark",
    "PhononBenchmark",
    "RelaxResult",
    "Simulator",
    "SinglePointResult",
    "SofteningBenchmark",
    "clear_cache",
    "load_mace",
    "run_benchmarks",
]
