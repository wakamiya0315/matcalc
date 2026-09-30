"""MatCalc benchmarks: compare a machine-learning interatomic potential (MLIP) with DFT and coupled-cluster data.

The benchmarks (see ``docs/benchmarks.md``):

- ``EquilibriumBenchmark``: relaxed structures and formation energies (WBM, PBE);
- ``ElasticityBenchmark``: bulk and shear moduli (Materials Project, PBE);
- ``PhononBenchmark``: heat capacity at 300 K from harmonic phonons (Alexandria, PBE);
- ``SofteningBenchmark``: systematic softening of forces on high-energy configurations (WBM);
- ``DiscoveryBenchmark``: stability of hypothetical crystals and their relaxed geometry (WBM, Matbench
  Discovery);
- ``KappaBenchmark``: lattice thermal conductivity at 300 K (PhononDB, Matbench Discovery's κ_SRME);
- ``DiatomicsBenchmark``: potential-energy curves of homonuclear dimers (PBE, Matbench Discovery);
- ``NoncovalentBenchmark``: interaction energies of noncovalent complexes (NCI Atlas, CCSD(T)/CBS);
- ``ConformerBenchmark``: relative energies of conformers of drug-like molecules (DLPNO-CCSD(T));
- ``ReactionBenchmark``: barrier heights and reaction energies of organic reactions (RDB7, CCSD(T)-F12a).

Each benchmark asks a *simulator* for relaxations and single points: ``ASESimulator`` works with any
ASE calculator, ``matcalc.simulation.TorchSimSimulator`` with any TorchSim model. The MLIP itself is
provided by the user.

Example:
    >>> import matcalc
    >>> calculator = ...  # the ASE calculator of the MLIP to benchmark
    >>> table = matcalc.ElasticityBenchmark(n_samples=10).run(calculator, "my-mlip")
"""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version

from .benchmarks import (
    BENCHMARKS,
    Benchmark,
    ConformerBenchmark,
    DiatomicsBenchmark,
    DiscoveryBenchmark,
    ElasticityBenchmark,
    EquilibriumBenchmark,
    KappaBenchmark,
    NoncovalentBenchmark,
    PhononBenchmark,
    ReactionBenchmark,
    SofteningBenchmark,
    run_benchmarks,
)
from .config import clear_cache
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
    "ConformerBenchmark",
    "DiatomicsBenchmark",
    "DiscoveryBenchmark",
    "ElasticityBenchmark",
    "EquilibriumBenchmark",
    "KappaBenchmark",
    "NoncovalentBenchmark",
    "PhononBenchmark",
    "ReactionBenchmark",
    "RelaxResult",
    "Simulator",
    "SinglePointResult",
    "SofteningBenchmark",
    "clear_cache",
    "run_benchmarks",
]
