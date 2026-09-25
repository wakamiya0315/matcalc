"""TorchSimSimulator against ASESimulator, both driving the same TorchSim Lennard-Jones model (CPU).

Wrapping one TorchSim model in a tiny ASE calculator makes the two simulators evaluate exactly the same
potential, so any difference comes from batching and from the optimizer implementation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from ase.calculators.calculator import Calculator, all_changes
from numpy.testing import assert_allclose

from matcalc import ASESimulator, ElasticityBenchmark, SofteningBenchmark

from .helpers import structure

ts = pytest.importorskip("torch_sim")
torch = pytest.importorskip("torch")

from torch_sim.models.lennard_jones import LennardJonesModel  # noqa: E402

from matcalc.simulation import as_simulator  # noqa: E402
from matcalc.simulation.torchsim import TorchSimSimulator  # noqa: E402
from matcalc.structures import to_ase_atoms  # noqa: E402


def lj_model() -> LennardJonesModel:
    # sigma = 2.3 A puts the Lennard-Jones minimum (2^(1/6) sigma = 2.58 A) near the Cu-Cu distance.
    return LennardJonesModel(
        sigma=2.3, epsilon=0.1, cutoff=6.0, device=torch.device("cpu"), dtype=torch.float64, compute_stress=True
    )


class TorchSimModelCalculator(Calculator):
    """ASE calculator that evaluates a TorchSim model on one structure."""

    implemented_properties = ("energy", "free_energy", "forces", "stress")

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model

    def calculate(self, atoms: Any = None, properties: Any = None, system_changes: Any = all_changes) -> None:
        super().calculate(atoms, properties, system_changes)
        out = self.model(ts.io.atoms_to_state([self.atoms], device=self.model.device, dtype=self.model.dtype))
        s = out["stress"][0].detach().cpu().numpy()
        energy = float(out["energy"][0])
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": out["forces"].detach().cpu().numpy(),
            "stress": np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]]),  # ASE Voigt order
        }


def rattled(formula: str, seed: int) -> Any:
    return structure(formula).copy().perturb(0.05, seed=seed)


def starting_structures() -> list[Any]:
    return [rattled("Cu", 0), rattled("Cu", 1), rattled("NiAl", 2), rattled("Cu3Au", 3)]


def test_single_point_matches_ase() -> None:
    model = lj_model()
    cells = [*starting_structures(), structure("Cu1")]
    reference = ASESimulator(TorchSimModelCalculator(model), show_progress=False).single_point(cells)
    batched = TorchSimSimulator(model, show_progress=False).single_point(cells)
    for ref, got in zip(reference, batched, strict=True):
        assert got.energy == pytest.approx(ref.energy, abs=1e-10)
        assert_allclose(got.forces, ref.forces, atol=1e-10)
        assert_allclose(got.stress, ref.stress, atol=1e-12)
    no_stress = TorchSimSimulator(model, show_progress=False).single_point(cells, compute_stress=False)
    assert all(r.stress is None for r in no_stress)
    assert model.compute_stress  # switched back on afterwards


def test_relax_matches_ase_fire() -> None:
    model = lj_model()
    starts = starting_structures()
    reference = ASESimulator(TorchSimModelCalculator(model), show_progress=False).relax(
        starts, fmax=0.01, max_steps=300
    )
    batched = TorchSimSimulator(model, show_progress=False).relax(starts, fmax=0.01, max_steps=300)
    for ref, got in zip(reference, batched, strict=True):
        assert got.converged == ref.converged
        assert got.energy == pytest.approx(ref.energy, abs=1e-6)
        assert_allclose(got.structure.lattice.matrix, ref.structure.lattice.matrix, atol=1e-4)
        assert abs(got.n_steps - ref.n_steps) <= 1


def test_structures_joining_a_running_batch_follow_the_same_path() -> None:
    """With room for two structures, later structures join a batch in which another one is running."""
    model = lj_model()
    starts = starting_structures()
    alone = [TorchSimSimulator(model, show_progress=False).relax([s], fmax=0.01, max_steps=300)[0] for s in starts]
    metric = ts.autobatching.calculate_memory_scalers(
        ts.io.atoms_to_state([to_ase_atoms(s) for s in starts], device=model.device, dtype=model.dtype)
    )
    squeezed = TorchSimSimulator(model, max_memory_scaler=2.2 * max(metric), show_progress=False)
    together = squeezed.relax(starts, fmax=0.01, max_steps=300)
    for one, many in zip(alone, together, strict=True):
        assert many.n_steps == one.n_steps
        assert many.energy == pytest.approx(one.energy, abs=1e-9)
        assert_allclose(many.structure.cart_coords, one.structure.cart_coords, atol=1e-7)


def test_max_steps_is_respected() -> None:
    (result,) = TorchSimSimulator(lj_model(), show_progress=False).relax([rattled("Cu", 5)], fmax=1e-8, max_steps=4)
    assert result.n_steps == 4
    assert not result.converged


def test_benchmarks_run_with_torchsim(elasticity_dataset: Any, softening_dataset: Any) -> None:
    model = lj_model()
    assert isinstance(as_simulator(model), TorchSimSimulator)
    reference = ASESimulator(TorchSimModelCalculator(model), show_progress=False)
    batched = TorchSimSimulator(model, show_progress=False)
    for benchmark, column in ((ElasticityBenchmark, "K_vrh"), (SofteningBenchmark, "softening_scale")):
        dataset = elasticity_dataset if benchmark is ElasticityBenchmark else softening_dataset
        ase_table = benchmark(dataset).run(reference, "lj")
        ts_table = benchmark(dataset).run(batched, "lj")
        assert list(ts_table["status_lj"]) == list(ase_table["status_lj"])
        assert_allclose(ts_table[f"{column}_lj"], ase_table[f"{column}_lj"], rtol=1e-3)
