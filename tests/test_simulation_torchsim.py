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

from torch_sim.autobatching import calculate_memory_scalers  # noqa: E402
from torch_sim.models.lennard_jones import LennardJonesModel  # noqa: E402

from matcalc.simulation import SplitSimulator, as_simulator  # noqa: E402
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
        assert got.energy == pytest.approx(ref.energy, abs=1e-9)
        assert_allclose(got.structure.lattice.matrix, ref.structure.lattice.matrix, atol=1e-7)
        assert got.n_steps == ref.n_steps  # same FIRE and same convergence test as ASE


def test_already_relaxed_structures_are_not_moved() -> None:
    model = lj_model()
    (relaxed,) = TorchSimSimulator(model, show_progress=False).relax([rattled("Cu", 0)], fmax=0.01, max_steps=300)
    for simulator in (
        ASESimulator(TorchSimModelCalculator(model), show_progress=False),
        TorchSimSimulator(model, show_progress=False),
    ):
        (again,) = simulator.relax([relaxed.structure], fmax=0.01, max_steps=300)
        assert again.n_steps == 0
        assert again.energy == pytest.approx(relaxed.energy, abs=1e-10)


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


def test_out_of_memory_is_retried_with_half_the_capacity() -> None:
    simulator = TorchSimSimulator(lj_model(), max_memory_scaler=1e9, show_progress=False)
    state = simulator._state(starting_structures())
    capacities = []

    def run(capacity: float) -> str:
        capacities.append(capacity)
        if len(capacities) == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.62 GiB")
        return "done"

    assert simulator._batched(state, run) == "done"
    assert capacities == [1e9, 5e8]

    at_largest = TorchSimSimulator(lj_model(), max_memory_scaler=1e-3, show_progress=False)
    attempts = []

    def always_out_of_memory(capacity: float) -> str:
        attempts.append(capacity)
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        at_largest._batched(state, always_out_of_memory)
    assert len(attempts) == 2  # one more attempt at the smallest possible capacity, then give up
    tiny = TorchSimSimulator(lj_model(), max_memory_scaler=1e-3, show_progress=False)
    assert tiny._batched(state, lambda capacity: capacity) > 1e-3  # never below the largest structure

    def broken(capacity: float) -> str:
        raise RuntimeError(f"not memory ({capacity})")

    with pytest.raises(RuntimeError, match="not memory"):  # other errors are not retried
        simulator._batched(state, broken)


class _OutOfMemoryModel:
    """A model that 'runs out of memory' on batches of more than ``max_structures`` structures, or with a
    structure of more than ``max_atoms`` atoms, and records the size of every batch it sees."""

    def __init__(self, inner: Any, *, max_structures: int = 1000, max_atoms: int = 1000) -> None:
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "limits", (max_structures, max_atoms))
        object.__setattr__(self, "batches", [])

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self.inner, name, value)

    def __call__(self, state: Any) -> Any:
        self.batches.append(state.n_systems)
        max_structures, max_atoms = self.limits
        if state.n_systems > max_structures or int(torch.bincount(state.system_idx).max()) > max_atoms:
            raise RuntimeError("CUDA out of memory (test)")
        return self.inner(state)


def test_single_point_lowers_the_capacity_after_running_out_of_memory() -> None:
    model = lj_model()
    cells = [*starting_structures(), structure("Cu1")]
    reference = TorchSimSimulator(model, show_progress=False).single_point(cells)
    small = _OutOfMemoryModel(model, max_structures=2)
    simulator = TorchSimSimulator(small, show_progress=False)
    results = simulator.single_point(cells)
    for ref, got in zip(reference, results, strict=True):
        assert got.error is None
        assert got.energy == pytest.approx(ref.energy, abs=1e-12)
        assert_allclose(got.forces, ref.forces, atol=1e-12)
    assert simulator.capacities == sorted(simulator.capacities, reverse=True)  # only ever lowered

    # A capacity somewhat too large costs one failed batch, not one per batch.
    copies = [rattled("Cu", seed) for seed in range(40)]
    metric = calculate_memory_scalers(simulator._state(copies), memory_scales_with=model.memory_scales_with)
    four = _OutOfMemoryModel(model, max_structures=4)
    TorchSimSimulator(four, max_memory_scaler=6.5 * max(metric), show_progress=False).single_point(copies)
    assert sum(n > 4 for n in four.batches) == 1


def test_single_point_evaluates_structures_larger_than_the_capacity_alone() -> None:
    model = lj_model()
    cells = [*starting_structures(), structure("Cu1")]
    reference = TorchSimSimulator(model, show_progress=False).single_point(cells)
    recorder = _OutOfMemoryModel(model)
    results = TorchSimSimulator(recorder, max_memory_scaler=1e-3, show_progress=False).single_point(cells)
    assert recorder.batches == [1] * len(cells)
    for ref, got in zip(reference, results, strict=True):
        assert got.energy == pytest.approx(ref.energy, abs=1e-12)


def test_structures_that_do_not_fit_alone_fail_without_stopping_the_others() -> None:
    cells = [*starting_structures(), structure("Cu1")]  # 4, 4, 2, 4 and 1 atoms
    small = _OutOfMemoryModel(lj_model(), max_atoms=2)
    results = TorchSimSimulator(small, show_progress=False).single_point(cells)
    assert [r.error for r in results] == ["GPU out of memory", "GPU out of memory", None, "GPU out of memory", None]
    # One structure per batch: the two rattled fcc Cu cells have the same size, so after the first one
    # failed the second is not tried.
    alone = _OutOfMemoryModel(lj_model(), max_atoms=2)
    TorchSimSimulator(alone, max_memory_scaler=1e-3, show_progress=False).single_point(cells)
    assert len(alone.batches) == len(cells) - 1
    assert np.isfinite(results[2].energy)
    assert np.isfinite(results[4].energy)


def test_split_simulator_relaxes_with_one_and_evaluates_with_the_other() -> None:
    relaxer = TorchSimSimulator(lj_model(), show_progress=False)
    evaluator = TorchSimSimulator(_OutOfMemoryModel(lj_model()), show_progress=False)
    split = SplitSimulator(relaxer, evaluator)
    assert split.batched
    relaxed = split.relax(starting_structures()[:2], fmax=0.05, max_steps=50)
    assert [r.energy for r in relaxed] == [
        r.energy for r in relaxer.relax(starting_structures()[:2], fmax=0.05, max_steps=50)
    ]
    assert evaluator.model.batches == []  # relaxations did not touch the evaluator
    split.single_point(starting_structures())
    assert evaluator.model.batches
