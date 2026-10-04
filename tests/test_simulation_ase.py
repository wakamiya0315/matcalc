from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from ase.calculators.calculator import Calculator

from matcalc import ASESimulator
from matcalc.simulation import as_simulator

from .helpers import structure

if TYPE_CHECKING:
    import pytest


def test_relax_converges_to_emt_lattice(emt_simulator: ASESimulator) -> None:
    start = structure("Cu")  # a = 3.62 A, EMT prefers about 3.59 A
    (result,) = emt_simulator.relax([start], fmax=0.01, max_steps=500)
    assert result.converged
    assert result.error is None
    assert result.max_force <= 0.01
    assert result.n_steps > 0
    assert result.structure is not None
    assert 3.55 < result.structure.lattice.a < 3.62
    assert np.abs(result.stress).max() < 2e-3  # the cell relaxed too (eV/A^3)


def test_relax_reports_non_convergence(emt_simulator: ASESimulator) -> None:
    start = structure("NiAl").copy().perturb(0.1, seed=0)
    (result,) = emt_simulator.relax([start], fmax=1e-6, max_steps=3)
    assert not result.converged
    assert result.n_steps == 3


def test_single_point_shapes(emt_simulator: ASESimulator) -> None:
    with_stress, without = (
        emt_simulator.single_point([structure("Cu")])[0],
        emt_simulator.single_point([structure("Cu")], compute_stress=False)[0],
    )
    assert with_stress.forces.shape == (4, 3)
    assert with_stress.stress.shape == (3, 3)
    assert without.stress is None
    assert with_stress.energy == without.energy


class _Broken(Calculator):
    implemented_properties = ("energy", "forces", "stress")

    def calculate(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("model exploded")


def test_failures_are_reported_not_raised(caplog: pytest.LogCaptureFixture) -> None:
    simulator = ASESimulator(_Broken(), show_progress=False)
    (relaxed,) = simulator.relax([structure("Cu")], fmax=0.05, max_steps=10)
    (point,) = simulator.single_point([structure("Cu")])
    assert not relaxed.converged
    assert "model exploded" in relaxed.error
    assert "model exploded" in point.error
    assert np.isnan(point.energy)
    assert "failed" in caplog.text


def test_as_simulator_accepts_calculators_and_simulators(emt_simulator: ASESimulator) -> None:
    assert as_simulator(emt_simulator) is emt_simulator
    assert isinstance(as_simulator(emt_simulator.calculator), ASESimulator)


def test_relax_in_a_fixed_cell_keeps_fixed_atoms_in_place(emt_simulator: ASESimulator) -> None:
    from ase.build import add_adsorbate, fcc111
    from ase.constraints import FixAtoms

    slab = fcc111("Cu", size=(2, 2, 3), a=3.62, vacuum=6.0)
    add_adsorbate(slab, "Cu", 2.2, "fcc")
    slab.positions += np.random.default_rng(0).normal(0.0, 0.03, slab.positions.shape)
    fixed = [atom.index for atom in slab if atom.tag == 3]
    slab.set_constraint(FixAtoms(indices=fixed))
    (result,) = emt_simulator.relax([slab], fmax=0.01, max_steps=300, relax_cell=False)
    assert result.converged
    assert result.optimizer_converged
    assert result.stress is None
    np.testing.assert_allclose(result.structure.lattice.matrix, slab.cell[:], atol=1e-12)
    np.testing.assert_allclose(result.structure.cart_coords[fixed], slab.positions[fixed], atol=1e-12)
    assert np.abs(result.forces[fixed]).max() == 0.0  # FIRE sees no force on fixed atoms
    assert len(slab.constraints) == 1  # the caller's structure is untouched
