from __future__ import annotations

import numpy as np
import pytest
from ase.calculators.emt import EMT
from numpy.testing import assert_allclose
from pymatgen.core import Composition
from pymatgen.core.elasticity import ElasticTensor
from scipy.optimize import curve_fit

from matcalc.properties.elasticity import fit_elastic_tensor, strained_structures
from matcalc.properties.energetics import formation_energy_per_atom
from matcalc.properties.phonon import ThermalProperties, displaced_supercells, make_phonopy, thermal_properties
from matcalc.properties.softening import softening_scale

from .helpers import structure


def test_elastic_fit_recovers_known_cubic_constants() -> None:
    c11, c12, c44 = 1.1, 0.6, 0.35  # eV/A^3
    voigt = np.zeros((6, 6))
    voigt[:3, :3] = c12
    np.fill_diagonal(voigt[:3, :3], c11)
    voigt[3, 3] = voigt[4, 4] = voigt[5, 5] = c44
    exact = ElasticTensor.from_voigt(voigt)

    cells, strains = strained_structures(structure("Cu"))
    assert len(cells) == len(strains) == 24
    stresses = [np.asarray(exact.calculate_stress(strain)) for strain in strains]
    fit = fit_elastic_tensor(strains, stresses, np.zeros((3, 3)))

    to_gpa = 1 / exact.GPa_to_eV_A3
    assert_allclose(fit.elastic_tensor, voigt * to_gpa, atol=1e-8)
    assert fit.bulk_modulus_vrh == pytest.approx((c11 + 2 * c12) / 3 * to_gpa)
    assert fit.mean_r2 == pytest.approx(1.0)


def test_softening_scale_equals_curve_fit_slope() -> None:
    rng = np.random.default_rng(1)
    dft = [rng.normal(size=(8, 3)) for _ in range(5)]
    mlip = [0.83 * f + rng.normal(scale=0.05, size=f.shape) for f in dft]

    popt, _ = curve_fit(lambda x, a: a * x, np.ravel(dft), np.ravel(mlip))  # upstream's fit
    assert softening_scale(dft, mlip) == pytest.approx(popt[0], rel=1e-7)


def test_formation_energy_per_atom() -> None:
    # 2 Cu at -4 eV and 2 Au at -3 eV: E_form = (-15 - (2*-4 + 2*-3)) / 4 = -0.25 eV/atom
    assert formation_energy_per_atom(-15.0, Composition("Cu2Au2"), {"Cu": -4.0, "Au": -3.0}) == pytest.approx(-0.25)


def test_heat_capacity_at_grid_temperature() -> None:
    temperatures = np.arange(0, 1001, 10.0)
    thermal = ThermalProperties(temperatures, temperatures / 100, temperatures, temperatures, 1.0)
    assert thermal.heat_capacity_at(300) == pytest.approx(3.0)
    with pytest.raises(ValueError, match="not on the temperature grid"):
        thermal.heat_capacity_at(305)


def test_compact_force_constants_give_the_same_heat_capacity() -> None:
    """Deliberate change vs upstream: compact force constants and no eigenvectors, same C_V."""
    cell = structure("Cu1")
    calc = EMT()

    def forces_of(phonon: object) -> list[np.ndarray]:
        out = []
        for atoms in displaced_supercells(phonon, displacement=0.015):
            atoms.calc = calc
            out.append(atoms.get_forces())
        return out

    compact = make_phonopy(cell, min_supercell_length=8.0)
    thermal = thermal_properties(compact, forces_of(compact))

    full = make_phonopy(cell, min_supercell_length=8.0)
    full.forces = forces_of(full)
    full.produce_force_constants()  # upstream: full force constants
    full.run_mesh(with_eigenvectors=True)  # upstream: with eigenvectors
    full.run_thermal_properties(t_step=10, t_max=1000, t_min=0)

    assert_allclose(thermal.heat_capacity, full.thermal_properties.heat_capacity, rtol=1e-8, atol=1e-12)
    assert thermal.heat_capacity_at(300) == pytest.approx(full.thermal_properties.heat_capacity[30], rel=1e-8)
