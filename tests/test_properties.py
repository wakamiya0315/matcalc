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
from matcalc.properties.phonon import (
    IMAGINARY_THRESHOLD_THZ,
    HarmonicProperties,
    displaced_supercells,
    harmonic_properties,
    make_phonopy,
)
from matcalc.properties.softening import softening_scale

from .helpers import FCC_PRIMITIVE, structure


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


def _fcc_cu_phonopy() -> object:
    """Conventional fcc Cu (4 atoms) in a 2 x 2 x 2 supercell, with fcc's primitive matrix."""
    return make_phonopy(structure("Cu"), 2 * np.eye(3), FCC_PRIMITIVE, symprec=1e-5)


def test_displaced_supercells_are_those_of_phonopy() -> None:
    reference = _fcc_cu_phonopy()
    reference.generate_displacements(distance=0.01)
    rows = [[d["number"], *d["displacement"]] for d in reference.dataset["first_atoms"]]
    cells = displaced_supercells(_fcc_cu_phonopy(), rows)
    expected = reference.supercells_with_displacements
    assert len(cells) == len(expected) == 1  # fcc: one symmetry-distinct displacement
    for atoms, cell in zip(cells, expected, strict=True):
        assert_allclose(atoms.positions, cell.positions, atol=1e-12)
        assert list(atoms.numbers) == list(cell.numbers)


def test_compact_force_constants_give_the_same_heat_capacity() -> None:
    """Compact force constants and no eigenvectors give the same C_V as the full arrays."""
    calc = EMT()
    reference = _fcc_cu_phonopy()
    reference.generate_displacements(distance=0.01)
    rows = [[d["number"], *d["displacement"]] for d in reference.dataset["first_atoms"]]

    def forces_of(cells: list) -> list[np.ndarray]:
        out = []
        for atoms in cells:
            atoms.calc = calc
            out.append(atoms.get_forces())
        return out

    compact = _fcc_cu_phonopy()
    harmonic = harmonic_properties(compact, forces_of(displaced_supercells(compact, rows)), mesh=(20, 20, 20))

    full = _fcc_cu_phonopy()
    full.forces = forces_of(displaced_supercells(full, rows))
    full.produce_force_constants()
    full.run_mesh([20, 20, 20], with_eigenvectors=True)
    full.run_thermal_properties(temperatures=[300.0])
    assert harmonic.heat_capacity == pytest.approx(full.get_thermal_properties_dict()["heat_capacity"][0], rel=1e-8)
    assert 0.9 * 3 * 8.314 < harmonic.heat_capacity < 3 * 8.314  # one atom per primitive cell
    assert harmonic.dynamically_stable


def test_imaginary_modes_below_minus_50_kelvin_mean_unstable() -> None:
    assert pytest.approx(1.0418, abs=1e-4) == IMAGINARY_THRESHOLD_THZ
    assert HarmonicProperties(20.0, 300.0, min_frequency=-1.0).dynamically_stable
    assert not HarmonicProperties(20.0, 300.0, min_frequency=-1.1).dynamically_stable
