from __future__ import annotations

import numpy as np
import pytest
from ase.calculators.emt import EMT
from numpy.testing import assert_allclose
from pymatgen.core import Composition, Lattice, Structure
from pymatgen.core.elasticity import ElasticTensor
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from scipy.optimize import curve_fit

from matcalc.properties.elasticity import fit_elastic_tensor, strained_structures
from matcalc.properties.energetics import formation_energy_per_atom
from matcalc.properties.phonon import (
    IMAGINARY_THRESHOLD_THZ,
    HarmonicProperties,
    displaced_supercells,
    generated_displacements,
    harmonic_properties,
    make_phonopy,
    stability_qpoints,
    undisplaced_supercell,
)
from matcalc.properties.softening import softening_scale
from matcalc.properties.stability import geometry_metrics, mp2020_correction_change, stability_metrics

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


def test_displacements_without_symmetry_go_both_ways_along_the_lattice_for_every_primitive_atom() -> None:
    unit_cell = structure("NiAl")  # B2: two atoms, primitive
    supercell = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    rows = generated_displacements(make_phonopy(unit_cell, supercell, use_symmetry=False), 0.01, plus_minus=True)
    assert len(rows) == 6 * len(unit_cell)
    assert len({row[0] for row in rows}) == len(unit_cell)
    vectors = np.array([row[1:4] for row in rows])
    assert_allclose(np.linalg.norm(vectors, axis=1), 0.01)
    assert_allclose(vectors[0::2], -vectors[1::2])  # each displacement is followed by its opposite
    assert len(generated_displacements(make_phonopy(unit_cell, supercell), 0.01)) == 2  # one per atom


def test_without_symmetry_the_forces_left_on_the_relaxed_cell_cancel() -> None:
    """Forces that the relaxed cell keeps and that do not follow its symmetry (as an SO(3)-equivariant MLIP
    can leave on a cell relaxed under a symmetry constraint) cancel in the central differences without
    symmetry, while phonopy's symmetric construction takes them for a response to the displacement, unless
    the forces on the undisplaced supercell are subtracted."""
    calc = EMT()
    unit_cell = structure("NiAl")
    supercell = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]

    def heat_capacity(*, use_symmetry: bool, residual: float, subtract: bool = False) -> float:
        phonon = make_phonopy(unit_cell, supercell, use_symmetry=use_symmetry)
        rows = generated_displacements(phonon, 0.01, plus_minus=True if not use_symmetry else "auto")
        cells = displaced_supercells(phonon, rows)
        if subtract:
            cells.append(undisplaced_supercell(phonon))
        # the same force on every atom of a sublattice, opposite on the two (no net force), along [111]: its
        # projection on the displacement is what the symmetric construction cannot tell from a response
        left = np.outer(np.where(phonon.supercell.numbers == 28, residual, -residual), [1.0, 1.0, 1.0]) / np.sqrt(3)
        forces = []
        for atoms in cells:
            atoms.calc = calc
            forces.append(atoms.get_forces() + left)
        if subtract:
            *forces, on_undisplaced = forces
            forces = [f - on_undisplaced for f in forces]
        return harmonic_properties(phonon, forces, mesh=(8, 8, 8)).heat_capacity

    exact = heat_capacity(use_symmetry=False, residual=0.0)
    assert heat_capacity(use_symmetry=True, residual=0.0) == pytest.approx(exact, rel=1e-6)
    assert heat_capacity(use_symmetry=False, residual=0.01) == pytest.approx(exact, rel=1e-9)
    assert abs(heat_capacity(use_symmetry=True, residual=0.01) - exact) > 0.1  # J/(K mol)
    assert heat_capacity(use_symmetry=True, residual=0.01, subtract=True) == pytest.approx(exact, rel=1e-6)


def test_imaginary_modes_below_minus_50_kelvin_mean_unstable() -> None:
    assert pytest.approx(1.0418, abs=1e-4) == IMAGINARY_THRESHOLD_THZ
    assert HarmonicProperties(20.0, 300.0, min_frequency=-1.0).dynamically_stable
    assert not HarmonicProperties(20.0, 300.0, min_frequency=-1.1).dynamically_stable


def test_stability_qpoints_are_those_of_the_reference() -> None:
    """(n1/S1, n2/S2, n3/S3) for a diagonal supercell matrix S, as reduced coordinates."""
    qpoints = stability_qpoints(make_phonopy(structure("Cu"), [[2, 0, 0], [0, 2, 0], [0, 0, 1]], FCC_PRIMITIVE))
    assert len(qpoints) == 4
    assert {tuple(np.round(q % 1, 6)) for q in qpoints} == {(0, 0, 0), (0.5, 0, 0), (0, 0.5, 0), (0.5, 0.5, 0)}


def test_strained_cells_are_those_of_pymatgen() -> None:
    from pymatgen.core.elasticity import DeformedStructureSet

    cell = structure("CuAu")
    cells, strains = strained_structures(cell)
    expected = DeformedStructureSet(cell, symmetry=False)
    assert len(cells) == len(expected) == len(strains) == 24
    for atoms, reference in zip(cells, expected, strict=True):
        assert_allclose(atoms.cell.array, reference.lattice.matrix, atol=1e-12)
        assert_allclose(atoms.positions, reference.cart_coords, atol=1e-12)
        assert list(atoms.numbers) == list(reference.atomic_numbers)


def test_stability_metrics_count_missing_predictions_as_unstable() -> None:
    true = np.array([-0.1, -0.05, 0.0, 0.2, 0.3, 0.02])
    pred = np.array([-0.2, 0.1, np.nan, 0.1, -0.01, 0.05])  # TP, FN, FN (missing), TN, FP, TN
    metrics = stability_metrics(true, pred)
    assert (metrics["TP"], metrics["FN"], metrics["FP"], metrics["TN"]) == (1, 2, 1, 2)
    assert metrics["Precision"] == pytest.approx(0.5)
    assert metrics["Recall"] == pytest.approx(1 / 3)
    assert metrics["F1"] == pytest.approx(0.4)
    assert metrics["DAF"] == pytest.approx(0.5 / 0.5)
    assert metrics["missing"] == 1
    errors = np.array([-0.1, 0.15, -0.1, -0.31, 0.03])
    assert metrics["MAE"] == pytest.approx(np.mean(np.abs(errors)))
    assert metrics["RMSE"] == pytest.approx(np.sqrt(np.mean(errors**2)))
    present = ~np.isnan(pred)
    total = np.sum((true[present] - true[present].mean()) ** 2)
    assert metrics["R2"] == pytest.approx(1 - np.sum(errors**2) / total)
    assert stability_metrics(true, pred, prevalence=0.25)["DAF"] == pytest.approx(2.0)


def test_geometry_metrics_follow_matbench_discovery() -> None:
    metrics = geometry_metrics(
        rmsd=[0.01, np.nan, 0.03, 0.02],
        space_groups=[225, 221, 12, np.nan],
        reference_space_groups=[225, 225, 2, 221],
        n_operations=[48, 48, 4, np.nan],
        reference_n_operations=[48, 192, 2, 48],
    )
    assert metrics["rmsd"] == pytest.approx((0.01 + 1.0 + 0.03 + 0.02) / 4)  # unmatched counts as 1
    assert metrics["n_structures"] == 3
    assert metrics["symmetry_match"] == pytest.approx(1 / 3)
    assert metrics["symmetry_decrease"] == pytest.approx(1 / 3)
    assert metrics["symmetry_increase"] == pytest.approx(1 / 3)
    assert metrics["n_sym_ops_mae"] == pytest.approx((0 + 144 + 2) / 3)


def test_mp2020_correction_change_follows_the_oxide_type() -> None:
    def li2o2(o_o: float) -> Structure:  # two O atoms o_o apart along z
        return Structure(
            Lattice.cubic(6.0),
            ["Li", "Li", "O", "O"],
            [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0.5, 0, 0.5 + o_o / 6]],
        )

    parameters = {
        "run_type": "GGA",
        "is_hubbard": False,
        "hubbards": {},
        "potcar_symbols": ["PAW_PBE Li_sv 10Sep2004", "PAW_PBE O 08Apr2002"],
        "potcar_spec": [
            {"titel": "PAW_PBE Li_sv 10Sep2004", "hash": None},
            {"titel": "PAW_PBE O 08Apr2002", "hash": None},
        ],
    }
    peroxide, oxide = li2o2(1.5), li2o2(2.8)
    correction = MaterialsProject2020Compatibility().comp_correction
    change = mp2020_correction_change(oxide, peroxide, -20.0, parameters)
    assert change == pytest.approx(2 * (correction["oxide"] - correction["peroxide"]))
    assert mp2020_correction_change(li2o2(1.45), peroxide, -20.0, parameters) == 0
    assert mp2020_correction_change(peroxide, oxide, -20.0, parameters) == pytest.approx(-change)
