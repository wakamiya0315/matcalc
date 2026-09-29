"""The benchmark pipelines on tiny EMT datasets (offline, CPU)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from monty.serialization import dumpfn

from matcalc import (
    ElasticityBenchmark,
    EquilibriumBenchmark,
    PhononBenchmark,
    SofteningBenchmark,
    run_benchmarks,
)
from matcalc.datasets import sample_subset

from .helpers import FCC_PRIMITIVE, SOFTENING_FACTOR, phonon_entry, structure

if TYPE_CHECKING:
    from pathlib import Path

    from matcalc import ASESimulator

R_GAS = 8.314462618  # J/(K mol)


def test_elasticity(elasticity_dataset: Path, emt_simulator: ASESimulator) -> None:
    benchmark = ElasticityBenchmark(elasticity_dataset)
    table = benchmark.run(emt_simulator, "emt")
    assert list(table.columns[:6]) == ["mp_id", "formula", "K_vrh_DFT", "G_vrh_DFT", "K_vrh_emt", "G_vrh_emt"]
    assert list(table["status_emt"]) == ["ok", "ok"]
    assert (table["K_vrh_emt"] > 50).all()
    assert (table["G_vrh_emt"] > 0).all()
    assert (table["relax_steps_emt"] > 0).all()
    summary = benchmark.summarize(table, "emt")
    assert summary["n_ok"] == 2
    assert set(summary["K_vrh"]) == {"MAE", "STDAE", "n"}
    assert {"relax", "single points", "fit"} <= set(summary["timings_s"])


def test_elasticity_marks_unconverged_relaxations(emt_simulator: ASESimulator, tmp_path: Path) -> None:
    # As upstream, convergence is judged on the atomic forces only; they vanish by symmetry in perfect
    # crystals, so the atoms are displaced here to leave forces after a single FIRE step.
    rattled = structure("Cu").copy().perturb(0.05, seed=3)
    path = tmp_path / "rattled.json.gz"
    dumpfn(
        [{"mp_id": "t-Cu", "formula": "Cu", "structure": rattled, "bulk_modulus_vrh": 1, "shear_modulus_vrh": 1}], path
    )
    table = ElasticityBenchmark(path, max_steps=1).run(emt_simulator, "emt")
    assert table.loc[0, "status_emt"].startswith("relaxation not converged")
    assert np.isnan(table.loc[0, "K_vrh_emt"])
    assert table.loc[0, "relax_steps_emt"] == 1


def test_checkpoint_resumes(elasticity_dataset: Path, emt_simulator: ASESimulator, tmp_path: Path) -> None:
    checkpoint = tmp_path / "elasticity_emt.json.gz"
    first = ElasticityBenchmark(elasticity_dataset).run(emt_simulator, "emt", checkpoint_file=checkpoint, chunk_size=1)
    assert checkpoint.exists()

    class Exploding:
        def relax(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("finished materials must not be recomputed")

        single_point = relax

    second = ElasticityBenchmark(elasticity_dataset).run(Exploding(), "emt", checkpoint_file=checkpoint)
    assert second.equals(first)
    with pytest.raises(ValueError, match="belongs to another run"):
        ElasticityBenchmark(elasticity_dataset).run(emt_simulator, "other-model", checkpoint_file=checkpoint)


def test_phonon(phonon_dataset: Path, emt_simulator: ASESimulator) -> None:
    benchmark = PhononBenchmark(phonon_dataset)
    table = benchmark.run(emt_simulator, "emt")
    assert list(table["status_emt"]) == ["ok", "ok"]
    assert list(table["stable_DFT"]) == [True, False]
    assert "min_frequency_DFT" in table.columns
    heat_capacity = table.loc[0, "CV_emt"]
    assert 0.8 * 3 * R_GAS < heat_capacity < 3 * R_GAS  # one atom per primitive cell: below Dulong-Petit
    assert table.loc[0, "stable_emt"]  # fcc Cu is dynamically stable
    assert (table["relax_steps_emt"] > 0).all()
    assert (table["residual_force_emt"] < 1e-6).all()  # both cells have their forces zero by symmetry
    summary = benchmark.summarize(table, "emt")
    assert summary["CV"]["n"] == 2
    assert summary["CV (DFT-stable)"]["n"] == 1
    assert summary["CV (DFT-stable)"]["MAE"] == pytest.approx(abs(heat_capacity - 24.4))
    assert sum(summary["stability"].values()) == 2


def test_phonon_without_symmetry_gives_the_same_results_for_an_o3_invariant_potential(
    phonon_dataset: Path, emt_simulator: ASESimulator
) -> None:
    with_symmetry = PhononBenchmark(phonon_dataset).run(emt_simulator, "emt")
    benchmark = PhononBenchmark(phonon_dataset, use_symmetry=False)
    table = benchmark.run(emt_simulator, "emt")
    assert list(table["status_emt"]) == ["ok", "ok"]
    np.testing.assert_allclose(table["CV_emt"], with_symmetry["CV_emt"], rtol=1e-6)
    np.testing.assert_allclose(table["min_frequency_emt"], with_symmetry["min_frequency_emt"], atol=1e-3)
    assert list(table["stable_emt"]) == list(with_symmetry["stable_emt"])
    assert benchmark.summarize(table, "emt")["use_symmetry"] is False
    assert PhononBenchmark(phonon_dataset).summarize(with_symmetry, "emt")["use_symmetry"] is True


def test_phonon_residual_forces_change_nothing_for_an_o3_invariant_potential(
    phonon_dataset: Path, emt_simulator: ASESimulator
) -> None:
    subtracted = PhononBenchmark(phonon_dataset).run(emt_simulator, "emt")
    benchmark = PhononBenchmark(phonon_dataset, subtract_residual_forces=False)
    table = benchmark.run(emt_simulator, "emt")
    np.testing.assert_allclose(table["CV_emt"], subtracted["CV_emt"], rtol=1e-9)
    np.testing.assert_allclose(table["min_frequency_emt"], subtracted["min_frequency_emt"], atol=1e-6)
    assert "residual_force_emt" not in table.columns
    assert benchmark.summarize(table, "emt")["subtract_residual_forces"] is False


@pytest.mark.parametrize(
    ("first", "second"),
    [({"use_symmetry": False}, {}), ({"subtract_residual_forces": False}, {})],
    ids=["use_symmetry", "subtract_residual_forces"],
)
def test_phonon_checkpoint_is_not_resumed_with_other_settings(
    phonon_dataset: Path, emt_simulator: ASESimulator, tmp_path: Path, first: dict, second: dict
) -> None:
    checkpoint = tmp_path / "phonon.json"
    PhononBenchmark(phonon_dataset, **first).run(emt_simulator, "emt", checkpoint_file=checkpoint)
    with pytest.raises(ValueError, match="belongs to another run"):
        PhononBenchmark(phonon_dataset, **second).run(emt_simulator, "emt", checkpoint_file=checkpoint)


def test_phonon_without_converged_relaxation_gives_nan(tmp_path: Path, emt_simulator: ASESimulator) -> None:
    strained = phonon_entry(
        "t-Cu", "Cu", [[2, 0, 0], [0, 2, 0], [0, 0, 2]], FCC_PRIMITIVE, 24.4, stable=True, strain=0.05
    )
    dumpfn({"entries": [strained]}, tmp_path / "strained.json.gz")
    table = PhononBenchmark(tmp_path / "strained.json.gz", max_steps=2).run(emt_simulator, "emt")
    assert list(table["status_emt"]) == ["relaxation not converged"]
    assert table["CV_emt"].isna().all()
    assert table.loc[0, "relax_steps_emt"] == 2


def test_phonon_displacements_are_regenerated_when_the_space_group_changes(
    tmp_path: Path, emt_simulator: ASESimulator
) -> None:
    entry = phonon_entry("t-Cu", "Cu", [[2, 0, 0], [0, 2, 0], [0, 0, 2]], FCC_PRIMITIVE, 24.4, stable=True)
    reference = PhononBenchmark(_dump(tmp_path / "a.json.gz", entry)).run(emt_simulator, "emt")
    entry["space_group"] = 221  # pretend the DFT structure had another space group than fcc (225)
    table = PhononBenchmark(_dump(tmp_path / "b.json.gz", entry)).run(emt_simulator, "emt")
    assert table.loc[0, "status_emt"] == "ok (space group 221 -> 225; displacements generated by phonopy)"
    assert table.loc[0, "CV_emt"] == pytest.approx(reference.loc[0, "CV_emt"], rel=1e-6)


def _dump(path: Path, entry: dict) -> Path:
    dumpfn({"entries": [entry]}, path)
    return path


def test_softening(softening_dataset: Path, emt_simulator: ASESimulator) -> None:
    table = SofteningBenchmark(softening_dataset).run(emt_simulator, "emt")
    assert list(table.columns) == ["material_id", "formula", "softening_scale_emt", "status_emt"]
    assert table.loc[0, "formula"] == "Cu"
    assert table.loc[0, "softening_scale_emt"] == pytest.approx(1 / SOFTENING_FACTOR, rel=1e-10)


def test_equilibrium(equilibrium_dataset: Path, emt_simulator: ASESimulator) -> None:
    pytest.importorskip("matminer")
    benchmark = EquilibriumBenchmark(equilibrium_dataset)
    table = benchmark.run(emt_simulator, "emt")
    assert set(benchmark.reference_energies) == {"Cu", "Au"}
    assert list(table["status_emt"]) == ["ok", "ok"]
    assert np.isfinite(table["Eform_emt"]).all()
    assert (table["d_emt"] >= 0).all()
    assert table.loc[0, "structure_emt"].composition.reduced_formula == "Cu3Au"

    again = EquilibriumBenchmark(equilibrium_dataset).run(emt_simulator, "emt")  # seeded displacements
    assert np.array_equal(again["Eform_emt"], table["Eform_emt"])


def test_run_benchmarks_merges_models_on_the_material_id(
    softening_dataset: Path, emt_simulator: ASESimulator, tmp_path: Path
) -> None:
    tables = run_benchmarks(
        [SofteningBenchmark(softening_dataset)], {"a": emt_simulator, "b": emt_simulator}, output_dir=tmp_path
    )
    merged = tables["softening"]
    assert {"softening_scale_a", "softening_scale_b", "status_a", "status_b"} <= set(merged.columns)
    assert len(merged) == 1
    assert (tmp_path / "softening_a.json.gz").exists()


def test_parallel_post_processing_gives_the_same_numbers(
    phonon_dataset: Path, elasticity_dataset: Path, equilibrium_dataset: Path, emt_simulator: ASESimulator
) -> None:
    serial = PhononBenchmark(phonon_dataset).run(emt_simulator, "emt")
    parallel = PhononBenchmark(phonon_dataset, workers=2).run(emt_simulator, "emt")
    assert parallel.equals(serial)
    serial = ElasticityBenchmark(elasticity_dataset).run(emt_simulator, "emt")
    parallel = ElasticityBenchmark(elasticity_dataset, workers=2).run(emt_simulator, "emt")
    assert parallel.equals(serial)
    pytest.importorskip("matminer")
    serial = EquilibriumBenchmark(equilibrium_dataset).run(emt_simulator, "emt")
    parallel = EquilibriumBenchmark(equilibrium_dataset, workers=2).run(emt_simulator, "emt")
    assert np.array_equal(parallel["d_emt"], serial["d_emt"])


def test_discovery(discovery_dataset: Path, emt_simulator: ASESimulator) -> None:
    from pymatgen.core import Composition

    from matcalc import DiscoveryBenchmark

    from .conftest import DISCOVERY_ENTRIES

    benchmark = DiscoveryBenchmark(discovery_dataset)
    table = benchmark.run(emt_simulator, "emt")
    assert list(table["material_id"]) == [entry[0] for entry in DISCOVERY_ENTRIES]
    assert all(status.startswith("ok") for status in table["status_emt"])
    relaxed = emt_simulator.relax([m.structure for m in benchmark.materials], fmax=0.05, max_steps=500)
    for (_, formula, e_form_dft, e_hull_dft, correction, _), result, row in zip(
        DISCOVERY_ENTRIES, relaxed, table.itertuples(), strict=True
    ):
        composition = Composition(formula)
        reference = sum(benchmark.reference_energies[el.symbol] * n for el, n in composition.items())
        e_form = (result.energy - reference) / composition.num_atoms + correction
        assert row.e_form_per_atom_emt == pytest.approx(e_form, abs=1e-10)
        assert row.e_above_hull_emt == pytest.approx(e_hull_dft + e_form - e_form_dft, abs=1e-10)
        assert row.rmsd_emt < 0.2  # EMT relaxes the perturbed start back towards the reference cell
    summary = benchmark.summarize(table, "emt")
    full, unique = summary["discovery"]["full_test_set"], summary["discovery"]["unique_prototypes"]
    assert full["TP"] + full["FP"] + full["TN"] + full["FN"] == 4
    assert unique["TP"] + unique["FP"] + unique["TN"] + unique["FN"] == 3
    assert summary["geo_opt"]["symprec=1e-2"]["n_structures"] == 4
    assert summary["timings_s"]["relax"] > 0

    subset = DiscoveryBenchmark(discovery_dataset, n_samples=2, seed=3)
    ids = [entry[0] for entry in DISCOVERY_ENTRIES]
    assert [m.material_id for m in subset.materials] == sample_subset(ids, 2, 3)


def test_kappa(kappa_dataset: Path, emt_simulator: ASESimulator) -> None:
    from matcalc import KappaBenchmark

    benchmark = KappaBenchmark(kappa_dataset, workers=1)
    table = benchmark.run(emt_simulator, "emt")
    ok, censored = table.iloc[0], table.iloc[1]
    assert ok["status_emt"] == "ok"
    # EMT against its own reference. Phonon lifetimes on a q-point mesh are not continuous in the input:
    # the 1e-16 A the relaxation moves the atoms change kappa by up to about 1 % here.
    assert ok["kappa_emt"] == pytest.approx(ok["kappa_DFT"], rel=0.03)
    assert ok["srme_emt"] < 0.05
    assert censored["status_emt"] == "censored: space group 221 -> 225"
    assert censored["srme_emt"] == 2.0
    summary = benchmark.summarize(table, "emt")
    assert summary["kappa_SRME"] == pytest.approx((ok["srme_emt"] + 2.0) / 2)
    assert summary["failure_rate"] == pytest.approx(0.5)


def test_settings_that_change_the_results_are_kept_in_the_checkpoint(
    discovery_dataset: Path, kappa_dataset: Path
) -> None:
    from matcalc import DiscoveryBenchmark, KappaBenchmark

    assert DiscoveryBenchmark(discovery_dataset).run_settings() == {}
    assert DiscoveryBenchmark(discovery_dataset, fmax=0.02).run_settings() == {"fmax": "0.02"}
    assert KappaBenchmark(kappa_dataset).run_settings() == {}
    assert KappaBenchmark(kappa_dataset, temperature=500.0).run_settings() == {"temperature": "500.0"}


def test_diatomics(diatomics_dataset: Path, emt_simulator: ASESimulator) -> None:
    from matcalc import DiatomicsBenchmark

    benchmark = DiatomicsBenchmark(diatomics_dataset)
    assert [m.material_id for m in benchmark.materials] == ["Al", "Ni", "Cu"]
    assert benchmark.rough_references == {"Ni"}
    table = benchmark.run(emt_simulator, "emt").set_index("element")
    assert (table["status_emt"] == "ok").all()
    # EMT against its own curves: the errors come only from the different separations of the two grids.
    for element in ("Al", "Cu"):
        assert table.loc[element, "pbe_energy_mae_emt"] < 0.02
        assert table.loc[element, "pbe_bond_length_error_emt"] < 0.02
        assert table.loc[element, "pbe_wall_dist_mae_emt"] < 0.005
    assert table["tortuosity_emt"].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert table["force_flips_emt"].tolist() == [1.0, 1.0, 1.0]
    assert np.isnan(table.loc["Ni", "pbe_energy_mae_emt"])  # rough reference: smoothness metrics only
    summary = benchmark.summarize(table.reset_index(), "emt")
    assert summary["pbe_energy_mae"] == pytest.approx(table.loc[["Al", "Cu"], "pbe_energy_mae_emt"].mean())
    assert summary["pbe_vib_freq_coverage"] == {"n_valid": 2, "n_eligible": 2}


def test_diatomics_curve_with_non_finite_scored_points_gets_no_metrics(diatomics_dataset: Path) -> None:
    from matcalc import DiatomicsBenchmark
    from matcalc.properties.diatomics import DIMER_DISTANCES

    benchmark = DiatomicsBenchmark(diatomics_dataset)
    material = benchmark.materials[0]
    energies = 1.0 / DIMER_DISTANCES**2
    forces = np.zeros((len(DIMER_DISTANCES), 2, 3))
    energies[0] = np.nan  # 0.1 Å, below the scored range: left out
    assert benchmark._metrics(material, energies, forces)["status"] == "ok"
    energies[90] = np.nan  # 2.3 Å, inside it
    assert benchmark._metrics(material, energies, forces)["status"].startswith("non-finite")
