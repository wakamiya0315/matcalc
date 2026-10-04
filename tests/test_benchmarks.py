"""The benchmark pipelines on tiny EMT datasets (offline, CPU)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
    # The first compound's lowest frequency is an acoustic mode at Γ: zero up to a few 1e-6 THz of numerical
    # noise, whose sign and size depend on the platform and the phonopy version.
    np.testing.assert_allclose(table["min_frequency_emt"], subtracted["min_frequency_emt"], atol=1e-4)
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


def test_kappa_in_two_stages_gives_the_same_table(
    kappa_dataset: Path, emt_simulator: ASESimulator, tmp_path: Path
) -> None:
    from matcalc import KappaBenchmark

    table = KappaBenchmark(kappa_dataset).run(emt_simulator, "emt")
    KappaBenchmark(kappa_dataset).save_forces(emt_simulator, tmp_path / "forces.pkl")
    two_stages = KappaBenchmark(kappa_dataset, workers=2).run_saved_forces(tmp_path / "forces.pkl", "emt")
    assert two_stages.equals(table)
    with pytest.raises(ValueError, match="was saved for"):
        KappaBenchmark(kappa_dataset, temperature=400.0).run_saved_forces(tmp_path / "forces.pkl", "emt")


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


MOLECULAR_EMT_ELEMENTS = {"H", "C", "N", "O", "Al", "Ni", "Cu", "Pd", "Ag", "Pt", "Au"}
"""The elements of ASE's EMT."""


def test_noncovalent(ncia_dataset: Path, emt_simulator: ASESimulator) -> None:
    from matcalc import NoncovalentBenchmark

    benchmark = NoncovalentBenchmark(
        ncia_dataset, sets=["D442x10", "IHB100x10", "R739x5"], elements=MOLECULAR_EMT_ELEMENTS
    )
    assert [m.material_id for m in benchmark.materials] == [
        "D442x10:1.01.01",
        "D442x10:1.02.01",
        "IHB100x10:01.001",
        "R739x5:001.01",
    ]
    ion = benchmark.materials[2]
    assert ion.structure.info["charge"] == 1
    assert all(point.info["charge"] == 1 for point in ion.settings["points"])
    table = benchmark.run(emt_simulator, "emt").set_index("system_id")
    assert table["status_emt"].tolist() == ["ok", "skipped: B not in the elements", "ok", "ok"]
    assert table["dataset"].tolist() == ["Dispersion", "Dispersion", "Ionic hydrogen bonds", "Repulsive contacts"]
    assert table["group"].tolist() == ["HCNO", "Boron", "OH(+)-O", "HCNO"]
    # EMT against its own curves; the repulsive contact is the highest point of its curve, not the lowest.
    ok = table["status_emt"] == "ok"
    assert table.loc[ok, "interaction_energy_emt"].tolist() == pytest.approx(
        table.loc[ok, "interaction_energy_ref"].tolist(), abs=1e-6
    )
    reference = table["interaction_energy_ref"]
    assert reference["R739x5:001.01"] > 0 > reference["D442x10:1.01.01"]
    summary = benchmark.summarize(table.reset_index(), "emt")
    assert (summary["n_systems"], summary["n_ok"], summary["n_skipped"]) == (4, 3, 1)
    assert summary["interaction_energy"]["MAE"] == pytest.approx(0.0, abs=1e-6)
    assert summary["interaction_energy"]["n"] == 3
    assert set(summary["datasets"]) == {"Dispersion", "Ionic hydrogen bonds", "Repulsive contacts"}
    assert set(summary["subsets"]) == {"Dispersion: HCNO", "Ionic hydrogen bonds: OH(+)-O", "Repulsive contacts: HCNO"}


def test_conformers(conformer_dataset: Path, emt_simulator: ASESimulator) -> None:
    from matcalc import ConformerBenchmark

    benchmark = ConformerBenchmark(conformer_dataset, elements={"Cu"})
    assert [m.material_id for m in benchmark.materials] == ["neutral", "cation", "gold"]  # "pair" has 2 conformers
    table = benchmark.run(emt_simulator, "emt").set_index("molecule")
    assert table["status_emt"].tolist() == ["ok", "ok", "skipped: Au not in the elements"]
    assert table["charge"].tolist() == [0, 1, 0]
    assert table["n_conformers"].tolist() == [3, 4, 3]
    neutral, cation = table.loc["neutral"], table.loc["cation"]
    assert neutral["mae_emt"] == pytest.approx(0.0, abs=1e-6)
    assert neutral["spearman_emt"] == pytest.approx(1.0)
    # The reference relative energies of the cation are twice EMT's: the errors are EMT's relative energies,
    # counted from the conformer lowest in the reference.
    reference = np.array(cation["energies_ref"])
    predicted = np.array(cation["energies_emt"])
    assert reference.min() == 0.0
    assert predicted[int(np.argmin(reference))] == 0.0
    assert predicted == pytest.approx(reference / 2, abs=1e-6)
    assert cation["mae_emt"] == pytest.approx(np.abs(predicted).mean(), abs=1e-6)
    assert cation["spearman_emt"] == pytest.approx(1.0)
    summary = benchmark.summarize(table.reset_index(), "emt")
    assert (summary["n_molecules"], summary["n_ok"], summary["n_skipped"]) == (3, 2, 1)
    assert summary["mae"] == pytest.approx((neutral["mae_emt"] + cation["mae_emt"]) / 2)


def test_molecular_settings_are_kept_in_the_checkpoint(ncia_dataset: Path, conformer_dataset: Path) -> None:
    from matcalc import ConformerBenchmark, NoncovalentBenchmark

    assert NoncovalentBenchmark(ncia_dataset, sets=["R739x5"], elements=["O", "H"]).run_settings() == {
        "sets": "R739x5",
        "elements": "H,O",
    }
    assert NoncovalentBenchmark(ncia_dataset, sets="R739x5").sets == ("R739x5",)
    with pytest.raises(ValueError, match="Unknown NCI Atlas data sets"):
        NoncovalentBenchmark(ncia_dataset, sets=["S66x8"])
    assert ConformerBenchmark(conformer_dataset).run_settings() == {}
    assert ConformerBenchmark(conformer_dataset, elements=["Cu"]).run_settings() == {"elements": "Cu"}


def test_reactions(rdb7_dataset: Path, emt_simulator: ASESimulator) -> None:
    from ase.calculators.emt import EMT

    from matcalc import ReactionBenchmark
    from matcalc.properties.molecules import KCAL_PER_MOL

    benchmark = ReactionBenchmark(rdb7_dataset, elements={"Cu"})
    assert [m.material_id for m in benchmark.materials] == ["000000", "000001", "000002"]
    table = benchmark.run(emt_simulator, "emt").set_index("reaction")
    assert table["status_emt"].tolist() == ["ok", "ok", "skipped: Au not in the elements"]
    assert table["n_products"].tolist() == [1, 2, 1]
    ok = table["status_emt"] == "ok"
    # EMT against its own energies; the energies of the products are summed.
    for quantity in ("barrier", "reaction_energy"):
        assert table.loc[ok, f"{quantity}_emt"].tolist() == pytest.approx(
            table.loc[ok, f"{quantity}_ref"].tolist(), abs=1e-6
        )
    material = benchmark.materials[1]
    energies = []
    for original in [material.structure, *material.settings["products"]]:
        atoms = original.copy()
        atoms.calc = EMT()
        energies.append(atoms.get_potential_energy() / KCAL_PER_MOL)
    assert table.loc["000001", "reaction_energy_emt"] == pytest.approx(energies[1] + energies[2] - energies[0])
    assert table.loc["000001", "barrier_ref"] > 0
    summary = benchmark.summarize(table.reset_index(), "emt")
    assert (summary["n_reactions"], summary["n_ok"], summary["n_skipped"]) == (3, 2, 1)
    assert summary["barrier"]["MAE"] == pytest.approx(0.0, abs=1e-6)
    assert set(summary["reaction_energy"]) == {"MAE", "RMSE", "ME", "n"}
    assert ReactionBenchmark(rdb7_dataset).run_settings() == {}
    assert ReactionBenchmark(rdb7_dataset, elements=["Cu"]).run_settings() == {"elements": "Cu"}


def test_discovery_shards_split_the_draw_and_merge_back(discovery_dataset: Path) -> None:
    import pandas as pd

    from matcalc import DiscoveryBenchmark

    whole = [m.material_id for m in DiscoveryBenchmark(discovery_dataset).materials]
    shards = [DiscoveryBenchmark(discovery_dataset, shard=(k, 3)) for k in range(3)]
    assert [[m.material_id for m in shard.materials] for shard in shards] == [whole[k::3] for k in range(3)]
    assert shards[1].run_settings() == {"shard": "1/3"}
    tables = [pd.DataFrame({"material_id": [m.material_id for m in shard.materials]}) for shard in shards]
    assert DiscoveryBenchmark.merge_shards(tables)["material_id"].tolist() == whole
    with pytest.raises(ValueError, match="shard order"):
        DiscoveryBenchmark.merge_shards(tables[::-1])
    with pytest.raises(ValueError, match="more than one shard"):
        DiscoveryBenchmark.merge_shards([tables[0], tables[0]])
    with pytest.raises(ValueError, match="0 <= k < n"):
        DiscoveryBenchmark(discovery_dataset, shard=(3, 3))


def test_gmtkn55(gmtkn55_dataset: Path, emt_simulator: ASESimulator) -> None:
    from matcalc import GMTKN55Benchmark

    benchmark = GMTKN55Benchmark(gmtkn55_dataset)
    assert [m.material_id for m in benchmark.materials] == ["BH76:1", "BH76RC:1", "IL16:1", "W4-11:1"]
    assert benchmark.materials[1].formula == "r -> a + b"
    assert benchmark.molecules["W4-11/cu"].info["spin"] == 2
    assert benchmark.molecules["IL16/cation"].info["charge"] == 1
    evaluated: list[int] = []

    class Counting:
        def relax(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

        def single_point(self, structures: list, **kwargs: Any) -> Any:
            evaluated.append(len(structures))
            return emt_simulator.single_point(structures, **kwargs)

    table = benchmark.run(Counting(), "emt").set_index("reaction")
    assert evaluated == [len(benchmark.molecules)]  # each molecule once; BH76 and BH76RC share theirs
    assert (table["status_emt"] == "ok").all()
    deviation = table["energy_emt"] - table["energy_ref"]
    assert deviation.drop("IL16:1").abs().max() < 1e-6
    assert deviation["IL16:1"] == pytest.approx(1.0)
    summary = benchmark.summarize(table.reset_index(), "emt")
    subsets = summary["subsets"]
    assert {name: s["N"] for name, s in subsets.items()} == {"BH76": 1, "BH76RC": 1, "IL16": 1, "W4-11": 1}
    mean_abs = np.mean([s["mean_abs_reference"] for s in subsets.values()])
    assert summary["mean_abs_reference"] == pytest.approx(mean_abs)
    assert summary["WTMAD-2"]["total"] == pytest.approx(mean_abs * 1.0 / subsets["IL16"]["mean_abs_reference"] / 4)
    assert summary["WTMAD-2"]["intermolecular NCI"] == pytest.approx(mean_abs / subsets["IL16"]["mean_abs_reference"])
    assert summary["WTMAD-2"]["barrier heights"] == pytest.approx(0.0, abs=1e-6)
    assert np.isnan(summary["WTMAD-2"]["intramolecular NCI"])


def test_gmtkn55_filters_skip_whole_reactions(gmtkn55_dataset: Path, emt_simulator: ASESimulator) -> None:
    from matcalc import GMTKN55Benchmark

    def skipped(**filters: Any) -> list[str]:
        table = GMTKN55Benchmark(gmtkn55_dataset, **filters).run(emt_simulator, "emt")
        return table.loc[table["status_emt"].str.startswith("skipped"), "reaction"].tolist()

    assert skipped(charges=(0, 0)) == ["IL16:1"]
    assert skipped(max_unpaired_electrons=0) == ["BH76RC:1", "W4-11:1"]
    assert skipped(elements={"Cu"}) == ["IL16:1"]
    benchmark = GMTKN55Benchmark(gmtkn55_dataset, elements=["Cu", "Au"], charges=(-1, 1), max_unpaired_electrons=2)
    assert benchmark.run_settings() == {"elements": "Au,Cu", "charges": "-1,1", "max_unpaired_electrons": "2"}


def test_gmtkn55_reaction_lines_are_read_as_the_evaluator_reads_them() -> None:
    from matcalc.benchmarks.gmtkn55 import reactions_of

    text = "f=$1\n# a comment\n$tmer {a,b}1/$f c/$f  x -1 -1 2 $w 3.5 0 1 # W4\n"
    assert reactions_of(text) == [(["a1", "b1", "c"], [-1.0, -1.0, 2.0], 3.5)]
    with pytest.raises(ValueError, match="added energy"):
        reactions_of("$tmer a/$f x 1 $w 3.5 2.0 1\n")


def test_adsorption(adsorption_dataset: Path, emt_simulator: ASESimulator) -> None:
    from ase.build import bulk, fcc111
    from ase.constraints import FixAtoms

    from matcalc import AdsorptionBenchmark
    from matcalc.structures import molecule_in_box

    benchmark = AdsorptionBenchmark(adsorption_dataset)
    table = benchmark.run(emt_simulator, "emt")
    assert list(table["reaction"]) == ["T-1", "T-2", "T-3", "T-4"]
    assert list(table.columns[:4]) == ["reaction", "formula", "energy_exp", "energy_emt"]
    assert table["status_emt"].str.startswith("ok").all()
    assert list(table["subset"]) == ["ADS41", "ADS41", "ADS41", "Surf13"]
    assert list(table["adsorbates"]) == [2, 1, 2, 2]

    # T-1 by hand: the same EMT relaxations of the Pt crystal, the slab, H on fcc and H2 in a box.
    (crystal,) = emt_simulator.relax([bulk("Pt", "fcc", a=3.92, cubic=True)], fmax=0.02, max_steps=1000)
    slab = fcc111("Pt", size=(2, 2, 3), a=crystal.structure.lattice.a, vacuum=10.0)
    slab.pbc = True
    slab.set_constraint(FixAtoms(indices=[a.index for a in slab if a.tag == 3]))
    (clean,) = emt_simulator.relax([slab], fmax=0.02, max_steps=1000, relax_cell=False)
    with_h = slab.copy()
    with_h.positions = clean.structure.cart_coords
    fcc = with_h.info["adsorbate_info"]
    site = np.dot(fcc["sites"]["fcc"], fcc["cell"])
    top = with_h.positions[with_h.get_tags() == 1]
    centre = 0.5 * (with_h.cell[0] + with_h.cell[1])[:2]
    xy = min((atom[:2] + site for atom in top), key=lambda p: np.linalg.norm(p - centre))
    with_h.append("H")
    with_h.positions[-1] = [*xy, top[:, 2].mean() + 1.0]
    h2 = molecule_in_box(["H", "H"], [[0, 0, 0], [0, 0, 0.74]])
    adsorbed, molecule = emt_simulator.relax([with_h, h2], fmax=0.02, max_steps=1000, relax_cell=False)
    by_hand = 2 * (adsorbed.energy - clean.energy) - molecule.energy
    assert table["energy_emt"][0] == pytest.approx(by_hand, abs=1e-6)

    summary = benchmark.summarize(table, "emt")
    assert summary["n_ok"] == 4
    assert summary["all"]["n"] == 4
    assert set(summary["ADS41"]) == {"all", "chemisorption", "per_adsorbate"}
    error = table["energy_emt"] - table["energy_exp"]
    assert summary["ADS41"]["per_adsorbate"]["ME"] == pytest.approx(np.mean(error[:3] / table["adsorbates"][:3]))
    assert summary["Surf13"]["all"]["n"] == 1
    assert set(summary["timings_s"]) == {"bulk relaxation", "slab relaxation", "adsorbate relaxation"}


def test_adsorption_subsets_and_elements(adsorption_dataset: Path, emt_simulator: ASESimulator) -> None:
    from matcalc import AdsorptionBenchmark

    only_surf13 = AdsorptionBenchmark(adsorption_dataset, subsets=["Surf13"])
    assert [m.material_id for m in only_surf13.materials] == ["T-4"]
    assert only_surf13.run_settings() == {"subsets": "Surf13"}
    without_ni = AdsorptionBenchmark(adsorption_dataset, elements=["Pt", "Cu", "H", "C", "O"])
    table = without_ni.run(emt_simulator, "emt")
    assert list(table["status_emt"]) == ["ok", "ok", "skipped: N, Ni not in the elements", "ok"]
    assert np.isnan(table["energy_emt"][2])
    with pytest.raises(ValueError, match="Unknown subsets"):
        AdsorptionBenchmark(adsorption_dataset, subsets=["CE39"])


def test_adsorption_failures_reach_every_reaction_of_the_structure(
    adsorption_dataset: Path, emt_simulator: ASESimulator
) -> None:
    from matcalc import AdsorptionBenchmark

    class NoNickel(type(emt_simulator)):
        def relax(self, structures: Any, **kwargs: Any) -> Any:
            results = super().relax(structures, **kwargs)
            for i, s in enumerate(structures):
                if "Ni" in s.get_chemical_symbols() and len(s) == 4:  # the Ni crystal
                    results[i] = type(results[i]).failed("RuntimeError: no nickel")
            return results

    table = AdsorptionBenchmark(adsorption_dataset).run(NoNickel(emt_simulator.calculator, show_progress=False), "emt")
    reason = "N/Ni(100): relaxation of Ni(100): relaxation of the Ni crystal: RuntimeError: no nickel"
    assert table["status_emt"][2] == reason
    assert list(table["status_emt"][[0, 1, 3]]) == ["ok", "ok", "ok"]


def test_adsorption_dataset_is_consistent() -> None:
    """Every reaction of the packaged dataset balances its elements and names structures that exist."""
    from collections import Counter

    from matcalc import AdsorptionBenchmark
    from matcalc.surfaces import add_adsorbates, build_slab, bulk_crystal

    benchmark = AdsorptionBenchmark()
    data = benchmark.definitions
    reactions = data["reactions"]
    assert Counter(r["subset"] for r in reactions) == {"ADS41": 41, "Surf13": 13}
    categories = Counter(r["category"] for r in reactions if r["subset"] == "ADS41")
    assert categories == {"chemisorption": 26, "dispersion": 15}
    assert len({r["id"] for r in reactions}) == len(reactions)
    crystals = {
        name: bulk_crystal(s["lattice"], s["symbols"], s["a"], s.get("c"), s.get("u"))
        for name, s in data["crystals"].items()
    }
    slabs = {name: build_slab(crystals[s["crystal"]], s) for name, s in data["slabs"].items()}
    compositions = {name: Counter(slab.get_chemical_symbols()) for name, slab in slabs.items()}
    compositions |= {name: Counter(m["symbols"]) for name, m in data["molecules"].items()}
    for name, entry in data["adsorbed"].items():
        compositions[name] = Counter(add_adsorbates(slabs[entry["slab"]], entry["adsorbates"]).get_chemical_symbols())
    for reaction in reactions:
        balance: Counter[str] = Counter()
        for name, coefficient in reaction["terms"].items():
            for symbol, count in compositions[name].items():
                balance[symbol] += coefficient * count
        assert all(abs(v) < 1e-9 for v in balance.values()), reaction["id"]
        assert np.isfinite(reaction["reference"]["energy"])
        assert reaction["reference"]["energy"] < 0
