"""Smoke tests with a real MACE model and the real datasets (GPU node, network).

Run with ``pytest -m mace``; they are skipped when mace-torch is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from matcalc import ElasticityBenchmark, PhononBenchmark, SofteningBenchmark, load_mace

pytestmark = [pytest.mark.mace, pytest.mark.network]
pytest.importorskip("mace")


@pytest.fixture(scope="module")
def mace_calculator() -> object:
    return load_mace("MACE-MatPES-PBE-0", dtype="float64")


def test_softening_with_mace(mace_calculator: object) -> None:
    table = SofteningBenchmark(n_samples=3, seed=1).run(mace_calculator, "mace")
    assert (table["status_mace"] == "ok").all()
    assert ((table["softening_scale_mace"] > 0.3) & (table["softening_scale_mace"] < 1.5)).all()


def test_elasticity_with_mace(mace_calculator: object) -> None:
    table = ElasticityBenchmark(n_samples=3, seed=101).run(mace_calculator, "mace")
    ok = table["status_mace"] == "ok"
    assert ok.any()
    errors = np.abs(table.loc[ok, "K_vrh_mace"] - table.loc[ok, "K_vrh_DFT"])
    assert (errors < 60).all()


def test_phonon_with_mace(mace_calculator: object) -> None:
    table = PhononBenchmark(n_samples=2, seed=0).run(mace_calculator, "mace")
    assert table["status_mace"].str.startswith("ok").all()
    assert (np.abs(table["CV_mace"] - table["CV_DFT"]) < 20).all()
