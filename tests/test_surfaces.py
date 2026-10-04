"""Crystals, slabs and adsorbate placement of the adsorption benchmark."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from ase.geometry import find_mic

from matcalc.surfaces import VACUUM, add_adsorbates, anchor_point, build_slab, bulk_crystal

PT = bulk_crystal("fcc", ["Pt"], 3.924)
RU = bulk_crystal("hcp", ["Ru"], 2.706, 4.282)
MGO = bulk_crystal("rocksalt", ["Mg", "O"], 4.214)
RUTILE = bulk_crystal("rutile", ["Ti", "O"], 4.587, 2.954, 0.3048)
ANATASE = bulk_crystal("anatase", ["Ti", "O"], 3.782, 9.502, 0.2081)


def _layers(slab: object, tolerance: float = 0.3) -> list[list[int]]:
    """Atoms grouped into planes of equal height, bottom first."""
    heights = slab.positions[:, 2]
    order = np.argsort(heights)
    planes = [[int(order[0])]]
    for i in order[1:]:
        if heights[i] - heights[planes[-1][-1]] < tolerance:
            planes[-1].append(int(i))
        else:
            planes.append([int(i)])
    return planes


def _lateral_distance(slab: object, atoms: np.ndarray, point: np.ndarray) -> float:
    vectors = np.zeros((len(atoms), 3))
    vectors[:, :2] = atoms[:, :2] - point[:2]
    _, lengths = find_mic(vectors, slab.cell, pbc=[True, True, False])
    return float(lengths.min())


def test_crystals_have_their_conventional_cells() -> None:
    assert len(PT) == 4
    assert PT.cell.lengths() == pytest.approx([3.924] * 3)
    assert len(RU) == 2
    assert RU.cell.lengths()[2] == pytest.approx(4.282)
    assert Counter(MGO.get_chemical_symbols()) == {"Mg": 4, "O": 4}
    assert Counter(RUTILE.get_chemical_symbols()) == {"Ti": 2, "O": 4}
    assert Counter(ANATASE.get_chemical_symbols()) == {"Ti": 4, "O": 8}
    o = RUTILE.positions[np.array(RUTILE.get_chemical_symbols()) == "O"]
    ti = RUTILE.positions[0]
    assert np.linalg.norm(o - ti, axis=1).min() == pytest.approx(1.4142 * 0.3048 * 4.587, abs=1e-3)


@pytest.mark.parametrize(("crystal", "facet"), [(PT, "111"), (PT, "100"), (RU, "0001")])
def test_metal_slabs_are_periodic_with_their_bottom_layers_fixed(crystal: object, facet: str) -> None:
    slab = build_slab(crystal, {"facet": facet, "size": [2, 2], "layers": 4, "fixed_layers": 2})
    assert len(slab) == 16
    assert slab.pbc.all()
    fixed = slab.constraints[0].get_indices()
    assert sorted(fixed) == sorted(i for i in range(16) if slab[i].tag in (3, 4))
    planes = _layers(slab)
    assert len(planes) == 4
    assert {slab[i].tag for i in planes[-1]} == {1}
    assert slab.cell.lengths()[2] == pytest.approx(np.ptp(slab.positions[:, 2]) + 2 * VACUUM)


def test_sites_of_a_111_surface() -> None:
    slab = build_slab(PT, {"facet": "111", "size": [3, 3], "layers": 4, "fixed_layers": 2})
    tags = slab.get_tags()
    top, second, third = (slab.positions[tags == layer] for layer in (1, 2, 3))
    a_nn = 3.924 / np.sqrt(2)
    ontop, bridge, fcc, hcp = (anchor_point(slab, {"site": s}) for s in ("ontop", "bridge", "fcc", "hcp"))
    assert _lateral_distance(slab, top, ontop) == pytest.approx(0.0, abs=1e-9)
    assert _lateral_distance(slab, top, bridge) == pytest.approx(a_nn / 2)
    assert _lateral_distance(slab, third, fcc) == pytest.approx(0.0, abs=1e-9)  # fcc: an atom two layers down
    assert _lateral_distance(slab, second, hcp) == pytest.approx(0.0, abs=1e-9)  # hcp: an atom one layer down
    assert fcc[2] == pytest.approx(top[:, 2].mean())
    shifted = anchor_point(slab, {"site": "ontop", "shift": [1, 0]})
    vector = np.zeros((1, 3))
    vector[0, :2] = (shifted - ontop)[:2]
    _, length = find_mic(vector, slab.cell, pbc=[True, True, False])
    assert length[0] == pytest.approx(a_nn)


def test_hollow_site_of_a_100_surface_is_over_the_second_layer() -> None:
    slab = build_slab(PT, {"facet": "100", "size": [2, 2], "layers": 4, "fixed_layers": 2})
    hollow = anchor_point(slab, {"site": "hollow"})
    assert _lateral_distance(slab, slab.positions[slab.get_tags() == 2], hollow) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    ("crystal", "definition", "formula", "top"),
    [
        (
            MGO,
            {"in_plane": [[1, 0, 0], [0, 1, 0]], "stacking": [0, 0, 1], "size": [2, 2], "layers": 2},
            {"Mg": 32, "O": 32},
            {"Mg": 8, "O": 8},
        ),
        (
            RUTILE,
            {"in_plane": [[-1, 1, 0], [0, 0, 1]], "stacking": [0, 1, 0], "size": [2, 4], "layers": 5},
            {"Ti": 80, "O": 160},
            {"O": 8},  # bridging O rows
        ),
        (
            ANATASE,
            {"in_plane": [[1, 0, -1], [0, 1, 0]], "stacking": [0, 0, 1], "size": [1, 3], "layers": 4},
            {"Ti": 48, "O": 96},
            {"O": 6},  # two-fold O
        ),
    ],
)
def test_oxide_slabs_are_stoichiometric_and_non_polar(
    crystal: object, definition: dict, formula: dict, top: dict
) -> None:
    slab = build_slab(crystal, definition | {"fixed_layers": 1})
    assert Counter(slab.get_chemical_symbols()) == formula
    planes = _layers(slab, tolerance=0.1)
    assert Counter(slab[i].symbol for i in planes[-1]) == top
    assert Counter(slab[i].symbol for i in planes[0]) == top  # both surfaces alike
    charges = np.array([{"Mg": 2, "Ti": 4, "O": -2}[s] for s in slab.get_chemical_symbols()])
    centre = slab.positions[:, 2].mean()
    assert abs(np.dot(charges, slab.positions[:, 2] - centre)) < 1e-6  # no dipole across the slab
    unit = len(slab) // definition["layers"]
    assert len(slab.constraints[0].get_indices()) == unit  # one stacked unit fixed


def test_five_fold_titanium_of_rutile_110() -> None:
    slab = build_slab(
        RUTILE,
        {"in_plane": [[-1, 1, 0], [0, 0, 1]], "stacking": [0, 1, 0], "size": [2, 4], "layers": 5, "fixed_layers": 3},
    )
    ti5c = anchor_point(slab, {"atom": "Ti", "neighbours": 5})
    bridging = anchor_point(slab, {"atom": "O", "neighbours": 2})
    assert bridging[2] - ti5c[2] == pytest.approx(0.7071 * (1 - 2 * 0.3048) * 4.587, abs=1e-3)
    oxygens = slab.positions[np.array(slab.get_chemical_symbols()) == "O"]
    assert np.sort(np.linalg.norm(oxygens - ti5c, axis=1))[:6][-1] > 3.0  # nothing above a five-fold Ti
    with pytest.raises(ValueError, match="no anchor"):
        anchor_point(slab, {"atom": "Ti", "neighbours": 4})


def test_adsorbates_keep_their_offsets_from_the_anchor() -> None:
    slab = build_slab(PT, {"facet": "111", "size": [2, 2], "layers": 4, "fixed_layers": 2})
    groups = [{"anchor": {"site": "fcc"}, "symbols": ["C", "O"], "positions": [[0, 0, 1.3], [0, 0, 2.47]]}]
    adsorbed = add_adsorbates(slab, groups)
    assert len(adsorbed) == len(slab) + 2
    assert list(adsorbed.get_tags()[-2:]) == [0, 0]
    assert sorted(adsorbed.constraints[0].get_indices()) == sorted(slab.constraints[0].get_indices())
    fcc = anchor_point(slab, {"site": "fcc"})
    assert adsorbed.positions[-2] == pytest.approx(fcc + np.array([0, 0, 1.3]))
    assert adsorbed.get_distance(len(slab), len(slab) + 1) == pytest.approx(1.17)
