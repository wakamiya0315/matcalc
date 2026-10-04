"""Bulk crystals, slabs and adsorbed structures for the adsorption benchmark.

Slabs are cut from a relaxed bulk crystal, so that every MLIP works with its own lattice constants (as the
DFT calculations of the reference data did with their functionals):

- metal surfaces, fcc(111), fcc(100) and hcp(0001), with ASE's builders;
- oxide surfaces, MgO(001), rutile TiO2(110) and anatase TiO2(101), by stacking an oriented cell of the
  bulk crystal (``oriented_slab``), cut where the stacked unit carries no dipole (Tasker's criterion).

Every slab is periodic in all three directions, with ``VACUUM`` Å of vacuum on each side. Atoms are
tagged with their layer, counted from the top (1 = top layer; adsorbate atoms get 0), and the bottom
``fixed_layers`` layers are held by a ``FixAtoms`` constraint.

Adsorbates are placed relative to an *anchor* of the top layer: a site of a metal surface (``ontop``,
``bridge``, ``fcc``, ``hcp`` or ``hollow``) or a surface atom (for example a five-fold coordinated Ti of
rutile(110)). Their positions are offsets from the anchor (Å), in the frame of the slab: x along the first
surface vector, z along the surface normal. Of the equivalent anchors of a supercell, the one nearest to the
centre of the surface cell is used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from ase import Atoms
from ase.build import bulk, fcc100, fcc111, hcp0001, make_supercell
from ase.constraints import FixAtoms
from ase.neighborlist import neighbor_list
from ase.spacegroup import crystal

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

VACUUM = 10.0
"""Vacuum on each side of a slab (Å): 20 Å between a surface and the periodic image of the slab above it."""

FORMAL_CHARGES = {"Mg": 2, "Ti": 4, "O": -2}
"""Formal ionic charges, used to find the non-polar termination of oxide slabs."""

BOND_LENGTH = 2.4
"""Largest metal-oxygen distance counted as a bond when atoms of an oxide surface are classified (Å)."""


def bulk_crystal(
    lattice: str, symbols: Sequence[str], a: float, c: float | None = None, u: float | None = None
) -> Atoms:
    """The conventional cell of a crystal.

    Args:
        lattice: ``"fcc"`` (4 atoms), ``"hcp"`` (2 atoms), ``"rocksalt"`` (8 atoms), ``"rutile"`` (TiO2, space
            group 136, 6 atoms) or ``"anatase"`` (TiO2, space group 141, 12 atoms).
        symbols: Chemical symbols of the cation and (oxides) of the anion.
        a: Lattice constant a (Å).
        c: Lattice constant c (Å; hcp, rutile, anatase).
        u: Internal coordinate of the anion (rutile: O at (u, u, 0); anatase: O at (0, 0, u)).

    Returns:
        The crystal, periodic in all directions.

    Raises:
        ValueError: For an unknown lattice.
    """
    if lattice == "fcc":
        return bulk(symbols[0], "fcc", a=a, cubic=True)
    if lattice == "hcp":
        return bulk(symbols[0], "hcp", a=a, c=c)
    if lattice == "rocksalt":
        return bulk("".join(symbols), "rocksalt", a=a, cubic=True)
    if lattice == "rutile":
        return crystal(list(symbols), basis=[(0, 0, 0), (u, u, 0)], spacegroup=136, cellpar=[a, a, c, 90, 90, 90])
    if lattice == "anatase":
        return crystal(list(symbols), basis=[(0, 0, 0), (0, 0, u)], spacegroup=141, cellpar=[a, a, c, 90, 90, 90])
    raise ValueError(f"unknown lattice {lattice!r}")


def build_slab(crystal_cell: Atoms, definition: Mapping[str, Any]) -> Atoms:
    """A slab of a relaxed crystal, with its bottom layers fixed.

    Args:
        crystal_cell: The relaxed conventional cell of the crystal (from ``bulk_crystal``).
        definition: Either a metal surface, ``{"facet": "111" | "100" | "0001", "size": [n1, n2], "layers",
            "fixed_layers"}``, or an oxide surface, ``{"in_plane": [v1, v2], "stacking": v3, "size": [n1, n2],
            "layers", "fixed_layers"}`` with the vectors in units of the conventional cell (``oriented_slab``;
            a layer is then one stacked unit).

    Returns:
        The slab, tagged by layer (1 = top), with a ``FixAtoms`` constraint on the bottom layers.
    """
    size, layers = definition["size"], definition["layers"]
    if "facet" in definition:
        slab = metal_slab(crystal_cell, definition["facet"], size, layers)
    else:
        slab = oriented_slab(crystal_cell, definition["in_plane"], definition["stacking"], layers, size)
    fixed = [atom.index for atom in slab if atom.tag > layers - definition["fixed_layers"]]
    slab.set_constraint(FixAtoms(indices=fixed))
    return slab


def metal_slab(crystal_cell: Atoms, facet: str, size: Sequence[int], layers: int) -> Atoms:
    """A metal slab built by ASE, with the lattice constants of the relaxed crystal.

    Args:
        crystal_cell: Conventional fcc cell (4 atoms) or hcp cell (2 atoms) of the metal.
        facet: ``"111"`` or ``"100"`` (fcc), ``"0001"`` (hcp).
        size: Repetitions of the surface cell (n1, n2).
        layers: Number of atomic layers.

    Returns:
        The slab (non-orthogonal surface cell for (111) and (0001)), tagged by layer (1 = top), with the
        site positions of ASE's builders in ``info["adsorbate_info"]``.

    Raises:
        ValueError: For an unknown facet.
    """
    symbol = crystal_cell[0].symbol
    lengths = crystal_cell.cell.lengths()
    if facet == "0001":
        slab = hcp0001(symbol, size=(*size, layers), a=lengths[0], c=lengths[2], vacuum=VACUUM)
    elif facet == "111":
        slab = fcc111(symbol, size=(*size, layers), a=lengths[0], vacuum=VACUUM)
    elif facet == "100":
        slab = fcc100(symbol, size=(*size, layers), a=lengths[0], vacuum=VACUUM)
    else:
        raise ValueError(f"unknown metal facet {facet!r}")
    slab.pbc = True
    return slab


def oriented_slab(
    crystal_cell: Atoms, in_plane: Sequence[Sequence[int]], stacking: Sequence[int], layers: int, size: Sequence[int]
) -> Atoms:
    """A slab made of ``layers`` stacked units of an oriented cell of the crystal.

    The oriented cell is spanned by the lattice vectors ``in_plane`` (the surface cell) and ``stacking`` (the
    translation from one unit to the next), all in units of the conventional cell. The unit starts at the
    atomic plane for which it carries no dipole along the surface normal (formal charges; Tasker type I or
    II), the widest gap between planes deciding between several such planes. The slab is rotated so that the
    first surface vector lies along x and the surface normal along z.

    Args:
        crystal_cell: Conventional cell of the crystal.
        in_plane: The two surface vectors v1, v2; v1 x v2 points out of the surface.
        stacking: The stacking vector v3, pointing out of the surface.
        layers: Number of stacked units.
        size: Repetitions of the surface cell (n1, n2).

    Returns:
        The slab, its units tagged from the top (1 = top unit).

    Raises:
        ValueError: If the stacking vector points into the surface.
    """
    unit = make_supercell(crystal_cell, np.array([*in_plane, stacking], dtype=float), wrap=True)
    v1, v2, v3 = np.array(unit.cell)
    normal = np.cross(v1, v2) / np.linalg.norm(np.cross(v1, v2))
    if np.dot(v3, normal) <= 0:
        raise ValueError(f"the stacking vector {stacking} points into the surface spanned by {in_plane}")
    heights = np.round(unit.get_scaled_positions(wrap=True)[:, 2], 6)
    scaled = unit.get_scaled_positions(wrap=True)
    scaled[:, 2] = (heights - _nonpolar_start(heights, unit.get_chemical_symbols())) % 1.0
    unit.set_scaled_positions(scaled)
    positions, symbols, tags = [], [], []
    for k in range(layers):
        for i in range(size[0]):
            for j in range(size[1]):
                positions.extend(unit.positions + i * v1 + j * v2 + k * v3)
                symbols.extend(unit.get_chemical_symbols())
                tags.extend([layers - k] * len(unit))
    x = v1 / np.linalg.norm(v1)
    rotation = np.array([x, np.cross(normal, x), normal])  # rows: the new x, y and z axes
    positions = np.array(positions) @ rotation.T
    thickness = np.ptp(positions[:, 2])
    slab = Atoms(
        symbols,
        positions=positions,
        cell=[size[0] * v1 @ rotation.T, size[1] * v2 @ rotation.T, [0.0, 0.0, thickness + 2 * VACUUM]],
        tags=tags,
        pbc=True,
    )
    slab.positions[:, 2] += VACUUM - positions[:, 2].min()
    return slab


def _nonpolar_start(heights: np.ndarray, symbols: Sequence[str]) -> float:
    """Fractional height of the lowest atomic plane of the stacked unit that carries no dipole."""
    charges = np.array([FORMAL_CHARGES[s] for s in symbols], dtype=float)
    planes = np.unique(heights)
    gaps_below = planes - np.roll(planes, 1) + (planes == planes[0])  # gap from the plane below (cyclic)
    best: tuple[tuple[float, float], float] | None = None
    for plane, gap in zip(planes, gaps_below, strict=True):
        dipole = abs(float(np.dot(charges, (heights - plane) % 1.0)))
        key = (round(dipole, 6), -gap)
        if best is None or key < best[0]:
            best = (key, float(plane))
    assert best is not None  # noqa: S101 - a crystal has at least one plane
    return best[1]


def anchor_point(slab: Atoms, anchor: Mapping[str, Any]) -> np.ndarray:
    """Position of an anchor of the top layer.

    Args:
        slab: The slab (relaxed or not), tagged by layer as by ``build_slab``.
        anchor: ``{"site": name}`` for a site of a metal slab (``ontop``, ``bridge``, ``fcc``, ``hcp``,
            ``hollow``; the height is the mean height of the top layer), optionally ``"shift": [i, j]`` surface
            cells away from the site nearest to the centre (to anchor the molecules of a layer each on its own
            site), or ``{"atom": symbol}`` for a surface atom, the topmost atom of that element (within 0.5 Å of
            the highest one), optionally with ``"neighbours": n`` O (or metal) atoms within ``BOND_LENGTH``.

    Returns:
        Cartesian position of the anchor (Å); of the equivalent anchors, the one nearest to the centre of the
        surface cell.

    Raises:
        ValueError: If the slab has no such anchor.
    """
    centre = 0.5 * (slab.cell[0] + slab.cell[1])
    if "site" in anchor:
        # ASE's site coordinates are relative to an atom of the top layer, in units of the surface cell; the
        # sites of the supercell are their translations by the top-layer atoms.
        info = slab.info["adsorbate_info"]
        top = slab.positions[slab.get_tags() == 1]
        site = np.dot(info["sites"][anchor["site"]], info["cell"])
        candidates = [np.array([*(atom[:2] + site), top[:, 2].mean()]) for atom in top]
        if "shift" in anchor:
            distances = [np.linalg.norm(_in_cell(slab, c)[:2] - centre[:2]) for c in candidates]
            base = _in_cell(slab, candidates[int(np.argmin(distances))])
            base[:2] += np.dot(anchor["shift"], info["cell"])
            return _in_cell(slab, base)
    else:
        symbols = np.array(slab.get_chemical_symbols())
        heights = slab.positions[:, 2]
        of_element = symbols == anchor["atom"]
        if not of_element.any():
            raise ValueError(f"no {anchor['atom']} in the slab")
        topmost = of_element & (heights > heights[of_element].max() - 0.5)
        if "neighbours" in anchor:
            i, j = neighbor_list("ij", slab, BOND_LENGTH)
            counted = (symbols[j] == "O") != (symbols[i] == "O")  # cation-anion pairs
            coordination = np.bincount(i[counted], minlength=len(slab))
            topmost &= coordination == anchor["neighbours"]
        candidates = list(slab.positions[topmost])
    if not candidates:
        raise ValueError(f"the slab has no anchor {dict(anchor)}")
    distances = [np.linalg.norm(_in_cell(slab, c)[:2] - centre[:2]) for c in candidates]
    return _in_cell(slab, candidates[int(np.argmin(distances))])


def add_adsorbates(slab: Atoms, adsorbates: Sequence[Mapping[str, Any]]) -> Atoms:
    """The slab with adsorbates placed at fixed offsets from their anchors.

    Args:
        slab: The slab (tagged by layer, with its constraints).
        adsorbates: Groups of atoms ``{"anchor": ..., "symbols": [...], "positions": [[dx, dy, dz], ...]}``,
            positions relative to the anchor (Å; see ``anchor_point``).

    Returns:
        A new structure: the slab with the adsorbate atoms appended (tag 0), the constraints of the slab kept,
        and every atom wrapped into the cell.
    """
    combined = slab.copy()
    for group in adsorbates:
        origin = anchor_point(slab, group["anchor"])
        atoms = Atoms(group["symbols"], positions=origin + np.asarray(group["positions"], dtype=float))
        atoms.set_tags(0)
        combined.extend(atoms)
    combined.wrap()
    return combined


def _in_cell(slab: Atoms, position: np.ndarray) -> np.ndarray:
    """The periodic image of a position inside the surface cell (the height is kept)."""
    scaled = np.linalg.solve(np.array(slab.cell).T, position)
    scaled[:2] %= 1.0
    return scaled @ np.array(slab.cell)
