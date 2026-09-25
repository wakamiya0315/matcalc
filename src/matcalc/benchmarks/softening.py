"""Softening benchmark: are the MLIP's forces on high-energy configurations systematically too small?

B. Deng et al., npj Comput. Mater. 11, 9 (2025), doi:10.1038/s41524-024-01500-6. The dataset holds
up to ten high-energy configurations ("frames") per WBM material with their DFT (GGA) forces
(figshare 27307776).

Recipe:

1. Forces on every frame of every material (single points at the DFT geometries).
2. Softening scale = slope ``a`` of F_MLIP ≈ a · F_DFT over all force components of a material's
   frames; ``a < 1`` means the MLIP's potential energy surface is too soft.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from matcalc.properties.softening import softening_scale

from ._common import OK, Benchmark, Material, failed

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matcalc.simulation import Simulator


class SofteningBenchmark(Benchmark):
    """Softening scale ``softening_scale`` (dimensionless) of WBM materials."""

    name = "softening"
    id_column = "material_id"
    default_dataset = "wbm-high-energy-states.json.gz"
    summary_metrics: ClassVar[dict[str, str]] = {"softening_scale": "value"}
    default_chunk_size = 200

    def read_entries(self, raw: Any) -> list[Material]:
        """Read the high-energy frames.

        Args:
            raw: Dict of material id → {frame id → {``structure``, ``vasp_f`` (DFT forces, eV/Å), ...}}.

        Returns:
            One ``Material`` per material; ``reference["frames"]`` holds its frames.

        Raises:
            TypeError: If ``raw`` is not a dict.
        """
        if not isinstance(raw, dict):
            raise TypeError(f"SofteningBenchmark expects a dict of material id -> frames, got {type(raw).__name__}.")
        materials = []
        for material_id, frames_by_id in raw.items():
            frames = list(frames_by_id.values())
            first = frames[0]["structure"]
            materials.append(Material(material_id, first.composition.reduced_formula, first, {"frames": frames}))
        return materials

    def evaluate(self, materials: Sequence[Material], simulator: Simulator) -> list[dict[str, Any]]:
        """Steps 1-2 for some materials.

        Args:
            materials: Materials to evaluate.
            simulator: The simulator of this run.

        Returns:
            Per material: ``softening_scale`` and ``status``.
        """
        frames = [frame for material in materials for frame in material.reference["frames"]]
        with self.stage("single points"):
            forces = iter(simulator.single_point([frame["structure"] for frame in frames], compute_stress=False))

        predictions = []
        with self.stage("fit"):
            for material in materials:
                own_frames = material.reference["frames"]
                own = [next(forces) for _ in own_frames]
                errors = [r.error for r in own if r.error is not None]
                if errors:
                    predictions.append(failed(f"single point failed: {errors[0]}", ("softening_scale",)))
                    continue
                scale = softening_scale([frame["vasp_f"] for frame in own_frames], [r.forces for r in own])
                predictions.append({"softening_scale": scale, "status": OK})
        return predictions
