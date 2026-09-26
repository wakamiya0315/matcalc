"""V3: TorchSim MACE vs ASE MACE (same checkpoint, float64) on cells taken from the four datasets.

1. Single points on DFT, strained (+-1 % normal, +-6 % shear), high-energy and perturbed cells: energy,
   forces and stress, with TorchSim's default neighbour list and with its cell-list neighbour list.
2. Relaxations (FIRE + Frechet cell filter, fmax 0.05 eV/A, 500 steps) of 30 structures. TorchSim gets room
   for only about four structures at a time, so most structures join a batch that is already running.
   Compared: TorchSimSimulator, and the same with TorchSim's own FIRE step (without its two corrections).

Set V3_ONLY_RELAX=1 to run part 2 only. Phonon supercells are left out here because the shared iqrsh GPU
could not hold the largest float64 ones; they are compared material by material in V4/V5 on gpu_h.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

import matcalc  # before torch_sim, as the benchmark runner does
from mace_models import load_mace
from matcalc import ASESimulator
from matcalc.properties.elasticity import strained_structures
from matcalc.simulation import torchsim as tsim
from matcalc.structures import to_ase_atoms

import torch_sim as ts  # isort: skip
from torch_sim.autobatching import calculate_memory_scalers  # isort: skip
from torch_sim.models.mace import MaceModel  # isort: skip
from torch_sim.optimizers import fire_step  # isort: skip


def single_points(report: dict, calc: object, model: object) -> dict[str, list]:
    elastic = matcalc.ElasticityBenchmark(n_samples=10, seed=7).materials
    strained = [strained_structures(m.structure)[0] for m in elastic]
    softening = matcalc.SofteningBenchmark(n_samples=10, seed=3).materials
    equilibrium = matcalc.EquilibriumBenchmark(n_samples=10, seed=5).materials
    cells = {
        "elasticity: DFT cells": [m.structure for m in elastic],
        "elasticity: strained (+-1% normal, +-6% shear)": [s[3] for s in strained] + [s[23] for s in strained],
        "softening: high-energy frames": [f["structure"] for m in softening for f in m.reference["frames"][:2]],
        "equilibrium: perturbed 0.1 A": [m.structure.copy().perturb(0.1, seed=42) for m in equilibrium],
    }
    ase_sim = ASESimulator(calc, show_progress=False)
    for nl_name, nl in (("default", None), ("cell_list", ts.neighbors.alchemiops_nl_cell_list)):
        ts_model = model if nl is None else MaceModel(model=model.model, device=model.device, dtype=model.dtype, neighbor_list_fn=nl)
        ts_sim = tsim.TorchSimSimulator(ts_model, show_progress=False)
        for label, group in cells.items():
            ref, got = ase_sim.single_point(group), ts_sim.single_point(group)
            n_atoms = np.array([len(c) for c in group])
            report[f"single point [{nl_name}] {label}"] = {
                "n_cells": len(group),
                "atoms": [int(n_atoms.min()), int(n_atoms.max())],
                "max|dE|/atom (eV)": max(abs(r.energy - g.energy) / n for r, g, n in zip(ref, got, n_atoms)),
                "max|dF| (eV/A)": float(max(np.abs(r.forces - g.forces).max() for r, g in zip(ref, got))),
                "max|dsigma| (eV/A^3)": float(max(np.abs(r.stress - g.stress).max() for r, g in zip(ref, got))),
            }
            print(json.dumps({f"[{nl_name}] {label}": report[f"single point [{nl_name}] {label}"]}), flush=True)
            torch.cuda.empty_cache()
    return cells


def relaxations(report: dict, calc: object, model: object, perturbed: list) -> None:
    starts = [m.structure for m in matcalc.ElasticityBenchmark(n_samples=20, seed=11).materials] + perturbed
    t0 = time.perf_counter()
    ref = ASESimulator(calc, show_progress=False).relax(starts, fmax=0.05, max_steps=500)
    t_ase = time.perf_counter() - t0
    state = ts.io.atoms_to_state([to_ase_atoms(s) for s in starts], device=model.device, dtype=model.dtype)
    sim = tsim.TorchSimSimulator(model, max_memory_scaler=4 * float(np.median(calculate_memory_scalers(state))), show_progress=False)

    def compare(results: list, label: str, elapsed: float) -> None:
        de = np.array([abs(r.energy - g.energy) / len(s) for r, g, s in zip(ref, results, starts)])
        da = [np.abs(np.array(r.structure.lattice.abc) - np.array(g.structure.lattice.abc)).max() for r, g in zip(ref, results)]
        dn = np.array([g.n_steps - r.n_steps for r, g in zip(ref, results)])
        report[f"relax {label}"] = {
            "n": len(starts),
            "max|dE|/atom (eV)": float(de.max()),
            "max|d lattice| (A)": float(max(da)),
            "steps TS-ASE min/median/max": [int(dn.min()), float(np.median(dn)), int(dn.max())],
            "converged flips": int(sum(r.converged != g.converged for r, g in zip(ref, results))),
            "time ASE (s)": round(t_ase, 1),
            "time TorchSim (s)": round(elapsed, 1),
        }
        print(json.dumps({label: report[f"relax {label}"]}), flush=True)

    t0 = time.perf_counter()
    compare(sim.relax(starts, fmax=0.05, max_steps=500), "TorchSimSimulator", time.perf_counter() - t0)
    corrected = tsim.ase_consistent_fire_step
    tsim.ase_consistent_fire_step = fire_step  # relax() looks the step up at call time
    try:
        t0 = time.perf_counter()
        compare(sim.relax(starts, fmax=0.05, max_steps=500), "TorchSim's own FIRE step", time.perf_counter() - t0)
    finally:
        tsim.ase_consistent_fire_step = corrected


def main() -> None:
    calc = load_mace("ase")
    model = load_mace("torchsim")
    report: dict = {}
    if os.environ.get("V3_ONLY_RELAX"):
        equilibrium = matcalc.EquilibriumBenchmark(n_samples=10, seed=5).materials
        perturbed = [m.structure.copy().perturb(0.1, seed=42) for m in equilibrium]
    else:
        perturbed = single_points(report, calc, model)["equilibrium: perturbed 0.1 A"]
    relaxations(report, calc, model, perturbed)
    with open("v3_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
