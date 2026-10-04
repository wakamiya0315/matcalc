"""FIRE vs L-BFGS for the fixed-cell relaxations of the Adsorption benchmark (docs/validation.md, section 13).

python optimizer_check.py run --backend ase --optimizer fire --out ase-fire.json [--model medium-omat-0]
    [--dtype float64] [--device cuda] [--reactions ADS41-10,...] [--subsets ADS41]
python optimizer_check.py compare ase-fire.json ase-lbfgs.json

The crystals are relaxed with FIRE in every run (atoms and cell, keeping the space group); the optimizer under
test relaxes the slabs, the adsorbed slabs and the gas-phase molecules in their fixed cells, the bottom layers of
the slabs held by FixAtoms. L-BFGS is ASE's LBFGS with its defaults (memory 100, initial Hessian 70 eV/A^2, steps
of at most 0.2 A per atom) for ASE, and TorchSim's lbfgs with the same settings (its "ASE mode") for TorchSim.
The swap is made inside the simulators of matcalc (the FIRE of ASESimulator, the FIRE init and step functions of
TorchSimSimulator), so everything else (convergence test, fixed atoms, batching, results) is the benchmark's own.
Every relaxation is recorded: stage, formula, atoms, steps, energy, convergence.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd

STAGES = ("crystals", "slabs", "adsorbed slabs and molecules")
"""The three relax calls of the benchmark, in order."""

LBFGS_MEMORY, LBFGS_ALPHA, LBFGS_MAXSTEP = 100, 70.0, 0.2
"""ASE's LBFGS defaults, also given to TorchSim's lbfgs."""

SAME_ENERGY, SAME_REACTION = 1e-3, 0.01
"""Differences (eV) below which two relaxed structures, or two reaction energies, count as the same."""


class Recorder:
    """A simulator that relaxes fixed cells with the chosen optimizer and records every relaxation."""

    def __init__(self, inner: Any, backend: str, optimizer: str) -> None:
        self.inner, self.backend, self.optimizer = inner, backend, optimizer
        self.records: list[dict[str, Any]] = []
        self.stage_seconds: dict[str, float] = {}
        self._calls = 0

    @property
    def batched(self) -> bool:
        """Whether the wrapped simulator wants large chunks."""
        return getattr(self.inner, "batched", False)

    def single_point(self, structures: list, **kwargs: Any) -> list:
        """The wrapped simulator's single points."""
        return self.inner.single_point(structures, **kwargs)

    def relax(self, structures: list, *, relax_cell: bool = True, **kwargs: Any) -> list:
        """The wrapped simulator's relaxations, fixed cells with the chosen optimizer, each one recorded."""
        stage = STAGES[min(self._calls, len(STAGES) - 1)]
        self._calls += 1
        start = time.perf_counter()
        with self._optimizer(relax_cell=relax_cell):
            results = self.inner.relax(structures, relax_cell=relax_cell, **kwargs)
        self.stage_seconds[stage] = time.perf_counter() - start
        for structure, result in zip(structures, results, strict=True):
            formula = (
                structure.get_chemical_formula()
                if hasattr(structure, "get_chemical_formula")
                else structure.composition.formula.replace(" ", "")
            )
            self.records.append(
                {
                    "stage": stage,
                    "formula": formula,
                    "n_atoms": len(structure),
                    "steps": int(result.n_steps),
                    "energy": float(result.energy),
                    "converged": bool(result.optimizer_converged),
                    "error": result.error,
                }
            )
        return results

    def _optimizer(self, *, relax_cell: bool) -> contextlib.AbstractContextManager:
        if relax_cell or self.optimizer == "fire":
            return contextlib.nullcontext()
        stack = contextlib.ExitStack()
        if self.backend == "ase":
            from ase.optimize import LBFGS

            from matcalc.simulation import ase as ase_simulation

            lbfgs = partial(LBFGS, memory=LBFGS_MEMORY, alpha=LBFGS_ALPHA, maxstep=LBFGS_MAXSTEP)
            stack.enter_context(mock.patch.object(ase_simulation, "FIRE", lbfgs))
        else:
            from torch_sim.optimizers import lbfgs_init, lbfgs_step

            from matcalc.simulation import torchsim as torchsim_simulation

            init = partial(lbfgs_init, alpha=LBFGS_ALPHA)
            step = partial(lbfgs_step, max_history=LBFGS_MEMORY, max_step=LBFGS_MAXSTEP)
            stack.enter_context(mock.patch.object(torchsim_simulation, "fire_init", init))
            stack.enter_context(mock.patch.object(torchsim_simulation, "ase_consistent_fire_step", step))
        return stack


def run(args: argparse.Namespace) -> None:
    """One run of the benchmark with one backend and one optimizer."""
    import torch

    import matcalc
    from matcalc.benchmarks.adsorption import SUBSETS
    from matcalc.simulation import TorchSimSimulator

    sys.path.insert(0, str(Path(__file__).parent))
    from mace_models import load_mace

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    potential = load_mace(args.backend, model=args.model, dtype=args.dtype, device=device, head=args.head)
    inner = (
        TorchSimSimulator(potential, show_progress=False)
        if args.backend == "torchsim"
        else matcalc.ASESimulator(potential, show_progress=False)
    )
    simulator = Recorder(inner, args.backend, args.optimizer)
    subsets = tuple(args.subsets.split(",")) if args.subsets else SUBSETS
    bench = matcalc.AdsorptionBenchmark(subsets=subsets)
    if args.reactions:
        ids = set(args.reactions.split(","))
        bench.materials = [m for m in bench.materials if m.material_id in ids]
    start = time.perf_counter()
    table = bench.run(simulator, "mlip")
    seconds = time.perf_counter() - start
    out = {
        "backend": args.backend,
        "optimizer": args.optimizer,
        "model": args.model,
        "head": args.head,
        "dtype": args.dtype,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "seconds": round(seconds, 1),
        "stage_seconds": {k: round(v, 1) for k, v in simulator.stage_seconds.items()},
        "records": simulator.records,
        "table": table.to_dict(orient="records"),
        "summary": bench.summarize(table, "mlip"),
    }
    Path(args.out).write_text(json.dumps(out, indent=1, default=str))
    fixed = [r for r in simulator.records if r["stage"] != STAGES[0]]
    print(
        f"{args.backend} {args.optimizer}: {seconds:.0f} s, {len(simulator.records)} relaxations, "
        f"{sum(r['steps'] for r in fixed)} steps in the fixed cells, "
        f"{sum(not r['converged'] for r in simulator.records)} not converged"
    )


def compare(first_file: str, second_file: str) -> None:
    """Two runs relaxation by relaxation and reaction by reaction."""
    first, second = (json.loads(Path(f).read_text()) for f in (first_file, second_file))
    label_a = f"{first['backend']} {first['optimizer']}"
    label_b = f"{second['backend']} {second['optimizer']}"
    a, b = pd.DataFrame(first["records"]), pd.DataFrame(second["records"])
    if list(a["formula"]) != list(b["formula"]):
        raise ValueError("the two runs relaxed different structures")
    fixed = a["stage"] != STAGES[0]
    energy = (b["energy"] - a["energy"]) / a["n_atoms"]
    per_structure = b["energy"] - a["energy"]
    report: dict[str, Any] = {"runs": [label_a, label_b], "relaxations": len(a)}
    report["crystals (FIRE in both)"] = {
        "same steps": int((a["steps"][~fixed] == b["steps"][~fixed]).sum()),
        "max |dE|/atom (eV)": float(energy[~fixed].abs().max()),
    }
    steps_a, steps_b = a["steps"][fixed], b["steps"][fixed]
    report["fixed cells"] = {
        "relaxations": int(fixed.sum()),
        "steps": [int(steps_a.sum()), int(steps_b.sum())],
        "median steps": [float(steps_a.median()), float(steps_b.median())],
        "max steps": [int(steps_a.max()), int(steps_b.max())],
        "not converged": [int((~a["converged"][fixed]).sum()), int((~b["converged"][fixed]).sum())],
        "same steps": int((steps_a == steps_b).sum()),
        "fewer steps with the second": int((steps_b < steps_a).sum()),
        "max |dE| per structure (eV)": float(per_structure[fixed].abs().max()),
        "|dE| > 1 meV": int((per_structure[fixed].abs() > SAME_ENERGY).sum()),
        "lower energy with the second by > 1 meV": int((per_structure[fixed] < -SAME_ENERGY).sum()),
        "higher energy with the second by > 1 meV": int((per_structure[fixed] > SAME_ENERGY).sum()),
    }
    ta, tb = pd.DataFrame(first["table"]), pd.DataFrame(second["table"])
    d_reaction = (tb["energy_mlip"] - ta["energy_mlip"]).abs()
    report["reactions"] = {
        "n": len(ta),
        "max |dE| (eV)": float(d_reaction.max()),
        "mean |dE| (eV)": float(d_reaction.mean()),
        "|dE| > 0.01 eV": int((d_reaction > SAME_REACTION).sum()),
        "same status": bool((ta["status_mlip"] == tb["status_mlip"]).all()),
        "MAE vs experiment (eV)": [first["summary"]["all"]["MAE"], second["summary"]["all"]["MAE"]],
        "moved": [first["summary"]["n_moved"], second["summary"]["n_moved"]],
        "not converged": [first["summary"]["n_not_converged"], second["summary"]["n_not_converged"]],
    }
    report["seconds"] = [first["seconds"], second["seconds"]]
    report["stage seconds"] = [first["stage_seconds"], second["stage_seconds"]]
    print(json.dumps(report, indent=1, default=lambda x: round(float(x), 6)))
    worst = d_reaction.sort_values(ascending=False).index[:8]
    rows = ta.loc[worst, ["reaction", "formula", "energy_exp", "energy_mlip", "displacement_mlip"]].copy()
    rows["energy (second)"] = tb.loc[worst, "energy_mlip"]
    rows["displacement (second)"] = tb.loc[worst, "displacement_mlip"]
    with pd.option_context("display.width", 200):
        print(rows.round(3).to_string(index=False))
    largest = np.argsort(-per_structure[fixed].abs().to_numpy())[:8]
    rows = a[fixed].iloc[largest][["stage", "formula", "steps", "energy"]].copy()
    rows["steps (second)"] = b[fixed].iloc[largest]["steps"].to_numpy()
    rows["dE (eV)"] = per_structure[fixed].iloc[largest].to_numpy()
    with pd.option_context("display.width", 200):
        print(rows.round(4).to_string(index=False))


def main() -> None:
    """Run the command named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--backend", required=True, choices=["ase", "torchsim"])
    run_parser.add_argument("--optimizer", required=True, choices=["fire", "lbfgs"])
    run_parser.add_argument("--model", default="medium-omat-0")
    run_parser.add_argument("--head", default=None)
    run_parser.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    run_parser.add_argument("--device", default=None, choices=["cuda", "cpu"])
    run_parser.add_argument("--reactions", default=None, help="comma-separated reaction ids (default: all)")
    run_parser.add_argument("--subsets", default=None, help="ADS41, Surf13 or both (default)")
    run_parser.add_argument("--out", required=True)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("first")
    compare_parser.add_argument("second")
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        compare(args.first, args.second)


if __name__ == "__main__":
    main()
