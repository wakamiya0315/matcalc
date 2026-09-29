"""The Kappa benchmark in two stages: the MLIP on a GPU node, the conductivities (phono3py, CPU only) on a CPU node.

  forces JOBS.pkl [--n-samples N] [--model medium] [--workers W]
      Relaxations, harmonic check and forces on the FC3 supercells (TorchSim), as KappaBenchmark.evaluate does;
      the inputs of every conductivity calculation are saved instead of being computed.
  conductivity JOBS.pkl OUT.csv [--workers W]
      The conductivities of the saved jobs in W worker processes, then the table and the summary (OUT.json), as
      run_one.py writes them. Set RAYON_NUM_THREADS (phono3py's default Rust backend) so that W x threads fits the
      cores.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

MODEL_NAME = "mace"


class _Collector:
    """Stands in for the worker pool of the conductivity step: keeps the jobs and returns their numbers."""

    def __init__(self) -> None:
        self.jobs: list[Any] = []

    def map(self, function: Any, items: list[Any]) -> list[dict[str, int]]:
        start = len(self.jobs)
        self.jobs.extend(items)
        return [{"job": start + k} for k in range(len(items))]


def forces(args: argparse.Namespace) -> None:
    import matcalc
    import matcalc.benchmarks.kappa as kappa_module
    from matcalc.benchmarks._common import worker_pool
    from matcalc.simulation import TorchSimSimulator
    from mace_models import load_mace

    loaded = time.perf_counter()
    bench = matcalc.KappaBenchmark(n_samples=args.n_samples, seed=args.seed, workers=args.workers)
    simulator = TorchSimSimulator(load_mace("torchsim", model=args.model), show_progress=False)
    start = time.perf_counter()
    materials = bench.materials
    with bench.stage("relax"):
        relaxed = simulator.relax(
            [m.structure for m in materials],
            fmax=bench.fmax,
            max_steps=bench.max_steps,
            fix_symmetry=True,
            symprec=bench.relax_symprec,
        )
    predictions: list[dict[str, Any]] = [{} for _ in materials]
    cells = bench._symmetric_cells(materials, relaxed, predictions)
    with worker_pool(args.workers) as pool:
        harmonic = bench._harmonic_step(materials, cells, simulator, pool, predictions)
    collector = _Collector()
    # pool_map computes a single item at once; with the collector every job must go to it.
    kappa_module.pool_map = lambda pool, function, items: pool.map(function, items)
    bench._conductivity_step(materials, cells, harmonic, simulator, collector, predictions)
    saved = {
        "n_samples": args.n_samples,
        "seed": args.seed,
        "model": args.model,
        "predictions": predictions,
        "relax_steps": [r.n_steps for r in relaxed],
        "jobs": collector.jobs,
        "setup_s": round(start - loaded, 1),
        "run_s": round(time.perf_counter() - start, 1),
        "stage_times_s": {k: round(v, 1) for k, v in bench.timings.items()},
    }
    with Path(args.jobs).open("wb") as f:
        pickle.dump(saved, f)
    print(f"{len(collector.jobs)} conductivity jobs of {len(materials)} crystals saved in {saved['run_s']} s")


def conductivity(args: argparse.Namespace) -> None:
    import matcalc
    from matcalc.benchmarks._common import OK, pool_map, worker_pool
    from matcalc.benchmarks.kappa import _conductivity_or_error, _censored

    with Path(args.jobs).open("rb") as f:
        saved = pickle.load(f)  # noqa: S301 - written by the forces stage of this script
    bench = matcalc.KappaBenchmark(n_samples=saved["n_samples"], seed=saved["seed"], workers=args.workers)
    start = time.perf_counter()
    with worker_pool(args.workers) as pool:
        results = list(pool_map(pool, _conductivity_or_error, saved["jobs"]))
    elapsed = time.perf_counter() - start
    rows = []
    for material, prediction, steps in zip(bench.materials, saved["predictions"], saved["relax_steps"], strict=True):
        if "job" in prediction:
            outcome = results[prediction["job"]]
            prediction = _censored(f"phono3py failed: {outcome}") if isinstance(outcome, str) else outcome | {"status": OK}
        rows.append(bench._row(material, prediction | {"relax_steps": steps}, MODEL_NAME))
    table = bench._table(rows)
    table.to_csv(args.out, index=False)
    bench.timings = saved["stage_times_s"] | {"conductivity (CPU stage)": elapsed}
    info = {
        "model": saved["model"],
        "workers": args.workers,
        "n": len(table),
        "forces_stage_s": saved["run_s"],
        "conductivity_stage_s": round(elapsed, 1),
        "summary": bench.summarize(table, MODEL_NAME),
    }
    Path(args.out).with_suffix(".json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)
    first = stages.add_parser("forces")
    first.add_argument("jobs")
    first.add_argument("--n-samples", type=int, default=None)
    first.add_argument("--seed", type=int, default=42)
    first.add_argument("--model", default="medium")
    first.add_argument("--workers", type=int, default=2)
    second = stages.add_parser("conductivity")
    second.add_argument("jobs")
    second.add_argument("out")
    second.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    forces(args) if args.stage == "forces" else conductivity(args)


if __name__ == "__main__":
    main()
