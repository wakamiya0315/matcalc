"""The Kappa benchmark in two stages: the MLIP on a GPU node, the conductivities (phono3py, CPU only) on a CPU node.

  forces FORCES.pkl [--n-samples N] [--model medium] [--workers W]
      KappaBenchmark.save_forces with MACE (TorchSim): relaxations, harmonic check and FC3 forces.
  conductivity FORCES.pkl OUT.csv [--n-samples N] [--workers W]
      KappaBenchmark.run_saved_forces in W worker processes; writes the table and the summary (OUT.json) as
      run_one.py does. Set RAYON_NUM_THREADS (phono3py's default Rust backend) and OMP_NUM_THREADS so that
      W x threads fits the cores.
"""

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["forces", "conductivity"])
    parser.add_argument("forces_file")
    parser.add_argument("out", nargs="?")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="medium")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    import matcalc

    bench = matcalc.KappaBenchmark(n_samples=args.n_samples, seed=args.seed, workers=args.workers)
    start = time.perf_counter()
    if args.stage == "forces":
        from matcalc.simulation import TorchSimSimulator

        sys.path.insert(0, str(Path(__file__).parent))
        from mace_models import load_mace

        bench.save_forces(TorchSimSimulator(load_mace("torchsim", model=args.model), show_progress=False), args.forces_file)
        print(json.dumps({"forces_stage_s": round(time.perf_counter() - start, 1), "stage_times_s": bench.timings}))
        return
    table = bench.run_saved_forces(args.forces_file, "mace")
    table.to_csv(args.out, index=False)
    info = {
        "workers": args.workers,
        "n": len(table),
        "conductivity_stage_s": round(time.perf_counter() - start, 1),
        "summary": bench.summarize(table, "mace"),
    }
    Path(args.out).with_suffix(".json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
