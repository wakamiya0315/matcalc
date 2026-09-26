"""V4/V5: run one benchmark with upstream main (ASE, n_jobs=1) or this fork (ASE or TorchSim), MACE-MatPES-PBE-0."""

import argparse
import json
import logging
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("code", choices=["upstream", "fork-ase", "fork-torchsim"])
    parser.add_argument("benchmark", choices=["equilibrium", "elasticity", "phonon", "softening"])
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--min-supercell-length", type=float, default=None, help="phonon only (A)")
    parser.add_argument("--cueq", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import torch

    start = time.perf_counter()
    extra = {}
    if args.code == "upstream":
        from mace.calculators import mace_mp
        from pymatgen.core import Structure

        unseeded = Structure.perturb

        def seeded(self, *args_, seed=None, **kwargs):
            return unseeded(self, *args_, seed=args.seed if seed is None else seed, **kwargs)

        Structure.perturb = seeded  # upstream's perturbation is unseeded; use the fork's seed
        import matcalc.benchmark as mb

        calc = mace_mp(model="mace-matpes-pbe-0", device="cuda", default_dtype="float64")
        cls = {"equilibrium": mb.EquilibriumBenchmark, "elasticity": mb.ElasticityBenchmark,
               "phonon": mb.PhononBenchmark, "softening": mb.SofteningBenchmark}[args.benchmark]
        bench = cls(n_samples=args.n_samples, seed=args.seed)
        loaded = time.perf_counter()
        if args.benchmark == "softening":
            table = bench.run(calc, "mace", checkpoint_file=args.checkpoint)
        else:
            table = bench.run(calc, "mace", n_jobs=1, checkpoint_file=args.checkpoint, checkpoint_freq=50,
                              delete_checkpoint_on_finish=False)
    else:
        import matcalc
        from matcalc.simulation.torchsim import TorchSimSimulator

        backend = "torchsim" if args.code == "fork-torchsim" else "ase"
        model = matcalc.load_mace("MACE-MatPES-PBE-0", backend=backend, dtype=args.dtype, cueq=args.cueq)
        simulator = (TorchSimSimulator(model, show_progress=False) if backend == "torchsim"
                     else matcalc.ASESimulator(model, show_progress=False))
        options = {} if args.min_supercell_length is None else {"min_supercell_length": args.min_supercell_length}
        bench = matcalc.BENCHMARKS[args.benchmark](
            n_samples=args.n_samples, seed=args.seed, workers=args.workers, **options
        )
        loaded = time.perf_counter()
        table = bench.run(simulator, "mace", checkpoint_file=args.checkpoint)
        extra = {"stage_times_s": {k: round(v, 1) for k, v in bench.timings.items()}}
        if backend == "torchsim":
            extra["capacities"] = [round(c) for c in simulator.capacities]
    end = time.perf_counter()
    table = table[[c for c in table.columns if not c.startswith("structure_")]]
    table.to_csv(args.out, index=False)
    info = {"code": args.code, "dtype": args.dtype, "workers": args.workers, "cueq": args.cueq,
            "min_supercell_length": args.min_supercell_length,
            "benchmark": args.benchmark, "n": len(table), "setup_s": round(loaded - start, 1),
            "run_s": round(end - loaded, 1), "gpu": torch.cuda.get_device_name(0),
            "peak_gpu_mem_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2), **extra}
    Path(args.out).with_suffix(".json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
