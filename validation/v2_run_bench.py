"""V2: run one benchmark with the upstream code (main) or the refactored code, same MACE model."""

import argparse
import json
import time

parser = argparse.ArgumentParser()
parser.add_argument("code", choices=["upstream", "refactor"])
parser.add_argument("benchmark", choices=["equilibrium", "elasticity", "phonon", "softening"])
parser.add_argument("--n-samples", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", required=True)
args = parser.parse_args()

from mace.calculators import mace_mp  # noqa: E402

calc = mace_mp(model="mace-matpes-pbe-0", device="cuda", default_dtype="float64")
start = time.perf_counter()
if args.code == "upstream":
    from pymatgen.core import Structure

    unseeded = Structure.perturb

    def seeded(self, *args_, seed=None, **kwargs):
        """Upstream perturbs with seed=None; use the refactor's seed so both start from the same geometry.

        Everything else (in particular pymatgen's default min_distance) is passed through unchanged.
        """
        return unseeded(self, *args_, seed=args.seed if seed is None else seed, **kwargs)

    Structure.perturb = seeded
    import matcalc
    import matcalc.benchmark as mb

    cls = {
        "equilibrium": mb.EquilibriumBenchmark,
        "elasticity": mb.ElasticityBenchmark,
        "phonon": mb.PhononBenchmark,
        "softening": mb.SofteningBenchmark,
    }[args.benchmark]
    bench = cls(n_samples=args.n_samples, seed=args.seed)
    table = bench.run(calc, "mace", **({} if args.benchmark == "softening" else {"n_jobs": 1}))
else:
    import matcalc

    bench = matcalc.BENCHMARKS[args.benchmark](n_samples=args.n_samples, seed=args.seed)
    table = bench.run(calc, "mace")
wall = time.perf_counter() - start
table = table[[c for c in table.columns if not c.startswith("structure_")]]
table.to_csv(args.out, index=False)
print(json.dumps({"code": args.code, "matcalc": matcalc.__file__, "benchmark": args.benchmark, "wall_s": round(wall, 1), "rows": len(table)}))
