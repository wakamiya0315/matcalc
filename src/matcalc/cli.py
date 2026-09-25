"""Command line: ``matcalc-bench`` (or ``python -m matcalc.cli``).

Example::

    matcalc-bench --benchmark elasticity phonon --model MACE-MatPES-PBE-0 --n-samples 20 --out results/

For every benchmark this writes ``<benchmark>_<label>.csv`` (the result table without structures),
``<benchmark>_<label>.json.gz`` (all rows including structures; also the checkpoint that lets an
interrupted run resume) and ``summary.json`` (errors against DFT and wall times).
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .benchmarks import BENCHMARKS
from .models import load_mace

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="matcalc-bench", description="Run the MatCalc benchmarks for a MACE model.")
    parser.add_argument("--benchmark", nargs="+", choices=list(BENCHMARKS), default=list(BENCHMARKS))
    parser.add_argument("--model", default="MACE-MatPES-PBE-0", help="MACE model name or path to a .model file")
    parser.add_argument("--label", default=None, help="suffix of the predicted columns (default: the model name)")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    parser.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    parser.add_argument("--n-samples", type=int, default=None, help="random subset size (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=None, help="materials per checkpoint")
    parser.add_argument("--out", type=Path, default=Path("results"), help="output directory")
    return parser.parse_args(argv)


def _versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("matcalc", "ase", "pymatgen", "phonopy", "mace-torch", "torch"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def main(argv: Sequence[str] | None = None) -> None:
    """Run the requested benchmarks and write tables and a summary.

    Args:
        argv: Command-line arguments (default: ``sys.argv[1:]``).
    """
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    label = args.label or Path(args.model).stem
    args.out.mkdir(parents=True, exist_ok=True)
    calculator = load_mace(args.model, device=args.device, dtype=args.dtype)

    summary: dict[str, Any] = {
        "model": args.model,
        "label": label,
        "dtype": args.dtype,
        "simulator": "ase",
        "n_samples": args.n_samples,
        "seed": args.seed,
        "versions": _versions(),
        "benchmarks": {},
    }
    for name in args.benchmark:
        benchmark = BENCHMARKS[name](n_samples=args.n_samples, seed=args.seed)
        start = time.perf_counter()
        table = benchmark.run(
            calculator, label, checkpoint_file=args.out / f"{name}_{label}.json.gz", chunk_size=args.chunk_size
        )
        wall_time = time.perf_counter() - start
        csv_columns = [c for c in table.columns if not c.startswith("structure_")]
        table[csv_columns].to_csv(args.out / f"{name}_{label}.csv", index=False)
        summary["benchmarks"][name] = benchmark.summarize(table, label) | {"wall_time_s": wall_time}
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
        logger.info("%s done in %.1f s: %s", name, wall_time, summary["benchmarks"][name])
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
