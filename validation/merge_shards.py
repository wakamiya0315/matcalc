"""Join the tables of a Discovery run split into shards, and compute Matbench Discovery's metrics of the whole run.

  python merge_shards.py discovery_0of4.csv discovery_1of4.csv discovery_2of4.csv discovery_3of4.csv \\
      --out discovery.csv

The tables are those that run_one.py writes for --shard 0/n, 1/n, ..., (n-1)/n (column suffix "mace"), given in
this order. Writes the joined table (OUT) and its metrics (OUT with .json), and prints the metrics.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("shards", nargs="+", help="the tables of shards 0, 1, ..., n-1, in this order")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-name", default="mace")
    args = parser.parse_args()

    from matcalc import DiscoveryBenchmark

    table = DiscoveryBenchmark.merge_shards([pd.read_csv(path) for path in args.shards])
    table.to_csv(args.out, index=False)
    metrics = DiscoveryBenchmark.metrics(table, args.model_name)
    Path(args.out).with_suffix(".json").write_text(json.dumps(metrics, indent=1, default=str))
    print(json.dumps(metrics, default=str))


if __name__ == "__main__":
    main()
