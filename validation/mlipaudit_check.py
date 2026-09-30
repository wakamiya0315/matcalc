"""The molecular benchmarks against MLIPAudit's published results for the same model, system by system.

  noncovalent OURS.csv RESULT.json
      Interaction energy of every complex (our ids "<set>:<curve>" against MLIPAudit's curve ids), the
      skipped complexes, and MLIPAudit's summary (MAE and RMSE overall, per data set and per subset)
      against the one of NoncovalentBenchmark.summarize on our table.
  conformers OURS.csv RESULT.json
      Per molecule: the relative energy of every conformer, MAE, RMSE and Spearman correlation; the mean MAE
      and RMSE against MLIPAudit's.

OURS.csv is written by run_one.py (column suffix "mace"); RESULT.json is MLIPAudit's result file of the model,
https://huggingface.co/datasets/InstaDeepAI/mlipaudit-results/resolve/<revision>/<model>/<benchmark>/result.json
with <benchmark> = noncovalent_interactions or conformer_selection (MACE-OFF23 medium: <model> = MACE-OFF_ext).
"""

import argparse
import json

import numpy as np
import pandas as pd


def noncovalent(ours: pd.DataFrame, audit: dict) -> dict:
    """Complex-by-complex and summary differences (kcal/mol)."""
    import matcalc

    theirs = {s["system_id"]: s for s in audit["systems"]}
    ours = ours.assign(curve=ours["system_id"].str.split(":").str[1]).set_index("curve")
    ok = ours["status_mace"] == "ok"
    common = sorted(set(ours.index[ok]) & set(theirs))
    deviation = np.array([ours.loc[c, "interaction_energy_mace"] - theirs[c]["mlip_interaction_energy"] for c in common])
    reference = np.array([ours.loc[c, "interaction_energy_ref"] - theirs[c]["reference_interaction_energy"] for c in common])
    same = set(ours.index[ok]) == set(theirs)
    summary = matcalc.NoncovalentBenchmark().summarize(ours.reset_index(drop=True), "mace")
    worst: dict | None = {"all": 0.0, "datasets": 0.0, "subsets": 0.0} if same else None  # full runs only
    for kind in ("mae", "rmse") if same else ():
        key = kind.upper()
        worst["all"] = max(worst["all"], abs(summary["interaction_energy"][key] - audit[f"{kind}_interaction_energy_all"]))
        for name, value in audit[f"{kind}_interaction_energy_datasets"].items():
            worst["datasets"] = max(worst["datasets"], abs(summary["datasets"][name][key] - value))
        for name, value in audit[f"{kind}_interaction_energy_subsets"].items():
            worst["subsets"] = max(worst["subsets"], abs(summary["subsets"][name][key] - value))
    largest = np.argsort(-np.abs(deviation))[:5]
    return {
        "n_ok": int(ok.sum()),
        "n_audit": len(theirs),
        "same_complexes": same,
        "n_skipped": int(ours["status_mace"].str.startswith("skipped").sum()),
        "n_skipped_audit": audit["n_skipped_unallowed_elements"],
        "max_abs_reference_difference": float(np.abs(reference).max()),
        "interaction_energy_difference": _stats(deviation),
        "largest_differences": {common[i]: round(float(deviation[i]), 5) for i in largest},
        "MAE": [summary["interaction_energy"]["MAE"], audit["mae_interaction_energy_all"]],
        "RMSE": [summary["interaction_energy"]["RMSE"], audit["rmse_interaction_energy_all"]],
        "max_abs_summary_difference": worst,
    }


def conformers(ours: pd.DataFrame, audit: dict) -> dict:
    """Molecule-by-molecule differences (kcal/mol)."""
    theirs = {m["molecule_name"]: m for m in audit["molecules"]}
    ours = ours.set_index("molecule")
    ok = ours["status_mace"] == "ok"
    common = sorted(set(ours.index[ok]) & set(theirs))
    energies, metrics = [], {"mae": [], "rmse": [], "spearman": []}
    for molecule in common:
        mine = np.array(json.loads(ours.loc[molecule, "energies_mace"]))
        energies.extend(mine - np.array(theirs[molecule]["predicted_energy_profile"]))
        for metric, key in (("mae", "mae"), ("rmse", "rmse"), ("spearman", "spearman_correlation")):
            metrics[metric].append(ours.loc[molecule, f"{metric}_mace"] - theirs[molecule][key])
    return {
        "n_ok": int(ok.sum()),
        "n_audit": len(theirs),
        "same_molecules": set(ours.index[ok]) == set(theirs),
        "relative_energy_difference": _stats(np.array(energies)),
        **{f"{metric}_difference": _stats(np.array(values)) for metric, values in metrics.items()},
        "mean_mae": [float(ours.loc[ok, "mae_mace"].mean()), audit["avg_mae"]],
        "mean_rmse": [float(ours.loc[ok, "rmse_mace"].mean()), audit["avg_rmse"]],
    }


def _stats(values: np.ndarray) -> dict:
    return {"max_abs": float(np.abs(values).max()), "mean_abs": float(np.abs(values).mean()), "n": int(values.size)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("benchmark", choices=["noncovalent", "conformers"])
    parser.add_argument("ours")
    parser.add_argument("result")
    args = parser.parse_args()
    ours = pd.read_csv(args.ours)
    with open(args.result) as f:
        audit = json.load(f)
    report = noncovalent(ours, audit) if args.benchmark == "noncovalent" else conformers(ours, audit)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
