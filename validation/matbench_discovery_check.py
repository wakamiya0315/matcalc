"""Checks of the Discovery and Kappa benchmarks against the files Matbench Discovery publishes on Figshare.

Subcommands (file names as on Figshare, article 22715158):

  discovery RUN.csv PREDICTIONS.csv.gz
      Formation energies of a Discovery run vs published predictions of the same model
      (e.g. mace-mp-0-2023-12-11-discovery.csv.gz, rounded to 1e-4 eV/atom), crystal by crystal.
  geometry RUN.csv ANALYSIS_1e-5.csv.gz ANALYSIS_1e-2.csv.gz
      RMSD and symmetry of a Discovery run vs the published geometry analysis of the same model
      (e.g. mace-mp-0-2023-12-11-geo-opt-symprec=1e-5-moyo=0.4.2.csv.gz), crystal by crystal, and the
      geometry metrics on the crystals of that file.
  mp2020 COMPUTED_ENTRIES.jsonl.gz SUMMARY.csv.gz [N]
      How many of N random WBM DFT entries get a different MP2020 correction when it is recomputed with the
      installed pymatgen (2022-10-19-wbm-computed-structure-entries.jsonl.gz, 2023-12-13-wbm-summary.csv.gz).
  kappa-metric PREDICTIONS.json.gz REFERENCE.json.gz
      κ_SRME, κ_SRE, κ_SRD of published predictions (e.g. mace-mp-0-2024-11-09-phonons-kappa-103.json.gz)
      against the published DFT reference (2024-11-09-kappas-phononDB-PBE-noNAC.json.gz), computed with
      matcalc's metric functions: they should give the leaderboard's values.
  kappa RUN.csv PREDICTIONS.json.gz REFERENCE.json.gz
      A Kappa run vs published predictions of the same model, compound by compound.
  kappa-reference DATASET.json.gz REFERENCE.json.gz
      The packaged reference (recomputed with the phono3py matcalc uses) vs the published one.
"""

from __future__ import annotations

import gzip
import json
import random
import re
import sys
import warnings

import numpy as np
import pandas as pd

from matcalc.properties.stability import geometry_metrics
from matcalc.properties.thermal_conductivity import (
    KAPPA_ERROR_MAX,
    Conductivity,
    symmetric_relative_difference,
    symmetric_relative_mean_error,
)


def discovery(run: str, predictions: str) -> None:
    ours = pd.read_csv(run, index_col="material_id")
    published = pd.read_csv(predictions, index_col="material_id").e_form_per_atom.reindex(ours.index)
    diff = (ours.e_form_per_atom_mace - published).abs()
    print(f"{len(ours)} crystals, {int(diff.notna().sum())} compared")
    for tolerance in (1.5e-4, 1e-3, 1e-2, 0.1):
        print(f"  |E_form - published| > {tolerance:g} eV/atom: {int((diff > tolerance).sum())}")
    worst = diff.sort_values(ascending=False).head(10).index
    print(pd.DataFrame({"formula": ours.formula[worst], "ours": ours.e_form_per_atom_mace[worst],
                        "published": published[worst], "status": ours.status_mace[worst]}).to_string())


def geometry(run: str, *analyses: str) -> None:
    ours = pd.read_csv(run, index_col="material_id")
    for analysis in analyses:
        published = pd.read_csv(analysis, index_col="material_id")
        tolerance = f"{published.symprec.iloc[0]:.0e}".replace("e-0", "e-")
        o = ours.reindex(published.index)
        rmsd, rmsd_published = o.rmsd_mace, published.structure_rmsd_vs_dft
        matched = rmsd.notna() & rmsd_published.notna()
        diff = (rmsd - rmsd_published).abs()[matched]
        print(f"symprec {tolerance}: {len(published)} crystals ({len(ours.index.difference(published.index))} of the run are not in "
              f"the file); RMSD within 1e-4 for {np.mean(diff < 1e-4):.4%} of the {int(matched.sum())} matched ones, "
              f"unmatched in both {int((rmsd.isna() & rmsd_published.isna()).sum())}, in one only {int((rmsd.isna() != rmsd_published.isna()).sum())}; "
              f"same space group {np.mean(o[f'spg_num_{tolerance}_mace'] == published.spg_num):.4%}")
        metrics = geometry_metrics(rmsd, o[f"spg_num_{tolerance}_mace"], o[f"spg_num_{tolerance}_DFT"],
                                   o[f"n_sym_ops_{tolerance}_mace"], o[f"n_sym_ops_{tolerance}_DFT"])
        print("  metrics on these crystals:", {k: round(v, 4) for k, v in metrics.items()})


def mp2020(entries: str, summary: str, n: int = 4000) -> None:
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
    from pymatgen.entries.computed_entries import ComputedStructureEntry

    published = pd.read_csv(summary, index_col="material_id").e_correction_per_atom_mp2020
    wanted = set(random.Random(0).sample(list(published.index), n))
    compatibility = MaterialsProject2020Compatibility()
    changed = []
    with gzip.open(entries, "rt") as f:
        for line in f:
            match = re.search(r'"material_id"\s*:\s*"([^"]+)"', line[:200])
            if match is None or match.group(1) not in wanted:
                continue
            entry = ComputedStructureEntry.from_dict(json.loads(line)["computed_structure_entry"])
            entry.data = {}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                processed = compatibility.process_entry(entry, clean=True)
            per_atom = processed.correction / processed.composition.num_atoms if processed else np.nan
            if not abs(per_atom - published[match.group(1)]) <= 1e-4:
                changed.append((match.group(1), entry.composition.reduced_formula))
    print(f"{len(changed)} of {n} corrections change ({len(changed) / n:.2%}); e.g. {changed[:8]}")


def _published_kappa(row: pd.Series) -> float:
    value = row.kappa_tot_avg
    return float(np.ravel(value)[0]) if value is not None and np.size(value) else float("nan")


def _published_metrics(predictions: str, reference: str) -> pd.DataFrame:
    """SRME, SRE and SRD of every published prediction, with matcalc's functions (failures count as 2)."""
    dft = {e["material_id"]: e for e in json.load(gzip.open(reference, "rt"))}
    rows = {}
    for material_id, row in pd.read_json(predictions).set_index("material_id").iterrows():
        ref_kappa = float(np.ravel(dft[material_id]["kappa_tot_avg"])[0])
        kappa = _published_kappa(row)
        if not np.isfinite(kappa):
            rows[material_id] = (np.nan, KAPPA_ERROR_MAX, KAPPA_ERROR_MAX, -KAPPA_ERROR_MAX)
            continue
        mode_kappa = np.asarray(row.mode_kappa_tot_avg, dtype=float)
        predicted = Conductivity(
            kappa=np.array([kappa] * 3 + [0.0] * 3),
            kappa_particle=np.full(6, np.nan),
            kappa_coherence=np.full(6, np.nan),
            mode_kappa=mode_kappa[0] if mode_kappa.ndim == 3 else mode_kappa,
            weights=np.asarray(row.mode_weights if row.mode_weights is not None else dft[material_id]["weights"], dtype=float),
        )
        srd = symmetric_relative_difference(kappa, ref_kappa)
        srme = symmetric_relative_mean_error(predicted, np.asarray(dft[material_id]["mode_kappa_tot_avg"])[0], ref_kappa)
        rows[material_id] = (kappa, srme, abs(srd), srd)
    return pd.DataFrame.from_dict(rows, orient="index", columns=["kappa", "srme", "sre", "srd"])


def kappa_metric(predictions: str, reference: str) -> None:
    t = _published_metrics(predictions, reference)
    print(f"κ_SRME {t.srme.mean():.4f}, κ_SRE {t.sre.mean():.4f}, κ_SRD {t.srd.mean():.4f}, failures {int(t.kappa.isna().sum())}")


def kappa(run: str, predictions: str, reference: str) -> None:
    ours = pd.read_csv(run, index_col="mp_id")
    published = _published_metrics(predictions, reference).reindex(ours.index)
    ok = ours.status_mace == "ok"
    print(f"{int(ok.sum())} of {len(ours)} ok; others: {ours.status_mace[~ok].value_counts().to_dict()}")
    both = ok & published.kappa.notna()
    ratio = ours.kappa_mace[both] / published.kappa[both]
    print(f"κ / published κ ({int(both.sum())} compounds): median {ratio.median():.4f}, within 1 % "
          f"{np.mean(np.abs(ratio - 1) < 0.01):.0%}, within 5 % {np.mean(np.abs(ratio - 1) < 0.05):.0%}")
    srme = ours.srme_mace.fillna(KAPPA_ERROR_MAX)
    print(f"κ_SRME {srme.mean():.4f}, κ_SRE {ours.sre_mace.fillna(KAPPA_ERROR_MAX).mean():.4f}, "
          f"κ_SRD {ours.srd_mace.fillna(-KAPPA_ERROR_MAX).mean():.4f}")
    common = published.kappa.notna()
    print(f"on the {int(common.sum())} compounds of the published run: κ_SRME {srme[common].mean():.4f} "
          f"(published {published.srme[common].mean():.4f})")
    worst = (ratio - 1).abs().sort_values(ascending=False).head(8).index
    print(pd.DataFrame({"formula": ours.formula[worst], "ours": ours.kappa_mace[worst], "published": published.kappa[worst],
                        "srme": srme[worst], "srme_published": published.srme[worst]}).round(4).to_string())


def kappa_reference(dataset: str, reference: str) -> None:
    ours = json.load(gzip.open(dataset, "rt"))["entries"]
    dft = {e["material_id"]: e for e in json.load(gzip.open(reference, "rt"))}
    rows = []
    for entry in ours:
        ref = dft[entry["mp_id"]]
        ref_kappa = float(np.ravel(ref["kappa_tot_avg"])[0])
        mode = np.asarray(entry["mode_kappa"])
        error = np.abs(mode - np.asarray(ref["mode_kappa_tot_avg"])[0]).sum() / np.sum(entry["weights"])
        rows.append((entry["name"], entry["kappa"] / ref_kappa - 1, 2 * error / (entry["kappa"] + ref_kappa)))
    t = pd.DataFrame(rows, columns=["name", "deviation", "srme"]).set_index("name")
    print(f"{len(t)} compounds: κ / published κ - 1 median {t.deviation.median():+.4%}, mean |.| {t.deviation.abs().mean():.4%}, "
          f"{int((t.deviation.abs() < 0.01).sum())} within 1 %; SRME between the two: mean {t.srme.mean():.2e}, max {t.srme.max():.2e}")
    print(t.reindex(t.deviation.abs().sort_values(ascending=False).index).head(6).to_string())


if __name__ == "__main__":
    command, *arguments = sys.argv[1:]
    functions = {"discovery": discovery, "geometry": geometry, "mp2020": mp2020, "kappa-metric": kappa_metric, "kappa": kappa,
                 "kappa-reference": kappa_reference}
    if command == "mp2020" and len(arguments) == 3:
        arguments[2] = int(arguments[2])
    functions[command](*arguments)
