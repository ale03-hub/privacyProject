# Experiments: every protection configuration, evaluated on risk and utility
import os
import sys
import time
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.data import load_clean, sanitize, QI, QI_EXT, NUMERIC_QIS, TARGET, SEED
from src.hierarchies import generalize
from src.protection import (global_recoding, top_bottom_coding, top_bottom_series, local_suppression,
                            pram, additive_noise, rank_swapping, microaggregation)
from src.risk import (make_release_and_external, sample_uniqueness, record_linkage,
                      record_linkage_distance, attribute_disclosure_rate)
from src.utility import split_indices, evaluate_classifier, information_loss_total

os.makedirs("results", exist_ok=True)

# maximum tolerable risk (dist_linkage_rate) used to pick the compromise
MAX_RISK = {"QI": 0.01, "QI_EXT": 0.05}

SCENARIOS = {"QI": QI, "QI_EXT": QI_EXT}
BOTH = ("QI", "QI_EXT")
PERTURBATIVE_STEPS = {"pram", "noise", "rank_swap", "microaggregation"}
AGE_LEVEL = {5: 1, 10: 2}       # hierarchy level of the AGE bands of that width

# --- grid ---------------------------------------------------------------
# Each configuration is applied once to the release and evaluated in the
# `scenarios` listed (default both). `steps` are applied in order; only the
# deterministic steps (recoding, top/bottom coding, generalization) are known
# to the attacker, who applies them to the external dataset.
BOTH_COLS = ["AGE", "LIMIT_BAL"]
RECODE_E_M = [("global_recoding", {"col": "AGE", "width": 10}),
              ("generalization", {"levels": {"EDUCATION": 1, "MARRIAGE": 1}})]

GRID = [
    {"id": "original", "family": "reference", "technique": "original", "steps": []},
    {"id": "no_data", "family": "reference", "technique": "no_data", "steps": [("no_data", {})]},

    # non-perturbative
    *[{"id": f"recode_age_{w}", "family": "non-perturbative", "technique": "global_recoding",
       "steps": [("global_recoding", {"col": "AGE", "width": w})]} for w in (5, 10)],
    {"id": "recode_age_3bands", "family": "non-perturbative", "technique": "global_recoding",
     "steps": [("global_recoding", {"col": "AGE", "level": 3})]},
    {"id": "topcode_a", "family": "non-perturbative", "technique": "top_bottom_coding",
     "steps": [("top_bottom", {"bounds": {"AGE": (None, 60), "LIMIT_BAL": (None, 500_000)}})]},
    {"id": "topcode_b", "family": "non-perturbative", "technique": "top_bottom_coding",
     "steps": [("top_bottom", {"bounds": {"AGE": (25, 55), "LIMIT_BAL": (None, 300_000)}})]},
    *[{"id": f"generalize_{name}", "family": "non-perturbative", "technique": "generalization",
       "steps": [("generalization", {"levels": levels})]}
      for name, levels in [("EDU", {"EDUCATION": 1}), ("MAR", {"MARRIAGE": 1}),
                           ("EDU_MAR", {"EDUCATION": 1, "MARRIAGE": 1})]],
    *[{"id": f"local_supp_t{t}_{sc}", "family": "non-perturbative", "technique": "local_suppression",
       "scenarios": (sc,), "steps": [("local_suppression", {"threshold": t, "qis": sc})]}
      for t in (2, 5, 10) for sc in BOTH],

    # perturbative
    *[{"id": f"pram_{p}", "family": "perturbative", "technique": "pram",
       "steps": [("pram", {"cols": ["EDUCATION", "MARRIAGE"], "p_keep": p})]} for p in (0.9, 0.7, 0.5)],
    *[{"id": f"noise_{s}", "family": "perturbative", "technique": "additive_noise",
       "steps": [("noise", {"cols": BOTH_COLS, "sigma": s})]} for s in (0.05, 0.1, 0.2)],
    *[{"id": f"rankswap_{p}", "family": "perturbative", "technique": "rank_swapping",
       "steps": [("rank_swap", {"cols": BOTH_COLS, "p": p})]} for p in (5, 10, 20)],
    *[{"id": f"microagg_{k}", "family": "perturbative", "technique": "microaggregation",
       "steps": [("microaggregation", {"cols": BOTH_COLS, "size": k})]} for k in (3, 5, 10)],

    # combinations
    *[{"id": f"combo_recode_gen_locsup_{sc}", "family": "combination",
       "technique": "recoding+generalization+local_suppression", "scenarios": (sc,),
       "steps": RECODE_E_M + [("local_suppression", {"threshold": 5, "qis": sc})]} for sc in BOTH],
    {"id": "combo_noise_pram", "family": "combination", "technique": "noise+pram",
     "steps": [("noise", {"cols": BOTH_COLS, "sigma": 0.1}),
               ("pram", {"cols": ["EDUCATION", "MARRIAGE"], "p_keep": 0.9})]},
]


# --- running a configuration --------------------------------------------

def run_steps(df, steps):
    # returns the protected copy, what the attacker knows (scheme for align)
    # and the % of QI cells suppressed
    scheme, pct = {}, 0.0
    for name, p in steps:
        if name == "no_data":
            df = df.copy()
            df[QI_EXT] = np.nan
        elif name == "global_recoding":
            if "width" in p:
                df = global_recoding(df, p["col"], p["width"])
                scheme[p["col"]] = AGE_LEVEL[p["width"]]
            else:
                df = generalize(df, {p["col"]: p["level"]})
                scheme[p["col"]] = p["level"]
        elif name == "top_bottom":
            for col, (lower, upper) in p["bounds"].items():
                df = top_bottom_coding(df, col, lower, upper)
                scheme[col] = partial(top_bottom_series, lower=lower, upper=upper)
        elif name == "generalization":
            df = generalize(df, p["levels"])
            scheme.update(p["levels"])
        elif name == "local_suppression":
            df, pct = local_suppression(df, SCENARIOS[p["qis"]], p["threshold"])
        elif name == "pram":
            for i, col in enumerate(p["cols"]):
                df = pram(df, col, p["p_keep"], seed=SEED + i)
        elif name == "noise":
            for i, col in enumerate(p["cols"]):
                df = additive_noise(df, col, p["sigma"], seed=SEED + i)
        elif name == "rank_swap":
            for i, col in enumerate(p["cols"]):
                df = rank_swapping(df, col, p["p"], seed=SEED + i)
        elif name == "microaggregation":
            df = microaggregation(df, p["cols"], p["size"])
        else:
            raise ValueError(name)
    return df, scheme, pct


def fmt(v):
    if isinstance(v, dict):
        return ",".join(f"{k}{fmt(x)}" if isinstance(x, tuple) else f"{k}={fmt(x)}" for k, x in v.items())
    if isinstance(v, tuple):    # (lower, upper) bounds
        return "/".join(t for t in ("<%s" % v[0] if v[0] is not None else "", ">%s" % v[1] if v[1] is not None else "") if t)
    if isinstance(v, list):
        return "+".join(map(str, v))
    return str(v)


def describe(steps):
    texts = [", ".join(f"{k}={fmt(v)}" for k, v in p.items()) for _, p in steps]
    if len(steps) > 1:
        texts = [f"{name}({t})" for (name, _), t in zip(steps, texts)]
    return " + ".join(texts) or "-"


# --- summary ------------------------------------------------------------

def md_table(df):
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def compact_table(res):
    # one row per configuration, risk of both scenarios side by side
    out = []
    for cid, g in res.groupby("config_id", sort=False):
        row = {"config": cid}
        for sc, label in [("QI", "QI"), ("QI_EXT", "EXT")]:
            r = g[g["scenario"] == sc]
            for col, short in [("uniqueness", "uniq"), ("det_linkage_rate", "det"), ("dist_linkage_rate", "dist")]:
                row[f"{short} {label}"] = f"{r[col].iloc[0]:.4f}" if len(r) else "-"
        row["utility"] = f"{g['utility'].iloc[0]:.3f}"
        row["auc_qi"] = f"{g['auc_qi'].iloc[0]:.4f}"
        out.append(row)
    return pd.DataFrame(out)


def compromise(res, scenario):
    # highest utility among the configurations with dist_linkage_rate below the threshold
    ok = res[(res["scenario"] == scenario) & (res["dist_linkage_rate"] < MAX_RISK[scenario])]
    if ok.empty:
        return None
    return ok.sort_values(["utility", "dist_linkage_rate"], ascending=[False, True]).iloc[0]


def write_summary(res, total_time):
    lines = ["# Experiments summary", "",
             f"{len(res)} rows ({res['config_id'].nunique()} configurations), total time {total_time:.0f}s.", "",
             "Risk is measured on the release (80% of the population) against an external dataset "
             "(50% of the population, LIMIT_BAL in 50k bands). Utility = 1 - il_total.", ""]

    orig = res[res["config_id"] == "original"].set_index("scenario")
    lines += ["## Risk of the original data (identifiers removed)", "",
              md_table(orig[["uniqueness", "det_linkage_rate", "dist_linkage_rate", "attr_disclosure_rate",
                             "auc_full", "auc_qi"]].round(4).reset_index()), ""]

    lines += ["## Chosen compromise", "",
              "Highest utility among the configurations with `dist_linkage_rate` below the threshold.", ""]
    for sc in BOTH:
        best = compromise(res, sc)
        lines.append(f"### {sc} (dist_linkage_rate < {MAX_RISK[sc]})")
        if best is None:
            lines += ["", "No configuration is below the threshold.", ""]
            continue
        o = orig.loc[sc]
        lines += ["", f"**{best['config_id']}** ({best['technique']}: {best['params']})", "",
                  md_table(pd.DataFrame({
                      "": ["original", best["config_id"], "change"],
                      "uniqueness": [o["uniqueness"], best["uniqueness"], best["uniqueness"] - o["uniqueness"]],
                      "det_linkage": [o["det_linkage_rate"], best["det_linkage_rate"], best["det_linkage_rate"] - o["det_linkage_rate"]],
                      "dist_linkage": [o["dist_linkage_rate"], best["dist_linkage_rate"], best["dist_linkage_rate"] - o["dist_linkage_rate"]],
                      "utility": [1.0, best["utility"], best["utility"] - 1.0],
                      "auc_full": [o["auc_full"], best["auc_full"], best["auc_full"] - o["auc_full"]],
                      "auc_qi": [o["auc_qi"], best["auc_qi"], best["auc_qi"] - o["auc_qi"]],
                  }).set_index("").round(4).reset_index()), ""]

    lines += ["## All configurations", "",
              "uniq / det / dist = uniqueness and deterministic / distance-based linkage rate; "
              "QI and EXT are the two attack scenarios; `-` = not evaluated in that scenario.", "",
              md_table(compact_table(res)), ""]
    text = "\n".join(lines)
    with open("results/summary.md", "w") as f:
        f.write(text)
    return text


def main():
    start = time.perf_counter()
    population = sanitize(load_clean(with_identifiers=True))
    released, external = make_release_and_external(population)
    stds = population[NUMERIC_QIS].std()
    train_idx, test_idx = split_indices(released)
    print(f"released {len(released)}, external {len(external)}, {len(GRID)} configurations", flush=True)

    rows = []
    for n, cfg in enumerate(GRID, start=1):
        print(f"[{n}/{len(GRID)}] {cfg['id']}", flush=True)
        t0 = time.perf_counter()
        protected, scheme, pct = run_steps(released, cfg["steps"])
        protection_time = time.perf_counter() - t0

        # utility does not depend on the attack scenario: computed once
        loss = information_loss_total(released, protected)
        utility = {
            **{f"il_{c}": loss[c] for c in QI_EXT},
            "il_total": loss["il_total"],
            "utility": 1 - loss["il_total"],
            "auc_full": evaluate_classifier(protected, train_idx, test_idx)["auc"],
            "auc_qi": evaluate_classifier(protected, train_idx, test_idx, features=QI_EXT)["auc"],
        }
        # attribute disclosure is not meaningful with a perturbative part
        perturbative = any(name in PERTURBATIVE_STEPS for name, _ in cfg["steps"])
        for sc in cfg.get("scenarios", BOTH):
            qis = SCENARIOS[sc]
            row = {
                "config_id": cfg["id"], "family": cfg["family"], "technique": cfg["technique"],
                "params": describe(cfg["steps"]), "scenario": sc,
                "uniqueness": sample_uniqueness(protected, qis),
                "det_linkage_rate": record_linkage(protected, external, qis, scheme),
                "dist_linkage_rate": record_linkage_distance(protected, external, qis, stds, scheme=scheme),
                "attr_disclosure_rate": np.nan if perturbative else
                    attribute_disclosure_rate(protected, qis, TARGET, external, scheme),
                **utility,
                "suppressed_cells_pct": pct,
                "time_s": protection_time,
            }
            rows.append(row)
            print(f"    {sc:6s} uniq {row['uniqueness']:.4f}  det {row['det_linkage_rate']:.4f}  "
                  f"dist {row['dist_linkage_rate']:.4f}  utility {row['utility']:.3f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv("results/experiments.csv", index=False)
    total_time = time.perf_counter() - start
    text = write_summary(res, total_time)
    print(f"\n{'=' * 70}\n{text}\ntotal time: {total_time:.0f}s")


if __name__ == "__main__":
    main()
