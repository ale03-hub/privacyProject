# Baseline: risk and utility of the original data once the identifiers are removed
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.data import load_clean, sanitize, QI, QI_EXT, IDENTIFIERS, NUMERIC_QIS, TARGET
from src.risk import (sample_uniqueness, rare_combination_rate, make_release_and_external,
                      record_linkage, record_linkage_distance)
from src.utility import split_indices, evaluate_classifier

os.makedirs("results", exist_ok=True)


def main():
    # the 30,000 customers are the population; the sanitized release is a sample of it
    population = sanitize(load_clean(with_identifiers=True))
    released, external = make_release_and_external(population)
    print("identifiers left:", [c for c in IDENTIFIERS if c in released.columns])
    print(f"population {len(population)}, released {len(released)}, external {len(external)} "
          f"({external.index.isin(released.index).mean():.1%} of the external people are in the release)")

    stds = population[NUMERIC_QIS].std()

    # same classifier results for both scenarios: it does not depend on the attack
    train_idx, test_idx = split_indices(released)
    util = evaluate_classifier(released, train_idx, test_idx)
    trivial = 1 - released.loc[test_idx, TARGET].mean()
    print(f"trivial classifier (always 'no default') accuracy: {trivial:.4f}")

    rows = []
    for name, qis in [("QI", QI), ("QI_EXT", QI_EXT)]:
        row = {
            "scenario": name,
            "qis": "+".join(qis),
            "uniqueness": sample_uniqueness(released, qis),
            "rare_combinations_lt5": rare_combination_rate(released, qis, 5),
        }
        row["det_linkage_rate"] = record_linkage(released, external, qis)
        row["dist_linkage_rate"] = record_linkage_distance(released, external, qis, stds)
        row.update(util)
        rows.append(row)

    baseline = pd.DataFrame(rows)
    baseline.to_csv("results/baseline.csv", index=False)
    print()
    print(baseline.round(4).set_index("scenario").drop(columns="qis").T.to_string())
    print("\nWithout identifiers some records are still unique on the QIs, "
          "and knowing LIMIT_BAL (even approximately) makes many customers re-identifiable by linkage.")


if __name__ == "__main__":
    main()
