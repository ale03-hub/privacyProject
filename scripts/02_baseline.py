# Baseline: risk and utility of the original data once the identifiers are removed
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.data import load_clean, sanitize, QI, QI_EXT, IDENTIFIERS, SEED
from src.risk import uniqueness, small_class_rate, k_anonymity, make_external, linkage_rate
from src.utility import split_indices, evaluate_classifier

os.makedirs("results", exist_ok=True)


def main():
    full = load_clean(with_identifiers=True)
    released = sanitize(full)
    print("identifiers left:", [c for c in IDENTIFIERS if c in released.columns])

    # attacker's dataset: a sample of the same people with their QIs
    external = make_external(released, frac=0.5, seed=SEED)
    print(f"{len(released)} released records, {len(external)} in the external dataset")

    train_idx, test_idx = split_indices(released)

    rows = []
    for name, qis in [("QI", QI), ("QI_EXT", QI_EXT)]:
        util = evaluate_classifier(released, qis, train_idx, test_idx)
        rows.append({
            "scenario": name,
            "qis": "+".join(qis),
            "uniqueness": uniqueness(released, qis),
            "records_in_classes_lt5": small_class_rate(released, qis, 5),
            "k": k_anonymity(released, qis),
            "linkage_rate": linkage_rate(released, external, qis),
            "auc": util["auc"],
            "accuracy": util["accuracy"],
        })

    baseline = pd.DataFrame(rows)
    baseline.to_csv("results/baseline.csv", index=False)
    print()
    print(baseline.round(4).to_string(index=False))
    print("\nWithout identifiers k = 1 in both scenarios: some records are unique, "
          "and knowing LIMIT_BAL makes many customers re-identifiable by linkage.")


if __name__ == "__main__":
    main()
