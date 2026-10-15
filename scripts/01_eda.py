# Exploratory data analysis: attribute classification and QI distributions
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data import load_clean, QI, QI_EXT, TARGET
from src.risk import class_sizes

sns.set_theme(style="whitegrid", context="talk")
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)


def classify_attributes(df):
    rows = []

    def add(cols, role, why):
        for c in cols:
            rows.append({"column": c, "role": role, "motivation": why})

    add(["ID", "customer_code", "phone"], "identifier",
        "Directly identifies a customer (row id, synthetic customer code / phone simulating a real release)")
    add(["SEX"], "QI", "Publicly known / observable; not unique alone but identifying in combination")
    add(["EDUCATION"], "QI", "Available in external sources (e.g. registries, social profiles)")
    add(["MARRIAGE"], "QI", "Available in external sources (e.g. registries)")
    add(["AGE"], "QI", "Publicly known; very discriminating when combined with the other QIs")
    add([TARGET], "sensitive", "Main sensitive attribute: credit default (binary, ~22% positive)")
    add(["LIMIT_BAL"], "sensitive",
        "Financial information; an attacker who knows it approximately uses it as an extra QI (extended scenario)")
    add(["PAY_0"] + [f"PAY_{i}" for i in range(2, 7)], "sensitive", "Repayment history of the customer")
    add([f"BILL_AMT{i}" for i in range(1, 7)], "sensitive", "Monthly bill amounts: private financial data")
    add([f"PAY_AMT{i}" for i in range(1, 7)], "sensitive", "Monthly payments: private financial data")

    attr = pd.DataFrame(rows)
    assert set(attr["column"]) == set(df.columns), set(df.columns) ^ set(attr["column"])
    return attr


def plot_qi_distributions(df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, col in zip(axes.flat[:3], ["SEX", "EDUCATION", "MARRIAGE"]):
        sns.countplot(x=df[col], ax=ax, color="#4C72B0")
        ax.set_title(col)
        ax.set_xlabel("")
    sns.histplot(df["AGE"], bins=range(20, 82, 2), ax=axes[1, 0], color="#4C72B0")
    axes[1, 0].set_title("AGE")
    sns.histplot(df["LIMIT_BAL"], bins=40, ax=axes[1, 1], color="#4C72B0")
    axes[1, 1].set_title("LIMIT_BAL (extended QI)")
    sns.countplot(x=df[TARGET], ax=axes[1, 2], color="#C44E52")
    axes[1, 2].set_title(f"default (base rate {df[TARGET].mean():.1%})")
    axes[1, 2].set_xlabel("")
    plt.tight_layout()
    plt.savefig("figures/qi_distributions.png", dpi=300)
    plt.close()


def class_size_shares(df):
    bins = [0, 1, 4, 9, np.inf]
    labels = ["1 (unique)", "2-4", "5-9", ">=10"]
    shares = {}
    for name, qis in [("QI", QI), ("QI + LIMIT_BAL", QI_EXT)]:
        sizes = class_sizes(df, qis)
        shares[name] = pd.cut(sizes, bins=bins, labels=labels).value_counts(normalize=True).reindex(labels)
    shares = pd.DataFrame(shares)
    shares.index.name = "class size"
    return shares


def plot_class_sizes(shares):
    ax = shares.T.plot(kind="bar", stacked=True, figsize=(9, 6), colormap="viridis_r")
    ax.set_ylabel("share of records")
    ax.set_xlabel("")
    ax.set_title("Size of the equivalence class of each record")
    ax.legend(title="class size", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("figures/equivalence_class_sizes_original.png", dpi=300)
    plt.close()


def main():
    df = load_clean(with_identifiers=True)
    print(f"shape: {df.shape}")
    print(f"missing values: {int(df.isna().sum().sum())}")
    print(f"default base rate: {df[TARGET].mean():.3f}")
    print("\nQI value counts:")
    for col in ["SEX", "EDUCATION", "MARRIAGE"]:
        print(df[col].value_counts().sort_index().to_string(), "\n")
    print(df[["AGE", "LIMIT_BAL"]].describe().round(1).to_string())

    attr = classify_attributes(df)
    attr.to_csv("results/attribute_classification.csv", index=False)
    print("\nattribute roles:")
    for role, cols in attr.groupby("role")["column"]:
        print(f"  {role}: {', '.join(cols)}")

    plot_qi_distributions(df)

    shares = class_size_shares(df)
    plot_class_sizes(shares)
    print("\nshare of records by equivalence class size (identifiers removed):")
    print(shares.round(4).to_string())


if __name__ == "__main__":
    main()
