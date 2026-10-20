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

from matplotlib.ticker import FuncFormatter, PercentFormatter

from src.data import (load_clean, QI, QI_EXT, TARGET,
                      SEX_LABELS, EDUCATION_LABELS, MARRIAGE_LABELS)
from src.risk import combination_sizes

CATEGORY_LABELS = {"SEX": SEX_LABELS, "EDUCATION": EDUCATION_LABELS, "MARRIAGE": MARRIAGE_LABELS}

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


def bar_with_shares(ax, values, order, color):
    # count plot with readable category names and the share of records on each bar
    sns.countplot(x=values, order=order, ax=ax, color=color)
    ax.bar_label(ax.containers[0], labels=[f"{(values == v).mean():.1%}" for v in order], fontsize=12)
    ax.margins(y=0.12)
    ax.set_xlabel("")


def plot_qi_distributions(df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, col in zip(axes.flat[:3], ["SEX", "EDUCATION", "MARRIAGE"]):
        labels = CATEGORY_LABELS[col]
        bar_with_shares(ax, df[col].map(labels), list(labels.values()), "#4C72B0")
        ax.set_title(col)
        if col == "EDUCATION":
            ax.tick_params(axis="x", labelsize=13)
    sns.histplot(df["AGE"], bins=range(20, 82, 2), ax=axes[1, 0], color="#4C72B0")
    axes[1, 0].set_title("AGE")
    axes[1, 0].set_xlabel("years")
    sns.histplot(df["LIMIT_BAL"], bins=40, ax=axes[1, 1], color="#4C72B0")
    axes[1, 1].set_title("LIMIT_BAL (extended QI)")
    axes[1, 1].set_xlabel("credit limit (NT$)")
    axes[1, 1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    bar_with_shares(axes[1, 2], df[TARGET].map({0: "no default", 1: "default"}),
                    ["no default", "default"], "#C44E52")
    axes[1, 2].set_title(f"default (base rate {df[TARGET].mean():.1%})")
    plt.tight_layout()
    plt.savefig("figures/qi_distributions.png", bbox_inches='tight', dpi=300)
    plt.close()


def combination_size_shares(df):
    bins = [0, 1, 4, 9, np.inf]
    labels = ["1 (unique)", "2-4", "5-9", ">=10"]
    shares = {}
    for name, qis in [("QI", QI), ("QI + LIMIT_BAL", QI_EXT)]:
        sizes = combination_sizes(df, qis)
        shares[name] = pd.cut(sizes, bins=bins, labels=labels).value_counts(normalize=True).reindex(labels)
    shares = pd.DataFrame(shares)
    shares.index.name = "combination size"
    return shares


def plot_combination_sizes(shares, n):
    ax = shares.T.plot(kind="bar", stacked=True, figsize=(10, 7), colormap="viridis_r", width=0.6)
    # write the share inside each segment that is tall enough to hold it
    for container in ax.containers:
        ax.bar_label(container, labels=[f"{v:.1%}" if v >= 0.05 else "" for v in container.datavalues],
                     label_type="center", fontsize=13, color="white", fontweight="bold")
    # the unique share is often too thin to see: write it above each bar
    for i, unique in enumerate(shares.loc["1 (unique)"]):
        ax.annotate(f"unique: {unique:.2%}", (i, 1.0), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("share of records")
    ax.set_xlabel("")
    ax.set_title(f"How many records share each record's QI combination\n(all {n:,} customers, identifiers removed)",
                 fontsize=16)
    ax.legend(title="records sharing\nthe combination", bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=13, title_fontsize=13)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("figures/qi_combination_sizes_original.png", bbox_inches='tight', dpi=300)
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

    shares = combination_size_shares(df)
    plot_combination_sizes(shares, len(df))
    print("\nshare of records by QI combination size (identifiers removed):")
    print(shares.round(4).to_string())


if __name__ == "__main__":
    main()
