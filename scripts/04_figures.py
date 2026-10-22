# Final figures: privacy risk, utility and R-U confidentiality maps
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(ROOT, "results", "experiments.csv")
FIGURES_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Same illustrative thresholds used in scripts/03_experiments.py to identify
# one high-utility operating point. They are NOT universal privacy guarantees.
MAX_RISK = {"QI": 0.01, "QI_EXT": 0.05}
SCENARIO_TITLES = {
    "QI": "QI attacker",
    "QI_EXT": "Extended attacker (QI + LIMIT_BAL)",
}

# The figures are intended for the report, so labels are shorter than config_id.
PRETTY_LABELS = {
    "original": "Original",
    "no_data": "No QI information",
    "recode_age_5": "Age recoding (5y)",
    "recode_age_10": "Age recoding (10y)",
    "recode_age_3bands": "Age recoding (3 bands)",
    "topcode_a": "Top coding A",
    "topcode_b": "Top/bottom coding B",
    "generalize_EDU": "Generalize education",
    "generalize_MAR": "Generalize marriage",
    "generalize_EDU_MAR": "Generalize edu.+marriage",
    "local_supp_t2_QI": "Local suppression t=2 (QI)",
    "local_supp_t5_QI": "Local suppression t=5 (QI)",
    "local_supp_t10_QI": "Local suppression t=10 (QI)",
    "local_supp_t2_QI_EXT": "Local suppression t=2 (EXT)",
    "local_supp_t5_QI_EXT": "Local suppression t=5 (EXT)",
    "local_supp_t10_QI_EXT": "Local suppression t=10 (EXT)",
    "pram_0.9": "PRAM p=0.9",
    "pram_0.7": "PRAM p=0.7",
    "pram_0.5": "PRAM p=0.5",
    "noise_0.05": "Noise 0.05 sigma",
    "noise_0.1": "Noise 0.10 sigma",
    "noise_0.2": "Noise 0.20 sigma",
    "rankswap_5": "Rank swapping 5%",
    "rankswap_10": "Rank swapping 10%",
    "rankswap_20": "Rank swapping 20%",
    "microagg_3": "Microaggregation k=3",
    "microagg_5": "Microaggregation k=5",
    "microagg_10": "Microaggregation k=10",
    "combo_recode_gen_locsup_QI": "Rec.+gen.+loc.supp. (QI)",
    "combo_recode_gen_locsup_QI_EXT": "Rec.+gen.+loc.supp. (EXT)",
    "combo_noise_pram": "Noise + PRAM",
}

# label position in the R-U map: (offset in points, horizontal alignment);
# the default is just above-right of the point
LABEL_OFFSETS = {
    ("rankswap_10", "QI"): ((-8, 4), "right"),
    ("rankswap_20", "QI"): ((-8, 4), "right"),
    ("rankswap_10", "QI_EXT"): ((8, -10), "left"),
    ("rankswap_20", "QI_EXT"): ((-8, 4), "right"),
}

REQUIRED_COLUMNS = {
    "config_id", "family", "technique", "scenario", "uniqueness",
    "det_linkage_rate", "dist_linkage_rate", "utility", "auc_full", "auc_qi"
}

sns.set_theme(style="whitegrid", context="talk")


def load_results():
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(
            f"Missing {RESULTS_PATH}. Run `python main.py 03` first."
        )

    df = pd.read_csv(RESULTS_PATH)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"experiments.csv is missing columns: {sorted(missing)}")

    df = df.copy()
    df["label"] = df["config_id"].map(PRETTY_LABELS).fillna(df["config_id"])
    return df


def family_palette(df):
    families = [f for f in ["reference", "non-perturbative", "perturbative", "combination"]
                if f in set(df["family"])]
    colors = sns.color_palette("colorblind", n_colors=max(1, len(families)))
    return dict(zip(families, colors))


def add_family_legend(ax, palette, loc="best"):
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=9,
                   markerfacecolor=color, markeredgecolor="white", label=family)
        for family, color in palette.items()
    ]
    ax.legend(handles=handles, title="Family", loc=loc, fontsize=9, title_fontsize=10)


def plot_risk_by_technique(df):
    """Deterministic and distance-based record-linkage risk for every configuration."""
    palette = family_palette(df)
    fig, axes = plt.subplots(1, 2, figsize=(19, 13), constrained_layout=True)

    for ax, scenario in zip(axes, ["QI", "QI_EXT"]):
        d = df[df["scenario"] == scenario].copy()
        d = d.sort_values("dist_linkage_rate", ascending=True).reset_index(drop=True)
        y = np.arange(len(d))
        colors = [palette[f] for f in d["family"]]

        # the grey segment shows how much a distance-based attacker gains over exact matching
        ax.hlines(y, d["det_linkage_rate"], d["dist_linkage_rate"], color="0.8", linewidth=2, zorder=1)
        ax.scatter(d["det_linkage_rate"], y, s=60, facecolors="white", edgecolors=colors,
                   linewidth=1.5, zorder=3)
        ax.scatter(d["dist_linkage_rate"], y, s=75, c=colors,
                   edgecolor="white", linewidth=0.7, zorder=3)
        ax.axvline(MAX_RISK[scenario], linestyle="--", linewidth=1.5, color="0.35")

        original = d[d["config_id"] == "original"]
        if not original.empty:
            ax.axvline(original["dist_linkage_rate"].iloc[0], linestyle=":",
                       linewidth=1.3, color="0.55")

        ax.set_yticks(y)
        ax.set_yticklabels(d["label"], fontsize=9)
        ax.set_xlabel("Correct linkage rate")
        ax.set_title(SCENARIO_TITLES[scenario])
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlim(left=-0.002)
        ax.grid(axis="y", visible=False)

        # single legend in the lower right corner, which is empty
        # (rows are sorted by risk, so low-risk points sit on the left)
        handles = [
            plt.Line2D([0], [0], marker="o", linestyle="", markersize=12,
                       markerfacecolor=color, markeredgecolor="white", label=family)
            for family, color in palette.items()
        ] + [
            plt.Line2D([0], [0], marker="o", linestyle="", markersize=12, color="0.3",
                       label="distance-based linkage"),
            plt.Line2D([0], [0], marker="o", linestyle="", markersize=11, markerfacecolor="white",
                       markeredgecolor="0.3", markeredgewidth=1.8, label="deterministic linkage"),
            plt.Line2D([0], [0], linestyle="--", linewidth=2, color="0.35",
                       label=f"illustrative threshold = {MAX_RISK[scenario]:.0%}"),
            plt.Line2D([0], [0], linestyle=":", linewidth=2, color="0.55", label="original risk (distance)"),
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=14, framealpha=0.95)

    fig.suptitle("Privacy risk after each protection configuration\n(lower is better)", fontsize=20)
    out = os.path.join(FIGURES_DIR, "risk_by_technique.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_utility_by_technique(df):
    """Information-retention utility and predictive AUC for each configuration."""
    # Utility/AUC are scenario-independent except for scenario-specific local
    # suppression configurations, so one row per config_id is enough.
    d = df.drop_duplicates("config_id").copy()
    d = d.sort_values("utility", ascending=True).reset_index(drop=True)
    palette = family_palette(d)
    y = np.arange(len(d))
    colors = [palette[f] for f in d["family"]]

    fig, axes = plt.subplots(1, 2, figsize=(19, 13), sharey=True, constrained_layout=True)

    # Panel A: utility = 1 - information loss
    ax = axes[0]
    ax.scatter(d["utility"], y, s=75, c=colors, edgecolor="white", linewidth=0.7, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"], fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Utility = 1 - total information loss")
    ax.set_title("Information retention")
    ax.grid(axis="y", visible=False)
    add_family_legend(ax, palette, loc="lower right")

    # Panel B: predictive utility. Full-model AUC changes little because many
    # predictive financial variables are not modified; QI-only AUC is therefore
    # shown as well.
    ax = axes[1]
    ax.scatter(d["auc_full"], y, s=65, marker="o", label="AUC - full feature set", zorder=3)
    ax.scatter(d["auc_qi"], y, s=65, marker="s", label="AUC - QI + LIMIT_BAL", zorder=3)
    ax.set_xlabel("ROC-AUC")
    ax.set_title("Predictive utility")
    ax.set_xlim(0.48, 0.75)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=10)

    fig.suptitle("Utility after each protection configuration\n(higher is better)", fontsize=20)
    out = os.path.join(FIGURES_DIR, "utility_by_technique.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out

# per-column information loss shown in the heatmap (column -> header)
IL_COLUMNS = {"il_SEX": "SEX", "il_EDUCATION": "EDUCATION", "il_MARRIAGE": "MARRIAGE",
              "il_AGE": "AGE", "il_LIMIT_BAL": "LIMIT_BAL", "il_total": "total"}


def plot_attribute_disclosure(df):
    """Attribute disclosure rate of the original data and of the non-perturbative configurations."""
    # the measure is NaN for perturbative configurations (groups no longer match real people)
    d_all = df[df["attr_disclosure_rate"].notna() & (df["config_id"] != "no_data")]
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), constrained_layout=True)

    for ax, scenario in zip(axes, ["QI", "QI_EXT"]):
        d = d_all[d_all["scenario"] == scenario].sort_values("attr_disclosure_rate").reset_index(drop=True)
        original = d.loc[d["config_id"] == "original", "attr_disclosure_rate"].iloc[0]
        # red bars: the protection leaves more records exposed than the original data
        colors = ["0.6" if c == "original" else ("#C44E52" if v > original else "#4C72B0")
                  for c, v in zip(d["config_id"], d["attr_disclosure_rate"])]
        y = np.arange(len(d))
        ax.barh(y, d["attr_disclosure_rate"], color=colors, height=0.7)
        ax.axvline(original, linestyle=":", linewidth=1.5, color="0.3")
        for yi, v in zip(y, d["attr_disclosure_rate"]):
            ax.text(v, yi, f" {v:.2%}", va="center", fontsize=11)
        ax.set_yticks(y)
        ax.set_yticklabels(d["label"], fontsize=12)
        ax.set_xlabel("Records in a QI group where everyone has the same default value")
        ax.set_title(SCENARIO_TITLES[scenario])
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlim(0, d["attr_disclosure_rate"].max() * 1.2)
        ax.grid(axis="y", visible=False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=l) for c, l in
               [("0.6", "original"), ("#4C72B0", "lower than original"), ("#C44E52", "higher than original")]]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=13, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Attribute disclosure after non-perturbative protection\n(lower is better)", fontsize=20)
    out = os.path.join(FIGURES_DIR, "attribute_disclosure.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_information_loss_by_attribute(df):
    """Information loss of each QI column for every configuration."""
    d = df[~df["config_id"].isin(["original", "no_data"])].drop_duplicates("config_id")
    d = d.sort_values("il_total").set_index("label")[list(IL_COLUMNS)].rename(columns=IL_COLUMNS)
    # write only the losses that are not zero, so the table stays readable (the total always)
    values = d.to_numpy()
    annot = np.where(values >= 0.005, np.char.mod("%.2f", values), "")
    annot[:, -1] = np.char.mod("%.2f", values[:, -1])

    fig, ax = plt.subplots(figsize=(12, 14), constrained_layout=True)
    sns.heatmap(d, annot=annot, fmt="", cmap="Reds", vmin=0, vmax=1, linewidths=0.5,
                annot_kws={"fontsize": 12}, ax=ax,
                cbar_kws={"label": "information loss (0 = intact, 1 = no information left)", "shrink": 0.6})
    ax.axvline(len(IL_COLUMNS) - 1, color="black", linewidth=2.5)   # separates the total
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labeltop=True, labelbottom=False, labelsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_title("Where each protection loses information\n(rows sorted by total loss)", fontsize=18)
    out = os.path.join(FIGURES_DIR, "information_loss_by_attribute.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out

def pareto_frontier(d):
    """Return non-dominated points: maximize utility, minimize linkage risk."""
    d = d.sort_values(["utility", "dist_linkage_rate"], ascending=[False, True]).copy()
    keep = []
    best_risk = np.inf
    for idx, row in d.iterrows():
        risk = row["dist_linkage_rate"]
        if risk < best_risk - 1e-12:
            keep.append(idx)
            best_risk = risk
    return d.loc[keep].sort_values("utility")


def chosen_compromise(d, scenario):
    acceptable = d[d["dist_linkage_rate"] < MAX_RISK[scenario]].copy()
    if acceptable.empty:
        return None
    return acceptable.sort_values(
        ["utility", "dist_linkage_rate"], ascending=[False, True]
    ).iloc[0]


def plot_ru_map(df):
    """Risk-Utility confidentiality map for the two attacker scenarios."""
    palette = family_palette(df)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    for ax, scenario in zip(axes, ["QI", "QI_EXT"]):
        d = df[df["scenario"] == scenario].copy()

        # Feasible region under the illustrative risk threshold.
        ax.axhspan(0, MAX_RISK[scenario], alpha=0.08, color="0.4", zorder=0)
        ax.axhline(MAX_RISK[scenario], linestyle="--", linewidth=1.4,
                   color="0.35", label=f"illustrative threshold ({MAX_RISK[scenario]:.0%})")

        for family, g in d.groupby("family", sort=False):
            ax.scatter(g["utility"], g["dist_linkage_rate"], s=70,
                       color=palette[family], edgecolor="white", linewidth=0.7,
                       alpha=0.9, label=family, zorder=3)

        # Pareto frontier: there is no other evaluated point with both higher
        # utility and lower risk.
        frontier = pareto_frontier(d)
        ax.plot(frontier["utility"], frontier["dist_linkage_rate"],
                color="0.25", linewidth=1.3, linestyle=":", zorder=2,
                label="Pareto frontier")

        # Label only a few representative points to keep the map readable.
        important_ids = {
            "original", "no_data", "recode_age_5", "recode_age_10",
            "rankswap_10", "rankswap_20",
        }
        best = chosen_compromise(d, scenario)
        if best is not None:
            important_ids.add(best["config_id"])

        for _, row in d[d["config_id"].isin(important_ids)].iterrows():
            # points in the lower right corner are crowded: rank swapping labels go on the left
            offset, ha = LABEL_OFFSETS.get((row["config_id"], scenario), ((6, 6), "left"))
            ax.annotate(
                row["label"],
                (row["utility"], row["dist_linkage_rate"]),
                xytext=offset, textcoords="offset points", fontsize=10, ha=ha,
                fontweight="bold" if best is not None and row["config_id"] == best["config_id"] else "normal",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8),
                zorder=5,
            )

        if best is not None:
            ax.scatter([best["utility"]], [best["dist_linkage_rate"]],
                       s=220, facecolors="none", edgecolors="black", linewidths=1.7,
                       zorder=4)

        ax.set_title(SCENARIO_TITLES[scenario])
        ax.set_xlabel("Utility = 1 - total information loss")
        ax.set_ylabel("Correct distance-based linkage rate")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlim(-0.02, 1.02)
        ymax = max(d["dist_linkage_rate"].max() * 1.12, MAX_RISK[scenario] * 1.5)
        ax.set_ylim(-0.002, ymax)

        handles, labels = ax.get_legend_handles_labels()
        # Remove duplicate labels while preserving order.
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=14, markerscale=1.4,
                  loc="upper left", framealpha=0.95)

    fig.suptitle("Risk-Utility confidentiality map\n(right = more utility, down = less re-identification risk)",
                 fontsize=19)
    out = os.path.join(FIGURES_DIR, "ru_map.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def print_selected_points(df):
    print("\nSelected operating points (highest utility below the illustrative threshold):")
    for scenario in ["QI", "QI_EXT"]:
        d = df[df["scenario"] == scenario]
        best = chosen_compromise(d, scenario)
        if best is None:
            print(f"  {scenario}: none below {MAX_RISK[scenario]:.0%}")
        else:
            print(
                f"  {scenario}: {best['config_id']}  "
                f"risk={best['dist_linkage_rate']:.3%}, "
                f"utility={best['utility']:.3f}, "
                f"AUC(full)={best['auc_full']:.4f}, AUC(QI)={best['auc_qi']:.4f}"
            )


def main():
    df = load_results()
    outputs = [
        plot_risk_by_technique(df),
        plot_utility_by_technique(df),
        plot_ru_map(df),
        plot_attribute_disclosure(df),
        plot_information_loss_by_attribute(df),
    ]
    print_selected_points(df)
    print("\nSaved figures:")
    for path in outputs:
        print(f"  {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()