"""
02_exploratory_analysis.py
--------------------------
Step 2. Describes the cohort and checks possible confounds (age, sex, head
motion) before any model is fit. These plots are also used in DATA/README.md.

Input  : DATA/final/analysis_cohort.csv   (written by script 01)
Output : OUTPUT/tables/cohort_summary_by_site_and_diagnosis.csv
         OUTPUT/tables/group_comparisons_asd_vs_tdc.csv
         OUTPUT/figures/site_by_diagnosis_counts.png
         OUTPUT/figures/mean_fd_by_diagnosis.png
         OUTPUT/figures/mean_fd_by_site_and_diagnosis.png
         OUTPUT/figures/age_by_diagnosis.png

Statistical tests
    * Age and mean FD: Welch's t-test (does not assume equal variances).
    * Recorded sex: Fisher's exact test on the 2 x 2 diagnosis-by-sex table.
    These are descriptive checks on 60 people, not confirmatory tests.

Run from the repository root:  python SCRIPTS/02_exploratory_analysis.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, ttest_ind

import utils

utils.ensure_output_dirs()
cohort = pd.read_csv(utils.COHORT_PATH, dtype={"subject_id": str})
groups = ["ASD", "TDC"]
colors = {"ASD": "#C8553D", "TDC": "#2E6F95"}
rng = np.random.default_rng(utils.RANDOM_STATE)  # only used for point jitter

# --- Summary table by site and diagnosis -------------------------------------
summary = (cohort.groupby(["site", "diagnosis"])
           .agg(n=("subject_id", "size"),
                age_mean=("age", "mean"), age_sd=("age", "std"),
                n_female=("sex", lambda s: int((s == "F").sum())),
                n_male=("sex", lambda s: int((s == "M").sum())),
                mean_fd_mean=("mean_fd", "mean"), mean_fd_sd=("mean_fd", "std"),
                n_fd_above_threshold=("mean_fd",
                                      lambda s: int((s > utils.FD_THRESHOLD_MM).sum())))
           .round(3).reset_index())
summary.to_csv(utils.TAB_DIR / "cohort_summary_by_site_and_diagnosis.csv", index=False)
print(summary.to_string(index=False))

# --- ASD vs TDC comparisons ----------------------------------------------------
asd, tdc = cohort[cohort.diagnosis == "ASD"], cohort[cohort.diagnosis == "TDC"]
rows = []
for var, unit in [("age", "years"), ("mean_fd", "mm")]:
    t, p = ttest_ind(asd[var], tdc[var], equal_var=False)
    rows.append({"variable": var, "test": "Welch t-test",
                 "ASD_mean": asd[var].mean(), "TDC_mean": tdc[var].mean(),
                 "statistic": t, "p_value": p, "unit": unit})
sex_table = pd.crosstab(cohort["diagnosis"], cohort["sex"]).reindex(
    index=groups, columns=["F", "M"], fill_value=0)
odds, p_sex = fisher_exact(sex_table.values)
rows.append({"variable": "sex (F/M)", "test": "Fisher exact",
             "ASD_mean": f"{sex_table.loc['ASD', 'F']}F/{sex_table.loc['ASD', 'M']}M",
             "TDC_mean": f"{sex_table.loc['TDC', 'F']}F/{sex_table.loc['TDC', 'M']}M",
             "statistic": odds, "p_value": p_sex, "unit": "count"})
comparisons = pd.DataFrame(rows)
comparisons.to_csv(utils.TAB_DIR / "group_comparisons_asd_vs_tdc.csv", index=False)
print("\n" + comparisons.to_string(index=False))


def jitter_points(ax, values, x, color):
    """Overlay individual participants on a box plot."""
    ax.scatter(x + rng.uniform(-0.08, 0.08, len(values)), values,
               s=16, color=color, alpha=0.7, zorder=3, edgecolor="none")


# --- Figure: participants per site and diagnosis ---------------------------------
counts = pd.crosstab(cohort["site"], cohort["diagnosis"]).reindex(columns=groups)
fig, ax = plt.subplots(figsize=(6, 4))
xs = np.arange(len(counts))
for k, g in enumerate(groups):
    bars = ax.bar(xs + (k - 0.5) * 0.38, counts[g], width=0.36, label=g, color=colors[g])
    ax.bar_label(bars, fontsize=9)
ax.set_xticks(xs, counts.index)
ax.set_ylabel("Participants")
ax.set_xlabel("Site")
ax.set_title("Participants by site and diagnosis")
ax.set_ylim(0, counts.values.max() * 1.35)
ax.legend(frameon=False, ncol=2, loc="upper center")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(utils.FIG_DIR / "site_by_diagnosis_counts.png", dpi=200)

# --- Figure: mean FD and age by diagnosis -----------------------------------------
for var, label, fname in [("mean_fd", "Mean framewise displacement (mm)", "mean_fd_by_diagnosis.png"),
                          ("age", "Age (years)", "age_by_diagnosis.png")]:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    data = [cohort.loc[cohort.diagnosis == g, var] for g in groups]
    ax.boxplot(data, tick_labels=groups, widths=0.5, showfliers=False)
    for k, g in enumerate(groups):
        jitter_points(ax, data[k], k + 1, colors[g])
    if var == "mean_fd":
        ax.axhline(utils.FD_THRESHOLD_MM, ls="--", color="grey", lw=1)
        ax.text(2.45, utils.FD_THRESHOLD_MM, f"{utils.FD_THRESHOLD_MM} mm",
                va="bottom", ha="right", fontsize=8, color="grey")
    p = comparisons.loc[comparisons.variable == var, "p_value"].iloc[0]
    ax.set_title(f"{label.split(' (')[0]} by diagnosis\n(Welch p = {p:.3f})")
    ax.set_ylabel(label)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(utils.FIG_DIR / fname, dpi=200)

# --- Figure: mean FD by site and diagnosis -----------------------------------------
sites = sorted(cohort["site"].unique())
fig, ax = plt.subplots(figsize=(6.5, 4))
for s_i, site in enumerate(sites):
    for k, g in enumerate(groups):
        vals = cohort.loc[(cohort.site == site) & (cohort.diagnosis == g), "mean_fd"]
        pos = s_i * 3 + k + 1
        ax.boxplot([vals], positions=[pos], widths=0.6, showfliers=False)
        jitter_points(ax, vals, pos, colors[g])
ax.axhline(utils.FD_THRESHOLD_MM, ls="--", color="grey", lw=1)
ax.set_xticks([s_i * 3 + 1.5 for s_i in range(len(sites))], sites)
ax.set_ylabel("Mean framewise displacement (mm)")
ax.set_title("Head motion by site (ASD left, TDC right)")
ax.scatter([], [], color=colors["ASD"], label="ASD")
ax.scatter([], [], color=colors["TDC"], label="TDC")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(utils.FIG_DIR / "mean_fd_by_site_and_diagnosis.png", dpi=200)
print("\nSaved exploratory figures to OUTPUT/figures/")
