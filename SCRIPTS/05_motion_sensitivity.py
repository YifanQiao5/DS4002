"""
05_motion_sensitivity.py
------------------------
Step 5. Head-motion sensitivity analysis planned in MI2. Head motion is known
to create spurious FC differences [Power et al., 2012], and in this cohort
mean FD is higher in ASD than TDC. If the classifier were mostly picking up
motion, performance should change when high-motion people are removed.

Input  : DATA/final/fc_features.npz, DATA/final/analysis_cohort.csv
Output : OUTPUT/tables/motion_excluded_participants.csv  who was removed
         OUTPUT/tables/motion_cohort_balance.csv         site x diagnosis counts before/after
         OUTPUT/tables/motion_cv_fold_metrics.csv
         OUTPUT/tables/motion_cv_summary.csv
         OUTPUT/tables/motion_loso_metrics_by_site.csv
         OUTPUT/tables/motion_sensitivity_comparison.csv  full vs low-motion cohort
         OUTPUT/figures/motion_sensitivity_comparison.png

What it does
    1. Removes everyone with mean FD > 0.15 mm (threshold set in MI2).
    2. Reports how many were removed and whether ASD/TDC balance per site changed.
    3. Re-runs exactly the same repeated CV (step 3) and LOSO (step 4) on the
       remaining people and compares the results with the full cohort.
    The analysis is only run if at least 5 ASD and 5 TDC remain (5-fold CV
    needs at least one person of each group per fold).

Run from the repository root:  python SCRIPTS/05_motion_sensitivity.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils

utils.ensure_output_dirs()
X, cohort = utils.load_features()
thr = utils.FD_THRESHOLD_MM
metrics = ["balanced_accuracy", "roc_auc", "sensitivity", "specificity"]

# --- 1-2. Exclusion and balance ---------------------------------------------------
keep = (cohort["mean_fd"] <= thr).values
excluded = cohort.loc[~keep, ["subject_id", "site", "diagnosis", "mean_fd"]]
excluded.to_csv(utils.TAB_DIR / "motion_excluded_participants.csv", index=False)

balance = pd.concat({
    "full_cohort": pd.crosstab(cohort["site"], cohort["diagnosis"]),
    f"mean_fd_le_{thr}": pd.crosstab(cohort.loc[keep, "site"], cohort.loc[keep, "diagnosis"]),
}, axis=1).fillna(0).astype(int)
balance.to_csv(utils.TAB_DIR / "motion_cohort_balance.csv")
print(f"Excluded {(~keep).sum()} of {len(cohort)} participants with mean FD > {thr} mm")
print(excluded.groupby("diagnosis").size().rename("n_excluded").to_string())
print("\n" + balance.to_string())

X_low, cohort_low = X[keep], cohort.loc[keep].reset_index(drop=True)
n_by_group = cohort_low["diagnosis"].value_counts()
if min(n_by_group.get("ASD", 0), n_by_group.get("TDC", 0)) < utils.N_SPLITS:
    raise SystemExit("Too few participants left in one group; sensitivity analysis not run.")

# --- 3. Re-run CV and LOSO on the low-motion cohort ----------------------------------
fold_low, _ = utils.run_repeated_cv(X_low, cohort_low)
_, cv_low = utils.summarize_by_repeat(fold_low)
loso_low, _ = utils.run_leave_one_site_out(X_low, cohort_low)
fold_low.round(4).to_csv(utils.TAB_DIR / "motion_cv_fold_metrics.csv", index=False)
cv_low.round(4).to_csv(utils.TAB_DIR / "motion_cv_summary.csv", index_label="metric")
loso_low.round(4).to_csv(utils.TAB_DIR / "motion_loso_metrics_by_site.csv", index=False)

# Full-cohort results for comparison (same code, same seed as steps 3 and 4)
fold_full, _ = utils.run_repeated_cv(X, cohort)
_, cv_full = utils.summarize_by_repeat(fold_full)
loso_full, _ = utils.run_leave_one_site_out(X, cohort)

rows = []
for name, cv, loso, n in [("Full cohort", cv_full, loso_full, len(cohort)),
                          (f"Mean FD <= {thr} mm", cv_low, loso_low, len(cohort_low))]:
    for m in metrics:
        rows.append({"cohort": name, "n": n, "metric": m,
                     "cv_mean": cv.loc[m, "mean"], "cv_sd": cv.loc[m, "sd_across_repeats"],
                     "loso_mean_across_sites": loso[m].mean()})
comparison = pd.DataFrame(rows)
comparison.round(4).to_csv(utils.TAB_DIR / "motion_sensitivity_comparison.csv", index=False)
print("\n" + comparison.round(3).to_string(index=False))

# --- Figure: full vs low-motion -------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), sharey=True)
cohorts = comparison["cohort"].unique()
cols = ["#2E6F95", "#E0A458"]
for ax, (col_mean, title) in zip(axes, [("cv_mean", "Repeated 5-fold CV"),
                                        ("loso_mean_across_sites", "Leave-one-site-out")]):
    xs = np.arange(2)
    for k, c in enumerate(cohorts):
        sub = comparison[(comparison.cohort == c) &
                         comparison.metric.isin(["balanced_accuracy", "roc_auc"])]
        n = sub["n"].iloc[0]
        err = sub["cv_sd"].values if col_mean == "cv_mean" else None
        bars = ax.bar(xs + (k - 0.5) * 0.38, sub[col_mean], width=0.36, yerr=err,
                      capsize=3, color=cols[k], label=f"{c} (n = {n})")
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.axhline(0.5, ls=":", color="grey", lw=1)
    ax.set_xticks(xs, ["Balanced accuracy", "ROC-AUC"])
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Score")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=8, ncol=2, loc="lower center")
fig.suptitle("Head-motion sensitivity analysis (CV error bars: SD across repeats)", fontsize=10)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(utils.FIG_DIR / "motion_sensitivity_comparison.png", dpi=200)
print("Saved motion sensitivity tables and figure to OUTPUT/")
