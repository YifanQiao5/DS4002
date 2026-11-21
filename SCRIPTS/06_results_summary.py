"""
06_results_summary.py
---------------------
Step 6. Collects the main numbers from steps 3-5 into one table for the
README and the presentation, and checks them against the MI2 goal
(balanced accuracy >= 70% under repeated subject-level CV).

Input  : OUTPUT/tables/cv_summary.csv
         OUTPUT/tables/loso_metrics_by_site.csv
         OUTPUT/tables/motion_sensitivity_comparison.csv
Output : OUTPUT/tables/main_results_summary.csv

Run from the repository root:  python SCRIPTS/06_results_summary.py
"""

import pandas as pd

import utils

cv = pd.read_csv(utils.TAB_DIR / "cv_summary.csv", index_col="metric")
loso = pd.read_csv(utils.TAB_DIR / "loso_metrics_by_site.csv")
motion = pd.read_csv(utils.TAB_DIR / "motion_sensitivity_comparison.csv")
metrics = ["balanced_accuracy", "roc_auc", "sensitivity", "specificity"]


def fmt(mean, sd=None):
    return f"{mean:.3f}" if sd is None or pd.isna(sd) else f"{mean:.3f} ± {sd:.3f}"


rows = []
rows.append({"evaluation": "Repeated 5-fold CV, full cohort (mean ± SD over 10 repeats)",
             **{m: fmt(cv.loc[m, "mean"], cv.loc[m, "sd_across_repeats"]) for m in metrics}})
for _, r in loso.iterrows():
    rows.append({"evaluation": f"Leave-one-site-out, {r['held_out_site']}",
                 **{m: fmt(r[m]) for m in metrics}})
low = motion[motion.cohort != "Full cohort"].set_index("metric")
if len(low):
    label, n = low["cohort"].iloc[0], int(low["n"].iloc[0])
    rows.append({"evaluation": f"Repeated 5-fold CV, {label} (n = {n})",
                 **{m: fmt(low.loc[m, "cv_mean"], low.loc[m, "cv_sd"]) for m in metrics}})
    rows.append({"evaluation": f"Leave-one-site-out mean, {label} (n = {n})",
                 **{m: fmt(low.loc[m, "loso_mean_across_sites"]) for m in metrics}})

table = pd.DataFrame(rows)
table.to_csv(utils.TAB_DIR / "main_results_summary.csv", index=False)
print(table.to_string(index=False))

ba = cv.loc["balanced_accuracy", "mean"]
print(f"\nMI2 goal: balanced accuracy >= {utils.SUCCESS_BALANCED_ACC:.0%} under repeated CV. "
      f"Observed {ba:.1%}: goal {'met' if ba >= utils.SUCCESS_BALANCED_ACC else 'not met'}.")
