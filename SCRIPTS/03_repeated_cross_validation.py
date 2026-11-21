"""
03_repeated_cross_validation.py
-------------------------------
Step 3. Main analysis: how well do FC edges separate ASD from TDC when the
model is tested on people it has not seen?

Input  : DATA/final/fc_features.npz, DATA/final/analysis_cohort.csv
Output : OUTPUT/tables/cv_fold_metrics.csv         one row per repeat x fold (50 rows)
         OUTPUT/tables/cv_repeat_metrics.csv       5 folds averaged within each repeat
         OUTPUT/tables/cv_summary.csv              mean and SD across the 10 repeats
         OUTPUT/tables/cv_out_of_fold_predictions.csv
         OUTPUT/figures/cv_roc_curve.png
         OUTPUT/figures/cv_balanced_accuracy_by_repeat.png

Design (from MI2)
    * Repeated stratified 5-fold CV, 10 repeats, seed 42. Folds are stratified
      by site x diagnosis, so each test fold has a similar mix of sites and groups.
    * Model: StandardScaler + L2 logistic regression (C = 1). The scaler sits
      inside the pipeline, so it is fit on training folds only (no leakage).
    * Metrics: balanced accuracy, ROC-AUC, sensitivity, specificity. Fold
      metrics are averaged within each repeat; we report the mean and SD of
      those 10 repeat-level values.
    * The ROC curve uses each subject's out-of-fold probability averaged over
      the 10 repeats (for display only; the reported AUC is the fold average).

Run from the repository root:  python SCRIPTS/03_repeated_cross_validation.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

import utils

utils.ensure_output_dirs()
X, cohort = utils.load_features()
y = cohort["diagnosis"].map(utils.LABEL_MAP).values
print(f"X: {X.shape} | ASD = {y.sum()}, TDC = {(1 - y).sum()}")

# --- Run CV ---------------------------------------------------------------------
fold_df, oof_prob = utils.run_repeated_cv(X, cohort)
per_repeat, summary = utils.summarize_by_repeat(fold_df)

fold_df.round(4).to_csv(utils.TAB_DIR / "cv_fold_metrics.csv", index=False)
per_repeat.round(4).to_csv(utils.TAB_DIR / "cv_repeat_metrics.csv")
summary.round(4).to_csv(utils.TAB_DIR / "cv_summary.csv", index_label="metric")

mean_prob = oof_prob.mean(axis=0)
pd.DataFrame({"subject_id": cohort["subject_id"], "site": cohort["site"],
              "true_label": y, "mean_oof_prob_ASD": mean_prob.round(4)}
             ).to_csv(utils.TAB_DIR / "cv_out_of_fold_predictions.csv", index=False)

print("\nRepeated 5-fold CV (mean +/- SD across 10 repeats):")
for metric, row in summary.iterrows():
    print(f"  {metric:18s} {row['mean']:.3f} +/- {row['sd_across_repeats']:.3f}")
met = summary.loc["balanced_accuracy", "mean"] >= utils.SUCCESS_BALANCED_ACC
print(f"\nMI2 goal (balanced accuracy >= {utils.SUCCESS_BALANCED_ACC:.0%}): "
      f"{'met' if met else 'not met'}")

# --- ROC curve figure -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.8, 4.5))
for r in range(oof_prob.shape[0]):  # thin lines: each repeat's pooled out-of-fold ROC
    fpr, tpr, _ = roc_curve(y, oof_prob[r])
    ax.plot(fpr, tpr, color="#9DB4C0", lw=0.8, alpha=0.7)
fpr, tpr, _ = roc_curve(y, mean_prob)
ax.plot(fpr, tpr, color="#2E6F95", lw=2,
        label=f"Averaged over repeats (AUC = {roc_auc_score(y, mean_prob):.2f})")
ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="Chance")
ax.set_xlabel("False positive rate (1 - specificity)")
ax.set_ylabel("True positive rate (sensitivity)")
ax.set_title("ROC, repeated 5-fold CV")
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(utils.FIG_DIR / "cv_roc_curve.png", dpi=200)

# --- Balanced accuracy per repeat ----------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.bar(per_repeat.index, per_repeat["balanced_accuracy"], color="#2E6F95")
ax.axhline(utils.SUCCESS_BALANCED_ACC, ls="--", color="#C8553D", lw=1.2,
           label=f"Goal ({utils.SUCCESS_BALANCED_ACC:.0%})")
ax.axhline(0.5, ls=":", color="grey", lw=1, label="Chance (50%)")
ax.set_xticks(per_repeat.index)
ax.set_ylim(0, 1)
ax.set_xlabel("CV repeat")
ax.set_ylabel("Balanced accuracy (mean of 5 folds)")
ax.set_title("Balanced accuracy across CV repeats")
ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(utils.FIG_DIR / "cv_balanced_accuracy_by_repeat.png", dpi=200)
print("Saved CV tables and figures to OUTPUT/")
