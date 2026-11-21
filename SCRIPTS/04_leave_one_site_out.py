"""
04_leave_one_site_out.py
------------------------
Step 4. Cross-site generalization: train on two imaging sites and test on the
third, which the model never saw. This checks whether the CV result in
step 3 holds up when scanner and protocol differ between training and test.

Input  : DATA/final/fc_features.npz, DATA/final/analysis_cohort.csv
Output : OUTPUT/tables/loso_metrics_by_site.csv   one row per held-out site + mean row
         OUTPUT/tables/loso_predictions.csv       per-subject held-out predictions
         OUTPUT/figures/loso_performance_by_site.png

Same model as step 3 (StandardScaler + L2 logistic regression, C = 1).
Each held-out site has 20 people (10 ASD, 10 TDC), so each site-level
number is noisy: one person changes balanced accuracy by 5 points.

Run from the repository root:  python SCRIPTS/04_leave_one_site_out.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils

utils.ensure_output_dirs()
X, cohort = utils.load_features()

site_df, pred_df = utils.run_leave_one_site_out(X, cohort)
metrics = ["balanced_accuracy", "roc_auc", "sensitivity", "specificity"]
mean_row = {"held_out_site": "Mean across sites", **site_df[metrics].mean().to_dict()}
out = pd.concat([site_df, pd.DataFrame([mean_row])], ignore_index=True)
for c in ["n_train", "n_test", "n_test_ASD", "n_test_TDC"]:
    out[c] = out[c].astype("Int64")  # keep counts as integers next to the blank mean row
out.round(4).to_csv(utils.TAB_DIR / "loso_metrics_by_site.csv", index=False)
pred_df.round(4).to_csv(utils.TAB_DIR / "loso_predictions.csv", index=False)
print(out.round(3).to_string(index=False))

# --- Figure: balanced accuracy and AUC per held-out site ------------------------
fig, ax = plt.subplots(figsize=(6, 4))
xs = np.arange(len(site_df))
for k, (m, lab, col) in enumerate([("balanced_accuracy", "Balanced accuracy", "#2E6F95"),
                                    ("roc_auc", "ROC-AUC", "#88A9C3")]):
    bars = ax.bar(xs + (k - 0.5) * 0.38, site_df[m], width=0.36, label=lab, color=col)
    ax.bar_label(bars, fmt="%.2f", fontsize=8)
ax.axhline(utils.SUCCESS_BALANCED_ACC, ls="--", color="#C8553D", lw=1.2,
           label=f"Goal ({utils.SUCCESS_BALANCED_ACC:.0%} bal. acc.)")
ax.axhline(0.5, ls=":", color="grey", lw=1, label="Chance")
ax.set_xticks(xs, [f"Test: {s}" for s in site_df["held_out_site"]])
ax.set_ylim(0, 1.25)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_ylabel("Score on held-out site")
ax.set_title("Leave-one-site-out performance")
ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(utils.FIG_DIR / "loso_performance_by_site.png", dpi=200)
print("Saved LOSO tables and figure to OUTPUT/")
