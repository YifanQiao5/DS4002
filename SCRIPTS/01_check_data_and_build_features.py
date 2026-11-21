"""
01_check_data_and_build_features.py
-----------------------------------
Step 1 of the pipeline. Run this first.

Input  : DATA/initial/participants.csv
         DATA/initial/timeseries/{subject_id}_rois_cc200.1D  (one per subject)
Output : DATA/final/fc_features.npz            60 x 19,900 Fisher-z FC edges
         DATA/final/analysis_cohort.csv        metadata in the same row order
         OUTPUT/tables/data_quality_checks.csv per-subject checks
         OUTPUT/figures/mean_fc_matrix.png     mean FC matrix across subjects

What it does
    1. Checks the metadata: unique IDs, valid labels, no missing values.
    2. For every subject, loads the CC200 time series (time points x 200 ROIs)
       and checks shape, non-finite values and constant (all-zero) ROIs,
       because a constant ROI makes its correlations undefined.
    3. Computes the 200 x 200 Pearson correlation matrix, clips to +/-0.999999
       so arctanh stays finite, and applies the Fisher-z transform.
    4. Keeps the 19,900 unique upper-triangle edges as the feature vector.

Run from the repository root:  python SCRIPTS/01_check_data_and_build_features.py
"""

import matplotlib
matplotlib.use("Agg")  # write figures to files, no screen needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils

utils.ensure_output_dirs()

# --- 1. Metadata checks -----------------------------------------------------
participants = pd.read_csv(utils.PARTICIPANTS_PATH, dtype={"subject_id": str})
required = ["subject_id", "diagnosis", "age", "sex", "site", "mean_fd"]
missing_cols = [c for c in required if c not in participants.columns]
if missing_cols:
    raise ValueError(f"participants.csv is missing columns: {missing_cols}")

assert participants["subject_id"].is_unique, "Duplicate subject IDs in participants.csv"
assert set(participants["diagnosis"]) <= set(utils.LABEL_MAP), "Unexpected diagnosis labels"
n_missing = int(participants[required].isna().sum().sum())
print(f"Subjects: {len(participants)} | missing metadata values: {n_missing}")
print(pd.crosstab(participants["site"], participants["diagnosis"]))

# --- 2-4. Time-series checks and FC features ----------------------------------
upper_idx = np.triu_indices(utils.N_ROIS, k=1)  # 200*199/2 = 19,900 edges
qc_rows, features = [], []

for subject_id in participants["subject_id"]:
    path = utils.TIMESERIES_DIR / f"{subject_id}_rois_cc200.1D"
    if not path.exists():
        raise FileNotFoundError(f"Missing time-series file for subject {subject_id}: {path}")

    ts = np.loadtxt(path)  # rows = time points, columns = ROIs
    if ts.ndim != 2 or ts.shape[1] != utils.N_ROIS:
        raise ValueError(f"{subject_id}: shape {ts.shape}, expected (time points, 200)")
    n_nonfinite = int((~np.isfinite(ts)).sum())
    n_constant_rois = int((ts.std(axis=0) == 0).sum())
    qc_rows.append({"subject_id": subject_id, "n_timepoints": ts.shape[0],
                    "n_nonfinite_values": n_nonfinite,
                    "n_constant_rois": n_constant_rois})
    if n_nonfinite or n_constant_rois:
        raise ValueError(f"{subject_id}: {n_nonfinite} non-finite values, "
                         f"{n_constant_rois} constant ROIs; FC would be undefined")

    fc = np.corrcoef(ts, rowvar=False)          # 200 x 200 Pearson r
    np.fill_diagonal(fc, 0)                     # self-correlations are not features
    fc_z = np.arctanh(np.clip(fc, -0.999999, 0.999999))  # Fisher-z
    features.append(fc_z[upper_idx])

X = np.vstack(features)
qc = pd.DataFrame(qc_rows).merge(participants[["subject_id", "site", "diagnosis"]],
                                 on="subject_id")
qc.to_csv(utils.TAB_DIR / "data_quality_checks.csv", index=False)
print("\nTime points per site:")
print(qc.groupby("site")["n_timepoints"].agg(["count", "min", "max", "mean"]))

# --- Save final analysis data -------------------------------------------------
np.savez_compressed(utils.FEATURES_PATH, X=X.astype(np.float32),
                    subject_id=np.array(participants["subject_id"].tolist(), dtype="U"))
participants.to_csv(utils.COHORT_PATH, index=False)
print(f"\nFeature matrix {X.shape} saved to {utils.FEATURES_PATH}")

# --- Mean FC matrix figure ------------------------------------------------------
# Average in Fisher-z space, then convert back to r for display.
mean_z = np.zeros((utils.N_ROIS, utils.N_ROIS))
mean_z[upper_idx] = X.mean(axis=0)
mean_r = np.tanh(mean_z + mean_z.T)
np.fill_diagonal(mean_r, 1)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(mean_r, vmin=-1, vmax=1, cmap="RdBu_r")
fig.colorbar(im, ax=ax, label="Mean Pearson r")
ax.set_xlabel("CC200 ROI")
ax.set_ylabel("CC200 ROI")
ax.set_title(f"Mean functional connectivity (n = {len(X)})")
fig.tight_layout()
fig.savefig(utils.FIG_DIR / "mean_fc_matrix.png", dpi=200)
print("Saved OUTPUT/figures/mean_fc_matrix.png")
