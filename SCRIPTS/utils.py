"""
utils.py
--------
Shared paths, settings, and helper functions used by every numbered script
in SCRIPTS/. This file is not run on its own; the numbered scripts import it.

Project : ASD vs. TDC classification from resting-state functional connectivity
Course  : DS 4002, Project 3 (Group 8: Yifan, Emma)

What lives here
    * Folder paths (DATA/initial, DATA/final, OUTPUT), resolved relative to the
      repository root so the scripts work from any working directory.
    * Analysis settings (CV folds/repeats, random seed, motion threshold,
      logistic regression strength) so that every script uses the same values.
    * The classification pipeline (standardize -> L2 logistic regression).
    * Metric helpers (balanced accuracy, ROC-AUC, sensitivity, specificity).
    * Evaluation loops for repeated stratified 5-fold CV and
      leave-one-site-out testing, reused by the main and motion analyses.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths (all relative to the repository root, one level above SCRIPTS/)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_INITIAL = REPO_ROOT / "DATA" / "initial"
DATA_FINAL = REPO_ROOT / "DATA" / "final"
OUTPUT_DIR = REPO_ROOT / "OUTPUT"
FIG_DIR = OUTPUT_DIR / "figures"
TAB_DIR = OUTPUT_DIR / "tables"

PARTICIPANTS_PATH = DATA_INITIAL / "participants.csv"
TIMESERIES_DIR = DATA_INITIAL / "timeseries"
FEATURES_PATH = DATA_FINAL / "fc_features.npz"        # written by script 01
COHORT_PATH = DATA_FINAL / "analysis_cohort.csv"       # written by script 01

# ---------------------------------------------------------------------------
# Analysis settings (fixed in MI2; change here only if the plan changes)
# ---------------------------------------------------------------------------
N_ROIS = 200                 # CC200 atlas
N_SPLITS = 5                 # 5-fold CV
N_REPEATS = 10               # repeated 10 times with different shuffles
RANDOM_STATE = 42            # seed for every random step
LOGREG_C = 1.0               # inverse L2 strength (sklearn default)
FD_THRESHOLD_MM = 0.15       # motion sensitivity analysis cut-off
LABEL_MAP = {"TDC": 0, "ASD": 1}   # ASD is the positive class
SUCCESS_BALANCED_ACC = 0.70  # MI2 success criterion


def ensure_output_dirs():
    """Create OUTPUT/figures, OUTPUT/tables and DATA/final if missing."""
    for d in (FIG_DIR, TAB_DIR, DATA_FINAL):
        d.mkdir(parents=True, exist_ok=True)


def load_features():
    """Load the FC feature matrix and the matching cohort table.

    Returns
    -------
    X : (n_subjects, 19900) array of Fisher-z FC edges
    cohort : DataFrame with subject_id, diagnosis, age, sex, site, mean_fd,
             row-aligned with X
    """
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURES_PATH} not found. Run 01_check_data_and_build_features.py first.")
    npz = np.load(FEATURES_PATH, allow_pickle=False)
    cohort = pd.read_csv(COHORT_PATH, dtype={"subject_id": str})
    # Safety check: rows of X must be in the same order as the cohort table
    if not np.array_equal(npz["subject_id"].astype(str),
                          cohort["subject_id"].astype(str).values):
        raise ValueError("Subject order in fc_features.npz and analysis_cohort.csv differs.")
    return npz["X"], cohort


def make_model():
    """Standardize each edge (fit on training data only, because it sits
    inside the pipeline) and fit an L2-regularized logistic regression."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=LOGREG_C, solver="liblinear", max_iter=1000)),
    ])


def binary_metrics(y_true, y_pred, y_prob):
    """Balanced accuracy, ROC-AUC, sensitivity (ASD recall) and specificity
    (TDC recall). ROC-AUC is NaN when the test set has only one class."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred)
        if len(np.unique(y_true)) == 2 else np.nan,
        "roc_auc": auc,
        "sensitivity": sens,
        "specificity": spec,
    }


def run_repeated_cv(X, cohort):
    """Repeated stratified 5-fold CV, stratified jointly by site and diagnosis
    so that every fold keeps the site/diagnosis mix of the full cohort.

    Returns
    -------
    fold_df : one row per (repeat, fold) with the four metrics
    oof_prob : (n_repeats, n_subjects) out-of-fold ASD probabilities
    """
    y = cohort["diagnosis"].map(LABEL_MAP).values
    strata = (cohort["site"] + "_" + cohort["diagnosis"]).values
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                   random_state=RANDOM_STATE)
    rows = []
    oof_prob = np.full((N_REPEATS, len(y)), np.nan)
    for i, (tr, te) in enumerate(rskf.split(X, strata)):
        rep, fold = divmod(i, N_SPLITS)
        model = make_model().fit(X[tr], y[tr])
        prob = model.predict_proba(X[te])[:, 1]
        pred = (prob >= 0.5).astype(int)
        oof_prob[rep, te] = prob
        rows.append({"repeat": rep + 1, "fold": fold + 1, "n_test": len(te),
                     **binary_metrics(y[te], pred, prob)})
    return pd.DataFrame(rows), oof_prob


def summarize_by_repeat(fold_df):
    """Average the 5 folds inside each repeat, then report mean and SD across
    the 10 repeats (the SD reflects how much results depend on the split)."""
    metrics = ["balanced_accuracy", "roc_auc", "sensitivity", "specificity"]
    per_repeat = fold_df.groupby("repeat")[metrics].mean()
    summary = pd.DataFrame({"mean": per_repeat.mean(),
                            "sd_across_repeats": per_repeat.std(ddof=1)})
    return per_repeat, summary


def run_leave_one_site_out(X, cohort):
    """Train on two sites, test on the held-out third site, for each site."""
    y = cohort["diagnosis"].map(LABEL_MAP).values
    sites = cohort["site"].values
    rows, preds = [], []
    for test_site in sorted(np.unique(sites)):
        tr, te = sites != test_site, sites == test_site
        model = make_model().fit(X[tr], y[tr])
        prob = model.predict_proba(X[te])[:, 1]
        pred = (prob >= 0.5).astype(int)
        rows.append({"held_out_site": test_site,
                     "n_train": int(tr.sum()), "n_test": int(te.sum()),
                     "n_test_ASD": int(y[te].sum()),
                     "n_test_TDC": int((1 - y[te]).sum()),
                     **binary_metrics(y[te], pred, prob)})
        preds.append(pd.DataFrame({"subject_id": cohort.loc[te, "subject_id"].values,
                                   "held_out_site": test_site,
                                   "true_label": y[te], "prob_ASD": prob,
                                   "pred_label": pred}))
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)
