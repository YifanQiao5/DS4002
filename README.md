# Autism Spectrum Disorder Classification Using Resting-State Functional Connectivity

DS 4002, Project 3 · Group 8: Yifan (group leader), Emma

**Research question.** Can resting-state functional connectivity (FC) tell individuals with autism spectrum disorder (ASD) apart from typically developing controls (TDC) across imaging sites, and how robust is the prediction to site differences and head motion?

**Goal (MI2).** At least 70% balanced accuracy under repeated subject-level cross-validation, plus checks of cross-site generalization and head motion.

## Contents of this repository

This repository holds everything needed to reproduce the MI3 analysis, organized following the TIER Protocol 4.0 [1]:

* `README.md`: this orientation file.
* `LICENSE.md`: MIT license for the code. The data keep their own license (see `DATA/README.md`).
* `SCRIPTS/`: Python code, numbered in the order it must be run.
* `DATA/`: the initial data (participant table and 60 regional time-series files), the final analysis data made from them, and the metadata file `DATA/README.md`.
* `OUTPUT/`: every figure and table the scripts produce.

### Main results

60 participants from ABIDE I (NYU, UCLA_1, UM_1; 10 ASD and 10 TDC per site). Model: standardized FC edges (19,900 per person) and L2-regularized logistic regression.

| Evaluation | Balanced accuracy | ROC-AUC | Sensitivity | Specificity |
|---|---|---|---|---|
| Repeated 5-fold CV, 10 repeats (mean ± SD across repeats) | 0.707 ± 0.046 | 0.787 ± 0.029 | 0.773 ± 0.073 | 0.640 ± 0.044 |
| Leave-one-site-out, test on NYU | 0.750 | 0.750 | 0.800 | 0.700 |
| Leave-one-site-out, test on UCLA_1 | 0.750 | 0.790 | 0.600 | 0.900 |
| Leave-one-site-out, test on UM_1 | 0.750 | 0.780 | 0.900 | 0.600 |
| Repeated 5-fold CV, mean FD ≤ 0.15 mm (n = 57) | 0.716 ± 0.025 | 0.775 ± 0.045 | 0.795 ± 0.044 | 0.637 ± 0.043 |
| Leave-one-site-out mean, mean FD ≤ 0.15 mm (n = 57) | 0.748 | 0.813 | 0.796 | 0.700 |

What these numbers say:

* **The 70% goal is met, but only just.** Mean balanced accuracy is 70.7%, and 4 of the 10 CV repeats fall below 70% (range 0.63–0.78; `OUTPUT/figures/cv_balanced_accuracy_by_repeat.png`). The model clearly does better than chance (50%); whether it sits above or below 70% depends on how the data are split.
* **The signal carries over to a new site.** Training on two sites and testing on the third gives 75% balanced accuracy for every held-out site, with ROC-AUC 0.75–0.79. With 20 people per test site, one person moves balanced accuracy by 5 points, so these site-level numbers are rough.
* **The model leans toward calling people ASD.** Sensitivity (0.77) is higher than specificity (0.64) in CV.
* **Head motion does not explain the result.** The three people with mean FD above 0.15 mm were all ASD. Removing them leaves performance essentially unchanged (CV balanced accuracy 0.716, LOSO 0.748).
* **Limits.** 60 people, mostly male, ages 9–24, three sites; one fixed model setting (C = 1). These results are not evidence of a clinical biomarker.

## Section 1: Software and platform

* **Language:** Python 3 (tested with Python 3.13.18).
* **Packages** (listed in `requirements.txt`; versions used for the results above in brackets):
  * `numpy` (2.5.3): arrays and FC computation
  * `pandas` (3.0.5): tables
  * `scipy` (1.18.1): Welch's t-test, Fisher's exact test
  * `scikit-learn` (1.9.1): standardization, logistic regression, cross-validation, metrics
  * `matplotlib` (3.11.2, needs ≥ 3.9): figures
* **Platform:** developed on macOS; the results above were produced on Linux (Ubuntu). The scripts use only relative paths and run the same way on Windows, macOS or Linux. A full run takes under one minute on a laptop.

## Section 2: Map of documentation

```
DS4002/
├── README.md                          orientation (this file)
├── LICENSE.md                         MIT license for the code
├── requirements.txt                   Python packages to install
├── run_all.sh                         runs scripts 01–06 in order
├── SCRIPTS/
│   ├── utils.py                       shared paths, settings, model and evaluation functions (imported, not run)
│   ├── 01_check_data_and_build_features.py   data checks; builds 60 × 19,900 Fisher-z FC features
│   ├── 02_exploratory_analysis.py     cohort tables; age / sex / head-motion comparisons; EDA figures
│   ├── 03_repeated_cross_validation.py        main analysis: repeated stratified 5-fold CV
│   ├── 04_leave_one_site_out.py       cross-site test: train on 2 sites, test on the 3rd
│   ├── 05_motion_sensitivity.py       re-runs 03 and 04 without participants with mean FD > 0.15 mm
│   └── 06_results_summary.py          collects the main results into one table
├── DATA/
│   ├── README.md                      metadata: summary, provenance, license, ethics, data dictionary, plots
│   ├── initial/
│   │   ├── participants.csv           60 rows: subject_id, diagnosis, age, sex, site, mean_fd
│   │   ├── SHA256SUMS.txt             checksums for all initial data files
│   │   └── timeseries/                60 files, {subject_id}_rois_cc200.1D (time points × 200 regions)
│   └── final/                         made by script 01
│       ├── fc_features.npz            X (60 × 19,900 Fisher-z FC) and subject IDs
│       └── analysis_cohort.csv        metadata in the same row order as X
└── OUTPUT/
    ├── figures/
    │   ├── site_by_diagnosis_counts.png          participants per site and diagnosis
    │   ├── age_by_diagnosis.png                  age, ASD vs TDC
    │   ├── mean_fd_by_diagnosis.png              head motion, ASD vs TDC
    │   ├── mean_fd_by_site_and_diagnosis.png     head motion by site
    │   ├── mean_fc_matrix.png                    mean 200 × 200 FC matrix
    │   ├── cv_roc_curve.png                      ROC curve from repeated CV
    │   ├── cv_balanced_accuracy_by_repeat.png    balanced accuracy for each CV repeat vs. 70% goal
    │   ├── loso_performance_by_site.png          leave-one-site-out results
    │   └── motion_sensitivity_comparison.png     full vs. low-motion cohort
    └── tables/
        ├── data_quality_checks.csv               time points, non-finite values, constant regions per subject
        ├── cohort_summary_by_site_and_diagnosis.csv
        ├── group_comparisons_asd_vs_tdc.csv      age, sex and mean FD tests
        ├── cv_fold_metrics.csv                   50 rows (10 repeats × 5 folds)
        ├── cv_repeat_metrics.csv                 fold average per repeat
        ├── cv_summary.csv                        mean and SD across repeats
        ├── cv_out_of_fold_predictions.csv        per-subject ASD probability, averaged over repeats
        ├── loso_metrics_by_site.csv
        ├── loso_predictions.csv
        ├── motion_excluded_participants.csv      the 3 participants removed
        ├── motion_cohort_balance.csv             site × diagnosis counts before and after removal
        ├── motion_cv_fold_metrics.csv
        ├── motion_cv_summary.csv
        ├── motion_loso_metrics_by_site.csv
        ├── motion_sensitivity_comparison.csv
        └── main_results_summary.csv              the table at the top of this README
```

## Section 3: Instructions for reproducing the results

1. **Get the repository.** Clone it (`git clone https://github.com/YifanQiao5/DS4002.git`) or download it from GitHub with *Code → Download ZIP* and unzip it. All data are included, so nothing else needs to be downloaded.

2. **Open a terminal in the repository folder** (the folder that contains this `README.md`). Every command below is run from there.

3. **(Optional) Check the data files.** On macOS or Linux:
   ```bash
   cd DATA/initial && shasum -a 256 -c SHA256SUMS.txt && cd ../..
   ```
   Every line should end in `OK`.

4. **Install Python 3.10 or newer**, then install the packages (a virtual environment is recommended but not required):
   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

5. **Run the scripts in order.** Either run everything at once:
   ```bash
   bash run_all.sh
   ```
   or run them one by one (on Windows, use this form):
   ```bash
   python SCRIPTS/01_check_data_and_build_features.py
   python SCRIPTS/02_exploratory_analysis.py
   python SCRIPTS/03_repeated_cross_validation.py
   python SCRIPTS/04_leave_one_site_out.py
   python SCRIPTS/05_motion_sensitivity.py
   python SCRIPTS/06_results_summary.py
   ```
   If your system says `python: command not found` (common on macOS outside a virtual environment), type `python3` instead of `python`.
   Script 01 must run first, because it writes `DATA/final/`, which every later script reads. Scripts 03–05 can then run in any order. Script 06 needs the tables from 03, 04 and 05.

6. **Check the output.** All figures go to `OUTPUT/figures/` and all tables to `OUTPUT/tables/` (existing files are overwritten). Script 06 prints the main results table; it should match the table in *Main results* above. Every random step uses the fixed seed 42 (set in `SCRIPTS/utils.py`), so the numbers should come out identical. Small differences in the third decimal place are possible with other package versions.

7. **Changing the analysis.** All settings (number of folds and repeats, seed, logistic regression strength `C`, motion threshold) are defined once at the top of `SCRIPTS/utils.py`.

## References

[1] Project TIER, "TIER Protocol 4.0." [Online]. Available: https://www.projecttier.org/tier-protocol/protocol-4-0/

[2] A. Di Martino *et al.*, "The autism brain imaging data exchange: Towards a large-scale evaluation of the intrinsic brain architecture in autism," *Molecular Psychiatry*, vol. 19, no. 6, pp. 659–667, 2014, doi: 10.1038/mp.2013.78.

[3] Preprocessed Connectomes Project, "ABIDE Preprocessed." [Online]. Available: https://preprocessed-connectomes-project.github.io/abide/

[4] R. C. Craddock, G. A. James, P. E. Holtzheimer, X. P. Hu, and H. S. Mayberg, "A whole brain fMRI atlas generated via spatially constrained spectral clustering," *Human Brain Mapping*, vol. 33, no. 8, pp. 1914–1928, 2012, doi: 10.1002/hbm.21333.

[5] J. D. Power, K. A. Barnes, A. Z. Snyder, B. L. Schlaggar, and S. E. Petersen, "Spurious but systematic correlations in functional connectivity MRI networks arise from subject motion," *NeuroImage*, vol. 59, no. 3, pp. 2142–2154, 2012, doi: 10.1016/j.neuroimage.2011.10.018.

[6] F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
