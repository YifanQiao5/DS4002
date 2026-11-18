# README File for Data Establishment
## Summary
The working cohort comprises 60 de-identified ABIDE participants: 20 from each of NYU, UCLA_1, and UM_1, with 10 ASD and 10 TDC per site. The supplied participants.csv contains subject ID, diagnosis, age, recorded sex, site, and mean FD; the analysis notebook expects one CC200 ROI time-series file per participant (data/timeseries/{subject_id}_rois_cc200.1D). The derived image-like representation is a 200 × 200 FC matrix per subject. The team stores project data in Google Drive and code in a private GitHub repository. Access is restricted to authorized collaborators; folder contents and permissions should be verified.
## Provenance
ABIDE aggregates resting-state neuroimaging and phenotypic information collected by independent research sites [1]. This project uses a selected 60-person subset with preprocessed CC200 regional time series, not raw MRI volumes. Site-specific acquisition and preprocessing may vary; the notebook derives subject-wise FC from these ROI series. The exact selection rules, pipeline version, and source-file identifiers remain to be documented.
## License
ABIDE data are distributed through a research data-sharing initiative subject to source repository terms, acknowledgments, and privacy protections [1], [3]. Redistribution permissions for this processed copy have not been independently verified; participant-level data should not be publicly reposted before checking applicable terms.
## Data Dictionary
| Variable | Type / Meaning | Uncertainty / Considerations |
|---|---|---|
| `subject_id` | De-identified participant ID | Verify one-to-one match to time-series file |
| `diagnosis` | ASD or TDC | Existing study label, not independent verification |
| `age` | Age in years | Range 8.79–23.66; possible confounding |
| `sex` | Recorded F or M | Imbalanced; source-defined recording |
| `site` | NYU, UCLA_1, UM_1 | Scanner/protocol differences |
| `mean_fd` | Mean framewise displacement (mm) | Motion measure; definition is pipeline-dependent |
| CC200 time series | T × 200 matrix per subject | Time-point counts vary; verify missing ROI |
| FC matrix / vector | 200 × 200 Fisher-z FC / 19,900 edges | High-dimensional; motion/noise sensitive |

## Ethics
Only de-identified subject IDs will be used; the team will not attempt re-identification or interpret predictions as clinical diagnoses. The sample is small, age and recorded sex distributions may differ, and scanner/site and motion effects can produce spurious correlations. We will report these limitations, restrict source-file access, and avoid claims of a validated diagnostic biomarker.

## Exploratory Plots

<img width="651" height="620" alt="Screenshot 2026-11-18 at 2 19 28 PM" src="https://github.com/user-attachments/assets/a146d151-9d6e-4c68-8a02-de61f382ea0c" />

