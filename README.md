# A6P1 code - explainable decision automation for solar interconnection approvals

## How to run

```
pip install -r requirements.txt      # plus Graphviz (neato) for Fig. 1
python3 run_all.py                    # 6-12 min on 2 shared CPU cores, fixed seeds
```

`run_all.py` reads `../data/A6P1_solar_interconnection_apps.csv` and writes everything to `../results/` and `../figures/`.
`make_fig1.py` (called at the end of `run_all.py`) draws the methodology flowchart with Graphviz.

## What the pipeline does

1. Pre-processing fitted on each training fold: median imputation, winsorisation (0.5/99.5 %), and three screen features
   (remaining hosting capacity, transformer utilisation, PV/service ratio). The post-decision column `processing_days`
   and the IDs are never used as features.
2. Proposed **PS-GAM**: two-stage (review, then approve | not review) logistic P-spline GAM. It uses a double penalty
   per term and per-term smoothing parameters estimated with generalised Fellner-Schall updates, and is fitted by PIRLS
   with step halving.
3. Baselines: manual screening rules (grid-searched thresholds), multinomial logistic regression, and LightGBM
   with TreeSHAP (`pred_contrib`).
4. Decision layer: class weights for macro-F1 and an auto-approval threshold tau with validation FAR <= 2 %, both
   chosen on an inner 80/20 split.
5. Protocols: stratified 5-fold CV (mean +- std, paired t-tests), and a temporal drift test (train 2025, test 2026;
   bootstrap CI, McNemar).
6. Explanations: deletion AOPC, distractor (roof age) share, cross-fold stability, counterfactual recourse, an example
   with reason codes, and tau recalibration on January 2026.
7. Ablation (5-fold CV) and fixed-lambda sensitivity.

## Outputs (`../results/`)

| File | Content |
|---|---|
| metrics.json | all numbers quoted in the paper |
| table_cv_summary.csv / cv_fold_metrics.csv | 5-fold CV summary and per-fold metrics (Table III) |
| table_temporal_metrics.csv | 2026 drift test metrics with bootstrap CIs (Table III) |
| table_explanation_metrics.csv | AOPC, distractor share, stability, timing (Table IV) |
| table_ablation.csv / ablation_folds.csv | ablation study (Table V) |
| smoothing_sensitivity.csv | fixed-lambda sensitivity |
| automation_curves_temporal.csv | coverage vs false-auto-approval curves (Fig. 4) |
| deletion_curves.csv | deletion curves (Fig. 5) |
| gam_term_edf.csv, gam_global_importance.csv | per-term edf and mean absolute contributions of PS-GAM |
| example_reason_codes.csv | reason codes of the example application |
| confusion_gam_temporal.csv | confusion matrix of PS-GAM on 2026 |
| fig*.jpg | JPG copies of all figures (PNG 300 dpi in ../figures) |

fig6_confusion is produced as supplementary material; it is not used in the paper.
