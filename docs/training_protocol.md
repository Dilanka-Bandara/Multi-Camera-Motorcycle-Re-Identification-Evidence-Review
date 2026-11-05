# Phase 10 training and evaluation protocol

`BUILD_GUIDE.md` section 6.13, Phase 10 and Phase 11 govern this work. The
required training is scikit-learn logistic calibration of the independent
verifier. YOLO, DINOv2, ALIKED and LightGlue retain their pretrained weights.
Motorcycle-specific ReID training is explicitly a limitation in Phase 11.

## Data and split

Run the real YOLO pipeline on VnTraffic's synthetic camera pair and a
RoundaboutHD camera pair. Map predicted tracks to annotation identities using
the configured frame IoU and vote fraction. An unmapped track is not a labelled
negative for calibration. Evaluation excludes unmapped queries and reports their
count; unmapped gallery candidates still count as unsuccessful retrieval/decision
proposals but are excluded from ROC-AUC and ECE.

SHA-256 identity parity assigns approximately half the identities to tuning and
half to held-out evaluation. All query tracks belonging to one identity stay in
one partition. Calibration fitting additionally requires both identities of a
negative pair to belong to the tuning partition. This prevents held-out identity
features from entering the fitted logistic regression. Query partitions share
the original candidate gallery for threshold selection and evaluation; this is
not a benchmark with completely disjoint gallery identities. Assignment is
recomputed independently within each query partition. Report these conditions
alongside results rather than claiming a separate-camera generalization test.

The split is fixed before tuning. RoundaboutHD transit windows also use tuning
identities only. Its synchronized source videos use a documented zero clock
offset. VnTraffic's synthetic B feed has an 18-second video delay and a separate
6-second clock offset. The available VnTraffic source is only 17 seconds long;
the committed sample cannot supply the guide's requested approximately 60
seconds of distinct source footage.

## What is fitted

`scripts/fit_calibrator.py` fits the ten verifier features listed in
`configs/calibrator.json`, using `LogisticRegression`. Regularization, seed,
iteration limit and the threshold search grid come from
`configs/data_preparation.yaml`. Coefficients are serialized as JSON, never
pickle. No detector or embedding model is fine-tuned.

Select `min_match_score`, `ambiguity_margin`, `p_accept`, and `p_reject` using
tuning-query accepted F1, then accepted precision as the tie-breaker. Evaluate
the frozen coefficients and thresholds on held-out queries. Do not choose a
model or change the grid based on held-out scores. Every fit writes a manifest,
coefficients, thresholds, split identity/query lists, and metrics to a fresh
output directory so earlier experiments remain available.

After both real-data evaluations are complete, publish the selected learned
coefficients to `configs/calibrator.json` and the selected thresholds to the
relevant configuration. Record the training provenance and held-out results.
Until then the shipped coefficients remain explicitly provisional hand-set
defaults; Phase 10 is not accepted.

## Current evidence

The strict VnTraffic fit (`data/user/calibration/vntraffic-v3`) has four labelled
training pairs: two positives and two negatives. There are eight tuning queries
and fifteen held-out queries, with eleven unmapped queries excluded. This is
insufficient evidence for a robust calibration claim. It is a preliminary
experiment, not the final trained model. RoundaboutHD processing is in progress.

The VnTraffic baseline uses the original hand-set coefficients. Its corrected
labelled evaluation is `reports/vntraffic_baseline_labelled_metrics.json` and
the companion Markdown table. The earlier baseline report is preserved but
included unmapped queries in its no-match denominator; use the labelled report.
Do not describe the all-query baseline as held-out trained-model performance.

## Reproduction

Run from the repository root in the Python 3.11 environment with `backend` on
`PYTHONPATH`:

```sh
python scripts/fit_calibrator.py --run data/user/runs/vntraffic-gpu \
  --gt-a data/user/datasets/vntraffic_synth/gt_A.csv \
  --gt-b data/user/datasets/vntraffic_synth/gt_B.csv \
  --out data/user/calibration/vntraffic-next
python scripts/evaluate.py --run data/user/runs/vntraffic-gpu \
  --gt-a data/user/datasets/vntraffic_synth/gt_A.csv \
  --gt-b data/user/datasets/vntraffic_synth/gt_B.csv \
  --out reports/vntraffic_next_metrics.json
```

Use fresh output names. Keep datasets, model weights, environment files and raw
run artifacts out of Git. Only the attributed VnTraffic sample videos under
`samples/` are permitted video commits.
