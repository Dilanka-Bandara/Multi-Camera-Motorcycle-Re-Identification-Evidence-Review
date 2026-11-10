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

For the final fit, pool tuning pairs from RoundaboutHD and VnTraffic using
`--additional-training RUN GT_A GT_B`. Dataset identity namespaces are separate;
the same numeric ID in different datasets does not assert a shared vehicle.
Fit once, then use `--coefficients PATH` for the second preset's threshold tuning
without refitting. Both held-out query sets remain excluded from fitting and
threshold selection. This protocol was selected before viewing RoundaboutHD's
held-out scores; it is not a choice between models based on test performance.

The fitted coefficients are now published in `configs/calibrator.json`, and
both real-data presets contain the tuning-selected thresholds. The original
coefficients are preserved in `reports/calibrator_handset_baseline.json`.

## Development fixture calibration

The fixture preset replaces DINOv2 and LightGlue with a deterministic dummy
embedder and SIFT, so the pooled real-data coefficients do not describe its
features: they placed every fixture candidate between 0.41 and 0.51, outside
both decision thresholds. `verifier.calibrator` therefore lets a preset name its
own coefficient file, and the fixture uses `configs/calibrator_dev_fixture.json`.

`scripts/fit_fixture_calibrator.py` generates tuning fixtures from the seeds in
`configs/data_preparation.yaml` (`fixture_calibration`), runs the unmodified
pipeline on each, fits `LogisticRegression` on every reviewed pair and searches
regularization, `min_match_score`, `ambiguity_margin`, `p_accept` and `p_reject`.
It maximizes pooled accepted F1, then precision, then prefers the wider ambiguity
margin and the stronger regularization. Settings that leave either lookalike
pair unflagged on any tuning fixture are discarded. The script refuses to run if
the baked seed appears among the tuning seeds.

The published fit used seeds 101, 202 and 303: 108 pairs, 26 positive. It chose
C = 10, minimum match score 0.65, ambiguity margin 0.03, accept 0.50 and reject
0.10, with 10 of 27 tuning matches accepted and none false. On the baked seed-42
fixture, evaluated after selection, 3 of 9 true matches are accepted with no
false accept, 16 wrong candidates are rejected, Rank-1 is 9/9, ROC-AUC 0.959 and
ECE 0.129. All fixtures share the same 14 sprite designs and differ in transit
times and noise; this is not evidence about unseen vehicles or real footage.

```sh
python scripts/fit_fixture_calibrator.py \
  --work data/user/calibration/fixture-next-work --out data/user/calibration/fixture-next
```

## Current evidence

The final pooled fit (`data/user/calibration/pooled-v1`) has fourteen labelled
training pairs: five positives and nine negatives. RoundaboutHD supplies ten
pairs (three positive); VnTraffic supplies four (two positive). RoundaboutHD has
eleven tuning queries and twenty-six held-out queries; VnTraffic has eight and
fifteen respectively. The earlier VnTraffic-only experiments remain preserved.
This is a small calibration sample, not evidence of robust deployment accuracy.

Both tuning grids select minimum match score 0.45, ambiguity margin 0.02,
accept probability 0.50 and reject probability 0.10. Accepted F1 stays zero,
so these are the first tied grid settings, not demonstrated optimal thresholds.
Held-out Rank-1 is 10/14 on RoundaboutHD and 2/12 on VnTraffic; neither has
accepted matches. ECE is 0.164 and 0.290 respectively. Full manifests, tuning
metrics and held-out metrics are in `reports/`. Reports replay the raw features
from actual YOLO runs; FPS and wall time refer to those source pipeline runs.

The VnTraffic baseline uses the original hand-set coefficients. Its corrected
labelled evaluation is `reports/vntraffic_baseline_labelled_metrics.json` and
the companion Markdown table. The earlier baseline report is preserved but
included unmapped queries in its no-match denominator; use the labelled report.
Do not describe the all-query baseline as held-out trained-model performance.

## Reproduction

Run from the repository root in the Python 3.11 environment with `backend` on
`PYTHONPATH`:

```sh
python scripts/fit_calibrator.py --run data/user/runs/roundabouthd-gpu \
  --gt-a data/user/datasets/roundabouthd_pair/gt_A.csv \
  --gt-b data/user/datasets/roundabouthd_pair/gt_B.csv \
  --additional-training data/user/runs/vntraffic-gpu \
    data/user/datasets/vntraffic_synth/gt_A.csv data/user/datasets/vntraffic_synth/gt_B.csv \
  --out data/user/calibration/pooled-next
python scripts/fit_calibrator.py --run data/user/runs/vntraffic-gpu \
  --gt-a data/user/datasets/vntraffic_synth/gt_A.csv \
  --gt-b data/user/datasets/vntraffic_synth/gt_B.csv \
  --coefficients data/user/calibration/pooled-next/calibrator.json \
  --out data/user/calibration/vntraffic-next
python scripts/evaluate.py --run data/user/runs/vntraffic-gpu \
  --gt-a data/user/datasets/vntraffic_synth/gt_A.csv \
  --gt-b data/user/datasets/vntraffic_synth/gt_B.csv \
  --out reports/vntraffic_next_metrics.json
```

Use fresh output names. Keep datasets, model weights, environment files and raw
run artifacts out of Git. Only the attributed VnTraffic sample videos under
`samples/` are permitted video commits.
