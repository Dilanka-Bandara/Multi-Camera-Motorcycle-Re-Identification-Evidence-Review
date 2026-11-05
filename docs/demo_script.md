# MotoReID demo — 4 minutes 20 seconds

Use the packaged app at localhost:8080. Keep a completed fixture and the
preserved VnTraffic baseline available before presenting. Explain that baseline
screens show hand-set probabilities; final held-out calibration results are in
the README and reports. Do not portray uncertain matches as accepted identities.

| Time | Screen/action | Narration |
|---|---|---|
| 0:00–0:30 | Home, then README architecture | Two camera feeds become tracks with diverse crops, DINO appearance, motion and timing. Hungarian assignment can choose no match. A separate verifier checks geometry, parts, timing and contradictions. |
| 0:30–1:00 | Start Development fixture; show progress and completion | The fixture is deterministic and uses ground-truth detections. Real footage used YOLO on an RTX 3050. Detection ran at 17–29 FPS; full real runs took about nine minutes because crop processing and verification also cost time. |
| 1:00–1:45 | VnTraffic baseline, A-0000 → B-0018; click candidate and Play both; show crops/geometry | Both tracks map to annotated identity 39. This is a correct first-ranked retrieval, with 40 geometric inliers. Competing candidates and another query's claim keep the verdict uncertain despite a high baseline probability. This is not an accepted-match success claim. |
| 1:45–2:30 | A-0006 → B-0018; then A-0017 and its rank-2 candidate | A-0006 is identity 44, but the top candidate is identity 39. High visual similarity is contradicted by zero geometric inliers. A-0017's true identity is second. Both cases remain uncertain; returning the top result unconditionally would make mistakes. |
| 2:30–3:00 | Fixture A-0010 no reliable match (15 s); RoundaboutHD baseline A-0000 → B-0039 (15 s) | The pipeline can abstain at query level. RoundaboutHD supplies a real rejected candidate: 60 geometric inliers conflict with a -33.9 s adjusted transit. This is rejection under the configured transit rule, not a GT-confirmed different identity. Overlapping views can violate that assumption. |
| 3:00–3:20 | Development fixture header and evidence | The fixture estimates +3.567 s against a true +4 s clock offset using a documented travel-time prior. Clock offset and transit time cannot be separated without an assumption. VnTraffic's +6 s is configured, not estimated. |
| 3:20–4:00 | README held-out table and near-miss screenshots | The required training is logistic calibration, not retraining YOLO or DINO. Fourteen tuning pairs fitted five positive and nine negative examples. Thresholds used tuning queries only. Held-out Rank-1 is 10/14 on RoundaboutHD and 2/12 on VnTraffic. Neither accepted matches: precision is undefined and recall is zero. Small samples, annotation gaps and viewpoint limitations remain. |
| 4:00–4:20 | Download Results JSON, Candidates CSV and Queries CSV | Exports include ranked candidates, scores, evidence and verdicts. Model weights are baked into Docker for offline inference. Tests and reproducible evaluation reports accompany the project. |

## Presenter checks

- Use actual saved run IDs from the current run history. The baseline UI copy
  is `5d25ad36e97a46c89684f030f663ee10` in the development workspace.
- Show the result's configuration before discussing probabilities. Do not mix
  baseline screenshots with claims about the learned coefficients.
- Keep the held-out results visible long enough to read. Zero accepted matches
  is a measured limitation, not evidence that precision is perfect.
- That statement refers to GT-mapped held-out queries. The original RoundaboutHD
  baseline has one accepted query that cannot be mapped to GT (A-0002 to B-0052);
  its correctness is unknown and it is excluded from the scored metrics.
- Credit VnTraffic/HTV and the dataset authors; footage and screenshot
  modifications are described in `samples/vntraffic_synth/README.md`.
- The source VnTraffic clip is 17 seconds, with 18 seconds of blank delay added
  to B. The real pipeline does not process a full 60 seconds of distinct source
  content for this pair.
