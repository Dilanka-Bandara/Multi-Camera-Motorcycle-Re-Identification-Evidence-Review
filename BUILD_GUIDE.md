# MotoReID — Codex Build Guide

Multi-Camera Motorcycle Re-Identification & Evidence Review
Deadline: 8 November 2026

This file is the single source of truth for building the project. Build it **phase by phase**, in order. Each phase lists what to build, the exact behaviour required, and acceptance checks. Do not start a phase until the previous phase's checks pass.

---

## 0. How to use this guide (for the human)

1. Create an empty Git repo named `moto-reid`, put this file in the root as `BUILD_GUIDE.md`.
2. Create `AGENTS.md` in the root with the text in section 0.1 (Codex reads `AGENTS.md` automatically).
3. Give Codex one phase at a time, e.g.:
   `Read BUILD_GUIDE.md and AGENTS.md. Implement Phase 2 exactly as specified. Run the acceptance checks and update PROGRESS.md.`
4. After each phase: run the checks yourself, commit (`git commit -m "phase N: ..."`), then move on.
5. Real datasets are only touched in **Phase 10**. Everything before that runs on a synthetic fixture the code generates itself.

Suggested schedule:

| Day | Phases |
|---|---|
| 4 Nov | 0, 1, 2 |
| 5 Nov | 3, 4, 5 |
| 6 Nov | 6, 7, 8 |
| 7 Nov | 9, 10 |
| 8 Nov | 11, final checks, submit |

### 0.1 `AGENTS.md` content

```text
# Agent rules for this repo
- BUILD_GUIDE.md is the specification. Follow it phase by phase. Do not skip ahead.
- Python 3.11, type hints everywhere, ruff-clean, pytest must pass at the end of every phase.
- Never add paid APIs or credentials. Everything must run offline after the Docker build.
- Never commit data/, models/ weights, or videos (except files under samples/ explicitly allowed).
- Keep pure logic (matching, temporal, verifier, metrics) free of torch/cv2 imports so it is unit-testable.
- All thresholds come from configs/*.yaml, never hard-coded in logic.
- Store bounding boxes in normalized [0,1] coordinates everywhere outside the detector module.
- After each phase, append a short entry to PROGRESS.md: what was built, how to test, known gaps.
- If a library API differs from this guide, adapt to the installed version and note it in PROGRESS.md.
```

---

## 1. Requirements recap (from the assignment)

The system must:

1. Detect motorcycles in two CCTV videos and build **track-level** representations (not single frames).
2. Match motorcycles across cameras despite appearance, lighting, resolution, viewpoint, occlusion and track-ID changes.
3. Use timestamps, movement direction and plausible travel time, including **unknown camera-clock offset**.
4. Rank candidates, flag **lookalikes**, and return **"no reliable match"** instead of always picking the top candidate.
5. Have a **separate, independent verifier** that checks visual similarity, timing, tracking quality, missing information and contradictions.
6. Explain every result: supporting and contradicting evidence, competing candidates, and a status of **accepted / uncertain / rejected**.

Deliverables:

- Working app processing two feeds.
- Functional web UI: side-by-side video playback, tracks, candidate comparison, ranked matches, evidence, statuses.
- `Dockerfile` + `compose.yaml`; `docker compose up --build` → UI at `http://localhost:8080`; no paid credentials.
- JSON + CSV export: candidate pairs, similarity scores, evidence, verdicts, confidence.
- Git repo + README: run instructions, automated tests, design choices, limitations.
- 3–5 min demo: correct re-ID, similar-but-different bikes, rejected and uncertain cases, metrics, false matches, processing speed.

---

## 2. Tech stack

| Area | Choice | Notes |
|---|---|---|
| Language | Python 3.11 | |
| Detection | Ultralytics YOLO26 (`yolo26s.pt`) | COCO classes: person=0, car=2, motorcycle=3, bus=5, truck=7. Fallback to `yolo11s.pt` if YOLO26 weights fail to load. |
| Tracking | Ultralytics built-in BoT-SORT (`botsort.yaml`) | ByteTrack (`bytetrack.yaml`) selectable in config. |
| Appearance embedding | DINOv2 ViT-S/14 via `transformers` (`facebook/dinov2-small`) | Public, Apache-2.0, no token. |
| Colour features | OpenCV HSV + Lab histograms | |
| Verifier keypoints | LightGlue + ALIKED (`pip install git+https://github.com/cvg/LightGlue.git`) | Do NOT use SuperPoint (restrictive license). Fallback: OpenCV SIFT + ratio test + RANSAC. |
| Assignment | `scipy.optimize.linear_sum_assignment` | |
| Calibration | `scikit-learn` LogisticRegression | Save coefficients as JSON, never pickle. |
| Backend | FastAPI + Uvicorn + Pydantic v2 | |
| Storage | Filesystem per run (`data/runs/<run_id>/`) with JSON files | No database needed. |
| Video tools | ffmpeg / ffprobe | Transcode to browser-safe H.264 MP4. |
| Frontend | React + Vite + TypeScript + Tailwind CSS + TanStack Query | |
| Tests | pytest, pytest-asyncio, httpx; Vitest optional for UI | |
| Lint | ruff | |
| Container | Single image, multi-stage (Node build → Python runtime), port 8080 | CPU default; GPU via override file. |

Hardware target: works on CPU; accelerated on an NVIDIA RTX 3050 Laptop (4 GB VRAM). On GPU use FP16 and run stages sequentially to stay under 4 GB.

---

## 3. Repository structure

```text
moto-reid/
├── AGENTS.md
├── BUILD_GUIDE.md
├── PROGRESS.md
├── README.md
├── Dockerfile
├── compose.yaml
├── compose.gpu.yaml
├── .dockerignore
├── .gitignore
├── pyproject.toml
├── configs/
│   ├── default.yaml
│   ├── presets/
│   │   ├── dev_fixture.yaml
│   │   ├── vntraffic_synth.yaml        # Phase 10
│   │   └── roundabouthd_pair.yaml      # Phase 10
│   └── calibrator.json                 # default/learned calibration coefficients
├── backend/
│   └── app/
│       ├── main.py                     # FastAPI app, static frontend, routers
│       ├── settings.py                 # env + paths
│       ├── config.py                   # YAML config loading + Pydantic config models
│       ├── schemas.py                  # all Pydantic data models
│       ├── api/
│       │   ├── runs.py
│       │   ├── media.py                # range-request video serving
│       │   └── presets.py
│       ├── jobs/
│       │   └── runner.py               # background job execution + progress
│       ├── pipeline/
│       │   ├── orchestrator.py         # runs all stages for a run
│       │   ├── ingest.py               # ffprobe, transcode, timestamps
│       │   ├── detect_track.py         # YOLO + tracker (real mode)
│       │   ├── gt_detector.py          # ground-truth "detector" (dev/test mode)
│       │   ├── rider_assoc.py          # attach rider (person) boxes to motorcycles
│       │   ├── stitching.py            # merge fragmented tracks within a camera
│       │   ├── crops.py                # crop extraction + quality scoring + top-K
│       │   ├── track_quality.py
│       │   ├── motion.py               # direction vectors, entry/exit
│       │   └── representation.py       # builds TrackRepresentation
│       ├── features/
│       │   ├── embedder.py             # DINOv2 embedder + DummyEmbedder
│       │   ├── color.py
│       │   └── keypoints.py            # LightGlue/ALIKED + SIFT fallback
│       ├── matching/
│       │   ├── temporal.py             # transit model, clock-offset estimation
│       │   ├── similarity.py           # set-to-set visual + colour similarity
│       │   ├── candidates.py           # candidate generation + scoring + ranking
│       │   └── assignment.py           # Hungarian with "no match" option, ambiguity flags
│       ├── verify/
│       │   ├── checks.py               # independent evidence checks
│       │   ├── calibrator.py           # logistic calibration
│       │   └── policy.py               # accepted / uncertain / rejected decision
│       ├── export/
│       │   └── exporters.py            # JSON + CSV
│       ├── evaluation/
│       │   ├── gt.py                   # GT loading + track→identity mapping
│       │   └── metrics.py
│       └── models_download.py          # pre-download weights at Docker build
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── api/client.ts
│       ├── types.ts                    # mirrors backend schemas
│       ├── pages/ (HomePage.tsx, RunPage.tsx, MetricsPage.tsx)
│       └── components/
│           ├── NewRunForm.tsx
│           ├── DualVideoPlayer.tsx
│           ├── TrackOverlayCanvas.tsx
│           ├── PresenceTimeline.tsx
│           ├── TrackList.tsx
│           ├── CandidateTable.tsx
│           ├── CompareView.tsx
│           ├── EvidencePanel.tsx
│           ├── StatusBadge.tsx
│           └── ExportButtons.tsx
├── scripts/
│   ├── make_dev_fixture.py             # synthetic two-camera fixture (Phase 1)
│   ├── run_pipeline.py                 # CLI: run a config end-to-end
│   ├── evaluate.py                     # metrics vs ground truth
│   ├── fit_calibrator.py               # Phase 10
│   ├── download_data.py                # Phase 10
│   ├── make_synthetic_pair.py          # Phase 10 (split one real video into 2 cameras)
│   └── convert_roundabouthd.py         # Phase 10
├── samples/                            # small committed demo clips (Phase 10, CC-BY only)
├── tests/
│   ├── conftest.py
│   ├── unit/ ...
│   └── integration/ ...
└── data/                               # gitignored: inputs, runs, datasets
```

---

## 4. Configuration (`configs/default.yaml`)

Create this file in Phase 1. All modules read their thresholds from it. Presets override parts of it (deep merge). The UI can override a few fields per run.

```yaml
target:
  classes: [motorcycle]          # or [car, motorcycle, bus, truck] for "all_vehicles" mode
  rider_class: person

detector:
  mode: yolo                     # yolo | groundtruth
  weights: yolo26s.pt
  fallback_weights: yolo11s.pt
  imgsz: 960
  conf: 0.25
  iou: 0.5
  half: auto                     # true on CUDA, false on CPU
  device: auto                   # cuda if available else cpu
  max_process_height: 1080       # downscale input before detection
  vid_stride: auto               # 1 if fps<=15, 2 if fps<=30, else round(fps/15)

tracker:
  type: botsort.yaml             # or bytetrack.yaml
  min_track_frames: 6
  stitching:
    enabled: true
    max_gap_s: 2.0
    max_center_jump: 0.15        # normalized distance
    min_appearance_sim: 0.75

crops:
  top_k: 8
  min_box_height_px: 32
  min_frame_gap: 3               # temporal diversity between chosen crops
  border_margin: 0.01            # box touching border within this margin = truncated
  occlusion_iou: 0.3
  weights: {size: 0.35, conf: 0.25, sharpness: 0.30, aspect: 0.10}

embedding:
  backend: dinov2                # dinov2 | dummy
  model: facebook/dinov2-small
  batch_size: 32
  parts: true                    # rider + vehicle part embeddings

color:
  hsv_bins: [16, 4, 4]

cameras:
  A:
    name: Camera A
    start_time_s: 0.0            # absolute time of frame 0 (seconds, any shared epoch)
  B:
    name: Camera B
    start_time_s: 0.0

topology:
  direction: A_to_B              # A_to_B | B_to_A | both
  min_transit_s: 5.0
  max_transit_s: 60.0
  expected_exit_dir_A: null      # e.g. [1, 0] = moving right in image A; null = unknown
  expected_entry_dir_B: null
  clock_offset:
    mode: estimate               # estimate | fixed | none
    fixed_s: 0.0
    search_range_s: 120.0
    min_confident_pairs: 3
    confident_visual_sim: 0.80

matching:
  top_n_candidates: 5
  set_sim_top_m: 5
  weights: {visual: 0.55, color: 0.15, temporal: 0.20, direction: 0.10}
  temporal_soft_tau_s: 10.0
  min_match_score: 0.55          # cost of choosing "no match" in assignment
  ambiguity_margin: 0.05         # runner-up within this = lookalike flag

verifier:
  keypoints:
    backend: lightglue           # lightglue | sift
    max_keypoints: 512
    pairs_to_check: 3
    min_inliers_pass: 15
    max_inliers_fail: 4
  part_disagreement_gap: 0.20    # vehicle similar but rider very different (or vice versa)
  min_track_quality: 0.35
  p_accept: 0.75
  p_reject: 0.35
  review_top_n: 3                # verifier reviews top-N candidates per query, not only the winner

runtime:
  data_dir: /app/data
  seed: 42
```

---

## 5. Data models (`backend/app/schemas.py`)

Implement with Pydantic v2. Mirror them in `frontend/src/types.ts`.

```text
BBox: x1, y1, x2, y2 (normalized floats 0..1)

Detection: frame_idx, t_s (absolute time incl. camera start), bbox, conf, cls, track_id (per camera, raw)

CropInfo: frame_idx, t_s, bbox, quality (0..1), sharpness, truncated(bool), occluded(bool),
          path_full, path_rider (optional), path_vehicle

Track:
  track_id (string, e.g. "A-0007"), camera ("A"|"B"),
  raw_track_ids (list[int], after stitching),
  first_frame, last_frame, t_start_s, t_end_s,
  boxes: list[{frame_idx, bbox}]       # for UI overlay
  n_frames, mean_conf, median_box_h_px,
  entry_point, exit_point (normalized xy), motion_vector (unit [dx, dy]),
  quality (0..1) + quality_breakdown dict,
  crops: list[CropInfo] (top-K),
  has_rider: bool

TrackRepresentation (not exported in full; stored as .npz):
  track_id, emb_full (K x D), emb_full_mean (D), emb_rider (K x D | null), emb_vehicle (K x D),
  color_hist (vector), lab_mean (3)

EvidenceItem:
  check: str             # e.g. "keypoint_inliers"
  polarity: "supports" | "contradicts" | "neutral" | "missing"
  severity: "low" | "medium" | "high" | "critical"
  value: float | str | null
  threshold: float | str | null
  message: str           # human-readable, shown in UI

Candidate:
  pair_id ("A-0007__B-0012"), query_track_id (A side), candidate_track_id (B side),
  rank, scores: {visual, color, temporal, direction, combined},
  delta_t_s (raw), delta_t_adjusted_s (after clock offset),
  ambiguous: bool, competing_pair_ids: list[str],
  assigned: bool         # chosen by Hungarian assignment
  verification: Verification | null

Verification:
  status: "accepted" | "uncertain" | "rejected",
  confidence: float (calibrated probability 0..1),
  features: dict[str, float],
  evidence: list[EvidenceItem],
  summary: str,
  keypoint_viz_path: str | null

QueryResult:   # one per Camera-A track (and per B track if direction=both)
  query_track_id, outcome: "matched" | "uncertain" | "no_reliable_match",
  best_pair_id | null, candidates: list[Candidate] (ranked)

RunStatus: run_id, state (queued|running|done|failed), stage, progress (0..1),
           error | null, timings_s: dict[stage, seconds], fps_processed: dict

RunResult: run_id, created_at, config_snapshot, videos (meta per camera),
           clock_offset: {mode, estimated_s, n_pairs_used, reliable: bool},
           tracks_A, tracks_B, queries: list[QueryResult], metrics | null, timings
```

---

## 6. Pipeline design (what each module must do)

### 6.1 Ingest (`ingest.py`)
- `ffprobe` each video: fps, frame count, width, height, duration.
- Create a **web copy**: `ffmpeg -i in -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -movflags +faststart -an -vf "scale=-2:'min(1080,ih)'" -fps_mode passthrough web.mp4`. Browsers cannot play `mp4v`; always serve this web copy.
- Timestamp of frame i = `cameras.<X>.start_time_s + i / fps`.

### 6.2 Detection and tracking (`detect_track.py`)
- Load YOLO with fallback weights. Detect target classes **and** `person` (for riders).
- Use `model.track(source=video, stream=True, persist=True, tracker=cfg, classes=[...], imgsz, conf, iou, half, vid_stride, device, verbose=False)`. Iterate results; read `boxes.xyxyn`, `boxes.conf`, `boxes.cls`, `boxes.id` (may be None).
- Keep only boxes with a track id. Record per-frame detections for target classes and riders separately.
- Measure and store FPS for this stage.

### 6.3 Ground-truth detector (`gt_detector.py`) — dev/test mode
- Reads a GT boxes CSV (`frame_idx, identity_id, x1, y1, x2, y2, cls` normalized) and emits detections as if from the tracker, using `identity_id` as raw track id.
- Options: `drop_rate` (randomly drop detections) and `fragment_prob` (split a track into two raw ids at a random gap) so stitching gets exercised.

### 6.4 Rider association (`rider_assoc.py`)
- For each motorcycle box in a frame, find the person box whose bottom half overlaps the motorcycle box (overlap / person area ≥ 0.3 and horizontal centre within motorcycle x-range). Pick the best one.
- Output per frame: `vehicle_box`, `rider_box | None`, `union_box`. The **union box** is the crop used for full-object embeddings.
- If no rider is found: rider part is missing (record it; it becomes "missing" evidence later).

### 6.5 Stitching (`stitching.py`)
- Within a camera, merge raw track B into A if: B starts within `max_gap_s` after A ends, centre jump ≤ `max_center_jump`, and mean-embedding cosine ≥ `min_appearance_sim` (use colour histogram if embeddings not yet computed). Greedy, earliest first.

### 6.6 Crops and quality (`crops.py`, `track_quality.py`)
- Crop quality per frame: weighted sum of normalized box height, detector conf, Laplacian-variance sharpness, aspect plausibility; minus penalties if truncated at border or occluded (IoU with another box > `occlusion_iou`).
- Choose top-K crops with at least `min_frame_gap` frames between them. Save JPEGs: full (union), rider, vehicle.
- Track quality (0..1): combine track length, mean conf, mean crop quality, fraction truncated, fraction occluded, median size. Keep the breakdown dict for evidence.
- Drop tracks shorter than `min_track_frames` or with no crop above `min_box_height_px`. Log dropped counts.

### 6.7 Motion (`motion.py`)
- Entry point = mean centre over first 3 boxes; exit point = mean over last 3. Motion vector = normalized (exit − entry). If the track barely moves, vector = [0, 0] and direction evidence becomes "missing".

### 6.8 Features
- `embedder.py`: `DinoEmbedder` — batch crops, resize via `AutoImageProcessor`, take CLS token and mean of patch tokens, concatenate, L2-normalize. FP16 autocast on CUDA. `DummyEmbedder` — deterministic vector from colour histogram + small CNN-free stats (for tests, no torch download).
- `color.py`: HSV histogram (normalized) on the vehicle crop centre region (inner 80%); mean Lab colour. Similarity = 1 − Bhattacharyya distance.

### 6.9 Temporal model and clock offset (`temporal.py`)
- Raw delta for pair (a in A, b in B): `Δ = b.t_start_s − a.t_end_s` (for A→B).
- **Clock-offset estimation** (`mode: estimate`):
  1. Take pairs where visual set-similarity ≥ `confident_visual_sim` AND they are mutual best matches visually.
  2. Their raw Δ values are `true_transit + offset`. Compute the mode with a 1-D kernel density (bandwidth 2 s) within `±search_range_s`.
  3. `offset_estimate = mode − median(min_transit_s, max_transit_s)` only if the mode falls outside the configured window; otherwise offset = 0.
  4. If fewer than `min_confident_pairs`, set `reliable=false`, offset = 0, and **widen** the window by `search_range_s`. Every candidate then gets a "timing uncertain: clock offset unknown" evidence item.
- Adjusted delta `Δ' = Δ − offset`. Temporal score: 1.0 if `min ≤ Δ' ≤ max`; else `exp(−(d/τ)²)` where d is distance to the window; `Δ' < −2 s` is **impossible** (score 0, flagged critical).
- Direction score: if expected vectors configured, cosine between track motion vector and expected vector mapped to 0..1; if unknown, 0.5 and evidence "missing".

### 6.10 Similarity (`similarity.py`)
- Visual set-to-set similarity between two tracks: compute K×K cosine matrix of crop embeddings, take the mean of the top-`m` values; average with cosine of mean embeddings. This is robust to a few bad crops.
- Part similarities (rider, vehicle) computed the same way when parts exist.

### 6.11 Candidates and ranking (`candidates.py`)
- For each query track in A, consider all B tracks whose Δ' is not impossible (keep impossible ones only if visual ≥ 0.85, so the UI can show "looks identical but timing impossible").
- Combined score = weighted sum (config weights). Keep top `top_n_candidates`, ranked.

### 6.12 Assignment and ambiguity (`assignment.py`)
- Build cost matrix `1 − combined`. Append one dummy "no match" column per query with cost `1 − min_match_score`. Solve with Hungarian. A query assigned to dummy → proposed "no match".
- Ambiguity flags: query's runner-up within `ambiguity_margin` of top; or the same B track is top-1 for more than one A track. Fill `competing_pair_ids`.

### 6.13 Independent verifier (`verify/`)
The verifier must not trust the matcher's combined score. It recomputes its own evidence from raw data. It reviews the top `review_top_n` candidates of every query (so the UI can show why competitors lost).

Checks (each emits one or more `EvidenceItem`):

| Check | How | Supports when | Contradicts when |
|---|---|---|---|
| `keypoint_inliers` | LightGlue+ALIKED on best `pairs_to_check` crop pairs (vehicle crops), RANSAC fundamental matrix, take max inliers. Save a match visualization PNG. | inliers ≥ `min_inliers_pass` | inliers ≤ `max_inliers_fail` (medium) |
| `vehicle_part_similarity` | part set-similarity, vehicle | ≥ 0.80 | < 0.55 (high) |
| `rider_part_similarity` | part set-similarity, rider | ≥ 0.75 | < 0.45 (medium; riders can change posture, passengers) |
| `part_disagreement` | |vehicle − rider| > `part_disagreement_gap` | — | yes (medium): "same-looking bike, different rider" or reverse |
| `color_consistency` | Lab mean ΔE (after per-camera grey-world normalization) and histogram similarity | ΔE < 12 | ΔE > 30 (high) |
| `timing_plausibility` | recompute Δ' from raw timestamps and clock offset | inside window | outside (medium) / impossible (critical) |
| `clock_offset_reliability` | from temporal module | reliable | — ("missing" if unreliable) |
| `direction_consistency` | motion vectors vs topology | cosine ≥ 0.5 | cosine < −0.3 (high) |
| `track_quality_A`, `track_quality_B` | track quality score | ≥ 0.6 | < `min_track_quality` (medium) |
| `missing_info` | no rider part, too few crops (<3), motion unknown, no keypoints extracted | — | each → "missing" (low/medium) |
| `ambiguity` | margin to runner-up and competing claims | margin ≥ 2×`ambiguity_margin` | flagged ambiguous (high) |
| `assignment_conflict` | B track also proposed for another A with higher score | — | yes (high) |

**Calibration** (`calibrator.py`): feature vector = [visual, vehicle_part, rider_part (or 0.5 if missing), color_sim, keypoint_inliers_norm, temporal, direction, track_quality_min, margin, missing_count]. Probability = logistic(w·x + b). Load coefficients from `configs/calibrator.json` (ship hand-set defaults in Phase 5; refit from labels in Phase 10).

**Decision policy** (`policy.py`), in this order:
1. Any `critical` contradiction → **rejected**.
2. `p < p_reject` → **rejected**.
3. `p ≥ p_accept` AND no `high` contradictions AND not ambiguous AND both track qualities ≥ `min_track_quality` AND keypoint check not contradicting → **accepted**.
4. Otherwise → **uncertain**.
- Write a one-sentence `summary` built from the top 2 supporting and top 2 contradicting items.

**Query outcome**: `matched` if the assigned candidate is accepted; `uncertain` if the best reviewed candidate is uncertain; `no_reliable_match` if all reviewed candidates are rejected or the assignment chose "no match" and no candidate is accepted.

### 6.14 Export (`exporters.py`)
- `results.json`: full `RunResult`.
- `candidates.csv` columns: `run_id, query_track_id, candidate_track_id, rank, assigned, visual, color, temporal, direction, combined, delta_t_s, delta_t_adjusted_s, ambiguous, status, confidence, supporting_evidence, contradicting_evidence, missing_evidence, summary` (evidence columns = `;`-joined `check:message`).
- `queries.csv`: `query_track_id, outcome, best_candidate_track_id, confidence`.

### 6.15 Metrics (`evaluation/`)
- GT format (unified for all sources): per camera a CSV `frame_idx, identity_id, x1, y1, x2, y2` normalized.
- Map each predicted track to a GT identity by majority vote of IoU ≥ 0.5 matches over its frames (unmapped if < 50% of frames match).
- Metrics: precision / recall / F1 of **accepted** matches; same counting uncertain as positive; Rank-1 and Rank-5 accuracy for queries that have a true match; **correct-rejection rate** (queries with no true match that ended `no_reliable_match` or `uncertain`); false-match list (pair ids, scores, evidence); confidence ROC-AUC and expected calibration error (10 bins); processing FPS per stage and total wall time.

---

## 7. API (`backend/app/api/`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | `{status:"ok", device, models_loaded}` |
| GET | `/api/presets` | List preset configs whose data exists on disk |
| POST | `/api/runs` | Multipart: `video_a`, `video_b`, optional `config_overrides` (JSON) → `{run_id}` |
| POST | `/api/runs/from-preset` | `{preset: "dev_fixture" \| ...}` → `{run_id}` |
| GET | `/api/runs` | List runs with status |
| GET | `/api/runs/{id}/status` | `RunStatus` (poll every 1 s) |
| GET | `/api/runs/{id}` | `RunResult` (without heavy boxes) |
| GET | `/api/runs/{id}/tracks/{camera}` | Tracks incl. boxes for overlay |
| GET | `/api/runs/{id}/queries/{track_id}` | Ranked candidates + verification for one query |
| GET | `/api/runs/{id}/metrics` | Metrics if GT available |
| GET | `/api/runs/{id}/export.json` / `export.csv` / `queries.csv` | Downloads |
| GET | `/media/runs/{id}/...` | Web videos, crops, keypoint PNGs — **must support HTTP Range (206)** |

Jobs run in a background thread pool (max 1 concurrent pipeline to protect 4 GB VRAM). Progress updates per stage. Results cached by SHA-256 of both videos + config, so re-running the same input is instant.

---

## 8. Frontend (`frontend/`)

Pages:

1. **Home**: New run form — upload two videos **or** pick a preset; fields: target mode (motorcycle / all vehicles), min/max transit seconds, direction, clock offset mode (estimate / fixed value / none). List of previous runs with status.
2. **Run page** (main screen):
   - Top: `DualVideoPlayer` — two `<video>` elements side by side, shared play/pause/seek, a clock-offset slider that shifts camera B playback. `TrackOverlayCanvas` draws boxes from track JSON using `requestVideoFrameCallback` (fallback `requestAnimationFrame` + `currentTime`). Frame index = `floor(currentTime * fps)`. Selected track highlighted; others thin.
   - Below the players: `PresenceTimeline` — horizontal bars of when each track is visible per camera; clicking a bar seeks both videos.
   - Left column: `TrackList` for Camera A with outcome badges (matched / uncertain / no reliable match) and filters.
   - Centre: `CandidateTable` — ranked candidates for the selected query: rank, B track, combined score, Δt, status badge, ambiguous icon. Clicking a row selects it and seeks Camera A to the query and Camera B to the candidate.
   - Right: `CompareView` — two crop strips (query vs candidate), part similarities, keypoint match image, then `EvidencePanel`: green "supports", red "contradicts", grey "missing", with severity and message; verdict badge + confidence + summary sentence. A "competing matches" section lists other queries claiming the same B track.
   - Header: run summary — counts of accepted / uncertain / rejected, estimated clock offset (and whether reliable), processing time, export buttons.
3. **Metrics page**: metric cards, false-match table, per-stage timing table.

Style: clean, dark-friendly, Tailwind. Status colours: accepted = green, uncertain = amber, rejected = red. Never rely on colour alone — badge text always shown.

---

## 9. Docker

`Dockerfile` (multi-stage):

```text
Stage 1 (node:20-alpine): copy frontend/, npm ci, npm run build → /frontend/dist
Stage 2 (python:3.11-slim):
  - apt: ffmpeg, libgl1, libglib2.0-0, git, curl
  - ARG TORCH_VARIANT=cpu  (gpu override passes cu124 or another cu12x matching the driver)
  - pip install torch torchvision --index-url https://download.pytorch.org/whl/${TORCH_VARIANT}
    (install torch FIRST so ultralytics does not pull a different build)
  - pip install the project (pyproject) + LightGlue from git
  - ENV HF_HOME=/app/models/hf, YOLO weights dir /app/models
  - RUN python -m app.models_download   (YOLO26 + fallback, DINOv2-small, ALIKED + LightGlue weights)
  - ENV HF_HUB_OFFLINE=1 at runtime
  - RUN python scripts/make_dev_fixture.py --out /app/data/fixtures/dev  (so a demo works with zero downloads)
  - copy frontend dist into /app/static
  - HEALTHCHECK curl /api/health
  - CMD uvicorn app.main:app --host 0.0.0.0 --port 8080
```

`compose.yaml`:

```yaml
services:
  app:
    build:
      context: .
      args:
        TORCH_VARIANT: cpu
    ports:
      - "8080:8080"
    volumes:
      - ./data:/app/data/user
    restart: unless-stopped
```

`compose.gpu.yaml` (used as `docker compose -f compose.yaml -f compose.gpu.yaml up --build`):

```yaml
services:
  app:
    build:
      args:
        TORCH_VARIANT: cu124
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Note: the fixture lives at `/app/data/fixtures` inside the image; user data and runs go to the mounted volume `/app/data/user` so the volume does not hide the baked fixture.

---

## 10. Build phases

Each phase ends with: `ruff check .`, `pytest -q` passing, and a `PROGRESS.md` entry.

### Phase 0 — Skeleton that runs in Docker
Build:
- `pyproject.toml` (deps from section 2), `.gitignore` (data/, models/, node_modules, dist, *.mp4 except samples/), `.dockerignore`.
- FastAPI app with `/api/health`, serving a Vite React "Hello MotoReID" page from `/`.
- Dockerfile + compose.yaml + compose.gpu.yaml as in section 9 (weights download step can be a stub for now).
Acceptance:
- `docker compose up --build` → `http://localhost:8080` shows the page; `/api/health` returns ok.
- `pytest -q` runs (one health test).

### Phase 1 — Config, schemas, synthetic dev fixture, GT detector
Build:
- `configs/default.yaml` (section 4), config loader with deep-merge presets and per-run overrides; Pydantic config models.
- `schemas.py` (section 5).
- `scripts/make_dev_fixture.py`: generates two 640×360, 15 fps, 60 s MP4s plus GT CSVs:
  - Grey road background with lane markings and mild noise; one static occluder (a pole) in each camera.
  - Motorcycle sprites: two wheels, coloured body polygon, rider (helmet circle + torso rectangle). Identity = (body colour, helmet colour, torso colour, body shape variant).
  - ~14 identities. Include **2 lookalike pairs** (same body + helmet colour, different torso colour or a small sticker), **3 identities only in A** (exit elsewhere), **2 only in B**.
  - Camera A: bikes move left→right. Camera B: left→right at 0.8× scale, colour/brightness shift, slight blur and noise.
  - Transit A→B uniform 8–15 s; Camera B clock is **+4 s ahead** (so offset estimation is tested).
  - Write `gt_A.csv`, `gt_B.csv` (format in 6.15) and `fixture_meta.json` (true offset, true pairs).
  - Encode with ffmpeg to H.264.
- `configs/presets/dev_fixture.yaml`: `detector.mode: groundtruth`, topology window 5–20 s, expected directions [1,0] both.
- `gt_detector.py` (6.3).
Acceptance:
- Running the fixture script creates both videos + GT; unit tests load the config and validate schemas; GT detector test checks drop/fragment options.

### Phase 2 — Ingest, detection/tracking, rider association, stitching, crops
Build: `ingest.py`, `detect_track.py`, `rider_assoc.py`, `stitching.py`, `crops.py`, `track_quality.py`, `motion.py`, and `orchestrator.py` running these stages and writing `tracks_A.json`, `tracks_B.json`, crops, web videos into `data/user/runs/<id>/`. `scripts/run_pipeline.py --preset dev_fixture` runs it from the CLI.
Acceptance:
- On the dev fixture (GT mode with `fragment_prob=0.2`), stitching recovers ≥ 90% of original identities as single tracks.
- Each track has ≤ K crops saved and a quality score; web MP4 plays in a browser.
- Real YOLO path: a `@pytest.mark.slow` test runs detection on 2 s of any video file without crashing (skip if weights missing).

### Phase 3 — Features and track representations
Build: `embedder.py` (Dino + Dummy), `color.py`, `representation.py` (saves `.npz` per track), `similarity.py`.
Acceptance:
- Unit tests: set-similarity is higher for same-identity fixture tracks than different-identity tracks (DummyEmbedder in fast tests; Dino in a slow test).
- GPU run of Dino on the fixture stays under 3 GB VRAM (log `torch.cuda.max_memory_allocated`).

### Phase 4 — Temporal model, clock offset, candidates, assignment
Build: `temporal.py`, `candidates.py`, `assignment.py`; orchestrator writes `candidates.json`.
Acceptance (on the dev fixture):
- Estimated clock offset within ±1.5 s of the true +4 s.
- Rank-1 ≥ 0.8 on queries that have a true match.
- A-only identities are assigned to "no match" in ≥ 2 of 3 cases.
- Both lookalike pairs are flagged `ambiguous`.
- Unit tests for: temporal scoring edges, impossible deltas, offset estimation with too few pairs (`reliable=false`), Hungarian with dummy column.

### Phase 5 — Independent verifier
Build: `keypoints.py` (LightGlue+ALIKED with SIFT fallback), `checks.py`, `calibrator.py` with hand-set defaults in `configs/calibrator.json`, `policy.py`; verification attached to the top-N candidates of each query; query outcomes computed.
Acceptance:
- Table-driven unit tests for `policy.py` covering: critical timing → rejected; high ambiguity → uncertain even with high p; clean high-p → accepted; low p → rejected.
- On the fixture: true pairs mostly accepted or uncertain; lookalike wrong pairs never accepted; A-only identities end as `no_reliable_match` or `uncertain`.
- Verifier module does not import from `matching/candidates.py` scoring (independence) — enforce with a test that inspects imports.

### Phase 6 — Exports, metrics, API, background jobs
Build: `exporters.py`, `evaluation/gt.py`, `evaluation/metrics.py`, `scripts/evaluate.py`, all API routes (section 7), job runner with progress, result caching, Range-capable media serving.
Acceptance:
- Integration test: POST `/api/runs/from-preset` with `dev_fixture` → poll until done → `results.json` validates against schema; CSV has the required columns.
- Range request test: GET a web video with `Range: bytes=0-1023` returns 206.
- Metrics computed for the fixture and stored in the run.

### Phase 7 — Frontend
Build all pages and components from section 8 against the real API.
Acceptance (manual, record in PROGRESS.md):
- Run the dev fixture from the UI, watch progress, then: videos play side by side in sync; boxes overlay correctly while playing; selecting a Camera A track shows ranked candidates; clicking a candidate seeks both videos and shows crops, keypoint image, evidence, verdict; export buttons download JSON/CSV; metrics page shows numbers.
- `npm run build` has no TypeScript errors.

### Phase 8 — Docker polish (CPU + GPU)
Build: real `models_download.py`; fixture baked into the image; healthcheck; `HF_HUB_OFFLINE=1`; verify GPU override.
Acceptance:
- Fresh clone → `docker compose up --build` → dev fixture preset runs end-to-end in the UI **with networking disabled after build**.
- `docker compose -f compose.yaml -f compose.gpu.yaml up --build` → `/api/health` reports `device: cuda`.

### Phase 9 — Test hardening and README draft
Build:
- Fill test gaps: crop quality, colour similarity, stitching, exporters, metrics (hand-computed small example), calibrator load/save.
- Mark slow tests; `pytest -q -m "not slow"` must finish in < 60 s.
- README draft sections: overview, architecture diagram (Mermaid), quick start (Docker CPU / GPU), using the UI, config reference, matching design, verifier design, export format, tests, limitations (placeholders for Phase 10 results).
Acceptance: all tests green; README renders.

### Phase 10 — Real datasets, evaluation and tuning (FINAL DATA STAGE)
Build:
1. `scripts/download_data.py` (writes to `data/user/datasets/`, never into Git):
   - **RoundaboutHD**: `huggingface_hub.hf_hub_download(repo_id="yl4300/RoundaboutHD", repo_type="dataset", filename="RoundaboutHD.zip")`, then unzip. Public, no token. Large (4K video) — show progress.
   - **VnTraffic**: download `Vehicle_Tracking.zip` from Zenodo record 18195750 (`https://zenodo.org/records/18195750/files/Vehicle_Tracking.zip?download=1`), unzip. License CC-BY 4.0 — credit in README.
2. `scripts/convert_roundabouthd.py`:
   - Read `Multi_CAM_Ground_Truth.txt` (`camera_id obj_id frame_id xmin ymin width height xworld yworld`), convert to our per-camera GT CSV (normalized by the camera's video resolution).
   - Let the user choose the camera pair (`--cam-a 1 --cam-b 2`); print how many shared identities each pair has and how many are motorcycles (use `vehicle_statistic.xlsx` if it has class info; otherwise note "class unknown").
   - Optional `--start-s/--duration-s` to cut a shorter segment for demo speed.
   - Generate `configs/presets/roundabouthd_pair.yaml` (`target.classes: [car, motorcycle, bus, truck]` by default; transit window estimated from GT pairs: 5th–95th percentile of true Δ, widened by 20%).
3. `scripts/make_synthetic_pair.py` (for VnTraffic, motorcycle-heavy):
   - Input one video + its annotations; output camera A = left 55% crop, camera B = right 55% crop of a **time-shifted** copy, with B degraded: 0.75× resize, colour temperature shift, gamma change, Gaussian blur, JPEG re-encode artefacts, and a configurable clock offset (default +6 s).
   - Produce GT CSVs in our format when source annotations exist (convert their format; if conversion is impossible, produce GT from overlap-region identity consistency and document it).
   - Generate `configs/presets/vntraffic_synth.yaml` (`target.classes: [motorcycle]`, detector mode yolo).
   - Write a ~60 s downscaled sample pair to `samples/vntraffic_synth/` (allowed in Git, < 25 MB total) with an attribution `README.md`.
4. Run the real YOLO pipeline on both presets. Then:
   - `scripts/fit_calibrator.py`: fit logistic calibration on one split of labelled candidate pairs, evaluate on the other half; save `configs/calibrator.json`.
   - Tune `min_match_score`, `ambiguity_margin`, `p_accept`, `p_reject` on the tuning split only; report on the held-out split.
5. `scripts/evaluate.py --run <id>` → `reports/<preset>_metrics.json` + a markdown table.
Acceptance:
- README results table with rows: dev fixture, VnTraffic synthetic pair (motorcycles), RoundaboutHD camera pair (all vehicles + motorcycle subset if any). Columns: precision, recall, F1 (accepted), Rank-1, correct-rejection rate, false matches, ECE, FPS, hardware.
- At least 3 documented false matches or near-misses with screenshots/evidence explanations.
- Presets for downloaded data appear in the UI preset list.

### Phase 11 — Demo and final checks
Build:
- `docs/demo_script.md` (3–5 min): 1) problem + architecture (30 s); 2) run a preset, show progress and speed (30 s); 3) correct match with evidence (45 s); 4) lookalike pair rejected/uncertain with competing matches (45 s); 5) a `no_reliable_match` query (30 s); 6) clock-offset estimate (20 s); 7) metrics page + false matches (40 s); 8) JSON/CSV export (20 s).
- Final README pass: limitations (domain gap, no motorcycle-specific ReID training, synthetic pair has limited viewpoint change, RoundaboutHD is mostly cars, 4K downscaling, rider-change cases), license credits (Ultralytics AGPL-3.0, DINOv2 Apache-2.0, LightGlue Apache-2.0, ALIKED BSD-3, datasets).
Acceptance: run through the "Definition of done" checklist below on a fresh clone.

---

## 11. Coding rules for Codex

- Small functions, docstrings on public functions, `logging` (not print) with run_id in log lines.
- Deterministic: seed numpy/torch/random from config.
- GPU memory: load one heavy model at a time; `torch.cuda.empty_cache()` between stages; batch sizes from config.
- Never block the API event loop with pipeline work; use the job runner.
- Every evidence message must be understandable by a non-expert, e.g. "Only 3 keypoint matches between the bikes' crops (pass needs 15)".
- Fail gracefully: if LightGlue fails, fall back to SIFT and add an evidence note; if YOLO26 fails, use fallback weights and record it in run metadata.
- No dataset downloads, no network calls at runtime, before Phase 10.

---

## 12. Definition of done (maps to the assignment)

- [ ] `docker compose up --build` → UI at `http://localhost:8080`, no credentials, works offline after build.
- [ ] Two feeds processed into tracks with track-level representations (top-K quality crops, part embeddings, motion, timing).
- [ ] Ranked cross-camera candidates; Hungarian assignment with explicit "no match".
- [ ] Timing window, direction checks, and clock-offset estimation with reliability flag.
- [ ] Lookalike / ambiguity flags and competing-match display.
- [ ] Independent verifier with evidence items and accepted / uncertain / rejected + calibrated confidence.
- [ ] UI: synced side-by-side playback with overlays, track list, ranked candidates, crop comparison, evidence, statuses, metrics.
- [ ] JSON + CSV exports with pairs, scores, evidence, verdicts, confidence.
- [ ] Automated tests pass (`pytest -q`), fast suite < 60 s.
- [ ] README: run instructions, design choices, results table, limitations, licenses.
- [ ] Demo video 3–5 min covering correct, lookalike, rejected, uncertain, metrics, false matches, speed.
