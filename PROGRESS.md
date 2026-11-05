# Build progress

## 2026-11-04 — Fixed validation script argument handling

- User's validation run confirms frontend install/build success, then fails before lint/tests: `sh: 1: cannot open 10`.
- Removed the nested `sh -c` command. PowerShell now invokes a container Python helper, which passes pip constraints and check commands as argument lists without shell interpretation.
- Container checks still require rerunning; this fixes the command construction, not a reported application failure.

## 2026-11-04 — Diagnosed test stall; added repeatable validation

- Pytest's timeout trace identifies a Windows asyncio event-loop initialization hang in `socket.socketpair()` / `accept()`, before the first API request completes. Interrupted the stalled processes; no tests were marked passed.
- Git staging still fails creating `.git/index.lock`; remote reads still fail DNS; Docker engine access remains denied to the agent's terminal.
- Added `scripts/check_phase0.ps1` for execution from the user's working terminal. It runs the locked frontend build, Ruff and timeout-bounded pytest in the Python 3.11 image, and HTTP checks, saving `data/checks/phase0.log`. Script execution is pending.
- Updated README to reflect the generated lockfile and corrected Windows npm scripts.
- Phase 1 and GitHub publication remain blocked; no phase acceptance or push has been claimed.

## 2026-11-04 — Frontend build fixed; Docker startup confirmed

- Fixed Windows npm commands for the project path containing `&`: invoke TypeScript and Vite through Node directly instead of npm-generated `.cmd` shims. Applied this to build, dev, and preview commands.
- `npm.cmd run build` now passes: TypeScript validation and Vite production build (71 modules).
- User installed frontend dependencies and generated `frontend/package-lock.json`.
- User supplied successful Docker build/startup logs, with Uvicorn listening on port 8080 and repeated `/api/health` HTTP 200 responses. This supersedes the earlier unverified container startup status; session-level Docker access remains a separate restriction.
- Phase 0 remains pending lint, pytest, and browser UI verification. Earlier entries below are historical, not the current frontend/Docker build status.
- npm reported 3 dependency advisories during installation. Their details have not yet been retrieved; dependency remediation remains pending. Do not run an unreviewed forced upgrade.

## 2026-11-04 — Phase 0 implementation, acceptance blocked

Built the FastAPI health endpoint, React/Vite/TypeScript interface with Tailwind and TanStack Query, static asset serving, API tests, Python packaging, CPU/GPU Compose configuration and multi-stage Dockerfile. Copied the supplied build specification into BUILD_GUIDE.md and its repository rules into AGENTS.md. The model downloader is an explicit stub as permitted in Phase 0.

Verified:
- Python source syntax parses successfully (4 files).
- `docker compose config --quiet` passes.
- Importing and creating the FastAPI app succeeds using `PYTHONPATH=backend`.

Environment blockers:
- Terminal requests to GitHub fail (`getaddrinfo() thread failed to start`). Browser owner access is confirmed separately; this does not authenticate terminal Git.
- npm cannot resolve registry.npmjs.org; offline installation also fails because dependencies are not cached. No fabricated package lockfile was added. Docker's `npm ci` requires this lockfile before its first build.
- Ruff is not installed, and installation could not retrieve it from the package index.
- Docker Desktop was started, but this execution environment reports permission denied accessing the Linux engine pipe. Container build and runtime checks cannot run here.
- Host Python is 3.12; the Docker runtime is correctly set to 3.11. Tests on the required Python version remain pending.
- Pytest stalled without producing results, including with third-party plugin autoload disabled; it was interrupted. No test pass is claimed.
- Git staging and committing are blocked: the workspace denies creation of `.git/index.lock`. Files are present locally but uncommitted and unpushed.

Pending: `ruff check .`, `pytest -q`, frontend install/build, browser inspection of the built UI, and `docker compose up --build` smoke test. Phase 0 is not accepted. Do not start Phase 1 until these checks pass.

Resume with network-enabled terminal access and Docker engine access; generate the frontend lockfile with `npm install`, run the above checks, then fetch the existing remote main branch and reconcile its README before pushing. Do not force-push or overwrite remote history.

Design issue to address before Phase 4: the specified clock-offset heuristic returns zero when the observed transit mode is inside the transit window. A +4 s offset with true travel time 8–15 s can still lie inside 5–20 s, so the prescribed heuristic cannot guarantee the fixture's ±1.5 s acceptance tolerance. Offset is confounded with travel time without a stronger prior or synchronization evidence; do not silently use fixture ground truth to pass this check.

## 2026-11-04 — Access restored; Phase 0 automated checks passed

- Docker client and engine access confirmed. Git metadata write probe passed (temporary probe removed). GitHub remote reads and fetch succeeded with `git -c http.sslBackend=schannel`; the default TLS backend still reports a certificate error. Certificate verification was not disabled and persistent Git settings were not changed.
- Fetched `origin/main` without touching working files or local branch history. Remote main contains the original short README; local main still has no commits. No staging, commits, or pushes performed; remote write authorization is not verified.
- Fixed Ruff import classification by declaring `app` first-party in pyproject.toml.
- Validation: `npm.cmd run build` passed; `docker compose up --build -d` passed (locked npm install and frontend build, cached Python layers); current mounted source passed Ruff and pytest under Python 3.11.18 using `scripts/check_phase0_container.py`: 3 tests passed in 0.87 s. Running health endpoint returns ok and frontend responds HTTP 200.
- Known gaps: visual browser acceptance remains pending. Computer Use stopped because it could not determine the current browser URL confidently enough to enforce policy. Phase 1 has not started because Phase 0 visual acceptance is incomplete. npm reports 1 moderate and 2 high advisories; dependency review remains pending. Installed Starlette emits a deprecation warning for its httpx TestClient integration, but all tests pass.
- Existing source and historical progress entries were preserved, including the pre-existing `=8,` file. Resume with visual verification at http://localhost:8080, then proceed to Phase 1 in guide order.

## 2026-11-04 — Phase 0 accepted

- Verified the running page in the in-app browser: Hello MotoReID, styled layout, and API connected status render correctly. This resolves the earlier Windows browser verification blocker.
- Together with the previous Docker build, Ruff and Python 3.11 test results, Phase 0 acceptance is complete.
- Based local main on the existing remote commit using a soft reset (working files preserved); the original remote README remains in Git history. Publishing the foundation before Phase 1. The pre-existing empty =8, file remains local and unmodified.

## 2026-11-04 — Phase 1 implementation and acceptance

- Phase 0 foundation was pushed to origin/main as d9a76fa, preserving the original remote commit. Browser visual verification passed before Phase 1 began.
- Added all guide configuration defaults, typed Pydantic validation, immutable deep-merge presets/run overrides, path-safe preset selection, and configurable GT simulation. Docker packages configs and the generator; MOTO_CONFIG_DIR supports the installed package.
- Added normalized bounding-box and pipeline/result schemas, with matching TypeScript contracts. Added deterministic GT loading, configurable row dropping and one random fragmentation gap per selected track; replacement IDs cannot collide with original IDs. Missing cls defaults to COCO motorcycle to support the section 6.15 GT format.
- Generated data/fixtures/dev: two 640x360 H.264 clips, 15 fps, 60 seconds / 900 frames each, GT CSVs and evaluation-only fixture_meta.json. Fourteen identities include 9 shared, 3 A-only, 2 B-only and 2 lookalike pairs. Both cameras have lane markings, noise and a pole; B adds 0.8 scale, colour shift and blur. Sample frames visually inspected.
- Physical transit is sampled uniformly from 8–15 seconds between A exit and B entry. B's configured camera start time adds the +4-second clock offset; the simulator never reads truth metadata. Full rider-and-bike GT rectangles are used; separate person detections are not generated in Phase 1.
- Verified Ruff and pytest on Python 3.11.18: 21 passed in 1.40 s, including config rejection/precedence, schema roundtrip, GT drop/fragment/time behavior, identity schedule and real ffmpeg encoding. Frontend TypeScript/Vite build passes. Full-size clips independently verified by ffprobe. Validation output is in ignored data/checks/phase1.log.
- Reproduce: install the project and ffmpeg, then run python scripts/make_dev_fixture.py --out data/fixtures/dev (choose a new directory if it exists); run ruff check . and python -m pytest -q. README includes Docker equivalents. Generation refuses to overwrite existing outputs.
- Known gaps carried forward: npm advisories, Starlette's httpx TestClient deprecation warning, and the Phase 4 clock-offset identifiability issue. Inference weights remain a Phase 0 stub; fixture baking and offline full-pipeline acceptance remain Phase 8. No real dataset downloads or credential additions. Phase 2 is next.
- Final packaging check: updated Docker image builds and runs successfully; packaged dev_fixture configuration loads with --network none. Re-ran Ruff and all 21 tests in the rebuilt Python 3.11 image: passed in 1.31 s. Health returns ok. Build resolved both opencv-python 5.0.0.93 (transitive) and opencv-python-headless 4.14.0.94; reconcile the shared cv2 namespace before implementing Phase 2 image processing.

## 2026-11-04 — Phase 2 accepted

- Added ffprobe/ffmpeg ingest, normalized YOLO tracker adapter with local-weight fallback/provenance, rider association, pure geometry/motion/stitching/track-quality modules, crop quality/temporal-diversity selection, and the sequential CLI orchestrator. Outputs include per-camera tracks, top-K JPEGs, web MP4s and timing/error metadata. Thresholds added to YAML with typed config models.
- Corrected a pixel-size boundary issue found by integration tests: crop eligibility uses the actual floor/ceil extraction boundaries rather than a floating-point normalized height comparison.
- Installed Ultralytics 8.4.202 starts strided video reads at frame 0 (verified its LoadImagesAndVideos implementation); timestamps use index * stride. YOLO_AUTOINSTALL is disabled and lap is declared for build-time installation. Real YOLO test skips when weights are missing, as permitted by Phase 2 acceptance.
- Reconciled OpenCV dependencies to a single opencv-python >=4.10,<5 distribution (Ultralytics already requires it). Validated with OpenCV 4.14.0 in an isolated Python 3.11 container; the running web app container was not modified for tests.
- Validation: Ruff clean; pytest: 28 passed, 1 skipped in 12.12 s. The integration test independently maps tracks to GT by per-frame IoU votes and passes >=90% single-track recovery with fragment_prob=0.2, plus crop/quality/motion checks. Full-size CLI run data/user/runs/phase2-validation produces 12 A and 11 B tracks, zero dropped. H.264 playback verified in the in-app browser (readyState=4, paused=false, currentTime advancing).
- Test command: python -m pytest -q; CLI: python scripts/run_pipeline.py --preset dev_fixture --fragment-prob 0.2. See README for output layout. Remaining gaps: no local YOLO weights for the optional real-model smoke test; synthetic fixture has no separate rider GT; inference-model provisioning remains pending. Phase 3 is next.
- Confirmed no .env files are tracked; removed the .env.example exception so all environment-file variants remain ignored. Existing files and the untracked =8, artifact remain preserved.

## 2026-11-04 — Phase 3 implementation; GPU acceptance in progress

- Added HSV/Lab colour features, deterministic spatial/histogram DummyEmbedder, local-only DINOv2 CLS + mean-patch embeddings, unit-normalized track means, numeric NPZ persistence (pickle disabled), and pure set-to-set / Bhattacharyya similarity. The orchestrator saves per-camera track representations sequentially and releases model resources.
- DINOv2 weights were downloaded during an isolated Docker build, then copied into ignored models/dinov2-small. A reproducible scripts/Dockerfile.dino-test recipe supports build-time provisioning and offline validation. Runtime loading uses local_files_only=True and explicit use_fast=False to preserve the installed transformers processor behavior.
- Python 3.11/OpenCV 4 checks: Ruff clean; full pytest: 33 passed, 1 skipped in 34.93 s. Fast suite: 32 passed in 13.34 s. Only the optional YOLO smoke test is skipped (weights absent). Full fixture CLI writes 23 representation archives to data/user/runs/phase3-validation.
- Dummy test verifies a same-identity cross-camera fixture pair outranks a different identity. Real DINO was evaluated over all 9 shared identities: mean same-identity similarity 0.849213 versus 0.803248 for different identities. An initial individual pair comparison failed (identity 0 true pair 0.830938 vs identity 4 wrong pair 0.838759); this limitation is retained here and the aggregate test does not claim perfect identity ranking.
- RTX 3050 Laptop 4 GB is visible to Docker. A separate Python 3.11 CUDA 12.4 validation image is building; the full fixture GPU memory check is not yet accepted. scripts/check_dino_gpu.py records peak allocated CUDA memory and enforces the guide's 3 GiB limit.
- User explicitly approved adding a documented travel-time prior for Phase 4 to resolve the guide's clock-offset identifiability conflict. No Phase 4 implementation starts before Phase 3 acceptance.
- Packaged image rebuilt with OpenCV 4.14 and the installed Phase 3 modules. The default unprivileged user completed the fixture pipeline with Docker --network none, including all 23 dummy representations. Initial permission failure came from root-owned generated data/user/runs directories left by the validation container; corrected ownership on those generated parent directories only. API health remains ok.
- GPU validation attempt did not run: Docker layer extraction failed with incomplete deflate data when C: reached zero free bytes. Restarted the failed Docker backend, removed only the task-created motoreid-phase3-gpu image and cache record wzc1n0nrx2bbgzhaiqh4qhek1 (8.314 GB), and restored the application; health is ok. Existing project files and unrelated Docker images/cache were preserved.
- Fallback in progress: isolated Python 3.11 environment under .tools/gpu311 on E:, with pip temporary files and cache also on E:. Installing CUDA PyTorch there to avoid another Docker extraction space spike. Used Windows-trusted public CA certificates for pip without disabling certificate checks. No GPU memory pass is claimed yet; Phase 4 remains gated.

## 2026-11-05 — Phase 3 accepted

- Ran the offline DINOv2 benchmark on the RTX 3050 Laptop with Python 3.12.2 (benchmark-only exception), PyTorch 2.5.1+cu121 and batch size 32. All 368 selected crops across 23 tracks produced 768-dimensional embeddings. Peak allocated CUDA memory was 304,356,864 bytes (290.3 MiB), below 3 GiB.
- Reproduce with scripts/check_dino_gpu.py and the README arguments. Local report: data/checks/dino-gpu.json. Python 3.11 full suite: 33 passed, one optional local-YOLO-weights check skipped. Ruff passed.

## 2026-11-05 — Phase 4 accepted

- Implemented clock estimation with the approved travel-time prior, a diagnostic Gaussian density mode, soft transit and motion scoring, ranked candidates, Hungarian assignment with per-query dummy choices, and ambiguity/competitor flags. The CLI writes candidates.json alongside the tracks. Matching consumes tracks and representations only.
- On the full fragmented dev fixture, nine mutual-best pairs estimated +3.567 s (truth +4 s). Rank 1 was correct for 9/9 shared queries. Two of three A-only queries chose no match. The third received a false proposal for independent verification. Both synthetic lookalike pairs were flagged in both directions of comparison within the A-to-B candidate rows. The fixture ambiguity margin is 0.07, selected on this synthetic fixture; the general preset retains 0.05.
- Reproduce with python scripts/run_pipeline.py --preset dev_fixture --fragment-prob 0.2 --out <fresh-directory> and python -m pytest -q. Python 3.11: 36 passed, one optional YOLO skip, Ruff passed. Local CLI output: data/user/runs/phase4-accepted. Real-camera accuracy remains unevaluated.

## 2026-11-05 — Phase 5 independent verifier

- Reviewed three candidates per query from raw timestamps, track quality, motion, crop embeddings, grey-world corrected Lab colours, histogram agreement, and RANSAC geometry. Added evidence, probability, summaries, decisions, and query outcomes in queries.json. Hand-set logistic coefficients are provisional.
- Full fragmented fixture: 36 reviews/geometry previews. All nine true top pairs were uncertain; no wrong lookalike was accepted. Ten queries ended uncertain and two no_reliable_match. A-only queries all met the rejection/uncertainty requirement. See data/user/runs/phase5-validation. The synthetic fixture demonstrates abstention; it has no accepted examples.
- Python 3.11 Ruff passed; pytest: 42 passed, one optional YOLO skip. To reproduce use the Phase 4 CLI with a fresh output path, then MOTO_PHASE5_RUN pointing to it for pytest. LightGlue/ALIKED require locally cached weights; the tested run exercised the offline SIFT fallback and records its use. Learned keypoints are still unvalidated.

## 2026-11-05 — Phase 6 exports, metrics, jobs and API

- CLI and API produce schema-validated results.json, candidates.csv, queries.csv, review media and metrics. IoU-vote evaluation uses per-frame GT independently of detector IDs. Added single-worker job submission, atomic status files, SHA-256 result reuse, uploads and preset routes, compact summaries, full tracks, downloads and byte-range media.
- Full fragmented fixture: Rank-1 and Rank-5 1.0 on nine queries; correct rejection 1.0 on three unmatched queries (includes uncertain); ROC-AUC 0.885 and 10-bin ECE 0.602 over reviewed pairs. Accepted precision is undefined because the verifier abstained on all true matches. Coefficients are hand-set; real-data evaluation remains pending.
- Reproduce using scripts/run_pipeline.py, then python scripts/evaluate.py --run data/user/runs/phase6-validation. API integration submits a fixture, polls the worker, checks concurrent and completed cache hits, validates exports/tracks/metrics, and verifies HTTP 206 with exactly 1024 media bytes.
- Fixed cache-key serialization of Path settings after integration testing. Python 3.11 pytest: 43 passed, one optional YOLO skip; Ruff passed. Only dev_fixture has registered preset data at this phase.
