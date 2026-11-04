# MotoReID

Multi-Camera Motorcycle Re-Identification & Evidence Review.

**Build status: Phases 0–2 complete. The CLI produces tracks, crops and web videos; cross-camera matching and evidence review are not yet implemented.** See [PROGRESS.md](PROGRESS.md) for verified checks and blockers, and [BUILD_GUIDE.md](BUILD_GUIDE.md) for the complete specification.

## Stack

Python 3.11, FastAPI, Pydantic v2, React, TypeScript, Vite, Tailwind CSS and TanStack Query. Subsequent phases add YOLO26 with YOLO11 fallback, BoT-SORT/ByteTrack, DINOv2, OpenCV, LightGlue/ALIKED with SIFT fallback, SciPy assignment and scikit-learn calibration. Runs will use filesystem storage.

## Development

Use Python 3.11 for the project environment:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8080
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

The generated `frontend/package-lock.json` locks frontend dependencies. Use `npm ci` for subsequent installs. The npm scripts invoke Node directly so Windows paths containing `&` work correctly.

## Docker

After starting Docker Desktop:

```sh
docker compose up --build
```

Open http://localhost:8080. Health endpoint: http://localhost:8080/api/health.

GPU configuration (inference is not implemented in Phase 0):

```sh
docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

The Docker model-download module is explicitly a Phase 0 stub. Offline inference and the baked synthetic fixture are later phase requirements, not current capabilities.

## Checks

```sh
ruff check .
pytest -q
cd frontend
npm run build
```

The API test suite checks health, missing frontend handling, static assets, and unknown API routes. Heavy inference models are not imported or downloaded by these tests.

With the Docker image built and the application running, Windows users can run
`& .\scripts\check_phase0.ps1` from the project root. This validates the frontend,
runs Ruff and pytest in a disposable Python 3.11 application container, and checks
the running HTTP endpoints. The transcript is saved to `data/checks/phase0.log`.
The script stops on errors and does not commit or push code.

## Data and credentials

### Synthetic development fixture (Phase 1)

After installing the project with Python 3.11 and installing ffmpeg, run:

```powershell
python scripts/make_dev_fixture.py --out data/fixtures/dev
```

Or use the existing application image against current source in PowerShell:

```powershell
docker compose run --rm --no-deps --volume "${PWD}:/workspace" --workdir /workspace -e PYTHONPATH=/workspace/backend app python scripts/make_dev_fixture.py --out data/fixtures/dev
```

This creates two 60-second H.264 videos, normalized GT CSVs and evaluation metadata.
Existing fixture outputs are never overwritten: choose a new output directory to regenerate.
There are 9 shared identities, 3 A-only identities, 2 B-only identities and 2 lookalike pairs.
Camera B has smaller sprites, a colour shift, blur and noise. Physical transit is 8â€“15 seconds;
the preset adds 4 seconds to B's camera timestamps. Metadata records truth for evaluation,
not as input to matching or clock estimation. The fixture uses full rider-and-bike rectangles;
it does not yet provide separate person detections.

`load_config("dev_fixture", overrides=...)` deep-merges defaults, a named preset and
per-run overrides, then validates the result. Set `MOTO_CONFIG_DIR` or pass `config_dir`
when loading config outside the source checkout. The Docker image includes the YAML files
and generator; baking the generated videos remains Phase 8 work.

`groundtruth.drop_rate`, `fragment_prob`, `fragment_gap_frames` and `confidence` control
the simulated detector. Fragmentation splits each selected track once, with a missing-frame
gap and collision-free replacement ID. Seeds come from `runtime.seed`.

Run `python -m pytest -q` and `ruff check .`, or use the existing container check helper
(`scripts/check_phase0_container.py`), which now discovers the Phase 1 tests too.
The encoding test uses small frames for speed; full-size generated videos were separately
verified with ffprobe. The frontend mirrors the backend contracts in `src/types.ts`.

### Storage policy

Do not commit recordings, model weights, generated run data, or secrets. Only explicitly attributed demo media under `samples/` may be committed. No paid API credentials are required.


### Track and crop pipeline (Phase 2)

Run `python scripts/run_pipeline.py --preset dev_fixture --fragment-prob 0.2`.
It writes a new directory under `data/user/runs/` containing `tracks_A.json`,
`tracks_B.json`, JPEG crops, browser-safe MP4s and `pipeline.json` with timings,
quality scores, dropped counts and detector provenance. Use `--out` for a specific
new directory. Existing runs are never overwritten.

Real detection uses locally provisioned YOLO weights only and falls back to the
configured second local model on failure. Runtime downloads and automatic tracker
dependency installation are disabled. Run the optional real-video test using
`MOTO_TEST_WEIGHTS` and `MOTO_TEST_VIDEO`; it skips when local assets are missing.
All `.env` and `.env.*` files are excluded from version control.

### Appearance representations (Phase 3)

The pipeline now also writes `representations/<track_id>.npz` with full, vehicle
and available rider embeddings, normalized mean embeddings, HSV histograms and
Lab colour means. Archives contain numeric arrays and are loaded with pickle disabled.
The fixture defaults to a deterministic histogram/spatial-statistics embedder.
DINOv2 concatenates CLS and mean patch tokens and uses CUDA FP16 autocast when available.
Its local weights must exist before execution; it never downloads at runtime.

Prepare a separate validation image (downloads occur only during Docker build):

```powershell
$applicationImage = docker compose images -q app
# For GPU validation add: --build-arg TORCH_VARIANT=cu124
docker build -f scripts/Dockerfile.dino-test --build-arg "APP_IMAGE=$applicationImage" -t motoreid-dino-test .
docker run --rm --network none --volume "${PWD}:/workspace" --workdir /workspace -e PYTHONPATH=/workspace/backend -e MOTO_CONFIG_DIR=/workspace/configs -e MOTO_DINO_MODEL=/opt/dinov2-small motoreid-dino-test python -m pytest tests/test_phase3.py -q
```

For a GPU-enabled validation image, run the full selected-crop memory benchmark:

```powershell
docker run --rm --network none --gpus all --volume "${PWD}:/workspace" --workdir /workspace -e PYTHONPATH=/workspace/backend -e MOTO_CONFIG_DIR=/workspace/configs motoreid-dino-test python scripts/check_dino_gpu.py --run data/user/runs/phase2-validation --model /opt/dinov2-small --out data/checks/dino-gpu.json
```

Use your existing fixture run path for `--run` and `MOTO_TEST_RUN` if different.
The benchmark requires Python 3.11 and checks peak allocated CUDA memory below 3 GiB.
Appearance alone does not establish identity: DINOv2 can prefer a wrong synthetic
lookalike. Temporal matching and independent verification are subsequent phases.
