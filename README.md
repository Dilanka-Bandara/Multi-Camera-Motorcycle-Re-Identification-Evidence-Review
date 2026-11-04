# MotoReID

Multi-Camera Motorcycle Re-Identification & Evidence Review.

**Build status: Phase 0 in progress. This is the application foundation, not a finished inference system.** See [PROGRESS.md](PROGRESS.md) for verified checks and blockers, and [BUILD_GUIDE.md](BUILD_GUIDE.md) for the complete specification.

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

Do not commit recordings, model weights, generated run data, or secrets. Only explicitly attributed demo media under `samples/` may be committed. No paid API credentials are required.
