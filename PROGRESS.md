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
