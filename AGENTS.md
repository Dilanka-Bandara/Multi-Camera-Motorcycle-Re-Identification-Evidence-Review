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
