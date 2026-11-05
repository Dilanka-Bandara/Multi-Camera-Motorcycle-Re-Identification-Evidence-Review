"""Check the packaged UI, fixture and exports from inside a network-isolated app container."""

import json
import time
from urllib.request import Request, urlopen


def read(path: str, payload: bytes | None = None) -> bytes:
    request = Request(
        "http://127.0.0.1:8080" + path, data=payload, headers={"Content-Type": "application/json"}
    )
    with urlopen(request, timeout=30) as response:
        return response.read()


def main() -> None:
    deadline = time.monotonic() + 120
    while True:
        try:
            health = json.loads(read("/api/health"))
            break
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(1)
    if b'id="root"' not in read("/") or "dev_fixture" not in json.loads(read("/api/presets")):
        raise RuntimeError("Packaged UI or baked fixture missing")
    run_id = json.loads(read("/api/runs/from-preset", b'{"preset":"dev_fixture"}'))["run_id"]
    base = f"/api/runs/{run_id}"
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        status = json.loads(read(base + "/status"))
        if status["state"] == "failed":
            raise RuntimeError(status["error"])
        if status["state"] == "done":
            break
        time.sleep(1)
    else:
        raise TimeoutError("Fixture job did not finish")
    result = json.loads(read(base + "/export.json"))
    if not result["queries"] or result["metrics"]["rank1"] < 0.8:
        raise RuntimeError("Packaged fixture retrieval failed")
    print(
        json.dumps(
            {
                "device": health["device"],
                "run_id": run_id,
                "queries": len(result["queries"]),
                "rank1": result["metrics"]["rank1"],
            }
        )
    )


if __name__ == "__main__":
    main()
