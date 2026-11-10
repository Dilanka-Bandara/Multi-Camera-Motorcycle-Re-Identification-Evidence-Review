import csv
import io
import shutil
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.exporters import COLUMNS
from app.main import create_app
from app.schemas import RunResult


@pytest.mark.slow
def test_job_review_download_and_range(tmp_path: Path) -> None:
    source = Path("data/fixtures/dev")
    if not (source / "camera_A.mp4").is_file():
        pytest.skip("Generate the dev fixture before the API acceptance test")
    fixtures = tmp_path / "fixtures/dev"
    fixtures.mkdir(parents=True)
    for name in ("camera_A.mp4", "camera_B.mp4", "gt_A.csv", "gt_B.csv"):
        shutil.copyfile(source / name, fixtures / name)
    with TestClient(create_app(data_dir=tmp_path)) as client:
        assert client.get("/api/presets").json() == ["dev_fixture"]
        body = {"preset": "dev_fixture"}

        def submit() -> str:
            response = client.post("/api/runs/from-preset", json=body)
            assert response.status_code == 200
            return response.json()["run_id"]

        run_id = submit()
        assert submit() == run_id
        api = f"/api/runs/{run_id}"
        deadline = time.monotonic() + 110
        while time.monotonic() < deadline:
            status = client.get(api + "/status").json()
            if status["state"] in ("done", "failed"):
                break
            time.sleep(0.25)
        assert status["state"] == "done", status
        assert submit() == run_id
        full = RunResult.model_validate(client.get(api + "/export.json").json())
        assert full.metrics and full.metrics["rank1"] >= 0.8
        accepted = full.metrics["decisions"]["accepted"]
        assert accepted["tp"] >= 1 and accepted["fp"] == 0
        verdicts = {c.verification.status for q in full.queries for c in q.candidates
                    if c.verification}
        assert verdicts == {"accepted", "uncertain", "rejected"}
        assert client.get(api + "/metrics").json() == full.metrics
        assert "boxes" not in client.get(api).json()["tracks_A"][0]
        assert client.get(api + "/tracks/A").json()[0]["boxes"]
        q = full.queries[0]
        assert client.get(api + "/queries/" + q.query_track_id).json()["outcome"] == q.outcome
        assert next(csv.reader(io.StringIO(client.get(api + "/export.csv").text))) == COLUMNS
        assert client.get(api + "/queries.csv").text.startswith("query_track_id,outcome,")
        media = client.get(f"/media/runs/{run_id}/camera_A.mp4", headers={"Range": "bytes=0-1023"})
        assert media.status_code == 206 and len(media.content) == 1024
        assert client.get(f"/media/runs/{run_id}/results.json").status_code == 404
