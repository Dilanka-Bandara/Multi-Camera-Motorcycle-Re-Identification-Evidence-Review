"""Exercise real API routing and static-build integration."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app


def test_health_without_loading_models(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok", "device": "not_initialized", "models_loaded": [],
    }


def test_frontend_requires_a_real_build(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    assert client.get("/").status_code == 503
    assert client.get("/api/unknown").status_code == 404


def test_serves_frontend_and_assets(tmp_path: Path) -> None:
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<h1>Hello MotoReID</h1>", encoding="utf-8")
    (tmp_path / "assets" / "app.js").write_text("export {};", encoding="utf-8")
    client = TestClient(create_app(tmp_path))
    assert "Hello MotoReID" in client.get("/").text
    assert client.get("/assets/app.js").status_code == 200
