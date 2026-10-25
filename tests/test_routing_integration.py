"""Exercise actual API/ORM/CV wrapper/urgency/fusion with neural outputs stubbed.

No custom YOLO weights exist in this checkout; these tests validate invariance,
not accuracy of either neural network. Routing uses the real trained NB model.
"""
import importlib
import io
from pathlib import Path
import sys
import types
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture
def api(monkeypatch, tmp_path):
    fake_yolo = Mock(return_value=Mock())
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=fake_yolo))
    real_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda p: True if p.name == "yolov8_civic.pt" else real_exists(p))
    main = importlib.import_module("backend.main")
    cv = importlib.import_module("backend.services.cv_service")
    nlp = importlib.import_module("backend.services.nlp_service")
    endpoints = importlib.import_module("backend.api.endpoints")
    from backend.database.models import MasterTicket
    from backend.database.session import Base, get_db
    boxes = [types.SimpleNamespace(cls=[0], conf=[0.84]), types.SimpleNamespace(cls=[1], conf=[0.4])]
    vision = Mock(return_value=[types.SimpleNamespace(boxes=boxes)])
    sentiment = Mock(return_value=[{"label": "NEGATIVE", "score": 0.9}])
    monkeypatch.setattr(cv, "model", vision)
    monkeypatch.setattr(nlp, "_NLP_PIPELINE", sentiment)
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)

    def get_test_db():
        with sessions() as session:
            yield session

    main.app.dependency_overrides[get_db] = get_test_db
    monkeypatch.chdir(tmp_path)
    with TestClient(main.app) as client:
        yield client, sessions, MasterTicket, endpoints, sentiment, vision
    main.app.dependency_overrides.clear()
    engine.dispose()


def submit(client, lat=19.076, title="Dangerous road pothole", description="Dangerous deep pothole blocks road traffic"):
    photo = io.BytesIO()
    Image.new("RGB", (12, 12), "gray").save(photo, format="PNG")
    return client.post("/api/v1/report", data={
        "title": title, "description": description,
        "latitude": str(lat), "longitude": "72.8777"},
        files={"image": ("photo.png", photo.getvalue(), "image/png")})


def test_enabled_routing_preserves_actual_legacy_scores(api, monkeypatch):
    from backend.config import settings
    client, sessions, model, _, sentiment, vision = api
    monkeypatch.setattr(settings, "ROUTING_ENABLED", False)
    baseline_response = submit(client)
    assert baseline_response.status_code == 201
    baseline = baseline_response.json()
    monkeypatch.setattr(settings, "ROUTING_ENABLED", True)
    routed_response = submit(client, 19.09)  # Outside duplicate radius.
    routed = routed_response.json()
    assert routed_response.status_code == 201
    for key in ("category", "text_urgency", "visual_severity", "final_priority_score", "priority_level"):
        assert routed[key] == baseline[key]
    assert routed["visual_severity"] == 0.84
    assert routed["text_urgency"] == 0.975
    assert routed["final_priority_score"] == 1.0  # Existing synergy boost.
    assert routed["predicted_department"] == "ROADS"
    assert sentiment.call_args_list[0] == sentiment.call_args_list[1]
    assert vision.call_count == 2
    assert routed["request_id"] == routed_response.headers["x-request-id"]
    with sessions() as db:
        ticket = db.get(model, routed["ticket_id"])
        assert ticket.predicted_department == "ROADS"
        assert ticket.routing_model_version == routed["routing_model_version"]
    listed = client.get("/api/v1/report").json()
    assert len(listed) == 2 and all("department_confidence" in t for t in listed)
    assert client.get("/api/v1/tickets").status_code == 200
    patched = client.patch(f'/api/v1/report/{routed["ticket_id"]}', json={"status": "RESOLVED"})
    assert patched.json()["predicted_department"] == "ROADS"


def test_semantic_duplicate_preserves_both_reports(api):
    client, sessions, model, _, _, _ = api
    first = submit(client).json()
    with sessions() as db:
        ticket = db.get(model, first["ticket_id"])
        ticket.predicted_department = "WATER_SUPPLY"
        db.commit()
    second = submit(client, description="A dangerous deep pothole is blocking traffic on the road").json()
    assert second["ticket_id"] != first["ticket_id"]
    assert second["is_duplicate"] and second["report_count"] == 1
    assert second["duplicate_similarity"] >= 0.80
    listed = client.get("/api/v1/report").json()
    assert len(listed) == 2
    assert all(t["duplicate_group_id"] == second["duplicate_group_id"] for t in listed)
    assert all(t["related_report_count"] == 1 and t["status"] == "OPEN" for t in listed)
    original = next(t for t in listed if t["ticket_id"] == first["ticket_id"])
    assert original["predicted_department"] == "WATER_SUPPLY"
    assert original["final_priority_score"] == first["final_priority_score"]


def test_semantic_nonduplicate_and_distance_smoke(api):
    client, _, _, _, _, _ = api
    submit(client)
    unrelated = submit(client, title="Street light outage", description="The streetlight bulb is dark every evening").json()
    assert not unrelated["is_duplicate"] and unrelated["duplicate_group_id"] is None
    distant = submit(client, lat=19.077).json()
    assert not distant["is_duplicate"]
    assert len(client.get("/api/v1/report").json()) == 3


def test_semantic_failure_still_saves_report(api, monkeypatch):
    from backend.services import semantic_duplicate_service as semantic
    client, _, _, _, _, _ = api
    submit(client)
    monkeypatch.setattr(semantic, "find_semantic_match", Mock(side_effect=RuntimeError("model unavailable")))
    response = submit(client)
    assert response.status_code == 201
    assert response.json()["duplicate_group_id"] is None
    assert len(client.get("/api/v1/report").json()) == 2


def test_escaped_routing_failure_does_not_break_submission(api, monkeypatch):
    client, _, _, endpoints, _, _ = api
    monkeypatch.setattr(endpoints, "predict_department", Mock(side_effect=RuntimeError("private complaint")))
    response = submit(client)
    assert response.status_code == 201
    assert response.json()["predicted_department"] == "MANUAL_REVIEW"
    assert response.json()["final_priority_score"] == 1.0


def test_validation_and_legacy_null_routing(api):
    client, sessions, model, _, _, _ = api
    response = client.post("/api/v1/report", data={})
    assert response.status_code == 422 and response.headers["x-request-id"]
    with sessions() as db:
        db.add(model(title="Historic", category="pothole", latitude=19.0, longitude=72.0))
        db.commit()
    assert client.get("/api/v1/report").json()[0]["predicted_department"] is None


def test_persistence_errors_do_not_expose_report_data(api, monkeypatch, caplog):
    from sqlalchemy.orm import Session
    client, _, _, _, _, _ = api
    monkeypatch.setattr(Session, "commit", Mock(side_effect=RuntimeError("PRIVATE_REPORT_SENTINEL")))
    response = submit(client)
    assert response.status_code == 500
    assert "PRIVATE_REPORT_SENTINEL" not in response.text
    assert "PRIVATE_REPORT_SENTINEL" not in caplog.text
