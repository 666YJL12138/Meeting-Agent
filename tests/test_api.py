from fastapi.testclient import TestClient
from apps.api.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_meeting_flow():
    r = client.post("/meetings", json={
        "title": "周会",
        "host": "张三",
        "language": "zh-CN",
        "participants": ["张三", "李四"]
    })
    assert r.status_code == 200
    meeting_id = r.json()["meeting_id"]

    r2 = client.get(f"/meetings/{meeting_id}/status")
    assert r2.status_code == 200
    assert r2.json()["status"] == "created"
