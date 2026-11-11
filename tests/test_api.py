from fastapi.testclient import TestClient

from api.main import app

c = TestClient(app)


def test_health_and_session():
    assert c.get("/health").json() == {"status": "ok"}
    sid = c.post("/sessions", json={"guarded": True}).json()["session_id"]
    assert len(sid) == 12


def test_guarded_session_handles_urgent_without_llm():
    sid = c.post("/sessions", json={"guarded": True}).json()["session_id"]
    r = c.post(f"/sessions/{sid}/chat", json={"message": "I have crushing chest pain right now"}).json()
    assert "911" in r["reply"] and r["escalated"]
