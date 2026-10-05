from fastapi.testclient import TestClient

from backend.app.main import app


def test_health_route_exists():
    client = TestClient(app)
    response = client.get('/health')
    assert response.status_code == 200
    data = response.json()
    assert data['ok'] is True
    assert 'domains' in data
