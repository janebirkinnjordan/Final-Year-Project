from fastapi.testclient import TestClient
from backend.app.main import app


def test_create_and_list_session():
    client = TestClient(app)

    # Create a session
    create = client.post('/sessions', json={'title': 'Test Session'})
    assert create.status_code == 200
    session_id = create.json()['id']
    assert create.json()['title'] == 'Test Session'

    # List sessions and confirm it exists
    listing = client.get('/sessions')
    assert listing.status_code == 200
    ids = [item['id'] for item in listing.json()]
    assert session_id in ids

    # Clean up — delete the session
    delete = client.delete(f'/sessions/{session_id}')
    assert delete.status_code == 204
    