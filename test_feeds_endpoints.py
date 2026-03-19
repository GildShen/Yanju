from fastapi.testclient import TestClient
from research_agent.web import create_app
from research_agent.config import AppConfig

app = create_app()
client = TestClient(app)

def test_feeds_endpoints():
    # Write a test feed
    feed_url = "https://example.com/feed"
    
    # Save the feed
    response = client.post("/api/feeds/save", json={"content": feed_url})
    assert response.status_code == 200, f"Save failed: {response.text}"
    print("Save OK")
    
    # Get the feed
    response = client.get("/api/feeds")
    assert response.status_code == 200, f"Get failed: {response.text}"
    assert response.json()["content"] == feed_url, f"Content mismatch: {response.json()}"
    print("Get OK")

if __name__ == "__main__":
    test_feeds_endpoints()
