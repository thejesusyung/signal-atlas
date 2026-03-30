from __future__ import annotations


def test_articles_endpoint_filters_by_keyword(client):
    response = client.get("/articles", params={"q": "orbital"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["items"][0]["title"] == "Acme launches orbital product"


def test_article_detail_returns_entities_and_topics(client):
    list_response = client.get("/articles")
    article_id = list_response.json()["items"][0]["id"]

    response = client.get(f"/articles/{article_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["cleaned_text"] == "Acme announced a new orbital platform in Lima."
    assert payload["entities"][0]["name"] == "Acme"
    assert payload["topics"][0]["name"] == "technology"


def test_entities_endpoint_lists_entities(client):
    response = client.get("/entities")

    assert response.status_code == 200
    assert response.json()["items"][0]["name"] == "Acme"


def test_entities_endpoint_rejects_invalid_entity_type(client):
    response = client.get("/entities", params={"entity_type": "not-a-real-type"})

    assert response.status_code == 422


def test_entity_articles_endpoint_returns_linked_articles(client):
    entity_id = client.get("/entities").json()["items"][0]["id"]

    response = client.get(f"/entities/{entity_id}/articles")

    assert response.status_code == 200
    assert response.json()["total"] == 1


def test_topics_and_stats_endpoints(client):
    topics_response = client.get("/topics")
    stats_response = client.get("/stats")

    assert topics_response.status_code == 200
    assert topics_response.json()["items"][0]["name"] == "technology"
    assert stats_response.status_code == 200
    assert stats_response.json()["total_articles"] == 1


def test_articles_endpoint_uses_inclusive_date_range(client):
    published_at = client.get("/articles").json()["items"][0]["published_at"]
    published_date = published_at.split("T", 1)[0]

    response = client.get("/articles", params={"date_to": published_date})

    assert response.status_code == 200
    assert response.json()["total"] == 1
