import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask_ui import app


pytestmark = pytest.mark.functional


@pytest.fixture
def client(tmp_path, monkeypatch):
    result_path = tmp_path / "acsd_compare_demo.json"
    result_path.write_text(json.dumps({"config": {"mode": "compare"}, "runs": {}}))
    (tmp_path / "fixed_window_compare_demo.json").write_text(
        json.dumps({"config": {"mode": "compare"}, "runs": {}})
    )
    monkeypatch.setattr("flask_ui.RESULTS_DIR", tmp_path)
    with app.test_client() as client:
        yield client


def test_index_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    text = response.get_data(as_text=True)
    assert "Speculative Decoding Inspector" in text
    assert "Load Server Result" in text
    assert "ACSD traces" in text


def test_results_api_lists_and_serves_files(client):
    listing = client.get("/api/results")
    assert listing.status_code == 200
    payload = listing.get_json()
    assert payload["results"][0]["name"] == "acsd_compare_demo.json"

    fetched = client.get("/api/results/acsd_compare_demo.json")
    assert fetched.status_code == 200
    assert fetched.get_json()["config"]["mode"] == "compare"
