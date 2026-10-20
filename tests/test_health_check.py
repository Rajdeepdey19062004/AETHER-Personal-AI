from unittest.mock import patch

import requests

from health_check import check_index_health, check_ollama_health


def test_index_health():
    with patch("health_check.load_index", return_value=([], __import__("numpy").empty((0, 0)), [])):
        result = check_index_health()
    assert result["status"] == "ok"
    assert result["documents"] == 0


def test_ollama_unavailable():
    with patch("health_check.requests.get", side_effect=requests.ConnectionError("Connection refused")):
        result = check_ollama_health()
    assert result["status"] == "error"
