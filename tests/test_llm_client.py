from __future__ import annotations

import json
from unittest import mock

import requests

from voice.config import LLMConfig
from voice.llm.ollama_client import OllamaClient


def _fake_stream_response(chunks, status_code=200):
    resp = mock.Mock()
    resp.status_code = status_code
    lines = [json.dumps(c).encode("utf-8") for c in chunks]
    resp.iter_lines.return_value = iter(lines)
    return resp


def test_query_builds_single_turn_messages_no_history():
    client = OllamaClient(LLMConfig(system_prompt="Be brief."))
    messages = client._build_messages("what time is it")
    assert messages == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "what time is it"},
    ]
    # Calling again with a different question must not accumulate history.
    messages2 = client._build_messages("second question")
    assert messages2 == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "second question"},
    ]


def test_query_assembles_streamed_content():
    client = OllamaClient(LLMConfig())
    chunks = [
        {"message": {"content": "The weather "}, "done": False},
        {"message": {"content": "is sunny."}, "done": False},
        {"done": True},
    ]
    fake_resp = _fake_stream_response(chunks)
    with mock.patch("requests.post", return_value=fake_resp):
        text = client.query("what's the weather?")
    assert text == "The weather is sunny."


def test_query_sends_keep_alive_and_num_predict():
    client = OllamaClient(LLMConfig(keep_alive="10m", num_predict=64))
    fake_resp = _fake_stream_response([{"done": True}])
    with mock.patch("requests.post", return_value=fake_resp) as post_mock:
        client.query("hi")
    _, kwargs = post_mock.call_args
    assert kwargs["json"]["keep_alive"] == "10m"
    assert kwargs["json"]["options"]["num_predict"] == 64


def test_query_returns_empty_on_connection_error():
    client = OllamaClient(LLMConfig())
    with mock.patch("requests.post", side_effect=requests.exceptions.ConnectionError()):
        assert client.query("hi") == ""


def test_query_returns_empty_on_timeout():
    client = OllamaClient(LLMConfig(timeout_s=1.0))
    with mock.patch("requests.post", side_effect=requests.exceptions.Timeout()):
        assert client.query("hi") == ""


def test_query_returns_empty_on_non_200():
    client = OllamaClient(LLMConfig())
    fake_resp = mock.Mock(status_code=404, text="model not found")
    with mock.patch("requests.post", return_value=fake_resp):
        assert client.query("hi") == ""
