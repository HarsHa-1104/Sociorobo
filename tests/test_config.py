from __future__ import annotations

import os

from voice.config import load_config


def test_defaults_load_without_a_yaml_file(tmp_path):
    cfg = load_config(str(tmp_path / "does_not_exist.yaml"))
    assert cfg.vad.max_session_duration_s == 20.0
    assert cfg.vad.no_speech_timeout_s == 6.0
    assert cfg.wake.enabled is True


def test_yaml_overrides_defaults(tmp_path):
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "llm:\n  model: \"qwen2.5:1.5b-instruct\"\n  timeout_s: 12.5\n"
        "vad:\n  max_session_duration_s: 25\n"
    )
    cfg = load_config(str(yaml_path))
    assert cfg.llm.model == "qwen2.5:1.5b-instruct"
    assert cfg.llm.timeout_s == 12.5
    assert cfg.vad.max_session_duration_s == 25
    # Untouched fields keep their defaults.
    assert cfg.stt.threads == 4


def test_env_vars_override_yaml(tmp_path, monkeypatch):
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("llm:\n  model: \"gemma3:270m\"\n")
    monkeypatch.setenv("VOICE_LLM_MODEL", "qwen2.5:1.5b-instruct")
    monkeypatch.setenv("VOICE_VAD_MAX_SESSION_DURATION_S", "25")
    monkeypatch.setenv("VOICE_WAKE_ENABLED", "false")

    cfg = load_config(str(yaml_path))
    assert cfg.llm.model == "qwen2.5:1.5b-instruct"  # env beats yaml
    assert cfg.vad.max_session_duration_s == 25.0
    assert cfg.wake.enabled is False
