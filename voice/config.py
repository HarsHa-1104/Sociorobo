"""Central, single-source-of-truth configuration for the Voice Manager.

Every tunable called out in the Phase 2 spec lives here as a dataclass field
with a sensible default. Values can be overridden three ways, in increasing
priority order:

    1. hard-coded defaults below
    2. a YAML file (config/voice_config.yaml by default)
    3. environment variables (VOICE_<SECTION>_<FIELD>, e.g. VOICE_WAKE_THRESHOLD)

Nothing in this module talks to hardware, models, or subprocesses -- it is
pure configuration plumbing so every other module can stay free of
hard-coded paths/values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - yaml is an optional convenience dep
    yaml = None


# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AudioConfig:
    """Shared microphone / speaker configuration."""

    input_device_index: Optional[int] = None  # None = system default; set explicitly on UNO Q
    output_device: str = "default"  # ALSA device string for aplay; NOT the Jetson "hw:0,3" default
    sample_rate: int = 16000
    channels: int = 1
    frame_duration_ms: int = 30  # WebRTC VAD only accepts 10/20/30 ms


@dataclass
class WakeConfig:
    """Wake-word detector configuration."""

    enabled: bool = True
    engine: str = "openwakeword"  # only supported engine for now; see docs/MODEL_DECISION.md
    model_path: Optional[str] = None  # None = use the engine's bundled stock model
    model_name: str = "hey_jarvis"  # stock fallback keyword; swap to a custom "hey_arduino" model once trained+validated
    threshold: float = 0.5  # detection confidence threshold, engine-specific scale [0,1]
    trigger_level: int = 4  # consecutive frames above threshold required before firing (debounce)
    inference_framework: str = "onnx"


@dataclass
class VADConfig:
    """Voice-activity detection + session timing configuration."""

    aggressiveness: int = 2  # webrtcvad 0-3, higher = more aggressive at cutting non-speech
    padding_duration_ms: int = 400  # hangover window used to decide "speech ended" -- generous
                                     # enough to survive brief mid-sentence pauses
    activation_ratio: float = 0.6
    deactivation_ratio: float = 0.85
    no_speech_timeout_s: float = 6.0     # Section 8A: 5-7s, if nothing detected after wake word
    max_session_duration_s: float = 20.0  # Section 8B: hard ceiling, not a target duration
    post_tts_cooldown_s: float = 0.5     # mic/wake re-arm grace period after playback ends


@dataclass
class STTConfig:
    """whisper.cpp configuration."""

    binary_path: str = "/opt/whisper.cpp/build/bin/whisper-cli"
    model_path: str = "/opt/whisper.cpp/models/ggml-base.en-q5_0.bin"
    threads: int = 4  # UNO Q has exactly 4 cores; benchmarked encode time scales near-linearly
                       # with threads (2->3->4 threads: ~19.6s->14.5s->10.1s on a 1.5s clip) and
                       # Phase 1 has no other CPU-heavy process competing for cores. Revisit if a
                       # future phase adds a concurrent vision/motor pipeline sharing this CPU.
    language: str = "en"
    timeout_s: float = 15.0  # shortened from the old project's 30s; short commands only


@dataclass
class LLMConfig:
    """Local Ollama LLM configuration."""

    url: str = "http://localhost:11434/api/chat"
    model: str = "gemma3:270m"  # see docs/MODEL_DECISION.md for the evidence behind this default
    stream: bool = True
    timeout_s: float = 20.0  # shortened hard from the old project's 120s -- robot is standing still
    num_predict: int = 96    # cap response length; keep replies short by construction, not just by prompt
    temperature: float = 0.6
    keep_alive: str = "10m"  # explicit residency window instead of relying on Ollama's implicit default
    system_prompt: str = (
        "You are a helpful voice assistant on a small mobile robot. "
        "Answer in one short sentence, two at most. "
        "Do not use lists, markdown, or emoji. Be direct and concise."
    )


@dataclass
class TTSConfig:
    """Piper configuration."""

    binary_path: str = "/opt/piper/piper"
    model_path: str = "/opt/piper/voices/en_US-lessac-medium.onnx"  # medium tier by default; see MODEL_DECISION.md
    model_json_path: Optional[str] = None  # derived from model_path + ".json" if unset
    sample_rate: int = 22050
    timeout_s: float = 20.0
    persistent: bool = False  # Milestone 8: PersistentPiperTTS keeps one Piper process alive
                               # across requests instead of reloading the ONNX model every call
                               # (measured directly: ~1.9-2.4s reload cost per call, eliminated
                               # after the first request). Default False until human-confirmed
                               # audible quality matches the original -- see the Milestone 8
                               # commit for the real-hardware benchmark and confirmation.
    bluetooth_speaker_mac: Optional[str] = None  # Milestone 8: the production speaker (e.g. a
                               # Bluetooth device like "HBTS001") only exists as a PipeWire node --
                               # confirmed on real hardware that `aplay` cannot reach it at all (no
                               # PipeWire-ALSA compat plugin installed). When set, playback is
                               # routed via pw-play to whichever PipeWire node currently has this
                               # MAC as its api.bluez5.address, looked up fresh on every call (never
                               # a cached/hardcoded sink id -- those are reassigned on every
                               # reconnect). None keeps the original aplay/ALSA path unchanged.


@dataclass
class IPCConfig:
    """Voice Manager <-> HumanFollower local IPC configuration."""

    socket_path: str = "/tmp/humanfollower_voice.sock"
    connect_timeout_s: float = 3.0
    message_timeout_s: float = 5.0
    heartbeat_interval_s: float = 2.0
    pause_confirm_timeout_s: float = 8.0  # how long Voice Manager waits for PAUSE_CONFIRMED before proceeding anyway


@dataclass
class WatchdogConfig:
    """HumanFollower-side watchdog (reference values; HumanFollower owns the real enforcement)."""

    max_paused_duration_s: float = 45.0  # absolute ceiling from PAUSE_REQUEST to forced resume
    heartbeat_timeout_s: float = 6.0     # no heartbeat for this long during a session => assume Voice Manager is dead


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_dir: str = "/var/log/voice_manager"  # falls back to ./logs if not writable


@dataclass
class VoiceSystemConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    ipc: IPCConfig = field(default_factory=IPCConfig)
    watchdog: WatchdogConfig = field(default_factory=WatchdogConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ---------------------------------------------------------------------------
# Loading / overlay logic
# ---------------------------------------------------------------------------

_ENV_PREFIX = "VOICE"


def _coerce(raw: str, target_type: type) -> Any:
    if target_type is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if target_type is int:
        return int(raw)
    if target_type is float:
        return float(raw)
    if target_type is Optional[int]:
        return None if raw.strip().lower() in ("", "none", "null") else int(raw)
    return raw


def _apply_env_overrides(cfg: VoiceSystemConfig) -> None:
    for section_field in fields(cfg):
        section_name = section_field.name
        section_obj = getattr(cfg, section_name)
        if not is_dataclass(section_obj):
            continue
        for value_field in fields(section_obj):
            env_key = f"{_ENV_PREFIX}_{section_name.upper()}_{value_field.name.upper()}"
            if env_key in os.environ:
                raw = os.environ[env_key]
                current = getattr(section_obj, value_field.name)
                target_type = type(current) if current is not None else str
                setattr(section_obj, value_field.name, _coerce(raw, target_type))


def _apply_yaml_overrides(cfg: VoiceSystemConfig, yaml_path: Path) -> None:
    if yaml is None or not yaml_path.exists():
        return
    with open(yaml_path, "r") as fh:
        data = yaml.safe_load(fh) or {}
    for section_name, section_values in data.items():
        if not hasattr(cfg, section_name):
            continue
        section_obj = getattr(cfg, section_name)
        if not isinstance(section_values, dict):
            continue
        for key, value in section_values.items():
            if hasattr(section_obj, key):
                setattr(section_obj, key, value)


def load_config(yaml_path: Optional[str] = None) -> VoiceSystemConfig:
    """Build the effective config: defaults -> YAML file -> environment variables."""

    cfg = VoiceSystemConfig()

    default_yaml = Path(__file__).resolve().parent.parent / "config" / "voice_config.yaml"
    path = Path(yaml_path) if yaml_path else default_yaml
    _apply_yaml_overrides(cfg, path)

    # Environment variables always win, so a systemd unit or Docker env can
    # override a shipped YAML file without editing it.
    _apply_env_overrides(cfg)

    return cfg


__all__ = [
    "AudioConfig",
    "WakeConfig",
    "VADConfig",
    "STTConfig",
    "LLMConfig",
    "TTSConfig",
    "IPCConfig",
    "WatchdogConfig",
    "LoggingConfig",
    "VoiceSystemConfig",
    "load_config",
]
