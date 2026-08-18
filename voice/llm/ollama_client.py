"""Local Ollama LLM client -- single-turn, stateless by design.

Ported from the old project's ``llm/ollama.py``. That file was already
stateless per call (no conversation history retained), which happens to
match Section 5 of the Phase 2 spec exactly ("one wake word = one question
= one response", no multi-turn memory) -- so the core request-building
logic is kept as-is.

Changes made here:

  * ``model`` now defaults to a config value decided in
    docs/MODEL_DECISION.md, not hard-coded to whatever the old project's
    author last tested with.
  * ``timeout`` shortened hard from the old project's 120s to a
    configurable default of 20s (Section 25) -- the robot is standing
    still waiting on this call; a 2-minute worst case is not acceptable
    next to a person-following robot.
  * ``keep_alive`` is now sent explicitly (Section 18D) instead of relying
    on whatever Ollama's own implicit default happens to be.
  * ``num_predict`` default lowered and the system prompt tightened so
    responses stay short by construction (Section 12/25), not just by
    hopeful prompting.
"""

from __future__ import annotations

import json
import logging
from typing import List

import requests

from voice.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def _build_messages(self, user_text: str) -> List[dict]:
        messages: List[dict] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": user_text})
        return messages

    def query(self, user_text: str) -> str:
        """One-shot query. No history is sent or retained -- see module docstring."""
        payload = {
            "model": self.config.model,
            "messages": self._build_messages(user_text),
            "stream": self.config.stream,
            "keep_alive": self.config.keep_alive,
            "options": {
                "num_predict": self.config.num_predict,
                "temperature": self.config.temperature,
            },
        }
        logger.info("Querying Ollama model=%s: %r", self.config.model, user_text)

        try:
            response = requests.post(
                self.config.url,
                json=payload,
                stream=self.config.stream,
                timeout=self.config.timeout_s,
            )
        except requests.exceptions.ConnectionError:
            logger.error("Ollama connection error -- is `ollama serve` running?")
            return ""
        except requests.exceptions.Timeout:
            logger.warning("Ollama request timed out after %.1fs.", self.config.timeout_s)
            return ""

        if response.status_code != 200:
            logger.error("Ollama returned %s: %s", response.status_code, response.text[:300])
            return ""

        response_text = ""
        try:
            if self.config.stream:
                for line in response.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line.decode("utf-8"))
                    if chunk.get("done"):
                        break
                    content = chunk.get("message", {}).get("content")
                    if content:
                        response_text += content
            else:
                data = response.json()
                response_text = data.get("message", {}).get("content", "")
        except (json.JSONDecodeError, requests.exceptions.ChunkedEncodingError) as exc:
            logger.error("Failed to parse Ollama response stream: %s", exc)
            return response_text.strip()

        return response_text.strip()


__all__ = ["OllamaClient", "LLMError"]
