"""Real socket integration test: HumanFollowerLink talking to
ReferenceHumanFollowerServer over an actual Unix domain socket (no mocks on
either side of the wire). This is the closest thing in this suite to a true
integration test, and it doesn't need any audio/model hardware to run.
"""

from __future__ import annotations

import time

import pytest

from voice.config import IPCConfig, WatchdogConfig
from voice.ipc.client import HumanFollowerLink
from voice.ipc.server_stub import ReferenceHumanFollowerServer


@pytest.fixture
def ipc_config(tmp_path):
    return IPCConfig(
        socket_path=str(tmp_path / "test_voice.sock"),
        connect_timeout_s=2.0,
        message_timeout_s=2.0,
        heartbeat_interval_s=0.2,
        pause_confirm_timeout_s=2.0,
    )


def test_pause_request_gets_confirmed(ipc_config):
    pause_calls = []
    resume_calls = []

    server = ReferenceHumanFollowerServer(
        ipc_config, WatchdogConfig(),
        on_pause=lambda: (pause_calls.append(1), True)[1],
        on_resume=lambda: resume_calls.append(1),
    )
    server.start()
    try:
        link = HumanFollowerLink(ipc_config)
        confirmed = link.request_pause()
        assert confirmed is True
        assert pause_calls == [1]

        link.session_complete(outcome="answered")
        time.sleep(0.2)  # let the server thread process it
        assert resume_calls == [1]
        link.close()
    finally:
        server.stop()


def test_pause_not_confirmed_when_on_pause_returns_false(ipc_config):
    server = ReferenceHumanFollowerServer(
        ipc_config, WatchdogConfig(),
        on_pause=lambda: False,
        on_resume=lambda: None,
    )
    server.start()
    try:
        link = HumanFollowerLink(ipc_config)
        confirmed = link.request_pause()
        assert confirmed is False
        link.close()
    finally:
        server.stop()


def test_no_server_running_fails_soft(ipc_config):
    """Voice Manager must not raise/crash just because HumanFollower isn't
    reachable -- Section 17.
    """
    link = HumanFollowerLink(ipc_config)
    confirmed = link.request_pause()
    assert confirmed is False
    link.close()


def test_watchdog_force_resumes_on_missing_heartbeat(ipc_config):
    """Section 17: if Voice Manager goes silent mid-session, HumanFollower's
    watchdog must resume on its own -- this test proves the reference
    watchdog implementation actually does that, using a very short
    heartbeat_timeout_s so the test runs fast.
    """
    resume_calls = []
    forced_reasons = []

    watchdog_cfg = WatchdogConfig(max_paused_duration_s=30.0, heartbeat_timeout_s=0.3)
    server = ReferenceHumanFollowerServer(
        ipc_config, watchdog_cfg,
        on_pause=lambda: True,
        on_resume=lambda: resume_calls.append(1),
        on_watchdog_forced_resume=lambda reason: forced_reasons.append(reason),
    )
    server.start()
    try:
        link = HumanFollowerLink(ipc_config)
        link.request_pause()
        # Simulate Voice Manager dying: no heartbeat, no session_complete ever sent.
        link.close()

        time.sleep(1.0)  # well past heartbeat_timeout_s=0.3
        assert resume_calls == [1]
        assert len(forced_reasons) == 1
        assert "heartbeat_timeout_s" in forced_reasons[0]
    finally:
        server.stop()


def test_watchdog_max_paused_duration_trips_even_with_heartbeats(ipc_config):
    """A Voice Manager stuck in a genuine infinite loop but still
    heartbeating must still be force-resumed eventually.
    """
    resume_calls = []
    forced_reasons = []

    watchdog_cfg = WatchdogConfig(max_paused_duration_s=0.5, heartbeat_timeout_s=5.0)
    server = ReferenceHumanFollowerServer(
        ipc_config, watchdog_cfg,
        on_pause=lambda: True,
        on_resume=lambda: resume_calls.append(1),
        on_watchdog_forced_resume=lambda reason: forced_reasons.append(reason),
    )
    server.start()
    try:
        link = HumanFollowerLink(ipc_config)
        link.request_pause()
        link.start_heartbeat()  # heartbeats keep flowing...

        time.sleep(1.0)  # ...but max_paused_duration_s=0.5 still trips
        assert resume_calls == [1]
        assert "max_paused_duration_s" in forced_reasons[0]
        link.close()
    finally:
        server.stop()
