"""Subprocess helper that guarantees a timeout kills the whole process
tree, not just the immediate child.

``subprocess.run(..., timeout=...)`` only signals the process it directly
spawned. If that process has itself spawned a child, the grandchild is
orphaned and keeps running indefinitely -- confirmed on real UNO Q
hardware during Milestone 7 fault-injection testing: a hung stand-in
binary left its own child process running after subprocess.run()'s
timeout fired and supposedly killed it. This matters here specifically
because Piper's espeak-ng phonemizer backend is a real example of a
voice-pipeline dependency that can shell out to a separate process.

Starting the child in its own process group (``start_new_session=True``)
and killing the whole group via ``os.killpg`` on timeout closes that gap.
"""

from __future__ import annotations

import os
import signal
import subprocess
from typing import Optional, Sequence


def run_with_group_kill(
    cmd: Sequence[str],
    *,
    input: Optional[bytes] = None,
    timeout: Optional[float] = None,
    text: bool = False,
) -> subprocess.CompletedProcess:
    """Drop-in replacement for ``subprocess.run(..., capture_output=True)``
    whose timeout kills the entire process group, not just the immediate
    child. Raises ``subprocess.TimeoutExpired`` on timeout, same as
    ``subprocess.run``.
    """
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if input is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        text=text,
    )
    try:
        stdout, stderr = proc.communicate(input=input, timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass  # already gone
        proc.communicate()  # reap it, avoid a zombie
        raise
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


__all__ = ["run_with_group_kill"]
