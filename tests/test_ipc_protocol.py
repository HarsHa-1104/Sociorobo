from __future__ import annotations

from voice.ipc.protocol import Message, MessageType, decode_lines, encode


def test_round_trip_encode_decode():
    msg = Message(type=MessageType.PAUSE_REQUEST, session_id="abc123")
    raw = encode(msg)
    messages, remainder = decode_lines(raw)
    assert remainder == b""
    assert len(messages) == 1
    assert messages[0].type == MessageType.PAUSE_REQUEST
    assert messages[0].session_id == "abc123"


def test_reason_and_extra_survive_round_trip():
    msg = Message(type=MessageType.ERROR, session_id="s1", reason="stt crashed")
    raw = encode(msg)
    messages, _ = decode_lines(raw)
    assert messages[0].reason == "stt crashed"

    msg2 = Message(type=MessageType.VOICE_SESSION_COMPLETE, session_id="s2",
                    extra={"outcome": "answered"})
    messages2, _ = decode_lines(encode(msg2))
    assert messages2[0].extra == {"outcome": "answered"}


def test_multiple_messages_in_one_buffer():
    raw = encode(Message(type=MessageType.HEARTBEAT, session_id="s1")) + \
          encode(Message(type=MessageType.HEARTBEAT, session_id="s1"))
    messages, remainder = decode_lines(raw)
    assert len(messages) == 2
    assert remainder == b""


def test_partial_trailing_line_is_buffered_not_dropped():
    complete = encode(Message(type=MessageType.HEARTBEAT, session_id="s1"))
    partial = b'{"type": "PAUSE_REQ'  # deliberately truncated, no trailing newline
    messages, remainder = decode_lines(complete + partial)
    assert len(messages) == 1
    assert remainder == partial

    # Feeding the rest later, prefixed with the remainder, completes it.
    rest = b'UEST", "session_id": "s2", "ts": 1.0}\n'
    messages2, remainder2 = decode_lines(remainder + rest)
    assert len(messages2) == 1
    assert messages2[0].type == MessageType.PAUSE_REQUEST
    assert remainder2 == b""


def test_malformed_json_is_dropped_not_fatal():
    raw = b"not valid json at all\n" + encode(Message(type=MessageType.HEARTBEAT, session_id="s1"))
    messages, _ = decode_lines(raw)
    assert len(messages) == 1
    assert messages[0].type == MessageType.HEARTBEAT
