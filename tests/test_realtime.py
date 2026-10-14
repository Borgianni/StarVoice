import json
import socket
import threading
import wave
from pathlib import Path

import pytest

from starvoice.opus import OpusEncoder, OpusDecoder, OpusConfig, OpusUnavailable
from starvoice.realtime import (
    KIND_AUDIO, KIND_START, Arrival, WirePacket, choose_playout,
    decode_wire, encode_wire, playout_deadline_ns, receive_speech, send_speech,
)


def test_sv02_packet_roundtrip():
    import uuid
    sid = uuid.uuid4()
    p = WirePacket(KIND_AUDIO, sid, 7, 123456, b"opus")
    assert decode_wire(encode_wire(p)) == p
    with pytest.raises(ValueError):
        decode_wire(encode_wire(p)[:-1])
    with pytest.raises(ValueError):
        decode_wire(b"bad")


def test_frame_selection_is_strictly_causal():
    packets = {
        0: Arrival(101, b"p0"),
        1: Arrival(99, b"lbrr"),
    }
    lbrr = lambda payload: payload == b"lbrr"
    # A late primary is unusable. In-band repair is carried by the NEXT packet.
    assert choose_playout(0, 100, packets, lbrr) == ("fec", b"lbrr", 1)
    assert choose_playout(0, 98, packets, lbrr) == ("plc", None, None)
    assert choose_playout(0, 101, packets, lbrr) == ("primary", b"p0", 0)
    assert choose_playout(0, 100, {0: Arrival(101, b"p0")}, lbrr)[0] == "plc"
    assert choose_playout(0, 100, {1: Arrival(100, b"no-lbrr")}, lbrr)[0] == "plc"


def test_receiver_monotonic_playout_anchor():
    assert playout_deadline_ns(1_000_000_000, 2, 2, 60) == 1_060_000_000
    assert playout_deadline_ns(1_000_000_000, 2, 3, 60) == 1_080_000_000
    assert playout_deadline_ns(1_000_000_000, 2, 0, 60) == 1_020_000_000
    with pytest.raises(ValueError):
        playout_deadline_ns(1, 0, 0, -1)


def test_localhost_real_opus_realtime_receiver(tmp_path: Path):
    try:
        cfg = OpusConfig()
        enc = OpusEncoder(cfg)
        dec = OpusDecoder(cfg)
    except OpusUnavailable:
        pytest.skip("libopus unavailable")
    else:
        enc.close()
        dec.close()

    wav_path = tmp_path / "speech.wav"
    with wave.open(str(wav_path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(48000)
        f.writeframes(b"\0\0" * (cfg.frame_samples * 55))

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    ready = threading.Event()
    receiver_output = tmp_path / "receiver"
    errors = []
    receiver_result = []

    def target():
        try:
            receiver_result.append(receive_speech(
                "127.0.0.1", port, receiver_output,
                buffer_ms=60, idle_timeout_s=5, linger_ms=50, ready_event=ready
            ))
        except Exception as ex:
            errors.append(ex)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    assert ready.wait(2)
    sender = send_speech(
        f"127.0.0.1:{port}", wav_path, tmp_path / "sender",
        policy="always-fec",
    )
    thread.join(timeout=5)
    assert not thread.is_alive(), "receiver failed to terminate"
    assert not errors, errors
    assert receiver_result[0]["frames_decoded"] == 55
    assert sender["frames"] == 55
    assert receiver_result[0]["primary_frames"] >= 50
    assert receiver_result[0]["primary_frames"] + receiver_result[0]["fec_frames"] + receiver_result[0]["plc_frames"] == 55
    rows = [json.loads(line) for line in (receiver_output / "frame-decisions.jsonl").read_text().splitlines()]
    assert len(rows) == 55
    assert [r["sequence"] for r in rows] == list(range(55))
    assert all(r["actual_playout_mono_ns"] >= r["scheduled_playout_mono_ns"] for r in rows)
    assert (receiver_output / "received.wav").exists()
