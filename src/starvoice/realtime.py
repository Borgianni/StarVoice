"""SV02 real-time speech transport; experimental receiver-local playout only.

This is deliberately separate from the frozen SV01 echo/probe protocol.
Receiver monotonic timestamps define a *jitter-buffer* deadline, NOT absolute OWD.
"""
from __future__ import annotations

import json
import math
import random
import select
import socket
import struct
import time
import uuid
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .opus import OpusConfig, OpusDecoder, OpusEncoder, packet_has_fec
from .predictor import PhaseModel
from .probe import parse_target
from .speech import _policy_fec
from .storage import JsonlWriter, dump_json, environment_snapshot, utc_now_iso

MAGIC = b"SV02"
KIND_START = 1
KIND_ACK = 2
KIND_AUDIO = 3
KIND_END = 4
HEADER = struct.Struct("!4sB16sIQH")
FRAME_MS = 20
SAMPLE_RATE = 48000
CHANNELS = 1


@dataclass(frozen=True)
class WirePacket:
    kind: int
    session_id: uuid.UUID
    sequence: int
    send_mono_ns: int
    payload: bytes = b""


@dataclass(frozen=True)
class Arrival:
    receive_mono_ns: int
    payload: bytes


def encode_wire(p: WirePacket) -> bytes:
    if p.kind not in (KIND_START, KIND_ACK, KIND_AUDIO, KIND_END):
        raise ValueError("invalid SV02 packet kind")
    if not 0 <= p.sequence <= 0xFFFFFFFF or not 0 <= p.send_mono_ns <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("SV02 sequence/timestamp out of range")
    if len(p.payload) > 65535:
        raise ValueError("SV02 payload too large")
    return HEADER.pack(
        MAGIC, p.kind, p.session_id.bytes, p.sequence, p.send_mono_ns, len(p.payload)
    ) + p.payload


def decode_wire(data: bytes) -> WirePacket:
    if len(data) < HEADER.size:
        raise ValueError("short SV02 packet")
    magic, kind, sid, seq, send_ns, size = HEADER.unpack_from(data)
    if magic != MAGIC or kind not in (KIND_START, KIND_ACK, KIND_AUDIO, KIND_END):
        raise ValueError("invalid SV02 header")
    if len(data) != HEADER.size + size:
        raise ValueError("invalid SV02 payload length")
    return WirePacket(kind, uuid.UUID(bytes=sid), seq, send_ns, data[HEADER.size:])


def choose_playout(
    seq: int,
    deadline_ns: int,
    arrivals: dict[int, Arrival],
    has_lbrr: Callable[[bytes], bool] = packet_has_fec,
) -> tuple[str, bytes | None, int | None]:
    """Select using only packets received no later than the frame deadline.

    The FEC for source frame seq is in packet seq+1, never packet seq itself.
    """
    primary = arrivals.get(seq)
    if primary is not None and primary.receive_mono_ns <= deadline_ns:
        return "primary", primary.payload, seq
    nxt = arrivals.get(seq + 1)
    if (
        nxt is not None
        and nxt.receive_mono_ns <= deadline_ns
        and has_lbrr(nxt.payload)
    ):
        return "fec", nxt.payload, seq + 1
    return "plc", None, None


def playout_deadline_ns(
    first_arrival_ns: int, first_sequence: int, sequence: int, buffer_ms: float
) -> int:
    if first_sequence < 0 or sequence < 0 or buffer_ms < 0:
        raise ValueError("invalid playout parameters")
    return (
        first_arrival_ns
        + (sequence - first_sequence) * FRAME_MS * 1_000_000
        + round(buffer_ms * 1_000_000)
    )


def _validated_start(payload: bytes) -> dict:
    try:
        obj = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid START metadata") from exc
    if not isinstance(obj, dict):
        raise TypeError("START metadata must be an object")
    if (
        not 0 < obj.get("frames", 0) <= 1_000_000
        or obj.get("frame_ms") != FRAME_MS
        or obj.get("sample_rate") != SAMPLE_RATE
        or obj.get("channels") != CHANNELS
    ):
        raise ValueError("invalid START stream parameters")
    return obj


def send_speech(
    target: str,
    input_wav: Path,
    output_dir: Path,
    policy: str = "plain",
    predictor: PhaseModel | None = None,
    bitrate: int = 24000,
    expected_loss_percent: int = 20,
    random_duty_cycle: float = 0.1,
    seed: int = 2027,
    drop_sequences: frozenset[int] = frozenset(),
) -> dict:
    """Send paced 20-ms Opus frames; optionally suppress selected datagrams for testing."""
    if policy == "predictive-fec" and predictor is None:
        raise ValueError("predictive-fec requires --predictor")
    if policy not in ("plain", "always-fec", "predictive-fec", "random-fec"):
        raise ValueError(f"unsupported realtime policy: {policy}")
    if not 0.0 <= random_duty_cycle <= 1.0:
        raise ValueError("invalid random duty cycle")
    if any(sequence < 0 for sequence in drop_sequences):
        raise ValueError("drop sequences must be nonnegative")

    cfg = OpusConfig(bitrate=bitrate)
    frame_bytes = cfg.frame_samples * cfg.channels * 2
    sid = uuid.uuid4()
    host, port = parse_target(target)
    addr = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_DGRAM)[0][4]
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=False)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.2)
    enc = OpusEncoder(cfg)
    writer = JsonlWriter(output_dir / "sender.jsonl")
    frames = protected = opus_bytes = payload_bytes = 0
    try:
        with wave.open(str(input_wav), "rb") as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (
                cfg.sample_rate, cfg.channels, 2
            ):
                raise ValueError("WAV must be 48 kHz, mono, PCM16")
            expected = math.ceil(wav.getnframes() / cfg.frame_samples)
            if expected == 0:
                raise ValueError("empty input WAV")
            if any(sequence >= expected for sequence in drop_sequences):
                raise ValueError("drop sequence out of input WAV range")
            start = json.dumps({
                "frames": expected, "frame_ms": FRAME_MS, "sample_rate": SAMPLE_RATE,
                "channels": CHANNELS, "bitrate": bitrate, "policy": policy,
                "expected_loss_percent": expected_loss_percent,
            }, sort_keys=True).encode()
            start_pkt = encode_wire(WirePacket(KIND_START, sid, 0, 0, start))
            acked = False
            for _ in range(15):
                sock.sendto(start_pkt, addr)
                try:
                    raw, peer = sock.recvfrom(65535)
                    ack = decode_wire(raw)
                    if peer == addr and ack.kind == KIND_ACK and ack.session_id == sid:
                        acked = True
                        break
                except (TimeoutError, ValueError):
                    pass
            if not acked:
                raise TimeoutError("SV02 receiver did not ACK START")

            sock.setblocking(False)
            tick0 = time.monotonic_ns() + 10_000_000
            seq = 0
            while True:
                pcm = wav.readframes(cfg.frame_samples)
                if not pcm:
                    break
                if len(pcm) < frame_bytes:
                    pcm += b"\0" * (frame_bytes - len(pcm))
                scheduled_ns = tick0 + seq * cfg.frame_ms * 1_000_000
                while True:
                    remaining_ns = scheduled_ns - time.monotonic_ns()
                    if remaining_ns <= 0:
                        break
                    time.sleep(min(remaining_ns / 1e9, 0.001))
                decision_ns = time.monotonic_ns()
                risk = predictor.risk(decision_ns) if predictor else 0.0
                fec = _policy_fec(policy, risk, rng, random_duty_cycle)
                enc.set_fec(fec, expected_loss_percent)
                opus = enc.encode(pcm)
                send_ns = time.monotonic_ns()
                payload = encode_wire(WirePacket(KIND_AUDIO, sid, seq, send_ns, opus))
                injected_drop = seq in drop_sequences
                if not injected_drop:
                    sock.sendto(payload, addr)
                writer.write({
                    "session_id": str(sid), "sequence": seq,
                    "scheduled_send_mono_ns": scheduled_ns,
                    "send_mono_ns": send_ns, "send_realtime_ns": time.time_ns(),
                    "fec_enabled": fec, "risk": risk, "has_lbrr": packet_has_fec(opus),
                    "injected_drop": injected_drop,
                    "opus_bytes": len(opus), "udp_payload_bytes": len(payload),
                })
                protected += int(fec)
                opus_bytes += len(opus)
                payload_bytes += len(payload)
                frames += 1
                seq += 1

            end = encode_wire(WirePacket(KIND_END, sid, expected, time.monotonic_ns()))
            for _ in range(3):
                sock.sendto(end, addr)
                time.sleep(0.02)
        summary = {
            "schema_version": 2, "session_id": str(sid), "role": "sender",
            "frames": frames, "fec_protected_frames": protected,
            "fec_duty_cycle": protected / frames, "opus_bytes": opus_bytes,
            "udp_payload_bytes": payload_bytes, "policy": policy,
            "target": target, "injected_drops": len(drop_sequences),
            "injected_drop_sequences": sorted(drop_sequences),
        }
        dump_json(output_dir / "manifest.json", {
            **environment_snapshot(), "session_id": str(sid), "role": "sender",
            "protocol": "SV02", "frame_ms": FRAME_MS,
            "sample_rate": SAMPLE_RATE, "channels": CHANNELS, "bitrate": bitrate,
            "policy": policy, "random_seed": seed, "target": target,
            "input_wav": str(input_wav), "frames": expected,
            "injected_drop_sequences": sorted(drop_sequences),
        })
        dump_json(output_dir / "summary.json", summary)
        return summary
    finally:
        writer.close()
        enc.close()
        sock.close()


def receive_speech(
    bind: str, port: int, output_dir: Path, buffer_ms: float = 60.0,
    idle_timeout_s: float = 30.0, linger_ms: float = 200.0,
    ready_event=None,
) -> dict:
    """Decode a single stream on receiver's monotonic playout timeline.

    Anchor: first observed AUDIO frame j arrives at t; frame j playouts at t+B.
    Frames before j are startup-uncertain, separately counted and excluded from
    later confirmatory outcomes. B is receiver buffering, NOT sender-to-playout OWD.
    """
    if not 0 <= buffer_ms <= 5000 or not 0 <= linger_ms <= 5000:
        raise ValueError("invalid timing parameters")
    output_dir.mkdir(parents=True, exist_ok=False)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((bind, port))
    sock.setblocking(False)
    if ready_event is not None:
        ready_event.set()

    session_id = None
    peer = None
    expected = 0
    decoder = None
    writer_rx = writer_decisions = wav_out = None
    arrivals: dict[int, Arrival] = {}
    first_seq = first_ns = None
    next_playout = 0
    counts = {"primary": 0, "fec": 0, "plc": 0}
    last_activity = time.monotonic_ns()
    final_at_ns = None
    duplicates = malformed = 0
    ended = False
    try:
        while True:
            now = time.monotonic_ns()
            if now - last_activity > idle_timeout_s * 1e9:
                raise TimeoutError("SV02 receiver idle timeout")
            if first_ns is not None:
                while next_playout < expected:
                    due = playout_deadline_ns(first_ns, first_seq, next_playout, buffer_ms)
                    now = time.monotonic_ns()
                    if now < due:
                        break
                    decision, payload, repair_seq = choose_playout(
                        next_playout, due, arrivals
                    )
                    try:
                        if decision == "primary":
                            pcm = decoder.decode(payload, fec=False)
                        elif decision == "fec":
                            pcm = decoder.decode(payload, fec=True)
                        else:
                            pcm = decoder.decode(None, fec=False)
                    except RuntimeError:
                        pcm = decoder.decode(None, fec=False)
                        decision = "plc"
                        repair_seq = None
                    wav_out.writeframes(pcm)
                    counts[decision] += 1
                    writer_decisions.write({
                        "session_id": str(session_id), "sequence": next_playout,
                        "scheduled_playout_mono_ns": due,
                        "actual_playout_mono_ns": time.monotonic_ns(),
                        "decision": decision, "source_packet_sequence": repair_seq,
                        "startup_uncertain": next_playout < first_seq,
                    })
                    next_playout += 1

                if next_playout >= expected:
                    if final_at_ns is None:
                        final_at_ns = time.monotonic_ns() + round(linger_ms * 1e6)
                    if time.monotonic_ns() >= final_at_ns:
                        break

            timeout = 0.02
            if first_ns is not None and next_playout < expected:
                next_due = playout_deadline_ns(first_ns, first_seq, next_playout, buffer_ms)
                timeout = max(0.0, min(timeout, (next_due - time.monotonic_ns()) / 1e9))
            readable, _, _ = select.select([sock], [], [], timeout)
            if not readable:
                continue

            while True:
                try:
                    raw, sender_addr = sock.recvfrom(65535)
                except BlockingIOError:
                    break
                receive_ns = time.monotonic_ns()
                try:
                    p = decode_wire(raw)
                except ValueError:
                    malformed += 1
                    continue

                if p.kind == KIND_START and (session_id is None or p.session_id == session_id):
                    if session_id is None:
                        metadata = _validated_start(p.payload)
                        session_id = p.session_id
                        peer = sender_addr
                        expected = metadata["frames"]
                        decoder = OpusDecoder(OpusConfig(bitrate=int(metadata["bitrate"])))
                        writer_rx = JsonlWriter(output_dir / "receiver.jsonl")
                        writer_decisions = JsonlWriter(output_dir / "frame-decisions.jsonl")
                        wav_out = wave.open(str(output_dir / "received.wav"), "wb")  # noqa: SIM115
                        wav_out.setnchannels(CHANNELS)
                        wav_out.setsampwidth(2)
                        wav_out.setframerate(SAMPLE_RATE)
                        dump_json(output_dir / "manifest.json", {
                            **environment_snapshot(), "role": "receiver",
                            "session_id": str(session_id), "protocol": "SV02",
                            "received_start_utc": utc_now_iso(),
                            "start_metadata": metadata, "playout_buffer_ms": buffer_ms,
                            "playout_anchor": "first_audio_arrival_receiver_monotonic",
                            "startup_exclusion_rule": "frames_before_first_received_audio",
                            "deadline_kind": "receiver_relative_buffer_not_absolute_owd",
                        })
                    if sender_addr == peer:
                        sock.sendto(encode_wire(WirePacket(KIND_ACK, session_id, 0, 0)), peer)
                        last_activity = receive_ns
                    continue

                if session_id is None or p.session_id != session_id or sender_addr != peer:
                    continue
                last_activity = receive_ns
                if p.kind == KIND_END:
                    ended = True
                elif p.kind == KIND_AUDIO:
                    if p.sequence >= expected:
                        continue
                    duplicate = p.sequence in arrivals
                    if duplicate:
                        duplicates += 1
                    else:
                        arrivals[p.sequence] = Arrival(receive_ns, p.payload)
                        if first_ns is None:
                            first_ns, first_seq = receive_ns, p.sequence
                    writer_rx.write({
                        "session_id": str(session_id), "sequence": p.sequence,
                        "receive_mono_ns": receive_ns,
                        "receive_realtime_ns": time.time_ns(),
                        "sender_mono_ns_uncomparable": p.send_mono_ns,
                        "opus_bytes": len(p.payload), "udp_payload_bytes": len(raw),
                        "has_lbrr": packet_has_fec(p.payload),
                        "duplicate": duplicate,
                    })

        summary = {
            "schema_version": 2, "session_id": str(session_id), "role": "receiver",
            "frames_expected": expected, "frames_decoded": next_playout,
            "primary_frames": counts["primary"], "fec_frames": counts["fec"],
            "plc_frames": counts["plc"], "primary_deadline_failures":
                counts["fec"] + counts["plc"],
            "initial_frames_before_first_arrival": first_seq,
            "unique_packets_observed": len(arrivals), "duplicate_packets": duplicates,
            "malformed_datagrams": malformed, "end_received": ended,
            "buffer_ms": buffer_ms, "endpoint": "receiver_relative_playout",
            "note": "No absolute OWD claimed. Startup prefix is flagged separately.",
        }
        dump_json(output_dir / "summary.json", summary)
        return summary
    finally:
        if wav_out is not None:
            wav_out.close()
        if writer_rx is not None:
            writer_rx.close()
        if writer_decisions is not None:
            writer_decisions.close()
        if decoder is not None:
            decoder.close()
        sock.close()
