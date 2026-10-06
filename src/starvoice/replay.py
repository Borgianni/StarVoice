from __future__ import annotations

import json
import math
import random
import wave
from pathlib import Path

from .opus import OpusConfig, OpusDecoder, OpusEncoder, packet_has_fec
from .predictor import PhaseModel
from .speech import _policy_fec
from .storage import JsonlWriter


def _trace_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty trace")
    rows.sort(key=lambda r: int(r["send_ns"]))
    return rows


def _exact_random_schedule(total_frames: int, duty_cycle: float, rng: random.Random) -> set[int]:
    if not 0.0 <= duty_cycle <= 1.0:
        raise ValueError("random duty cycle must be in [0, 1]")
    protected = int(round(total_frames * duty_cycle))
    if protected <= 0:
        return set()
    if protected >= total_frames:
        return set(range(total_frames))
    return set(rng.sample(range(total_frames), protected))



def replay_fixed_schedule_recovery_only(
    input_wav: Path,
    trace: Path,
    schedule: set[int],
    trace_offset_frames: int = 0,
    bitrate: int = 24000,
    expected_loss_percent: int = 20,
) -> dict:
    """Replay an exact externally supplied FEC schedule and count real LBRR recovery."""
    rows = _trace_rows(trace)
    cfg = OpusConfig(bitrate=bitrate)
    enc = OpusEncoder(cfg)
    frame_bytes = cfg.frame_samples * cfg.channels * 2
    trace_offset_frames %= len(rows)

    try:
        with wave.open(str(input_wav), "rb") as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (
                cfg.sample_rate,
                cfg.channels,
                2,
            ):
                raise ValueError("input WAV must be 48 kHz mono PCM16")

            total_frames = math.ceil(w.getnframes() / cfg.frame_samples)
            if any(i < 0 or i >= total_frames for i in schedule):
                raise ValueError("fixed schedule contains out-of-range frame")

            encoded: list[bytes] = []
            trace_indices: list[int] = []
            i = 0
            while True:
                pcm = w.readframes(cfg.frame_samples)
                if not pcm:
                    break
                if len(pcm) < frame_bytes:
                    pcm += b"\0" * (frame_bytes - len(pcm))
                fec = i in schedule
                enc.set_fec(fec, expected_loss_percent)
                encoded.append(enc.encode(pcm))
                trace_indices.append((trace_offset_frames + i) % len(rows))
                i += 1

        lost = 0
        recovered = 0
        lbrr_present = 0
        for i in range(len(encoded)):
            if not bool(rows[trace_indices[i]].get("lost")):
                continue
            lost += 1
            if i + 1 >= len(encoded):
                continue
            if bool(rows[trace_indices[i + 1]].get("lost")):
                continue
            if (i + 1) not in schedule:
                continue
            if packet_has_fec(encoded[i + 1]):
                lbrr_present += 1
                recovered += 1

        return {
            "frames": len(encoded),
            "network_lost_frames": lost,
            "fec_recovered_frames": recovered,
            "fec_protected_frames": len(schedule),
            "fec_duty_cycle": len(schedule) / len(encoded) if encoded else None,
            "scheduled_received_following_packets_with_lbrr": lbrr_present,
            "trace_offset_frames": trace_offset_frames,
        }
    finally:
        enc.close()

def replay_random_recovery_only(
    input_wav: Path,
    trace: Path,
    protected_frames: int,
    seed: int,
    trace_offset_frames: int = 0,
    bitrate: int = 24000,
    expected_loss_percent: int = 20,
) -> dict:
    """Evaluate one exact-budget random-FEC schedule without writing audio.

    This is intended for Monte Carlo baselines. It preserves the stateful Opus
    encoder behavior and verifies actual LBRR presence in the following packet,
    while skipping decode/output work that is irrelevant to recovery counting.
    """
    rows = _trace_rows(trace)
    cfg = OpusConfig(bitrate=bitrate)
    enc = OpusEncoder(cfg)
    rng = random.Random(seed)
    frame_bytes = cfg.frame_samples * cfg.channels * 2
    trace_offset_frames %= len(rows)

    try:
        with wave.open(str(input_wav), "rb") as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (
                cfg.sample_rate,
                cfg.channels,
                2,
            ):
                raise ValueError("input WAV must be 48 kHz mono PCM16")

            total_frames = math.ceil(w.getnframes() / cfg.frame_samples)
            if protected_frames < 0 or protected_frames > total_frames:
                raise ValueError(
                    f"protected_frames={protected_frames} outside [0,{total_frames}]"
                )
            schedule = (
                set(rng.sample(range(total_frames), protected_frames))
                if 0 < protected_frames < total_frames
                else (set(range(total_frames)) if protected_frames == total_frames else set())
            )

            encoded: list[bytes] = []
            trace_indices: list[int] = []

            i = 0
            while True:
                pcm = w.readframes(cfg.frame_samples)
                if not pcm:
                    break
                if len(pcm) < frame_bytes:
                    pcm += b"\0" * (frame_bytes - len(pcm))
                fec = i in schedule
                enc.set_fec(fec, expected_loss_percent)
                encoded.append(enc.encode(pcm))
                trace_indices.append((trace_offset_frames + i) % len(rows))
                i += 1

        lost = 0
        recovered = 0
        for i in range(len(encoded)):
            if not bool(rows[trace_indices[i]].get("lost")):
                continue
            lost += 1
            if i + 1 >= len(encoded):
                continue
            next_lost = bool(rows[trace_indices[i + 1]].get("lost"))
            if next_lost:
                continue
            if (i + 1) not in schedule:
                continue
            if packet_has_fec(encoded[i + 1]):
                recovered += 1

        return {
            "frames": len(encoded),
            "network_lost_frames": lost,
            "fec_recovered_frames": recovered,
            "fec_protected_frames": len(schedule),
            "fec_duty_cycle": len(schedule) / len(encoded) if encoded else None,
            "trace_offset_frames": trace_offset_frames,
            "seed": seed,
        }
    finally:
        enc.close()


def replay_speech(
    input_wav: Path,
    output_wav: Path,
    trace: Path,
    policy: str,
    predictor: PhaseModel | None,
    bitrate: int = 24000,
    expected_loss_percent: int = 20,
    random_duty_cycle: float = 0.1,
    seed: int = 1,
    packet_log: Path | None = None,
    trace_offset_frames: int = 0,
    reactive_threshold_ms: float = 50.0,
    reactive_hold_s: float = 0.2,
) -> dict:
    """Replay one recorded loss pattern against one speech-protection policy.

    The same trace offset can be reused across policies for paired comparisons.
    The replay currently models loss only; RTT/jitter are retained in the source
    trace but are not converted to a playout model.
    """
    rows = _trace_rows(trace)
    cfg = OpusConfig(bitrate=bitrate)
    enc = OpusEncoder(cfg)
    dec = OpusDecoder(cfg)
    rng = random.Random(seed)
    writer = JsonlWriter(packet_log) if packet_log else None
    frame_bytes = cfg.frame_samples * cfg.channels * 2
    trace_offset_frames %= len(rows)

    reactive_events = sorted(
        (
            (int(r["recv_ns"]), float(r["rtt_ms"]))
            for r in rows
            if r.get("recv_ns") is not None and r.get("rtt_ms") is not None
        ),
        key=lambda x: x[0],
    )
    reactive_event_index = 0
    reactive_until_ns = -1
    reactive_hold_ns = int(reactive_hold_s * 1e9)

    encoded: list[bytes] = []
    fec_flags: list[bool] = []
    risks: list[float] = []
    send_times: list[int] = []
    trace_indices: list[int] = []

    try:
        with wave.open(str(input_wav), "rb") as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (
                cfg.sample_rate,
                cfg.channels,
                2,
            ):
                raise ValueError("input WAV must be 48 kHz mono PCM16")

            total_frames = math.ceil(w.getnframes() / cfg.frame_samples)
            random_schedule = (
                _exact_random_schedule(total_frames, random_duty_cycle, rng)
                if policy == "random-fec"
                else None
            )

            i = 0
            while True:
                pcm = w.readframes(cfg.frame_samples)
                if not pcm:
                    break
                if len(pcm) < frame_bytes:
                    pcm += b"\0" * (frame_bytes - len(pcm))
                trace_index = (trace_offset_frames + i) % len(rows)
                trace_row = rows[trace_index]
                send_ns = int(trace_row.get("send_ns", i * cfg.frame_ms * 1_000_000))
                risk = predictor.risk(send_ns) if predictor else 0.0

                if policy == "random-fec":
                    fec = i in random_schedule
                elif policy == "reactive-fec":
                    while (
                        reactive_event_index < len(reactive_events)
                        and reactive_events[reactive_event_index][0] <= send_ns
                    ):
                        observed_ns, observed_rtt = reactive_events[reactive_event_index]
                        if observed_rtt >= reactive_threshold_ms:
                            reactive_until_ns = max(
                                reactive_until_ns,
                                observed_ns + reactive_hold_ns,
                            )
                        reactive_event_index += 1
                    fec = send_ns < reactive_until_ns
                else:
                    fec = _policy_fec(policy, risk, rng, random_duty_cycle)

                enc.set_fec(fec, expected_loss_percent)
                encoded.append(enc.encode(pcm))
                fec_flags.append(fec)
                risks.append(risk)
                send_times.append(send_ns)
                trace_indices.append(trace_index)
                i += 1

        received: list[bytes | None] = []
        for i, payload in enumerate(encoded):
            trace_row = rows[trace_indices[i]]
            received.append(None if bool(trace_row.get("lost")) else payload)

        recovered = 0
        lost = 0
        with wave.open(str(output_wav), "wb") as out:
            out.setnchannels(cfg.channels)
            out.setsampwidth(2)
            out.setframerate(cfg.sample_rate)
            for i, payload in enumerate(received):
                recovered_here = False
                if payload is not None:
                    pcm = dec.decode(payload, fec=False)
                else:
                    lost += 1
                    next_payload = received[i + 1] if i + 1 < len(received) else None
                    next_has_fec = (
                        next_payload is not None
                        and i + 1 < len(fec_flags)
                        and fec_flags[i + 1]
                        and packet_has_fec(next_payload)
                    )
                    if next_has_fec:
                        try:
                            pcm = dec.decode(next_payload, fec=True)
                            recovered += 1
                            recovered_here = True
                        except RuntimeError:
                            pcm = dec.decode(None, fec=False)
                    else:
                        pcm = dec.decode(None, fec=False)
                out.writeframes(pcm)
                if writer:
                    writer.write(
                        {
                            "sequence": i,
                            "send_ns": send_times[i],
                            "source_trace_index": trace_indices[i],
                            "lost": payload is None,
                            "recovered_fec": recovered_here,
                            "fec_enabled": fec_flags[i],
                            "next_packet_fec_enabled": (
                                fec_flags[i + 1] if i + 1 < len(fec_flags) else False
                            ),
                            "risk": risks[i],
                            "encoded_bytes": len(encoded[i]),
                            "policy": policy,
                        }
                    )

        protected = sum(fec_flags)
        return {
            "frames": len(encoded),
            "source_trace": str(trace),
            "trace_offset_frames": trace_offset_frames,
            "policy": policy,
            "network_lost_frames": lost,
            "network_loss_rate": lost / len(encoded) if encoded else None,
            "fec_recovered_frames": recovered,
            "fec_duty_cycle": protected / len(fec_flags) if fec_flags else None,
            "fec_protected_frames": protected,
            "encoded_bytes": sum(map(len, encoded)),
            "replay_models": ["loss"],
            "reactive_threshold_ms": reactive_threshold_ms if policy == "reactive-fec" else None,
            "reactive_hold_s": reactive_hold_s if policy == "reactive-fec" else None,
        }
    finally:
        if writer:
            writer.close()
        enc.close()
        dec.close()
