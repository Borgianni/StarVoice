from __future__ import annotations

import json
import math
import random
import wave
from pathlib import Path

from .opus import OpusConfig, OpusDecoder, OpusEncoder
from .predictor import PhaseModel
from .speech import _policy_fec
from .storage import JsonlWriter


def _trace_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty trace")
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
) -> dict:
    """Replay one recorded loss pattern against one speech-protection policy.

    This isolates policy effects: multiple policies can be evaluated against the
    same recorded loss events. The replay currently models loss only; RTT/jitter
    are retained in the source trace but are not converted to a playout model.

    For random-fec, the requested duty cycle is enforced as an exact frame count
    (up to rounding) rather than as a Bernoulli probability. This supports
    equal-budget comparisons against predictive protection.
    """
    rows = _trace_rows(trace)
    cfg = OpusConfig(bitrate=bitrate)
    enc = OpusEncoder(cfg)
    dec = OpusDecoder(cfg)
    rng = random.Random(seed)
    writer = JsonlWriter(packet_log) if packet_log else None
    frame_bytes = cfg.frame_samples * cfg.channels * 2

    encoded: list[bytes] = []
    fec_flags: list[bool] = []
    risks: list[float] = []
    send_times: list[int] = []

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
                trace_row = rows[i % len(rows)]
                send_ns = int(trace_row.get("send_ns", i * cfg.frame_ms * 1_000_000))
                risk = predictor.risk(send_ns) if predictor else 0.0

                if policy == "random-fec":
                    fec = i in random_schedule
                else:
                    fec = _policy_fec(policy, risk, rng, random_duty_cycle)

                enc.set_fec(fec, expected_loss_percent)
                encoded.append(enc.encode(pcm))
                fec_flags.append(fec)
                risks.append(risk)
                send_times.append(send_ns)
                i += 1

        received: list[bytes | None] = []
        for i, payload in enumerate(encoded):
            trace_row = rows[i % len(rows)]
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
                    next_has_fec = i + 1 < len(fec_flags) and fec_flags[i + 1]
                    if next_payload is not None and next_has_fec:
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
                            "source_trace_index": i % len(rows),
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

        return {
            "frames": len(encoded),
            "source_trace": str(trace),
            "policy": policy,
            "network_lost_frames": lost,
            "network_loss_rate": lost / len(encoded) if encoded else None,
            "fec_recovered_frames": recovered,
            "fec_duty_cycle": sum(fec_flags) / len(fec_flags) if fec_flags else None,
            "fec_protected_frames": sum(fec_flags),
            "encoded_bytes": sum(map(len, encoded)),
            "replay_models": ["loss"],
        }
    finally:
        if writer:
            writer.close()
        enc.close()
        dec.close()
