from __future__ import annotations

import json
import math
import wave
from pathlib import Path

from .counterfactual import _packet_log_from_output
from .opus import OpusConfig, OpusDecoder, OpusEncoder, packet_has_fec


POLICIES = ("plain", "always-fec", "random-fec", "predictive-fec", "reactive-fec")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _trace_rows(path: Path) -> list[dict]:
    rows = _read_jsonl(path)
    rows.sort(key=lambda r: int(r["send_ns"]))
    return rows


def _packet_rows(path: Path) -> list[dict]:
    rows = _read_jsonl(path)
    rows.sort(key=lambda r: int(r["sequence"]))
    for i, row in enumerate(rows):
        if int(row["sequence"]) != i:
            raise ValueError(f"non-contiguous packet log {path} at {i}")
    return rows


def _arrival_ns(
    packet_row: dict,
    trace_rows: list[dict],
    rtt_fraction: float,
) -> int | None:
    idx = int(packet_row["source_trace_index"])
    tr = trace_rows[idx]
    if bool(tr.get("lost")) or tr.get("rtt_ms") is None:
        return None
    return int(packet_row["send_ns"]) + int(
        round(float(tr["rtt_ms"]) * rtt_fraction * 1_000_000)
    )


def render_deadline_audio(
    input_wav: Path,
    output_wav: Path,
    packet_log: Path,
    trace: Path,
    playout_ms: float,
    rtt_fraction: float = 0.5,
    bitrate: int = 24000,
    expected_loss_percent: int = 20,
) -> dict:
    """Re-encode a frozen FEC schedule and decode only audio useful by playout."""
    if playout_ms < 0:
        raise ValueError("playout_ms must be non-negative")
    if not 0.0 <= rtt_fraction <= 1.0:
        raise ValueError("rtt_fraction must be in [0,1]")

    packets = _packet_rows(packet_log)
    trace_rows = _trace_rows(trace)
    cfg = OpusConfig(bitrate=bitrate)
    enc = OpusEncoder(cfg)
    dec = OpusDecoder(cfg)
    frame_bytes = cfg.frame_samples * cfg.channels * 2
    frame_ns = int(round(cfg.frame_ms * 1_000_000))
    playout_ns = int(round(playout_ms * 1_000_000))

    if not packets:
        raise ValueError(f"empty packet log: {packet_log}")

    encoded: list[bytes] = []
    try:
        with wave.open(str(input_wav), "rb") as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (
                cfg.sample_rate,
                cfg.channels,
                2,
            ):
                raise ValueError("input WAV must be 48 kHz mono PCM16")

            i = 0
            while True:
                pcm = w.readframes(cfg.frame_samples)
                if not pcm:
                    break
                if i >= len(packets):
                    raise ValueError("packet log shorter than audio")
                if len(pcm) < frame_bytes:
                    pcm += b"\0" * (frame_bytes - len(pcm))
                fec = bool(packets[i].get("fec_enabled"))
                enc.set_fec(fec, expected_loss_percent)
                encoded.append(enc.encode(pcm))
                i += 1

        if len(encoded) != len(packets):
            raise ValueError(
                f"packet log/audio length mismatch: {len(packets)} != {len(encoded)}"
            )

        arrivals = [_arrival_ns(r, trace_rows, rtt_fraction) for r in packets]
        first_send_ns = int(packets[0]["send_ns"])

        output_wav.parent.mkdir(parents=True, exist_ok=True)
        network_lost = 0
        primary_late = 0
        primary_failures = 0
        recovered = 0

        with wave.open(str(output_wav), "wb") as out:
            out.setnchannels(cfg.channels)
            out.setsampwidth(2)
            out.setframerate(cfg.sample_rate)

            for i, payload in enumerate(encoded):
                deadline_ns = first_send_ns + i * frame_ns + playout_ns
                lost = bool(packets[i].get("lost"))
                if lost:
                    network_lost += 1

                arrival = arrivals[i]
                primary_on_time = arrival is not None and arrival <= deadline_ns
                if not lost and arrival is not None and not primary_on_time:
                    primary_late += 1

                if primary_on_time:
                    pcm = dec.decode(payload, fec=False)
                else:
                    primary_failures += 1
                    fec_ok = False
                    if i + 1 < len(encoded):
                        next_arrival = arrivals[i + 1]
                        fec_ok = (
                            next_arrival is not None
                            and next_arrival <= deadline_ns
                            and packet_has_fec(encoded[i + 1])
                        )
                    if fec_ok:
                        try:
                            pcm = dec.decode(encoded[i + 1], fec=True)
                            recovered += 1
                        except RuntimeError:
                            pcm = dec.decode(None, fec=False)
                    else:
                        pcm = dec.decode(None, fec=False)

                out.writeframes(pcm)

        return {
            "frames": len(encoded),
            "network_lost_frames": network_lost,
            "primary_late_frames": primary_late,
            "primary_deadline_failures": primary_failures,
            "fec_recovered_frames": recovered,
            "final_deadline_misses": primary_failures - recovered,
            "final_deadline_miss_rate": (
                (primary_failures - recovered) / len(encoded) if encoded else None
            ),
            "playout_ms": playout_ms,
            "rtt_fraction": rtt_fraction,
            "encoded_bytes": sum(map(len, encoded)),
        }
    finally:
        enc.close()
        dec.close()


def run_deadline_audio_benchmark(
    benchmark_results: Path,
    reactive_results: Path,
    manifest: Path,
    output: Path,
    playout_ms: float = 60.0,
    rtt_fraction: float = 0.5,
    bitrate: int = 24000,
    expected_loss_percent: int = 20,
    loss_exposed_only: bool = False,
) -> dict:
    """Render deadline-aware audio for frozen baseline schedules."""
    benchmark = _read_jsonl(benchmark_results)
    reactive = _read_jsonl(reactive_results)
    corpus = {r["utterance_id"]: r for r in _read_jsonl(manifest)}

    canonical = [r for r in benchmark if r.get("policy") == "predictive-fec"]
    if not canonical:
        raise ValueError("benchmark contains no predictive-fec rows")

    allowed = {
        (r["trace_name"], r["utterance_id"])
        for r in canonical
        if not loss_exposed_only or int(r["network_lost_frames"]) > 0
    }

    output.mkdir(parents=True, exist_ok=True)
    audio_root = output / "audio"
    benchmark_out = output / "benchmark-results.jsonl"
    reactive_out = output / "reactive-results.jsonl"

    counts = {
        "conditions": len(allowed),
        "policies": 0,
        "frames": 0,
        "network_lost_frames": 0,
        "primary_late_frames": 0,
        "fec_recovered_frames": 0,
        "final_deadline_misses": 0,
    }

    def process(rows: list[dict], out_path: Path, wanted_policies: set[str]) -> int:
        written = 0
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                key = (row["trace_name"], row["utterance_id"])
                if key not in allowed or row["policy"] not in wanted_policies:
                    continue

                item = corpus.get(row["utterance_id"])
                if item is None:
                    raise ValueError(
                        f"utterance missing from manifest: {row['utterance_id']}"
                    )
                packet_log = _packet_log_from_output(row["output_wav"])
                out_wav = (
                    audio_root
                    / Path(row["trace"]).stem
                    / row["policy"]
                    / f"{row['utterance_id']}.wav"
                )
                metrics = render_deadline_audio(
                    input_wav=Path(item["wav_48k_mono_pcm16"]),
                    output_wav=out_wav,
                    packet_log=packet_log,
                    trace=Path(row["trace"]),
                    playout_ms=playout_ms,
                    rtt_fraction=rtt_fraction,
                    bitrate=bitrate,
                    expected_loss_percent=expected_loss_percent,
                )

                if int(metrics["frames"]) != int(row["frames"]):
                    raise RuntimeError(f"frame-count drift for {key}/{row['policy']}")
                if int(metrics["network_lost_frames"]) != int(row["network_lost_frames"]):
                    raise RuntimeError(f"loss-mask drift for {key}/{row['policy']}")

                rendered = {
                    **row,
                    **metrics,
                    "output_wav": str(out_wav),
                    "source_output_wav": row["output_wav"],
                    "source_packet_log": str(packet_log),
                    "deadline_audio": True,
                    "replay_models": ["loss", "RTT/one-way-proxy", "playout-deadline"],
                }
                f.write(json.dumps(rendered, sort_keys=True) + "\n")
                f.flush()

                counts["policies"] += 1
                counts["frames"] += int(metrics["frames"])
                counts["network_lost_frames"] += int(metrics["network_lost_frames"])
                counts["primary_late_frames"] += int(metrics["primary_late_frames"])
                counts["fec_recovered_frames"] += int(metrics["fec_recovered_frames"])
                counts["final_deadline_misses"] += int(metrics["final_deadline_misses"])
                written += 1
        return written

    benchmark_written = process(
        benchmark,
        benchmark_out,
        {"plain", "always-fec", "random-fec", "predictive-fec"},
    )
    reactive_written = process(
        reactive,
        reactive_out,
        {"reactive-fec"},
    )

    summary = {
        "schema_version": 1,
        "experiment": "deadline-aware end-to-end Opus audio replay",
        "source_benchmark_results": str(benchmark_results),
        "source_reactive_results": str(reactive_results),
        "manifest": str(manifest),
        "playout_ms": playout_ms,
        "rtt_fraction": rtt_fraction,
        "one_way_delay_model": f"{rtt_fraction:g} * RTT",
        "loss_exposed_only": loss_exposed_only,
        "benchmark_rows": benchmark_written,
        "reactive_rows": reactive_written,
        "benchmark_results": str(benchmark_out),
        "reactive_results": str(reactive_out),
        **counts,
        "caveat": (
            "RTT-derived one-way delay is a proxy. Audio is decoded only when the "
            "primary packet or following LBRR repair is useful by playout."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
