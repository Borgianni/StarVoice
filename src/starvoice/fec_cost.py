from __future__ import annotations

import json
import math
import statistics
import tempfile
import wave
from pathlib import Path

import numpy as np

from .counterfactual import _replay_schedule_no_loss
from .qoe import _stoi


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _frame_count(path: Path, frame_samples: int = 960) -> int:
    with wave.open(str(path), "rb") as w:
        return math.ceil(w.getnframes() / frame_samples)


def _window_starts(total_frames: int, window_frames: int, positions: int) -> list[int]:
    if total_frames <= 0:
        return []
    if window_frames <= 0:
        raise ValueError("window_frames must be positive")
    if positions <= 0:
        raise ValueError("positions must be positive")
    width = min(window_frames, total_frames)
    max_start = total_frames - width
    if max_start == 0:
        return [0]
    raw = np.linspace(0, max_start, num=min(positions, max_start + 1))
    return sorted({int(round(x)) for x in raw})


def _window_features(
    wav_path: Path,
    start_frame: int,
    window_frames: int,
    frame_samples: int = 960,
) -> dict:
    with wave.open(str(wav_path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"expected mono PCM16 WAV: {wav_path}")
        sample_rate = w.getframerate()
        start_sample = start_frame * frame_samples
        sample_count = window_frames * frame_samples
        w.setpos(min(start_sample, w.getnframes()))
        raw = w.readframes(sample_count)

    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if pcm.size == 0:
        return {
            "rms_dbfs": -120.0,
            "zero_crossing_rate": 0.0,
            "spectral_centroid_hz": 0.0,
        }

    rms = float(np.sqrt(np.mean(pcm * pcm)))
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-6))
    zcr = float(np.mean(np.signbit(pcm[1:]) != np.signbit(pcm[:-1]))) if pcm.size > 1 else 0.0

    windowed = pcm * np.hanning(pcm.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(pcm.size, d=1.0 / sample_rate)
    spectral_sum = float(np.sum(spectrum))
    centroid = (
        float(np.sum(freqs * spectrum) / spectral_sum)
        if spectral_sum > 0.0
        else 0.0
    )
    return {
        "rms_dbfs": rms_dbfs,
        "zero_crossing_rate": zcr,
        "spectral_centroid_hz": centroid,
    }


def _summary(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def run_fec_cost_benchmark(
    manifest: Path,
    output: Path,
    positions_per_utterance: int = 6,
    window_ms: int = 200,
    bitrate: int = 24000,
    expected_loss_percent: int = 20,
    limit: int | None = None,
) -> dict:
    """Measure no-loss marginal codec cost of one isolated FEC window.

    Every candidate is paired with an all-FEC-off re-encode of the same utterance.
    Window locations are deterministic and uniformly spaced; no result-dependent
    selection is performed.
    """
    rows = _read_jsonl(manifest)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    if window_ms <= 0 or window_ms % 20 != 0:
        raise ValueError("window_ms must be a positive multiple of 20 ms")

    window_frames = window_ms // 20
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.jsonl"
    result_rows: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="starvoice-fec-cost-", dir=str(output)) as tmpdir:
        tmp = Path(tmpdir)
        with results_path.open("w", encoding="utf-8") as f:
            for item in rows:
                utterance_id = item["utterance_id"]
                source = Path(item["wav_48k_mono_pcm16"])
                total_frames = _frame_count(source)
                starts = _window_starts(
                    total_frames,
                    window_frames,
                    positions_per_utterance,
                )

                baseline_wav = tmp / f"{utterance_id}-plain.wav"
                baseline_schedule = [False] * total_frames
                baseline_codec = _replay_schedule_no_loss(
                    input_wav=source,
                    output_wav=baseline_wav,
                    schedule=baseline_schedule,
                    bitrate=bitrate,
                    expected_loss_percent=expected_loss_percent,
                )
                baseline_stoi, _, _ = _stoi(source, baseline_wav)

                for start in starts:
                    width = min(window_frames, total_frames - start)
                    schedule = [False] * total_frames
                    for idx in range(start, start + width):
                        schedule[idx] = True

                    candidate_wav = tmp / f"{utterance_id}-{start}.wav"
                    codec = _replay_schedule_no_loss(
                        input_wav=source,
                        output_wav=candidate_wav,
                        schedule=schedule,
                        bitrate=bitrate,
                        expected_loss_percent=expected_loss_percent,
                    )
                    candidate_stoi, _, _ = _stoi(source, candidate_wav)
                    features = _window_features(source, start, width)

                    out = {
                        "utterance_id": utterance_id,
                        "speaker_id": item["speaker_id"],
                        "start_frame": start,
                        "start_ms": start * 20,
                        "window_frames": width,
                        "window_ms": width * 20,
                        "total_frames": total_frames,
                        "baseline_stoi": baseline_stoi,
                        "fec_window_stoi": candidate_stoi,
                        "stoi_cost": baseline_stoi - candidate_stoi,
                        "baseline_encoded_bytes": int(baseline_codec["encoded_bytes"]),
                        "fec_window_encoded_bytes": int(codec["encoded_bytes"]),
                        "encoded_byte_cost": (
                            int(codec["encoded_bytes"]) - int(baseline_codec["encoded_bytes"])
                        ),
                        **features,
                    }
                    f.write(json.dumps(out, sort_keys=True) + "\n")
                    f.flush()
                    result_rows.append(out)

    costs = [float(r["stoi_cost"]) for r in result_rows]
    byte_costs = [float(r["encoded_byte_cost"]) for r in result_rows]
    rms = [float(r["rms_dbfs"]) for r in result_rows]
    zcr = [float(r["zero_crossing_rate"]) for r in result_rows]
    centroid = [float(r["spectral_centroid_hz"]) for r in result_rows]

    order = sorted(result_rows, key=lambda r: float(r["rms_dbfs"]))
    quartiles = []
    if order:
        chunks = np.array_split(np.asarray(order, dtype=object), 4)
        for idx, chunk in enumerate(chunks, start=1):
            rr = list(chunk)
            quartiles.append(
                {
                    "quartile": idx,
                    "count": len(rr),
                    "rms_dbfs_mean": statistics.fmean(
                        float(r["rms_dbfs"]) for r in rr
                    ),
                    "stoi_cost_mean": statistics.fmean(
                        float(r["stoi_cost"]) for r in rr
                    ),
                    "encoded_byte_cost_mean": statistics.fmean(
                        float(r["encoded_byte_cost"]) for r in rr
                    ),
                }
            )

    summary = {
        "schema_version": 1,
        "experiment": "isolated marginal FEC window codec cost",
        "manifest": str(manifest),
        "utterances": len(rows),
        "conditions": len(result_rows),
        "positions_per_utterance": positions_per_utterance,
        "window_ms_requested": window_ms,
        "window_frames_requested": window_frames,
        "bitrate": bitrate,
        "expected_loss_percent": expected_loss_percent,
        "results_jsonl": str(results_path),
        "stoi_cost": _summary(costs),
        "encoded_byte_cost": _summary(byte_costs),
        "content_association": {
            "pearson_stoi_cost_vs_rms_dbfs": _pearson(costs, rms),
            "pearson_stoi_cost_vs_zero_crossing_rate": _pearson(costs, zcr),
            "pearson_stoi_cost_vs_spectral_centroid_hz": _pearson(costs, centroid),
            "rms_quartiles": quartiles,
        },
        "interpretation": (
            "stoi_cost = STOI(all-FEC-off no-loss) - STOI(single-window FEC no-loss); "
            "positive values mean the isolated FEC window reduced STOI. "
            "This is a codec perturbation experiment, not a network-policy comparison."
        ),
        "design_note": (
            "Each candidate uses a fresh encoder/decoder and differs from its paired "
            "baseline only by one deterministic FEC window. Window positions are "
            "uniformly spaced before outcomes are observed."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
