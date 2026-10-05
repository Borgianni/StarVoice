from __future__ import annotations

import hashlib
import json
import math
import wave
from pathlib import Path

from .predictor import phase_model_from_warmup, search_period
from .replay import replay_speech


POLICIES = ("plain", "always-fec", "predictive-fec", "random-fec")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _wav_frames(path: Path, frame_ms: int = 20) -> int:
    with wave.open(str(path), "rb") as w:
        samples_per_frame = int(w.getframerate() * frame_ms / 1000)
        return math.ceil(w.getnframes() / samples_per_frame)


def _stable_offset(
    utterance_id: str,
    trace_name: str,
    seed: int,
    minimum: int,
    maximum: int,
) -> int:
    if maximum < minimum:
        raise ValueError("trace is too short for requested utterance after warm-up")
    raw = f"{seed}|{trace_name}|{utterance_id}".encode()
    value = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
    return minimum + value % (maximum - minimum + 1)


def run_codec_benchmark(
    manifest: Path,
    calibration_trace: Path,
    validation_traces: list[Path],
    output: Path,
    seed: int = 2027,
    warmup_s: float = 120.0,
    threshold_ms: float = 50.0,
    risk_half_width_s: float = 0.2,
    bitrate: int = 24000,
    expected_loss_percent: int = 20,
) -> dict:
    """Run paired Opus/FEC replay over a frozen speech corpus and validation traces.

    The period is estimated only from the calibration trace. For each validation
    trace, phase is estimated only from its warm-up interval and then frozen.
    Every utterance uses one deterministic post-warm-up trace offset shared by all
    policies. random-fec receives exactly the predictive-fec protected-frame count
    for that utterance, up to integer arithmetic that is exact by construction.
    """
    output.mkdir(parents=True, exist_ok=True)
    audio_root = output / "audio"
    logs_root = output / "packet-logs"
    audio_root.mkdir(exist_ok=True)
    logs_root.mkdir(exist_ok=True)

    corpus = _read_jsonl(manifest)
    if not corpus:
        raise ValueError("empty corpus manifest")

    period_s, coherence = search_period(
        calibration_trace,
        threshold_ms=threshold_ms,
    )

    trace_models: dict[str, tuple] = {}
    for trace in validation_traces:
        model, meta = phase_model_from_warmup(
            trace,
            period_s=period_s,
            warmup_s=warmup_s,
            threshold_ms=threshold_ms,
            risk_half_width_s=risk_half_width_s,
        )
        trace_models[trace.name] = (trace, model, meta)

    results_path = output / "results.jsonl"
    result_rows: list[dict] = []

    with results_path.open("w", encoding="utf-8") as result_file:
        for trace_name, (trace, predictor, phase_meta) in trace_models.items():
            trace_rows = _read_jsonl(trace)
            trace_rows.sort(key=lambda r: int(r["send_ns"]))
            if len(trace_rows) < 2:
                raise ValueError(f"trace too short: {trace}")

            interval_s = (
                int(trace_rows[1]["send_ns"]) - int(trace_rows[0]["send_ns"])
            ) / 1e9
            warmup_frames = math.ceil(warmup_s / interval_s)

            for item in corpus:
                utterance_id = item["utterance_id"]
                wav = Path(item["wav_48k_mono_pcm16"])
                frames = _wav_frames(wav)
                max_offset = len(trace_rows) - frames - 1
                offset = _stable_offset(
                    utterance_id,
                    trace_name,
                    seed,
                    warmup_frames,
                    max_offset,
                )

                base = {
                    "utterance_id": utterance_id,
                    "speaker_id": item["speaker_id"],
                    "chapter_id": item["chapter_id"],
                    "transcript": item["transcript"],
                    "trace": str(trace),
                    "trace_name": trace_name,
                    "trace_offset_frames": offset,
                    "period_s": period_s,
                    "calibration_coherence": coherence,
                    **phase_meta,
                }

                pred_audio = audio_root / trace.stem / "predictive-fec" / f"{utterance_id}.wav"
                pred_log = logs_root / trace.stem / "predictive-fec" / f"{utterance_id}.jsonl"
                pred_audio.parent.mkdir(parents=True, exist_ok=True)
                pred_log.parent.mkdir(parents=True, exist_ok=True)
                pred = replay_speech(
                    input_wav=wav,
                    output_wav=pred_audio,
                    trace=trace,
                    policy="predictive-fec",
                    predictor=predictor,
                    bitrate=bitrate,
                    expected_loss_percent=expected_loss_percent,
                    seed=seed,
                    packet_log=pred_log,
                    trace_offset_frames=offset,
                )
                pred_row = {**base, **pred, "output_wav": str(pred_audio)}
                result_file.write(json.dumps(pred_row, sort_keys=True) + "\n")
                result_file.flush()
                result_rows.append(pred_row)

                random_duty = (
                    pred["fec_protected_frames"] / pred["frames"]
                    if pred["frames"]
                    else 0.0
                )

                for policy in ("plain", "always-fec", "random-fec"):
                    out_wav = audio_root / trace.stem / policy / f"{utterance_id}.wav"
                    packet_log = logs_root / trace.stem / policy / f"{utterance_id}.jsonl"
                    out_wav.parent.mkdir(parents=True, exist_ok=True)
                    packet_log.parent.mkdir(parents=True, exist_ok=True)
                    result = replay_speech(
                        input_wav=wav,
                        output_wav=out_wav,
                        trace=trace,
                        policy=policy,
                        predictor=None,
                        bitrate=bitrate,
                        expected_loss_percent=expected_loss_percent,
                        random_duty_cycle=random_duty,
                        seed=seed,
                        packet_log=packet_log,
                        trace_offset_frames=offset,
                    )
                    row = {**base, **result, "output_wav": str(out_wav)}
                    row["equal_budget_target_frames"] = pred["fec_protected_frames"]
                    result_file.write(json.dumps(row, sort_keys=True) + "\n")
                    result_file.flush()
                    result_rows.append(row)

    summary = {
        "schema_version": 1,
        "manifest": str(manifest),
        "calibration_trace": str(calibration_trace),
        "validation_traces": [str(x) for x in validation_traces],
        "seed": seed,
        "warmup_s": warmup_s,
        "threshold_ms": threshold_ms,
        "risk_half_width_s": risk_half_width_s,
        "period_s": period_s,
        "calibration_coherence": coherence,
        "utterances": len(corpus),
        "conditions": len(result_rows),
        "results_jsonl": str(results_path),
        "note": "Replay models packet loss only; RTT/jitter/playout are not yet modeled.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
