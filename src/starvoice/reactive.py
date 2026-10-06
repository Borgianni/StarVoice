from __future__ import annotations

import json
from pathlib import Path

from .replay import replay_speech


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_reactive_benchmark(
    benchmark_results: Path,
    manifest: Path,
    output: Path,
    reactive_threshold_ms: float = 50.0,
    reactive_hold_s: float = 0.2,
    bitrate: int = 24000,
    expected_loss_percent: int = 20,
    seed: int = 2027,
) -> dict:
    """Run a causal RTT-triggered FEC baseline on frozen benchmark conditions.

    The benchmark results define the exact trace/utterance/offset conditions.
    At each send time, reactive-fec may only use RTT samples whose replies have
    already arrived (recv_ns <= send_ns). A high-RTT observation activates FEC
    for reactive_hold_s after the observation time. Loss rows are not used as
    triggers because the trace does not record their timeout-observation time.
    """
    if reactive_hold_s < 0:
        raise ValueError("reactive_hold_s must be non-negative")

    rows = _read_jsonl(benchmark_results)
    predictive = [r for r in rows if r.get("policy") == "predictive-fec"]
    if not predictive:
        raise ValueError("benchmark contains no predictive-fec rows")

    corpus = {r["utterance_id"]: r for r in _read_jsonl(manifest)}
    output.mkdir(parents=True, exist_ok=True)
    audio_root = output / "audio"
    logs_root = output / "packet-logs"
    results_path = output / "results.jsonl"

    result_rows: list[dict] = []
    with results_path.open("w", encoding="utf-8") as f:
        for base in predictive:
            utterance_id = base["utterance_id"]
            item = corpus.get(utterance_id)
            if item is None:
                raise ValueError(f"utterance missing from manifest: {utterance_id}")

            trace = Path(base["trace"])
            trace_stem = trace.stem
            out_wav = audio_root / trace_stem / "reactive-fec" / f"{utterance_id}.wav"
            packet_log = logs_root / trace_stem / "reactive-fec" / f"{utterance_id}.jsonl"
            out_wav.parent.mkdir(parents=True, exist_ok=True)
            packet_log.parent.mkdir(parents=True, exist_ok=True)

            result = replay_speech(
                input_wav=Path(item["wav_48k_mono_pcm16"]),
                output_wav=out_wav,
                trace=trace,
                policy="reactive-fec",
                predictor=None,
                bitrate=bitrate,
                expected_loss_percent=expected_loss_percent,
                seed=seed,
                packet_log=packet_log,
                trace_offset_frames=int(base["trace_offset_frames"]),
                reactive_threshold_ms=reactive_threshold_ms,
                reactive_hold_s=reactive_hold_s,
            )

            if int(result["network_lost_frames"]) != int(base["network_lost_frames"]):
                raise RuntimeError(
                    "loss-mask drift for "
                    f"{base['trace_name']}/{utterance_id}: "
                    f"{result['network_lost_frames']} != {base['network_lost_frames']}"
                )
            if int(result["frames"]) != int(base["frames"]):
                raise RuntimeError(
                    "frame-count drift for "
                    f"{base['trace_name']}/{utterance_id}: "
                    f"{result['frames']} != {base['frames']}"
                )

            row = {
                "utterance_id": utterance_id,
                "speaker_id": base["speaker_id"],
                "chapter_id": base["chapter_id"],
                "transcript": base["transcript"],
                "trace": base["trace"],
                "trace_name": base["trace_name"],
                "trace_offset_frames": base["trace_offset_frames"],
                **result,
                "output_wav": str(out_wav),
                "reference_predictive_protected_frames": base["fec_protected_frames"],
                "reference_predictive_duty_cycle": base["fec_duty_cycle"],
            }
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            result_rows.append(row)

    frames = sum(int(r["frames"]) for r in result_rows)
    losses = sum(int(r["network_lost_frames"]) for r in result_rows)
    recovered = sum(int(r["fec_recovered_frames"]) for r in result_rows)
    protected = sum(int(r["fec_protected_frames"]) for r in result_rows)
    encoded = sum(int(r["encoded_bytes"]) for r in result_rows)

    summary = {
        "schema_version": 1,
        "benchmark_results": str(benchmark_results),
        "manifest": str(manifest),
        "conditions": len(result_rows),
        "policy": "reactive-fec",
        "causal_feedback": "received RTT observations only; recv_ns <= current send_ns",
        "loss_events_used_as_trigger": False,
        "reactive_threshold_ms": reactive_threshold_ms,
        "reactive_hold_s": reactive_hold_s,
        "frames": frames,
        "network_lost_frames": losses,
        "fec_recovered_frames": recovered,
        "fec_recovery_rate": recovered / losses if losses else None,
        "fec_protected_frames": protected,
        "fec_duty_cycle": protected / frames if frames else None,
        "encoded_bytes": encoded,
        "results_jsonl": str(results_path),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
