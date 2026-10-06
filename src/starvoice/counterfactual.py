from __future__ import annotations

import json
import statistics
import wave
from pathlib import Path

from .opus import OpusConfig, OpusDecoder, OpusEncoder
from .qoe import _stoi


POLICIES = ("plain", "always-fec", "random-fec", "predictive-fec", "reactive-fec")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _packet_log_from_output(output_wav: str) -> Path:
    path = Path(output_wav)
    parts = list(path.parts)
    try:
        idx = parts.index("audio")
    except ValueError as exc:
        raise ValueError(f"cannot infer packet log path from {path}") from exc
    parts[idx] = "packet-logs"
    return Path(*parts).with_suffix(".jsonl")


def _fec_schedule(packet_log: Path, expected_frames: int) -> list[bool]:
    rows = _read_jsonl(packet_log)
    rows.sort(key=lambda r: int(r["sequence"]))
    if len(rows) != expected_frames:
        raise ValueError(
            f"packet log length mismatch for {packet_log}: "
            f"{len(rows)} != {expected_frames}"
        )
    for i, row in enumerate(rows):
        if int(row["sequence"]) != i:
            raise ValueError(f"non-contiguous sequence in {packet_log} at {i}")
    return [bool(r["fec_enabled"]) for r in rows]


def _replay_schedule_no_loss(
    input_wav: Path,
    output_wav: Path,
    schedule: list[bool],
    bitrate: int,
    expected_loss_percent: int,
) -> dict:
    cfg = OpusConfig(bitrate=bitrate)
    enc = OpusEncoder(cfg)
    dec = OpusDecoder(cfg)
    frame_bytes = cfg.frame_samples * cfg.channels * 2
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    frames = 0
    encoded_bytes = 0
    try:
        with wave.open(str(input_wav), "rb") as w, wave.open(str(output_wav), "wb") as out:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (
                cfg.sample_rate,
                cfg.channels,
                2,
            ):
                raise ValueError(f"unexpected WAV format: {input_wav}")

            out.setnchannels(cfg.channels)
            out.setsampwidth(2)
            out.setframerate(cfg.sample_rate)

            while True:
                pcm = w.readframes(cfg.frame_samples)
                if not pcm:
                    break
                if frames >= len(schedule):
                    raise ValueError("schedule shorter than audio")
                if len(pcm) < frame_bytes:
                    pcm += b"\0" * (frame_bytes - len(pcm))

                fec = schedule[frames]
                enc.set_fec(fec, expected_loss_percent)
                payload = enc.encode(pcm)
                encoded_bytes += len(payload)
                out.writeframes(dec.decode(payload, fec=False))
                frames += 1

        if frames != len(schedule):
            raise ValueError(f"schedule longer than audio: {len(schedule)} != {frames}")

        return {
            "frames": frames,
            "fec_protected_frames": sum(schedule),
            "fec_duty_cycle": sum(schedule) / frames if frames else None,
            "encoded_bytes": encoded_bytes,
        }
    finally:
        enc.close()
        dec.close()


def _summary(values: list[float]) -> dict:
    return {
        "conditions": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def run_codec_counterfactual(
    benchmark_results: Path,
    reactive_results: Path,
    manifest: Path,
    output: Path,
    bitrate: int = 24000,
    expected_loss_percent: int = 20,
    limit: int | None = None,
) -> dict:
    """Measure codec-only quality under exact frozen FEC schedules.

    For every frozen condition/policy, recover the exact frame-level FEC schedule
    from its packet log, re-encode and decode the reference with no network loss,
    and compute STOI. The actual lossy STOI is computed from the already frozen
    degraded output. The difference (codec-only STOI - actual STOI) estimates
    impairment attributable to the recorded network loss under that policy.
    """
    benchmark = _read_jsonl(benchmark_results)
    reactive = _read_jsonl(reactive_results)
    corpus = {r["utterance_id"]: r for r in _read_jsonl(manifest)}

    canonical = [r for r in benchmark if r.get("policy") == "predictive-fec"]
    if not canonical:
        raise ValueError("benchmark contains no predictive-fec rows")

    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        allowed = {
            (r["trace_name"], r["utterance_id"])
            for r in canonical[:limit]
        }
        benchmark = [
            r for r in benchmark
            if (r["trace_name"], r["utterance_id"]) in allowed
        ]
        reactive = [
            r for r in reactive
            if (r["trace_name"], r["utterance_id"]) in allowed
        ]
        canonical = canonical[:limit]

    expected = {
        (r["trace_name"], r["utterance_id"]): r
        for r in canonical
    }

    rows_by_key: dict[tuple[str, str, str], dict] = {}
    for row in benchmark + reactive:
        key = (row["trace_name"], row["utterance_id"], row["policy"])
        if key in rows_by_key:
            raise ValueError(f"duplicate row: {key}")
        rows_by_key[key] = row

    output.mkdir(parents=True, exist_ok=True)
    audio_root = output / "audio"
    results_path = output / "results.jsonl"
    result_rows: list[dict] = []

    with results_path.open("w", encoding="utf-8") as f:
        for (trace_name, utterance_id), base in sorted(expected.items()):
            item = corpus.get(utterance_id)
            if item is None:
                raise ValueError(f"utterance missing from manifest: {utterance_id}")
            reference = Path(item["wav_48k_mono_pcm16"])
            loss_count = int(base["network_lost_frames"])

            for policy in POLICIES:
                row = rows_by_key.get((trace_name, utterance_id, policy))
                if row is None:
                    raise ValueError(
                        f"missing frozen row: {(trace_name, utterance_id, policy)}"
                    )
                if int(row["network_lost_frames"]) != loss_count:
                    raise RuntimeError(
                        f"loss-mask drift: {trace_name}/{utterance_id}/{policy}"
                    )

                packet_log = _packet_log_from_output(row["output_wav"])
                schedule = _fec_schedule(packet_log, int(row["frames"]))
                if sum(schedule) != int(row["fec_protected_frames"]):
                    raise RuntimeError(
                        f"FEC schedule drift: {trace_name}/{utterance_id}/{policy}"
                    )

                cf_wav = audio_root / trace_name.replace(".jsonl", "") / policy / f"{utterance_id}.wav"
                codec = _replay_schedule_no_loss(
                    input_wav=reference,
                    output_wav=cf_wav,
                    schedule=schedule,
                    bitrate=bitrate,
                    expected_loss_percent=expected_loss_percent,
                )

                actual_stoi, _, _ = _stoi(reference, Path(row["output_wav"]))
                clean_stoi, _, _ = _stoi(reference, cf_wav)

                out = {
                    "trace_name": trace_name,
                    "utterance_id": utterance_id,
                    "policy": policy,
                    "network_lost_frames": loss_count,
                    "loss_exposed": loss_count > 0,
                    "fec_protected_frames": int(row["fec_protected_frames"]),
                    "actual_stoi": actual_stoi,
                    "codec_only_stoi": clean_stoi,
                    "network_impairment_stoi": clean_stoi - actual_stoi,
                    "counterfactual_wav": str(cf_wav),
                    **codec,
                }
                f.write(json.dumps(out, sort_keys=True) + "\n")
                f.flush()
                result_rows.append(out)

    analysis = {}
    for subset_name, subset_rows in {
        "all": result_rows,
        "loss_exposed": [r for r in result_rows if r["loss_exposed"]],
    }.items():
        policies = {}
        for policy in POLICIES:
            rr = [r for r in subset_rows if r["policy"] == policy]
            policies[policy] = {
                "actual_stoi": _summary([float(r["actual_stoi"]) for r in rr]),
                "codec_only_stoi": _summary([float(r["codec_only_stoi"]) for r in rr]),
                "network_impairment_stoi": _summary(
                    [float(r["network_impairment_stoi"]) for r in rr]
                ),
            }

        def paired_delta(field: str, left: str, right: str) -> dict:
            by_cond: dict[tuple[str, str], dict[str, float]] = {}
            for r in subset_rows:
                key = (r["trace_name"], r["utterance_id"])
                by_cond.setdefault(key, {})[r["policy"]] = float(r[field])
            vals = [
                x[left] - x[right]
                for x in by_cond.values()
                if left in x and right in x
            ]
            return {
                "pairs": len(vals),
                "mean_delta": statistics.fmean(vals) if vals else None,
                "median_delta": statistics.median(vals) if vals else None,
            }

        analysis[subset_name] = {
            "conditions": len({(r["trace_name"], r["utterance_id"]) for r in subset_rows}),
            "policies": policies,
            "paired": {
                "codec_only_predictive_minus_random": paired_delta(
                    "codec_only_stoi", "predictive-fec", "random-fec"
                ),
                "network_impairment_predictive_minus_random": paired_delta(
                    "network_impairment_stoi", "predictive-fec", "random-fec"
                ),
                "network_impairment_predictive_minus_reactive": paired_delta(
                    "network_impairment_stoi", "predictive-fec", "reactive-fec"
                ),
                "network_impairment_predictive_minus_plain": paired_delta(
                    "network_impairment_stoi", "predictive-fec", "plain"
                ),
            },
        }

    summary = {
        "schema_version": 1,
        "benchmark_results": str(benchmark_results),
        "reactive_results": str(reactive_results),
        "manifest": str(manifest),
        "conditions": len(expected),
        "policies": list(POLICIES),
        "bitrate": bitrate,
        "expected_loss_percent": expected_loss_percent,
        "results_jsonl": str(results_path),
        "analysis_sets": analysis,
        "interpretation": (
            "codec_only_stoi isolates distortion caused by the exact FEC schedule "
            "with network loss removed. network_impairment_stoi = codec_only_stoi "
            "- actual_stoi, so larger positive values mean more degradation due to "
            "the recorded network loss under that policy."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
