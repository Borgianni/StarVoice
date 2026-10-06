from __future__ import annotations

import json
from pathlib import Path

from .counterfactual import _packet_log_from_output
from .replay import replay_fixed_schedule_recovery_only


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sorted_log(path: Path) -> list[dict]:
    rows = _read_jsonl(path)
    rows.sort(key=lambda r: int(r["sequence"]))
    for i, row in enumerate(rows):
        if int(row["sequence"]) != i:
            raise ValueError(f"non-contiguous packet log {path} at sequence {i}")
    return rows


def run_oracle_recovery_benchmark(
    benchmark_results: Path,
    manifest: Path,
    output: Path,
    bitrate: int = 24000,
    expected_loss_percent: int = 20,
) -> dict:
    """Replay the minimal sparse Oracle FEC schedule using real Opus/LBRR.

    For each frozen condition, the Oracle enables encoder FEC on source frame i
    iff the Always-FEC replay recovered lost frame i. Opus generates that frame's
    LBRR while encoding i and carries it in a later packet (normally i+1 at 20 ms).
    This is a non-causal upper bound over encoder-control timing.
    """
    benchmark = _read_jsonl(benchmark_results)
    corpus = {r["utterance_id"]: r for r in _read_jsonl(manifest)}

    by_condition: dict[tuple[str, str], dict[str, dict]] = {}
    for row in benchmark:
        key = (row["trace_name"], row["utterance_id"])
        by_condition.setdefault(key, {})[row["policy"]] = row

    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.jsonl"

    result_rows: list[dict] = []
    with results_path.open("w", encoding="utf-8") as f:
        for key, policies in sorted(by_condition.items()):
            predictive = policies.get("predictive-fec")
            always = policies.get("always-fec")
            if predictive is None or always is None:
                continue

            item = corpus.get(key[1])
            if item is None:
                raise ValueError(f"utterance missing from manifest: {key[1]}")

            always_log = _sorted_log(_packet_log_from_output(always["output_wav"]))
            frames = int(always["frames"])
            if len(always_log) != frames:
                raise RuntimeError(f"packet-log length mismatch for {key}")

            schedule = {
                int(row["sequence"])
                for row in always_log
                if bool(row.get("lost"))
                and bool(row.get("recovered_fec"))
                and int(row["sequence"]) + 1 < frames
            }

            replay = replay_fixed_schedule_recovery_only(
                input_wav=Path(item["wav_48k_mono_pcm16"]),
                trace=Path(predictive["trace"]),
                schedule=schedule,
                trace_offset_frames=int(predictive["trace_offset_frames"]),
                bitrate=bitrate,
                expected_loss_percent=expected_loss_percent,
            )

            out = {
                "trace_name": key[0],
                "utterance_id": key[1],
                "frames": frames,
                "network_lost_frames": int(predictive["network_lost_frames"]),
                "always_fec_recovered_frames": int(always["fec_recovered_frames"]),
                "oracle_schedule_frames": len(schedule),
                **replay,
            }
            f.write(json.dumps(out, sort_keys=True) + "\n")
            result_rows.append(out)

    total_frames = sum(int(r["frames"]) for r in result_rows)
    total_losses = sum(int(r["network_lost_frames"]) for r in result_rows)
    always_recovered = sum(int(r["always_fec_recovered_frames"]) for r in result_rows)
    oracle_schedule = sum(int(r["oracle_schedule_frames"]) for r in result_rows)
    oracle_recovered = sum(int(r["fec_recovered_frames"]) for r in result_rows)
    lbrr_present = sum(
        int(r["scheduled_received_following_packets_with_lbrr"])
        for r in result_rows
    )

    summary = {
        "schema_version": 1,
        "experiment": "minimal sparse Oracle FEC replay",
        "benchmark_results": str(benchmark_results),
        "manifest": str(manifest),
        "conditions": len(result_rows),
        "frames": total_frames,
        "network_lost_frames": total_losses,
        "always_fec_recovered_frames": always_recovered,
        "oracle_schedule_frames": oracle_schedule,
        "oracle_effective_duty_cycle": (
            oracle_schedule / total_frames if total_frames else None
        ),
        "oracle_actual_recovered_frames": oracle_recovered,
        "oracle_fraction_of_always_fec_recovery": (
            oracle_recovered / always_recovered if always_recovered else None
        ),
        "lbrr_realization": (
            lbrr_present / oracle_schedule if oracle_schedule else None
        ),
        "protection_reduction_vs_always_fec": (
            total_frames / oracle_schedule if oracle_schedule else None
        ),
        "results_jsonl": str(results_path),
        "interpretation": (
            "This is a non-causal upper bound. It tests whether the opportunity "
            "locations inferred from Always-FEC can be realized by actual sparse "
            "Opus encoding when FEC is enabled only in those following-packet slots."
        ),
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
