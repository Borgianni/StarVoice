from __future__ import annotations

import json
from pathlib import Path

from .counterfactual import _packet_log_from_output
from .replay import replay_fixed_schedule_recovery_only


DEFAULT_PREROLL_FRAMES = (1, 2, 3, 5, 10, 20, 50)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sorted_log(path: Path) -> list[dict]:
    rows = _read_jsonl(path)
    rows.sort(key=lambda r: int(r["sequence"]))
    return rows


def _schedule_for_targets(
    targets: set[int],
    frames: int,
    preroll_frames: int,
) -> set[int]:
    if preroll_frames <= 0:
        raise ValueError("preroll_frames must be positive")
    schedule: set[int] = set()
    for target in targets:
        start = max(0, target - preroll_frames + 1)
        schedule.update(range(start, target + 1))
    return {i for i in schedule if 0 <= i < frames}


def run_fec_actuation_benchmark(
    benchmark_results: Path,
    manifest: Path,
    output: Path,
    preroll_frames: tuple[int, ...] = DEFAULT_PREROLL_FRAMES,
    bitrate: int = 24000,
    expected_loss_percent: int = 20,
) -> dict:
    """Measure real Opus FEC actuation efficiency versus predictive lead time.

    Target frames are the lost source frames that Always-FEC actually recovered.
    For each target, FEC is enabled on a deterministic window ending at that
    source frame. Window length 1 means target-only activation; larger values
    provide increasing pre-roll before the predicted impairment.
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

            always_log = _sorted_log(_packet_log_from_output(always["output_wav"]))
            targets = {
                int(row["sequence"])
                for row in always_log
                if bool(row.get("lost")) and bool(row.get("recovered_fec"))
            }
            if not targets:
                continue

            item = corpus.get(key[1])
            if item is None:
                raise ValueError(f"utterance missing from manifest: {key[1]}")

            frames = int(predictive["frames"])
            for window in preroll_frames:
                schedule = _schedule_for_targets(targets, frames, window)
                replay = replay_fixed_schedule_recovery_only(
                    input_wav=Path(item["wav_48k_mono_pcm16"]),
                    trace=Path(predictive["trace"]),
                    schedule=schedule,
                    trace_offset_frames=int(predictive["trace_offset_frames"]),
                    bitrate=bitrate,
                    expected_loss_percent=expected_loss_percent,
                )
                row = {
                    "trace_name": key[0],
                    "utterance_id": key[1],
                    "frames": frames,
                    "target_opportunities": len(targets),
                    "preroll_frames": window,
                    "preroll_ms": window * 20,
                    "schedule_frames": len(schedule),
                    **replay,
                }
                f.write(json.dumps(row, sort_keys=True) + "\n")
                f.flush()
                result_rows.append(row)

    aggregate = []
    total_frames = sum(
        int(policies["predictive-fec"]["frames"])
        for policies in by_condition.values()
        if "predictive-fec" in policies
    )
    for window in preroll_frames:
        rr = [r for r in result_rows if int(r["preroll_frames"]) == window]
        opportunities = sum(int(r["target_opportunities"]) for r in rr)
        protected = sum(int(r["schedule_frames"]) for r in rr)
        recovered = sum(int(r["fec_recovered_frames"]) for r in rr)
        aggregate.append(
            {
                "preroll_frames": window,
                "preroll_ms": window * 20,
                "target_opportunities": opportunities,
                "protected_frames": protected,
                "effective_duty_cycle": protected / total_frames if total_frames else None,
                "actual_recovered_frames": recovered,
                "recovery_fraction_of_always_fec": (
                    recovered / opportunities if opportunities else None
                ),
                "protected_frames_per_target": (
                    protected / opportunities if opportunities else None
                ),
            }
        )

    summary = {
        "schema_version": 1,
        "experiment": "Opus FEC actuation response versus predictive pre-roll",
        "benchmark_results": str(benchmark_results),
        "manifest": str(manifest),
        "frame_ms": 20,
        "preroll_frames": list(preroll_frames),
        "aggregate": aggregate,
        "results_jsonl": str(results_path),
        "interpretation": (
            "Each window ends on an Always-FEC-recoverable lost source frame. "
            "Increasing pre-roll measures how much advance warning is required "
            "for sparse Opus FEC control to realize recoverable LBRR in practice. "
            "This is a non-causal actuation characterization, not a deployable policy."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
