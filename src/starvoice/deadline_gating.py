from __future__ import annotations

import json
from collections import deque
from pathlib import Path

from .counterfactual import _packet_log_from_output
from .deadline import timely_fec_opportunity_slots
from .foresight import _sorted_log


DEFAULT_PLAYOUT_MS = (40.0, 50.0, 60.0, 80.0)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _trace_rows(path: Path) -> list[dict]:
    rows = _read_jsonl(path)
    rows.sort(key=lambda r: int(r["send_ns"]))
    return rows


def _causal_delay_estimates(
    packet_log: list[dict],
    trace_rows: list[dict],
    window_ms: float = 250.0,
) -> list[dict]:
    """Causal RTT estimators available at each packet send time."""
    received = sorted(
        (
            (int(r["recv_ns"]), float(r["rtt_ms"]))
            for r in trace_rows
            if r.get("recv_ns") is not None and r.get("rtt_ms") is not None
        ),
        key=lambda x: x[0],
    )
    idx = 0
    observed: deque[tuple[int, float]] = deque()
    window_ns = int(round(window_ms * 1_000_000))
    out = []

    for row in packet_log:
        now = int(row["send_ns"])
        while idx < len(received) and received[idx][0] <= now:
            observed.append(received[idx])
            idx += 1
        while observed and observed[0][0] < now - window_ns:
            observed.popleft()

        values = [v for _, v in observed]
        out.append(
            {
                "last_rtt_ms": values[-1] if values else None,
                "max_rtt_window_ms": max(values) if values else None,
            }
        )
    return out


def _keep_by_feasibility(
    estimate_rtt_ms: float | None,
    playout_ms: float,
    frame_ms: float,
    rtt_fraction: float,
) -> bool:
    # With no causal RTT observation, fail open: preserve the protection decision.
    if estimate_rtt_ms is None:
        return True
    estimated_repair_arrival_ms = frame_ms + rtt_fraction * estimate_rtt_ms
    return estimated_repair_arrival_ms <= playout_ms


def run_deadline_gating_analysis(
    benchmark_results: Path,
    output: Path,
    playout_ms: tuple[float, ...] = DEFAULT_PLAYOUT_MS,
    rtt_fraction: float = 0.5,
    frame_ms: float = 20.0,
    window_ms: float = 250.0,
) -> dict:
    """Gate the frozen predictive schedule by causal FEC deadline feasibility.

    This is a schedule-level diagnostic. The underlying predictive risk decisions
    are unchanged. A protected source frame is retained only if a causal RTT
    estimate suggests that the next packet carrying LBRR can still arrive before
    that source frame's playout deadline.

    Two estimator variants are reported:
      * last_rtt: most recently received RTT sample.
      * max_window: conservative maximum received RTT over the recent window.

    Both use only replies received by the current frame send time.
    """
    if not playout_ms:
        raise ValueError("at least one playout delay is required")
    if not 0.0 <= rtt_fraction <= 1.0:
        raise ValueError("rtt_fraction must be in [0,1]")

    rows = _read_jsonl(benchmark_results)
    by_condition: dict[tuple[str, str], dict[str, dict]] = {}
    for row in rows:
        key = (row["trace_name"], row["utterance_id"])
        by_condition.setdefault(key, {})[row["policy"]] = row

    trace_cache: dict[Path, list[dict]] = {}
    delay_acc = {
        float(d): {
            "frames": 0,
            "opportunities": 0,
            "baseline_protected": 0,
            "baseline_hits": 0,
            "last_rtt_protected": 0,
            "last_rtt_hits": 0,
            "max_window_protected": 0,
            "max_window_hits": 0,
        }
        for d in playout_ms
    }

    for key, policies in sorted(by_condition.items()):
        pred = policies.get("predictive-fec")
        always = policies.get("always-fec")
        if pred is None or always is None:
            continue

        pred_log = _sorted_log(_packet_log_from_output(pred["output_wav"]))
        always_log = _sorted_log(_packet_log_from_output(always["output_wav"]))
        trace = Path(pred["trace"])
        if trace not in trace_cache:
            trace_cache[trace] = _trace_rows(trace)
        trace_rows = trace_cache[trace]

        estimates = _causal_delay_estimates(
            pred_log,
            trace_rows,
            window_ms=window_ms,
        )
        protected = {
            int(row["sequence"])
            for row in pred_log
            if bool(row.get("fec_enabled"))
        }

        for delay in playout_ms:
            d = float(delay)
            opportunities = timely_fec_opportunity_slots(
                always_log,
                trace_rows,
                playout_ms=d,
                rtt_fraction=rtt_fraction,
                frame_ms=frame_ms,
            )

            last_schedule = set()
            max_schedule = set()
            for row, est in zip(pred_log, estimates):
                seq = int(row["sequence"])
                if seq not in protected:
                    continue
                if _keep_by_feasibility(
                    est["last_rtt_ms"], d, frame_ms, rtt_fraction
                ):
                    last_schedule.add(seq)
                if _keep_by_feasibility(
                    est["max_rtt_window_ms"], d, frame_ms, rtt_fraction
                ):
                    max_schedule.add(seq)

            a = delay_acc[d]
            a["frames"] += len(pred_log)
            a["opportunities"] += len(opportunities)
            a["baseline_protected"] += len(protected)
            a["baseline_hits"] += len(protected & opportunities)
            a["last_rtt_protected"] += len(last_schedule)
            a["last_rtt_hits"] += len(last_schedule & opportunities)
            a["max_window_protected"] += len(max_schedule)
            a["max_window_hits"] += len(max_schedule & opportunities)

    results = []
    for delay in sorted(delay_acc):
        a = delay_acc[delay]
        base_p = int(a["baseline_protected"])
        base_h = int(a["baseline_hits"])

        def policy(prefix: str) -> dict:
            p = int(a[f"{prefix}_protected"])
            h = int(a[f"{prefix}_hits"])
            return {
                "protected_frames": p,
                "effective_duty_cycle": p / int(a["frames"]) if a["frames"] else None,
                "timely_hits": h,
                "timely_recall": h / int(a["opportunities"]) if a["opportunities"] else None,
                "hit_retention_vs_baseline": h / base_h if base_h else None,
                "protection_reduction_vs_baseline": 1.0 - p / base_p if base_p else None,
                "timely_hits_per_1000_protected": 1000.0 * h / p if p else None,
            }

        baseline = {
            "protected_frames": base_p,
            "effective_duty_cycle": base_p / int(a["frames"]) if a["frames"] else None,
            "timely_hits": base_h,
            "timely_recall": base_h / int(a["opportunities"]) if a["opportunities"] else None,
            "timely_hits_per_1000_protected": (
                1000.0 * base_h / base_p if base_p else None
            ),
        }

        results.append(
            {
                "playout_ms": delay,
                "frames": int(a["frames"]),
                "timely_opportunities": int(a["opportunities"]),
                "baseline_predictive": baseline,
                "last_rtt_gate": policy("last_rtt"),
                "max_recent_rtt_gate": policy("max_window"),
                "oracle_keep_only_baseline_hits": {
                    "protected_frames": base_h,
                    "timely_hits": base_h,
                    "protection_reduction_vs_baseline": (
                        1.0 - base_h / base_p if base_p else None
                    ),
                },
            }
        )

    summary = {
        "schema_version": 1,
        "experiment": "causal deadline-feasibility gating of predictive FEC schedule",
        "benchmark_results": str(benchmark_results),
        "frame_ms": frame_ms,
        "playout_ms": [float(x) for x in playout_ms],
        "rtt_fraction": rtt_fraction,
        "recent_window_ms": window_ms,
        "gate_rule": (
            "Keep a predictive FEC decision iff frame_ms + rtt_fraction * "
            "causal_RTT_estimate <= playout_ms. Missing estimate fails open."
        ),
        "causality": (
            "RTT estimates include only replies with recv_ns <= current send_ns."
        ),
        "caution": (
            "This is a schedule-level placement diagnostic. If a gate materially "
            "improves efficiency, the selected schedule should next be replayed "
            "through the real Opus encoder/decoder before becoming a paper result."
        ),
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
