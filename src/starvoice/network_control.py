from __future__ import annotations

import json
import statistics
from pathlib import Path

from .predictor import _collapse_bursts, _event_times, _read_probe, search_period


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def analyze_network_control_trace(
    trace: Path,
    threshold_ms: float = 50.0,
    min_period_s: float = 10.0,
    max_period_s: float = 20.0,
    step_s: float = 0.001,
    collapse_gap_s: float = 0.5,
) -> dict:
    rows = _read_probe(trace)
    if len(rows) < 2:
        raise ValueError(f"trace too short: {trace}")

    send_times = [int(r["send_ns"]) / 1e9 for r in rows]
    intervals_ms = [
        (send_times[i] - send_times[i - 1]) * 1000.0
        for i in range(1, len(send_times))
    ]
    received = [r for r in rows if not bool(r.get("lost")) and r.get("rtt_ms") is not None]
    rtts = [float(r["rtt_ms"]) for r in received]
    lost = sum(bool(r.get("lost")) for r in rows)
    threshold_events = sum(
        (not bool(r.get("lost")))
        and r.get("rtt_ms") is not None
        and float(r["rtt_ms"]) >= threshold_ms
        for r in rows
    )

    event_times = _event_times(rows, threshold_ms=threshold_ms)
    bursts = _collapse_bursts(event_times, gap_s=collapse_gap_s)
    best_period_s, coherence = search_period(
        trace,
        min_period_s=min_period_s,
        max_period_s=max_period_s,
        step_s=step_s,
        threshold_ms=threshold_ms,
    )
    median_interval_ms = statistics.median(intervals_ms)

    return {
        "trace": str(trace),
        "packets": len(rows),
        "duration_s": send_times[-1] - send_times[0],
        "lost": lost,
        "loss_rate": lost / len(rows),
        "rtt_ms_mean": statistics.fmean(rtts) if rtts else None,
        "rtt_ms_median": statistics.median(rtts) if rtts else None,
        "rtt_ms_p95": _percentile(rtts, 0.95),
        "rtt_threshold_ms": threshold_ms,
        "rtt_threshold_events": threshold_events,
        "event_count": len(event_times),
        "collapsed_bursts": len(bursts),
        "collapse_gap_s": collapse_gap_s,
        "median_probe_interval_ms": median_interval_ms,
        "best_period_s": best_period_s,
        "period_coherence": coherence,
        "packets_per_period": best_period_s / (median_interval_ms / 1000.0),
        "period_search": {
            "min_period_s": min_period_s,
            "max_period_s": max_period_s,
            "step_s": step_s,
        },
    }


def run_network_control_analysis(
    traces: list[Path],
    output: Path,
    threshold_ms: float = 50.0,
    min_period_s: float = 10.0,
    max_period_s: float = 20.0,
    step_s: float = 0.001,
    collapse_gap_s: float = 0.5,
) -> dict:
    if not traces:
        raise ValueError("at least one trace is required")

    rows = [
        analyze_network_control_trace(
            trace,
            threshold_ms=threshold_ms,
            min_period_s=min_period_s,
            max_period_s=max_period_s,
            step_s=step_s,
            collapse_gap_s=collapse_gap_s,
        )
        for trace in traces
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "threshold_ms": threshold_ms,
        "traces": rows,
        "interpretation": (
            "A wall-clock network periodicity should keep best_period_s approximately "
            "stable when probe_interval_ms changes, while packets_per_period should "
            "scale inversely with the interval. A packet-count artifact would instead "
            "keep packets_per_period approximately stable."
        ),
        "causal_caution": (
            "Periodicity alone does not identify Starlink handovers. Compare against "
            "a terrestrial path to the same relay and repeat across days/sites before "
            "making a causal attribution."
        ),
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
