from __future__ import annotations

import json
import statistics
from pathlib import Path

from .predictor import _coherence, _read_probe


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


def _collapse_consecutive(times: list[float], gap_s: float) -> list[float]:
    """Collapse an event run when consecutive events are less than gap_s apart."""
    if not times:
        return []
    bursts = [times[0]]
    previous = times[0]
    for t in times[1:]:
        if t - previous >= gap_s:
            bursts.append(t)
        previous = t
    return bursts


def _search_period_times(
    times: list[float],
    min_period_s: float,
    max_period_s: float,
    step_s: float,
) -> tuple[float | None, float | None]:
    if not times:
        return None, None
    best_period = min_period_s
    best_score = -1.0
    p = min_period_s
    while p <= max_period_s + step_s / 2:
        score = _coherence(times, p)
        if score > best_score:
            best_period = p
            best_score = score
        p += step_s
    return best_period, best_score


def _period_view(
    event_times: list[float],
    collapse_gap_s: float,
    min_period_s: float,
    max_period_s: float,
    step_s: float,
    effective_interval_ms: float,
) -> dict:
    bursts = _collapse_consecutive(event_times, collapse_gap_s)
    period_s, coherence = _search_period_times(
        bursts,
        min_period_s=min_period_s,
        max_period_s=max_period_s,
        step_s=step_s,
    )
    return {
        "event_count": len(event_times),
        "collapsed_bursts": len(bursts),
        "best_period_s": period_s,
        "period_coherence": coherence,
        "packets_per_period_effective": (
            period_s / (effective_interval_ms / 1000.0)
            if period_s is not None and effective_interval_ms > 0
            else None
        ),
    }


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

    t0_ns = int(rows[0]["send_ns"])
    send_times = [int(r["send_ns"]) / 1e9 for r in rows]
    duration_s = send_times[-1] - send_times[0]
    intervals_ms = [
        (send_times[i] - send_times[i - 1]) * 1000.0
        for i in range(1, len(send_times))
    ]
    effective_interval_ms = duration_s * 1000.0 / (len(rows) - 1)

    received = [
        r for r in rows
        if not bool(r.get("lost")) and r.get("rtt_ms") is not None
    ]
    rtts = [float(r["rtt_ms"]) for r in received]

    loss_times: list[float] = []
    rtt_event_times: list[float] = []
    for row in rows:
        t = (int(row["send_ns"]) - t0_ns) / 1e9
        if bool(row.get("lost")):
            loss_times.append(t)
        elif row.get("rtt_ms") is not None and float(row["rtt_ms"]) >= threshold_ms:
            rtt_event_times.append(t)

    combined_times = sorted(loss_times + rtt_event_times)

    views = {
        "combined": _period_view(
            combined_times,
            collapse_gap_s,
            min_period_s,
            max_period_s,
            step_s,
            effective_interval_ms,
        ),
        "loss_only": _period_view(
            loss_times,
            collapse_gap_s,
            min_period_s,
            max_period_s,
            step_s,
            effective_interval_ms,
        ),
        "rtt_only": _period_view(
            rtt_event_times,
            collapse_gap_s,
            min_period_s,
            max_period_s,
            step_s,
            effective_interval_ms,
        ),
    }

    return {
        "trace": str(trace),
        "packets": len(rows),
        "duration_s": duration_s,
        "lost": len(loss_times),
        "loss_rate": len(loss_times) / len(rows),
        "rtt_ms_mean": statistics.fmean(rtts) if rtts else None,
        "rtt_ms_median": statistics.median(rtts) if rtts else None,
        "rtt_ms_p95": _percentile(rtts, 0.95),
        "rtt_threshold_ms": threshold_ms,
        "rtt_threshold_events": len(rtt_event_times),
        "collapse_gap_s": collapse_gap_s,
        "probe_timing": {
            "effective_interval_ms": effective_interval_ms,
            "median_interval_ms": statistics.median(intervals_ms),
            "p05_interval_ms": _percentile(intervals_ms, 0.05),
            "p95_interval_ms": _percentile(intervals_ms, 0.95),
            "note": (
                "effective_interval_ms = trace duration/(packets-1) and is used "
                "for packets-per-period. Median/p05/p95 expose scheduler batching."
            ),
        },
        "periodicity": views,
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
        "schema_version": 2,
        "threshold_ms": threshold_ms,
        "traces": rows,
        "interpretation": (
            "A wall-clock network periodicity should keep best_period_s approximately "
            "stable when the probe interval changes, while packets-per-period scales "
            "inversely with the effective interval. Loss-only periodicity is the "
            "strongest control because it does not depend on the RTT threshold."
        ),
        "timing_note": (
            "Scheduler catch-up can make the median inter-send interval differ from "
            "the configured interval. packets_per_period_effective therefore uses "
            "duration/(packets-1), while inter-send percentiles are reported as a "
            "probe-load diagnostic."
        ),
        "causal_caution": (
            "Periodicity alone does not identify Starlink handovers. Compare against "
            "a terrestrial path to the same relay and repeat across days/sites before "
            "making a causal attribution."
        ),
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
