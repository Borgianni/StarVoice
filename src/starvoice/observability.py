from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path

from .counterfactual import _packet_log_from_output
from .deadline import timely_fec_opportunity_slots
from .foresight import _sorted_log


DEFAULT_WINDOWS_MS = (100.0, 250.0, 500.0)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _trace_rows(path: Path) -> list[dict]:
    rows = _read_jsonl(path)
    rows.sort(key=lambda r: int(r["send_ns"]))
    return rows


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _topk_hits(slots: list[dict], feature: str, k: int) -> dict:
    ranked = sorted(
        slots,
        key=lambda x: (
            -float(x.get(feature, 0.0)),
            str(x["trace_name"]),
            str(x["utterance_id"]),
            int(x["sequence"]),
        ),
    )
    selected = ranked[: max(0, min(k, len(ranked)))]
    hits = sum(bool(x["opportunity"]) for x in selected)
    missed_hits = sum(
        bool(x["opportunity"]) and not bool(x["current_protected"])
        for x in selected
    )
    return {
        "feature": feature,
        "protected_frames": len(selected),
        "hits": hits,
        "missed_current_opportunity_hits": missed_hits,
    }


def _condition_features(
    pred_log: list[dict],
    trace_rows: list[dict],
    windows_ms: tuple[float, ...],
) -> list[dict]:
    """Build features available causally at each source-frame send time."""
    received = sorted(
        (
            (int(r["recv_ns"]), float(r["rtt_ms"]))
            for r in trace_rows
            if r.get("recv_ns") is not None and r.get("rtt_ms") is not None
        ),
        key=lambda x: x[0],
    )
    recv_index = 0
    observed: deque[tuple[int, float]] = deque()
    max_window_ns = int(max(windows_ms) * 1_000_000)

    out: list[dict] = []
    for row in pred_log:
        now = int(row["send_ns"])
        while recv_index < len(received) and received[recv_index][0] <= now:
            observed.append(received[recv_index])
            recv_index += 1
        while observed and observed[0][0] < now - max_window_ns:
            observed.popleft()

        feats: dict[str, float] = {
            "phase_risk": float(row.get("risk", 0.0)),
            "last_rtt_ms": observed[-1][1] if observed else 0.0,
            "last_rtt_age_ms": (
                (now - observed[-1][0]) / 1_000_000 if observed else 1e9
            ),
        }

        trace_index = int(row["source_trace_index"])
        for window_ms in windows_ms:
            cutoff = now - int(window_ms * 1_000_000)
            vals = [v for t, v in observed if t >= cutoff]
            suffix = str(int(window_ms))
            feats[f"rtt_max_{suffix}ms"] = max(vals) if vals else 0.0
            feats[f"rtt_mean_{suffix}ms"] = (
                sum(vals) / len(vals) if vals else 0.0
            )

            unanswered = 0
            send_cutoff = now - int(window_ms * 1_000_000)
            j = trace_index - 1
            while j >= 0 and int(trace_rows[j]["send_ns"]) >= send_cutoff:
                tr = trace_rows[j]
                recv_ns = tr.get("recv_ns")
                if recv_ns is None or int(recv_ns) > now:
                    unanswered += 1
                j -= 1
            feats[f"unanswered_{suffix}ms"] = float(unanswered)

        out.append(feats)

    return out


def run_causal_observability_audit(
    benchmark_results: Path,
    output: Path,
    playout_ms: float = 60.0,
    rtt_fraction: float = 0.5,
    frame_ms: float = 20.0,
    windows_ms: tuple[float, ...] = DEFAULT_WINDOWS_MS,
) -> dict:
    """Audit whether missed timely FEC opportunities have causal precursors.

    This is deliberately not a trained model. It ranks frozen frames by simple
    features that are available by each send time and reports top-k opportunity
    recall at the current predictive protection budget.
    """
    rows = _read_jsonl(benchmark_results)
    by_condition: dict[tuple[str, str], dict[str, dict]] = {}
    for row in rows:
        key = (row["trace_name"], row["utterance_id"])
        by_condition.setdefault(key, {})[row["policy"]] = row

    slots: list[dict] = []
    trace_cache: dict[Path, list[dict]] = {}
    current_protected = 0
    total_opportunities = 0
    current_hits = 0

    for key, policies in sorted(by_condition.items()):
        predictive = policies.get("predictive-fec")
        always = policies.get("always-fec")
        if predictive is None or always is None:
            continue

        pred_log = _sorted_log(_packet_log_from_output(predictive["output_wav"]))
        always_log = _sorted_log(_packet_log_from_output(always["output_wav"]))
        trace = Path(predictive["trace"])
        if trace not in trace_cache:
            trace_cache[trace] = _trace_rows(trace)
        trace_rows = trace_cache[trace]

        opportunities = timely_fec_opportunity_slots(
            always_log,
            trace_rows,
            playout_ms=playout_ms,
            rtt_fraction=rtt_fraction,
            frame_ms=frame_ms,
        )
        protected = {
            int(r["sequence"])
            for r in pred_log
            if bool(r.get("fec_enabled"))
        }
        features = _condition_features(pred_log, trace_rows, windows_ms)
        if len(features) != len(pred_log):
            raise RuntimeError("feature/log length mismatch")

        for row, feats in zip(pred_log, features):
            seq = int(row["sequence"])
            slots.append(
                {
                    "trace_name": key[0],
                    "utterance_id": key[1],
                    "sequence": seq,
                    "opportunity": seq in opportunities,
                    "current_protected": seq in protected,
                    **feats,
                }
            )

        total_opportunities += len(opportunities)
        current_protected += len(protected)
        current_hits += len(protected & opportunities)

    feature_names = [
        "phase_risk",
        "last_rtt_ms",
        "last_rtt_age_ms",
    ]
    for window_ms in windows_ms:
        suffix = str(int(window_ms))
        feature_names.extend(
            [
                f"rtt_max_{suffix}ms",
                f"rtt_mean_{suffix}ms",
                f"unanswered_{suffix}ms",
            ]
        )

    ranking = [
        _topk_hits(slots, feature, current_protected)
        for feature in feature_names
    ]
    ranking.sort(key=lambda r: (-int(r["hits"]), r["feature"]))

    opportunity_rows = [s for s in slots if s["opportunity"]]
    captured = [s for s in opportunity_rows if s["current_protected"]]
    missed = [s for s in opportunity_rows if not s["current_protected"]]

    feature_distributions = {}
    for feature in feature_names:
        def stats(group: list[dict]) -> dict:
            vals = [float(x[feature]) for x in group]
            return {
                "n": len(vals),
                "median": _quantile(vals, 0.5),
                "q25": _quantile(vals, 0.25),
                "q75": _quantile(vals, 0.75),
                "max": max(vals) if vals else None,
            }

        feature_distributions[feature] = {
            "captured_opportunities": stats(captured),
            "missed_opportunities": stats(missed),
        }

    summary = {
        "schema_version": 1,
        "experiment": "causal observability audit for missed timely voice opportunities",
        "benchmark_results": str(benchmark_results),
        "playout_ms": playout_ms,
        "rtt_fraction": rtt_fraction,
        "frame_ms": frame_ms,
        "windows_ms": list(windows_ms),
        "frames": len(slots),
        "timely_opportunities": total_opportunities,
        "current_protected_frames": current_protected,
        "current_hits": current_hits,
        "current_recall": (
            current_hits / total_opportunities if total_opportunities else None
        ),
        "missed_opportunities": total_opportunities - current_hits,
        "topk_at_current_budget": ranking,
        "feature_distributions": feature_distributions,
        "causality": (
            "RTT features use only replies with recv_ns <= current send_ns. "
            "unanswered features count prior probes in the lookback window whose "
            "reply has not arrived by the current send time; they do not use future "
            "loss labels or timeout outcomes."
        ),
        "interpretation": (
            "This is an observability audit, not a fitted predictor. Features that "
            "recover additional current-missed opportunities at fixed top-k budget "
            "justify inclusion in a subsequently frozen predictive model."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
