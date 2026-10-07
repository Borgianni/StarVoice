from __future__ import annotations

import json
from pathlib import Path

from .counterfactual import _packet_log_from_output
from .deadline import timely_fec_opportunity_slots
from .foresight import DEFAULT_BUDGETS, _frontier_point, _sorted_log


DEFAULT_PLAYOUT_MS = (30.0, 40.0, 50.0, 60.0, 80.0, 100.0)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _trace_rows(path: Path) -> list[dict]:
    rows = _read_jsonl(path)
    rows.sort(key=lambda r: int(r["send_ns"]))
    return rows


def run_deadline_foresight_analysis(
    benchmark_results: Path,
    output: Path,
    playout_ms: tuple[float, ...] = DEFAULT_PLAYOUT_MS,
    rtt_fraction: float = 0.5,
    frame_ms: float = 20.0,
    budgets: tuple[float, ...] = DEFAULT_BUDGETS,
) -> dict:
    """Value-of-foresight frontier using only FEC opportunities useful by playout.

    For each playout budget, Always-FEC packet logs define the empirical upper-bound
    opportunity set: source frames whose primary misses playout and whose following
    packet contains real LBRR that arrives by the source frame's deadline.

    The current predictive schedule, frozen causal risk scores, global random chance
    expectation, and oracle placement are then evaluated against the same timely
    opportunity set. Oracle/risk-ranked points are placement diagnostics, not newly
    decoded audio results.
    """
    if not playout_ms:
        raise ValueError("at least one playout delay is required")
    if not 0.0 <= rtt_fraction <= 1.0:
        raise ValueError("rtt_fraction must be in [0, 1]")

    rows = _read_jsonl(benchmark_results)
    by_condition: dict[tuple[str, str], dict[str, dict]] = {}
    for row in rows:
        key = (row["trace_name"], row["utterance_id"])
        by_condition.setdefault(key, {})[row["policy"]] = row

    condition_data: list[dict] = []
    trace_cache: dict[Path, list[dict]] = {}

    for key, policies in sorted(by_condition.items()):
        predictive = policies.get("predictive-fec")
        always = policies.get("always-fec")
        if predictive is None or always is None:
            continue

        pred_log = _sorted_log(_packet_log_from_output(predictive["output_wav"]))
        always_log = _sorted_log(_packet_log_from_output(always["output_wav"]))
        if len(pred_log) != len(always_log):
            raise RuntimeError(f"packet-log length mismatch for {key}")

        trace = Path(predictive["trace"])
        if trace not in trace_cache:
            trace_cache[trace] = _trace_rows(trace)

        condition_data.append(
            {
                "trace_name": key[0],
                "utterance_id": key[1],
                "trace_rows": trace_cache[trace],
                "predictive_log": pred_log,
                "always_log": always_log,
                "frames": len(pred_log),
            }
        )

    delay_results: list[dict] = []

    for delay in playout_ms:
        slots: list[dict] = []
        total_opportunities = 0
        current_protected = 0
        current_hits = 0
        current_actual_timely = 0
        condition_matched_random_expectation = 0.0

        for cond in condition_data:
            pred_log = cond["predictive_log"]
            always_log = cond["always_log"]
            trace_rows = cond["trace_rows"]
            frames = int(cond["frames"])

            opportunity_slots = timely_fec_opportunity_slots(
                always_log,
                trace_rows,
                playout_ms=float(delay),
                rtt_fraction=rtt_fraction,
                frame_ms=frame_ms,
            )
            actual_predictive_slots = timely_fec_opportunity_slots(
                pred_log,
                trace_rows,
                playout_ms=float(delay),
                rtt_fraction=rtt_fraction,
                frame_ms=frame_ms,
            )
            current_slots = {
                int(row["sequence"])
                for row in pred_log
                if bool(row.get("fec_enabled"))
            }

            for row in pred_log:
                sequence = int(row["sequence"])
                slots.append(
                    {
                        "trace_name": cond["trace_name"],
                        "utterance_id": cond["utterance_id"],
                        "sequence": sequence,
                        "risk": float(row.get("risk", 0.0)),
                        "opportunity": sequence in opportunity_slots,
                    }
                )

            hits = len(current_slots & opportunity_slots)
            total_opportunities += len(opportunity_slots)
            current_protected += len(current_slots)
            current_hits += hits
            current_actual_timely += len(actual_predictive_slots)
            condition_matched_random_expectation += (
                len(opportunity_slots) * len(current_slots) / frames
                if frames
                else 0.0
            )

        total_frames = len(slots)
        matched = _frontier_point(
            slots,
            total_opportunities,
            current_protected,
        )
        matched["target_duty_cycle"] = (
            current_protected / total_frames if total_frames else None
        )

        frontier = []
        for duty in budgets:
            if not 0.0 <= duty <= 1.0:
                raise ValueError(f"invalid budget duty cycle: {duty}")
            k = int(round(total_frames * duty))
            point = _frontier_point(slots, total_opportunities, k)
            point["target_duty_cycle"] = duty
            frontier.append(point)

        delay_results.append(
            {
                "playout_ms": float(delay),
                "frames": total_frames,
                "always_fec_timely_opportunities": total_opportunities,
                "current_predictive": {
                    "protected_frames": current_protected,
                    "effective_duty_cycle": (
                        current_protected / total_frames if total_frames else None
                    ),
                    "timely_opportunity_hits": current_hits,
                    "timely_opportunity_recall": (
                        current_hits / total_opportunities
                        if total_opportunities
                        else None
                    ),
                    "actual_timely_recoveries": current_actual_timely,
                    "actual_fraction_of_always_timely": (
                        current_actual_timely / total_opportunities
                        if total_opportunities
                        else None
                    ),
                    "lbrr_realization_given_timely_hit": (
                        current_actual_timely / current_hits
                        if current_hits
                        else None
                    ),
                },
                "condition_matched_random_expected_hits_at_current_schedule": (
                    condition_matched_random_expectation
                ),
                "risk_ranked_at_current_budget": matched,
                "frontier": frontier,
            }
        )

    summary = {
        "schema_version": 1,
        "experiment": "deadline-aware value of foresight under protection budget",
        "benchmark_results": str(benchmark_results),
        "conditions": len(condition_data),
        "frame_ms": frame_ms,
        "playout_ms": [float(x) for x in playout_ms],
        "rtt_fraction": rtt_fraction,
        "one_way_delay_model": f"{rtt_fraction:g} * RTT",
        "budget_semantics": (
            "Global long-run protection budget across all frozen frames. "
            "risk-ranked and oracle are placement diagnostics."
        ),
        "opportunity_semantics": (
            "A source frame is an opportunity iff its primary misses playout under "
            "the stated RTT-fraction model and Always-FEC's following packet carries "
            "actual LBRR that arrives by the source frame's playout deadline."
        ),
        "random_semantics": (
            "Frontier random is the exact global-uniform chance expectation. "
            "condition_matched_random_expected_hits_at_current_schedule preserves "
            "the current predictive policy's per-condition protected-frame counts."
        ),
        "caution": (
            "RTT-derived one-way delay is a proxy rather than synchronized OWD. "
            "Interpret the frontier jointly with RTT-fraction sensitivity. "
            "Oracle/risk-ranked hits are placement diagnostics; current_predictive "
            "actual_timely_recoveries are grounded in realized packet LBRR."
        ),
        "delays": delay_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
