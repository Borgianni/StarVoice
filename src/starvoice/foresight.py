from __future__ import annotations

import json
import math
from pathlib import Path

from .counterfactual import _packet_log_from_output


DEFAULT_BUDGETS = (0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.10)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sorted_log(path: Path) -> list[dict]:
    rows = _read_jsonl(path)
    rows.sort(key=lambda r: int(r["sequence"]))
    for i, row in enumerate(rows):
        if int(row["sequence"]) != i:
            raise ValueError(f"non-contiguous packet log {path} at sequence {i}")
    return rows


def _top_risk_slots(rows: list[dict], protected: int) -> set[int]:
    protected = max(0, min(protected, len(rows)))
    ranked = sorted(
        range(len(rows)),
        key=lambda i: (-float(rows[i].get("risk", 0.0)), i),
    )
    return set(ranked[:protected])


def run_foresight_analysis(
    benchmark_results: Path,
    output: Path,
    budgets: tuple[float, ...] = DEFAULT_BUDGETS,
) -> dict:
    """Quantify how useful future-risk information is under fixed FEC budgets.

    Recoverable opportunities are defined empirically from the frozen Always-FEC
    replay: a lost frame is an opportunity iff Always-FEC actually recovered it.
    The protection slot for lost frame i is packet i+1 because Opus in-band FEC
    carries the previous frame.

    This analysis does not claim that a sparse schedule would always emit LBRR
    whenever Always-FEC did. It is a placement/headroom diagnostic: Random is the
    chance baseline, risk-ranked uses the frozen causal risk score, and Oracle is
    the upper bound with perfect knowledge of opportunity locations.
    """
    rows = _read_jsonl(benchmark_results)
    by_condition: dict[tuple[str, str], dict[str, dict]] = {}
    for row in rows:
        key = (row["trace_name"], row["utterance_id"])
        by_condition.setdefault(key, {})[row["policy"]] = row

    conditions = []
    for key, policies in sorted(by_condition.items()):
        predictive = policies.get("predictive-fec")
        always = policies.get("always-fec")
        if predictive is None or always is None:
            continue

        pred_log = _sorted_log(_packet_log_from_output(predictive["output_wav"]))
        always_log = _sorted_log(_packet_log_from_output(always["output_wav"]))
        if len(pred_log) != len(always_log):
            raise RuntimeError(f"packet-log length mismatch for {key}")

        frames = len(pred_log)
        opportunity_slots = {
            int(row["sequence"]) + 1
            for row in always_log
            if bool(row.get("lost"))
            and bool(row.get("recovered_fec"))
            and int(row["sequence"]) + 1 < frames
        }
        current_slots = {
            int(row["sequence"])
            for row in pred_log
            if bool(row.get("fec_enabled"))
        }

        current_opportunity_hits = len(current_slots & opportunity_slots)
        current_actual_recovered = int(predictive["fec_recovered_frames"])
        current_budget = len(current_slots)

        frontier = []
        for duty in budgets:
            if not 0.0 <= duty <= 1.0:
                raise ValueError(f"invalid budget duty cycle: {duty}")
            k = int(round(frames * duty))
            risk_slots = _top_risk_slots(pred_log, k)
            m = len(opportunity_slots)
            risk_hits = len(risk_slots & opportunity_slots)
            oracle_hits = min(k, m)
            random_expected_hits = (m * k / frames) if frames else 0.0
            frontier.append(
                {
                    "duty_cycle": duty,
                    "protected_frames": k,
                    "recoverable_opportunities": m,
                    "risk_ranked_hits": risk_hits,
                    "oracle_hits": oracle_hits,
                    "random_expected_hits": random_expected_hits,
                }
            )

        conditions.append(
            {
                "trace_name": key[0],
                "utterance_id": key[1],
                "frames": frames,
                "network_lost_frames": int(predictive["network_lost_frames"]),
                "recoverable_opportunities": len(opportunity_slots),
                "current_predictive": {
                    "protected_frames": current_budget,
                    "duty_cycle": current_budget / frames if frames else None,
                    "opportunity_hits": current_opportunity_hits,
                    "actual_recovered_frames": current_actual_recovered,
                },
                "frontier": frontier,
            }
        )

    total_frames = sum(c["frames"] for c in conditions)
    total_losses = sum(c["network_lost_frames"] for c in conditions)
    total_opportunities = sum(c["recoverable_opportunities"] for c in conditions)
    current_protected = sum(
        c["current_predictive"]["protected_frames"] for c in conditions
    )
    current_hits = sum(
        c["current_predictive"]["opportunity_hits"] for c in conditions
    )
    current_actual = sum(
        c["current_predictive"]["actual_recovered_frames"] for c in conditions
    )

    aggregate_frontier = []
    for idx, duty in enumerate(budgets):
        protected = sum(c["frontier"][idx]["protected_frames"] for c in conditions)
        risk_hits = sum(c["frontier"][idx]["risk_ranked_hits"] for c in conditions)
        oracle_hits = sum(c["frontier"][idx]["oracle_hits"] for c in conditions)
        random_hits = sum(
            c["frontier"][idx]["random_expected_hits"] for c in conditions
        )
        aggregate_frontier.append(
            {
                "duty_cycle": duty,
                "protected_frames": protected,
                "effective_duty_cycle": protected / total_frames if total_frames else None,
                "recoverable_opportunities": total_opportunities,
                "risk_ranked_hits": risk_hits,
                "risk_ranked_opportunity_recall": (
                    risk_hits / total_opportunities if total_opportunities else None
                ),
                "oracle_hits": oracle_hits,
                "oracle_opportunity_recall": (
                    oracle_hits / total_opportunities if total_opportunities else None
                ),
                "random_expected_hits": random_hits,
                "random_expected_opportunity_recall": (
                    random_hits / total_opportunities if total_opportunities else None
                ),
                "risk_fraction_of_oracle": (
                    risk_hits / oracle_hits if oracle_hits else None
                ),
            }
        )

    summary = {
        "schema_version": 1,
        "experiment": "value of foresight under fixed protection budgets",
        "benchmark_results": str(benchmark_results),
        "conditions": len(conditions),
        "frames": total_frames,
        "network_lost_frames": total_losses,
        "always_fec_recoverable_opportunities": total_opportunities,
        "current_predictive": {
            "protected_frames": current_protected,
            "effective_duty_cycle": (
                current_protected / total_frames if total_frames else None
            ),
            "opportunity_hits": current_hits,
            "opportunity_recall": (
                current_hits / total_opportunities if total_opportunities else None
            ),
            "actual_recovered_frames": current_actual,
            "actual_fraction_of_always_fec_recovery": (
                current_actual / total_opportunities if total_opportunities else None
            ),
            "lbrr_realization_given_opportunity_hit": (
                current_actual / current_hits if current_hits else None
            ),
        },
        "frontier": aggregate_frontier,
        "interpretation": (
            "Random is the exact-budget chance expectation. risk-ranked spends the "
            "same budget on frames with the highest frozen causal StarVoice risk. "
            "Oracle knows future recoverable opportunity locations. The gap between "
            "risk-ranked and Oracle is prediction/placement headroom; the gap between "
            "opportunity hits and actual recovered frames reflects sparse-schedule "
            "Opus/LBRR realization and other codec constraints."
        ),
        "caution": (
            "Oracle and risk-ranked hits are placement diagnostics defined using "
            "opportunities observed under Always-FEC; they are not decoded QoE "
            "results and should not be presented as actual sparse-FEC recoveries."
        ),
        "conditions_detail": conditions,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
