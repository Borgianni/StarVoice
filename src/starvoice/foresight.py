from __future__ import annotations

import json
from pathlib import Path

from .counterfactual import _packet_log_from_output


DEFAULT_BUDGETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sorted_log(path: Path) -> list[dict]:
    rows = _read_jsonl(path)
    rows.sort(key=lambda r: int(r["sequence"]))
    for i, row in enumerate(rows):
        if int(row["sequence"]) != i:
            raise ValueError(f"non-contiguous packet log {path} at sequence {i}")
    return rows


def _frontier_point(
    slots: list[dict],
    opportunities: int,
    protected_frames: int,
) -> dict:
    k = max(0, min(protected_frames, len(slots)))
    ranked = sorted(
        slots,
        key=lambda x: (
            -float(x["risk"]),
            str(x["trace_name"]),
            str(x["utterance_id"]),
            int(x["sequence"]),
        ),
    )
    risk_hits = sum(bool(x["opportunity"]) for x in ranked[:k])
    oracle_hits = min(k, opportunities)
    random_hits = opportunities * k / len(slots) if slots else 0.0
    return {
        "protected_frames": k,
        "effective_duty_cycle": k / len(slots) if slots else None,
        "recoverable_opportunities": opportunities,
        "risk_ranked_hits": risk_hits,
        "risk_ranked_opportunity_recall": (
            risk_hits / opportunities if opportunities else None
        ),
        "oracle_hits": oracle_hits,
        "oracle_opportunity_recall": (
            oracle_hits / opportunities if opportunities else None
        ),
        "random_expected_hits": random_hits,
        "random_expected_opportunity_recall": (
            random_hits / opportunities if opportunities else None
        ),
        "risk_fraction_of_oracle": (
            risk_hits / oracle_hits if oracle_hits else None
        ),
    }


def run_foresight_analysis(
    benchmark_results: Path,
    output: Path,
    budgets: tuple[float, ...] = DEFAULT_BUDGETS,
) -> dict:
    """Quantify the value of future-risk information under a long-run FEC budget.

    The budget is global across the frozen stream rather than reset per utterance.
    This matches the behavior of the deployed threshold policy, which spends zero
    protection in safe regions and can spend heavily during short risky windows
    while respecting a small long-run average duty cycle.

    Recoverable opportunities are defined empirically from the frozen Always-FEC
    replay. A lost frame i is an opportunity iff Always-FEC actually recovered it;
    its protection slot is packet i+1 because Opus in-band FEC carries frame i in
    the following packet.

    Oracle and risk-ranked results are placement diagnostics. They do not assert
    that a newly synthesized sparse Opus schedule would necessarily emit LBRR in
    every selected slot.
    """
    rows = _read_jsonl(benchmark_results)
    by_condition: dict[tuple[str, str], dict[str, dict]] = {}
    for row in rows:
        key = (row["trace_name"], row["utterance_id"])
        by_condition.setdefault(key, {})[row["policy"]] = row

    slots: list[dict] = []
    condition_summaries: list[dict] = []
    total_losses = 0
    total_opportunities = 0
    current_protected = 0
    current_hits = 0
    current_actual = 0

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

        for row in pred_log:
            sequence = int(row["sequence"])
            slots.append(
                {
                    "trace_name": key[0],
                    "utterance_id": key[1],
                    "sequence": sequence,
                    "risk": float(row.get("risk", 0.0)),
                    "opportunity": sequence in opportunity_slots,
                }
            )

        hits = len(current_slots & opportunity_slots)
        actual = int(predictive["fec_recovered_frames"])
        losses = int(predictive["network_lost_frames"])

        total_losses += losses
        total_opportunities += len(opportunity_slots)
        current_protected += len(current_slots)
        current_hits += hits
        current_actual += actual

        condition_summaries.append(
            {
                "trace_name": key[0],
                "utterance_id": key[1],
                "frames": frames,
                "network_lost_frames": losses,
                "recoverable_opportunities": len(opportunity_slots),
                "current_predictive_protected_frames": len(current_slots),
                "current_predictive_opportunity_hits": hits,
                "current_predictive_actual_recovered_frames": actual,
            }
        )

    total_frames = len(slots)
    frontier = []
    for duty in budgets:
        if not 0.0 <= duty <= 1.0:
            raise ValueError(f"invalid budget duty cycle: {duty}")
        k = int(round(total_frames * duty))
        point = _frontier_point(slots, total_opportunities, k)
        point["target_duty_cycle"] = duty
        frontier.append(point)

    matched = _frontier_point(
        slots,
        total_opportunities,
        current_protected,
    )
    matched["target_duty_cycle"] = (
        current_protected / total_frames if total_frames else None
    )

    summary = {
        "schema_version": 2,
        "experiment": "value of foresight under a long-run protection budget",
        "budget_semantics": (
            "Global long-run average across all frozen frames. Budget is not reset "
            "per utterance; protection may concentrate in predicted-risk windows."
        ),
        "benchmark_results": str(benchmark_results),
        "conditions": len(condition_summaries),
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
        "risk_ranked_at_current_budget": matched,
        "frontier": frontier,
        "interpretation": (
            "Random is the exact global-budget chance expectation. risk-ranked "
            "spends the same long-run budget on the highest frozen causal StarVoice "
            "risk scores across the stream. Oracle knows future recoverable "
            "opportunity locations. The risk-ranked-to-Oracle gap is predictor/"
            "placement headroom. current_predictive is the actually replayed "
            "threshold policy, not a synthetic ranking."
        ),
        "caution": (
            "Oracle and risk-ranked hits are placement diagnostics defined using "
            "opportunities observed under Always-FEC; they are not decoded QoE "
            "results and should not be presented as actual sparse-FEC recoveries."
        ),
        "conditions_detail": condition_summaries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
