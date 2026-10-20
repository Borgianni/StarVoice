"""Causal risk-gating falsification on the archived nine-block StarVoice campaign.

Only previous-cycle fixed60 playout outcomes can activate a future ramp.
The rescue-only oracle knows future packet outcomes: NONCAUSAL, not deployable.
No live capture, codec change, audio rendering or independent-session claim.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .slackbank import (
    BASE_MS, PERIOD_FRAMES, TRAIN_FRAMES, arrival_data, calibrate_phase,
    evaluate, ramp_targets,
)


def decision(seq: int, budget: int, arrivals: dict, anchor: int, first_seq: int) -> str:
    deadline = anchor + (seq - first_seq) * 20_000_000 + budget * 1_000_000
    primary = arrivals.get(seq)
    if primary is not None and int(primary["receive_mono_ns"]) <= deadline:
        return "primary"
    carrier = arrivals.get(seq + 1)
    if (carrier is not None and bool(carrier.get("has_lbrr"))
            and int(carrier["receive_mono_ns"]) <= deadline):
        return "fec"
    return "plc"


def gated_schedule(always: list[int], events: list[int], enabled: set[int]) -> list[int]:
    result = [BASE_MS] * len(always)
    for event in events:
        if event in enabled:
            left = max(TRAIN_FRAMES, event - 25)
            right = min(len(always), event + 27)
            result[left:right] = always[left:right]
    if any(abs(a - b) > 1 for a, b in zip(result, result[1:])):
        raise AssertionError("gated schedule violates physical 1ms/frame slew")
    return result


def audit_block(campaign: Path, block: str) -> dict:
    n, first_seq, arrivals = arrival_data(campaign / block)
    phase, support = calibrate_phase(arrivals)
    if phase is None:
        raise RuntimeError(f"{block}: no training-only phase")
    anchor = int(arrivals[first_seq]["receive_mono_ns"])
    always = ramp_targets(n, phase)
    events = [e for e in range(phase, n, PERIOD_FRAMES)
              if e - 25 >= TRAIN_FRAMES and e + 27 < n]
    if not events:
        raise RuntimeError(f"{block}: no complete held-out cycles")

    def fixed_decision(seq: int) -> str:
        return decision(seq, 60, arrivals, anchor, first_seq)

    # Measure future/observed value ONLY to label the retrospective oracle.
    rescues_by_event = {}
    for event in events:
        rescues_by_event[event] = sum(
            fixed_decision(seq) == "plc"
            and decision(seq, always[seq], arrivals, anchor, first_seq) != "plc"
            for seq in range(event - 25, event + 27)
        )
    oracle_events = {e for e in events if rescues_by_event[e]}

    def previous_miss(event: int, cycles_ago: int) -> bool:
        prior = event - cycles_ago * PERIOD_FRAMES
        # All these receiver-relative deadlines have passed before event-25.
        if prior - 25 < first_seq or prior + 27 >= event - 25:
            return False
        return any(fixed_decision(seq) == "plc"
                   for seq in range(prior - 25, prior + 27))

    enabled = {
        "always": set(events),
        "previous1": {e for e in events if previous_miss(e, 1)},
        "previous2": {e for e in events
                      if previous_miss(e, 1) or previous_miss(e, 2)},
        "oracle_NONCAUSAL": oracle_events,
    }
    fixed = evaluate(n, first_seq, arrivals, [BASE_MS] * n)
    original = evaluate(n, first_seq, arrivals, always)
    result = {
        "block": block, "phase_training_only": phase,
        "supporting_training_cycles": support,
        "complete_cycles": len(events),
        "cycles_with_rescue": len(oracle_events),
        "rescue_frames_in_complete_cycles": sum(rescues_by_event.values()),
        "policies": {"fixed60": {
            "plc": fixed["plc"], "active_cycles": 0,
            "rescues_retained": 0, "edited_frames": 0,
            "mean_extra_budget_ms": 0.0,
        }},
    }
    for name, selected in enabled.items():
        schedule = gated_schedule(always, events, selected)
        stats = evaluate(n, first_seq, arrivals, schedule)
        result["policies"][name] = {
            "plc": stats["plc"],
            "active_cycles": len(selected),
            "rescues_retained": sum(rescues_by_event[e] for e in selected),
            "edited_frames": stats["frames_requiring_timescale_adjustment"],
            "mean_extra_budget_ms": stats["average_extra_budget_ms"],
        }
    if result["policies"]["always"]["plc"] != original["plc"]:
        raise AssertionError("baseline changed")
    if result["policies"]["oracle_NONCAUSAL"]["plc"] != original["plc"]:
        raise AssertionError("partial boundary event affected the score")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    manifest = json.loads((args.campaign / "campaign-manifest.json").read_text())
    if manifest.get("status") != "complete" or len(manifest.get("blocks", [])) != 9:
        parser.error("requires original complete nine-block campaign")
    blocks = [audit_block(args.campaign, r["block"]) for r in manifest["blocks"]]
    total_cycles = sum(b["complete_cycles"] for b in blocks)
    total_rescues = sum(b["rescue_frames_in_complete_cycles"] for b in blocks)
    policies = {}
    for name in blocks[0]["policies"]:
        rows = [b["policies"][name] for b in blocks]
        recovered = sum(x["rescues_retained"] for x in rows)
        active = sum(x["active_cycles"] for x in rows)
        policies[name] = {
            "plc": sum(x["plc"] for x in rows),
            "rescues_retained": recovered,
            "rescue_recall": round(recovered / total_rescues, 4) if total_rescues else None,
            "active_cycles": active,
            "active_fraction": round(active / total_cycles, 4),
            "edited_frames": sum(x["edited_frames"] for x in rows),
            "mean_extra_budget_ms": round(
                sum(x["mean_extra_budget_ms"] for x in rows) / len(rows), 5),
        }
    if policies["fixed60"]["plc"] != 82 or policies["always"]["plc"] != 15:
        raise AssertionError("replay fails frozen 82->15 accounting; no report written")
    gate = any(
        policies[name]["rescue_recall"] is not None
        and policies[name]["rescue_recall"] >= .9
        and policies[name]["active_fraction"] <= .5
        for name in ("previous1", "previous2")
    )
    report = {
        "schema": 1, "campaign": str(args.campaign),
        "blocks": blocks, "total_complete_cycles": total_cycles,
        "total_rescues_in_complete_cycles": total_rescues,
        "policies": policies, "simple_causal_gate_pass": gate,
        "gate_rule": ">=90% of phase-only rescues with <=50% of ramped cycles",
        "limitations": [
            "one site, one day; cycles within and between blocks correlated",
            "oracle_NONCAUSAL uses future information, not implementable",
            "past-cycle gates were fixed before this audit; no tuning on suffix",
            "packet/latency accounting only; NO speech-quality claim",
            "failure falsifies these TWO simple gates, not all possible predictors",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("POLICY             PLC  RESCUES  RECALL  ACTIVE/CYCLES  EDITS  EXTRA_MS")
    for name, row in policies.items():
        print(f"{name:18s} {row['plc']:3d}  {row['rescues_retained']:7d}  "
              f"{str(row['rescue_recall']):6s}  "
              f"{row['active_cycles']:4d}/{total_cycles:3d}       "
              f"{row['edited_frames']:5d}  {row['mean_extra_budget_ms']:.4f}")
    print("SIMPLE_CAUSAL_GATE_PASS:", gate)
    print("ORACLE NONCAUSAL; no independent-day or QoE claim.")
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
