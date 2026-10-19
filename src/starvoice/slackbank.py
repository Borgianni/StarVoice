"""Exploratory prospective playout-slack feasibility on SV02 natural arrival logs.

Models idealized 5% per-frame time-scale change; DOES NOT perform waveform
time stretching, use the real NetEQ implementation, or measure actual QoE.
Never treat a retrospectively chosen playout schedule as causal evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

FRAME_MS = 20
PERIOD_FRAMES = 750
TRAIN_FRAMES = 6000  # 120 seconds; no frame in this prefix contributes to scores
BASE_MS = 60
PEAK_MS = 80
SLEW_MS_PER_FRAME = 1


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def arrival_data(session_dir: Path) -> tuple[int, int, dict[int, dict]]:
    receiver = session_dir / "receiver"
    info = json.loads((receiver / "summary.json").read_text())
    if info["endpoint"] != "receiver_relative_playout" or info["buffer_ms"] != 60:
        raise ValueError("requires a native SV02 60ms receiver-relative session")
    n = int(info["frames_expected"])
    arrivals = {}
    for item in rows(receiver / "receiver.jsonl"):
        seq = int(item["sequence"])
        if seq not in arrivals or int(item["receive_mono_ns"]) < int(arrivals[seq]["receive_mono_ns"]):
            arrivals[seq] = item
    if not arrivals:
        raise ValueError("empty receiver log")
    first_seq, first = min(
        arrivals.items(), key=lambda x: int(x[1]["receive_mono_ns"])
    )
    if n <= TRAIN_FRAMES or first_seq >= n:
        raise ValueError("session too short for 120-second causal calibration")
    return n, first_seq, arrivals


def calibrate_phase(arrivals: dict[int, dict], training_frames: int = TRAIN_FRAMES) -> tuple[int | None, int]:
    """Infer 15s disturbance phase from past-only receiver gap excesses.

    Compare *within-host* time increments only. No subtraction of absolute
    sender and receiver monotonic clocks.
    """
    gap_phases = []
    seen = set()
    for seq in range(1, training_frames):
        last = arrivals.get(seq - 1)
        current = arrivals.get(seq)
        if last is None or current is None:
            continue
        tx_gap_ms = (
            int(current["sender_mono_ns_uncomparable"])
            - int(last["sender_mono_ns_uncomparable"])
        ) / 1e6
        rx_gap_ms = (
            int(current["receive_mono_ns"])
            - int(last["receive_mono_ns"])
        ) / 1e6
        if not (15 <= tx_gap_ms <= 25 and rx_gap_ms - tx_gap_ms >= 35):
            continue
        epoch = seq // PERIOD_FRAMES
        phase = seq % PERIOD_FRAMES
        # One candidate per epoch: largest excess receiver gap.
        previous = seen
        gap_phases.append((epoch, phase, rx_gap_ms - tx_gap_ms))

    by_epoch = {}
    for epoch, phase, magnitude in gap_phases:
        if epoch not in by_epoch or magnitude > by_epoch[epoch][1]:
            by_epoch[epoch] = (phase, magnitude)

    phases = [item[0] for item in by_epoch.values()]
    if len(phases) < 4:
        return None, len(phases)

    def close(a: int, b: int) -> bool:
        delta = (a - b) % PERIOD_FRAMES
        return min(delta, PERIOD_FRAMES - delta) <= 2

    best_phase = max(range(PERIOD_FRAMES), key=lambda i: sum(close(x, i) for x in phases))
    matches = sum(close(x, best_phase) for x in phases)
    if matches < 4 or matches / len(phases) < 0.5:
        return None, matches
    return best_phase, matches


def ramp_targets(n: int, phase: int | None, start: int = TRAIN_FRAMES) -> list[int]:
    """Smooth, bounded phase-triggered buffer targets; causal once phase fixed.

    A 20ms extra playout budget ramps in over 20 source frames, holds for
    five frames around the expected disruption, then ramps out over 20 frames.
    Each +/-1ms step *requires* ~5% audio time scaling in a real player.
    """
    current = BASE_MS
    schedule = [BASE_MS] * n
    if phase is None:
        return schedule
    for seq in range(start, n):
        until = (phase - seq) % PERIOD_FRAMES
        after = (seq - phase) % PERIOD_FRAMES
        if until <= 20:
            target = PEAK_MS - until
        elif after <= 5:
            target = PEAK_MS
        elif 5 < after <= 25:
            target = PEAK_MS - (after - 5)
        else:
            target = BASE_MS
        if target > current:
            current += min(SLEW_MS_PER_FRAME, target - current)
        elif target < current:
            current -= min(SLEW_MS_PER_FRAME, current - target)
        if not BASE_MS <= current <= PEAK_MS:
            raise AssertionError("buffer bound violated")
        schedule[seq] = current
    return schedule


def evaluate(n: int, first_seq: int, arrivals: dict[int, dict], schedule: list[int]) -> dict:
    first_ns = int(arrivals[first_seq]["receive_mono_ns"])
    total = Counter()
    extra = 0
    changed = 0
    max_delay = BASE_MS
    previous = schedule[TRAIN_FRAMES - 1]
    for seq in range(max(first_seq, TRAIN_FRAMES), n):
        budget = schedule[seq]
        if not BASE_MS <= budget <= 100:
            raise AssertionError("budget out of bounds")
        deadline = first_ns + (seq - first_seq) * FRAME_MS * 1_000_000 + budget * 1_000_000
        primary = arrivals.get(seq)
        if primary is not None and int(primary["receive_mono_ns"]) <= deadline:
            total["primary"] += 1
        else:
            total["primary_failures"] += 1
            carrier = arrivals.get(seq + 1)
            if (
                carrier is not None and bool(carrier.get("has_lbrr"))
                and int(carrier["receive_mono_ns"]) <= deadline
            ):
                total["fec"] += 1
            else:
                total["plc"] += 1
        total["frames"] += 1
        extra += budget - BASE_MS
        max_delay = max(max_delay, budget)
        if budget != previous:
            changed += 1
        if abs(budget - previous) > SLEW_MS_PER_FRAME and max_delay <= PEAK_MS:
            raise AssertionError("slew bound violated")
        previous = budget
    total["max_budget_ms"] = max_delay
    total["average_extra_budget_ms"] = round(extra / total["frames"], 5)
    total["frames_requiring_timescale_adjustment"] = changed
    return dict(total)


def analyze(session_dir: Path) -> dict:
    n, first_seq, arrivals = arrival_data(session_dir)
    phase, supporting_cycles = calibrate_phase(arrivals)
    schedules = {
        "fixed60": [60] * n,
        "fixed80": [80] * n,
        "fixed100": [100] * n,
        "causal_phase_ramp_60_80": ramp_targets(n, phase),
        "wrong_phase_ramp_60_80": ramp_targets(
            n, (phase + PERIOD_FRAMES // 2) % PERIOD_FRAMES if phase is not None else None
        ),
    }
    measures = {
        name: evaluate(n, first_seq, arrivals, schedule)
        for name, schedule in schedules.items()
    }
    return {
        "session_id": json.loads(
            (session_dir / "receiver" / "summary.json").read_text()
        )["session_id"],
        "phase_from_first_120s_only": phase,
        "calibration_cycles_supporting_phase": supporting_cycles,
        "excluded_calibration_frames": TRAIN_FRAMES,
        "proposed_physical_slew": "1ms playout change per 20ms frame; waveform stretch NOT implemented",
        "outcomes": measures,
        "caveats": [
            "Exploratory, one day and site, never independent days.",
            "Scores apply to fixed received packets and actual Always-FEC payloads.",
            "Phase estimated only from preceding 120 seconds of the same block.",
            "Not a real audio playout implementation, NetEQ comparison, QoE measurement or OWD.",
            "The 5% time-scaling waveform cost is not yet measured.",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("output exists: will not overwrite")
    manifest = json.loads((a.campaign_dir / "campaign-manifest.json").read_text())
    if manifest["status"] != "complete" or len(manifest["blocks"]) != 9:
        p.error("nine audited blocks required")
    reports = [
        analyze(a.campaign_dir / entry["block"])
        for entry in manifest["blocks"]
    ]
    a.output.mkdir(parents=True)
    (a.output / "results.json").write_text(json.dumps({
        "schema_version": 1,
        "campaign": str(a.campaign_dir),
        "reports": reports,
    }, indent=2, sort_keys=True) + "\n")
    table = []
    for idx, report in enumerate(reports, 1):
        for key, metrics in report["outcomes"].items():
            table.append({
                "block": idx,
                "phase": report["phase_from_first_120s_only"],
                "method": key,
                **metrics,
            })
    cols = list(dict.fromkeys(k for row in table for k in row))
    with (a.output / "by-block.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, cols)
        writer.writeheader()
        writer.writerows(table)
    for name in reports[0]["outcomes"]:
        plc = sum(r["outcomes"][name]["plc"] for r in reports)
        extra = sum(
            r["outcomes"][name]["average_extra_budget_ms"] * r["outcomes"][name]["frames"]
            for r in reports
        ) / sum(r["outcomes"][name]["frames"] for r in reports)
        print(f"{name}: plc={plc}, mean_extra_buffer_ms={extra:.3f}")
    print(f"Saved {a.output}; idealized receiver-relative feasibility ONLY")


if __name__ == "__main__":
    main()
