import json
from pathlib import Path

from starvoice.slackbank import (
    BASE_MS,
    TRAIN_FRAMES,
    analyze,
    calibrate_phase,
    evaluate,
    ramp_targets,
)


def synthetic_arrivals(n=9000, phase=424):
    rows = {}
    for seq in range(n):
        tx = seq * 20_000_000
        mod = (seq - phase) % 750
        extra_ms = 70 if mod == 0 else (55 if mod == 1 else 0)
        rows[seq] = {
            "sequence": seq,
            "sender_mono_ns_uncomparable": tx,
            "receive_mono_ns": tx + extra_ms * 1_000_000,
            "has_lbrr": True,
        }
    return rows


def test_causal_training_and_slew_limited_phase_schedule():
    arrivals = synthetic_arrivals()
    phase, support = calibrate_phase(arrivals)
    assert phase == 424
    assert support >= 4

    n = len(arrivals)
    schedule = ramp_targets(n, phase)
    assert schedule[:TRAIN_FRAMES] == [BASE_MS] * TRAIN_FRAMES
    assert min(schedule) == 60
    assert max(schedule) == 80
    assert max(abs(a - b) for a, b in zip(schedule, schedule[1:])) <= 1

    fixed60 = evaluate(n, 0, arrivals, [60] * n)
    causal = evaluate(n, 0, arrivals, schedule)
    fixed80 = evaluate(n, 0, arrivals, [80] * n)
    control = evaluate(n, 0, arrivals, ramp_targets(n, (phase + 375) % 750))

    assert fixed60["plc"] > 0
    assert causal["plc"] < fixed60["plc"]
    assert control["plc"] >= causal["plc"]
    assert causal["plc"] == fixed80["plc"]
    assert causal["average_extra_budget_ms"] < 2
    assert causal["max_budget_ms"] == 80


def test_post_training_arrivals_do_not_change_phase_or_schedule():
    arrivals = synthetic_arrivals()
    phase, _ = calibrate_phase(arrivals)
    changed = {
        seq: dict(row) for seq, row in arrivals.items()
    }
    for seq in range(TRAIN_FRAMES, len(changed)):
        changed[seq]["receive_mono_ns"] += ((seq * 137) % 211) * 1_000_000
    phase2, _ = calibrate_phase(changed)
    assert phase2 == phase
    assert ramp_targets(len(arrivals), phase2) == ramp_targets(len(arrivals), phase)


def test_session_reader_and_aggregate(tmp_path: Path):
    session = tmp_path / "block-01"
    rx = session / "receiver"
    rx.mkdir(parents=True)
    arrivals = synthetic_arrivals()
    (rx / "summary.json").write_text(json.dumps({
        "session_id": "test", "endpoint": "receiver_relative_playout",
        "buffer_ms": 60, "frames_expected": len(arrivals)
    }) + "\n")
    (rx / "receiver.jsonl").write_text("".join(
        json.dumps(row) + "\n" for row in arrivals.values()
    ))
    out = analyze(session)
    assert out["phase_from_first_120s_only"] == 424
    assert out["outcomes"]["causal_phase_ramp_60_80"]["plc"] < out["outcomes"]["fixed60"]["plc"]
    assert out["outcomes"]["causal_phase_ramp_60_80"]["average_extra_budget_ms"] < 2
