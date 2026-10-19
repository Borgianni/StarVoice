import json
from pathlib import Path

from starvoice.recoverability import analyze


def _jsonl(path, data):
    path.write_text("".join(json.dumps(x) + "\n" for x in data))


def test_recoverability_deadlines_and_unchanged_carrier(tmp_path: Path):
    tx, rx = tmp_path / "sender", tmp_path / "receiver"
    tx.mkdir()
    rx.mkdir()
    (tx / "summary.json").write_text(json.dumps({
        "session_id": "s", "frames": 6, "injected_drops": 0, "policy": "always-fec"
    }))
    (rx / "summary.json").write_text(json.dumps({
        "session_id": "s", "frames_expected": 6, "frames_decoded": 6,
        "buffer_ms": 60, "endpoint": "receiver_relative_playout"
    }))
    stamp = {
        0: 1_000_000_000, 1: 1_020_000_000,
        2: 1_105_000_000, 3: 1_114_000_000, 5: 1_132_000_000
    }
    _jsonl(rx / "receiver.jsonl", [
        {"sequence": seq, "receive_mono_ns": time, "has_lbrr": seq in (3, 5)}
        for seq, time in stamp.items()
    ])
    _jsonl(rx / "frame-decisions.jsonl", [
        {"sequence": seq, "decision": decision}
        for seq, decision in enumerate(
            ("primary", "primary", "plc", "primary", "fec", "primary")
        )
    ])
    report, events = analyze(tmp_path, (60, 80))
    sixty, eighty = report["budgets"]
    assert sixty["counts"]["primary_failures"] == 2
    assert sixty["counts"]["eligible_lbrr_upper_bound"] == 1
    assert sixty["counts"]["modeled_plc"] == 1
    assert sixty["observed_decision_disagreements_at_recorded_buffer"] == []
    assert eighty["counts"]["primary_failures"] == 1
    assert eighty["counts"]["eligible_lbrr_upper_bound"] == 1
    assert len(events) == 3
    assert any(row["sequence"] == 2 and
               row["carrier_kind"] == "carrier_receiver_observed_late"
               for row in events)
