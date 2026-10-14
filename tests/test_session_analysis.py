import json
from pathlib import Path

from starvoice.session_analysis import analyze_session


def test_analyze_session_causal_and_detects_violation(tmp_path: Path):
    tx = tmp_path / "sender"
    rx = tmp_path / "receiver"
    tx.mkdir()
    rx.mkdir()
    sid = "test-session"
    (tx / "summary.json").write_text(json.dumps({"session_id": sid, "frames": 2}))
    (rx / "summary.json").write_text(json.dumps({
        "session_id": sid, "frames_expected": 2, "frames_decoded": 2,
        "primary_frames": 1, "fec_frames": 1, "plc_frames": 0,
    }))

    def write(folder, name, rows):
        (folder / name).write_text("".join(json.dumps(row) + "\n" for row in rows))

    write(tx, "sender.jsonl", [{"sequence": 0}, {"sequence": 1}])
    write(rx, "receiver.jsonl", [
        {"sequence": 0, "receive_mono_ns": 10, "has_lbrr": False},
        {"sequence": 2, "receive_mono_ns": 20, "has_lbrr": True},
    ])
    write(rx, "frame-decisions.jsonl", [
        {"sequence": 0, "decision": "primary", "source_packet_sequence": 0,
         "scheduled_playout_mono_ns": 100, "actual_playout_mono_ns": 101},
        {"sequence": 1, "decision": "fec", "source_packet_sequence": 2,
         "scheduled_playout_mono_ns": 110, "actual_playout_mono_ns": 111},
    ])
    write(tx, "clock.jsonl", [{"utc": "test"}])
    write(rx, "clock.jsonl", [{"utc": "test"}])
    result = analyze_session(tmp_path)
    assert result["valid"], result["errors"]
    assert result["fec"] == 1

    write(rx, "receiver.jsonl", [
        {"sequence": 0, "receive_mono_ns": 10, "has_lbrr": False},
        {"sequence": 2, "receive_mono_ns": 200, "has_lbrr": True},
    ])
    result = analyze_session(tmp_path)
    assert not result["valid"]
    assert "causality_1" in result["errors"]
