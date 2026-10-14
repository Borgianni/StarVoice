import json
from pathlib import Path

from starvoice.session_analysis import _timing_report, analyze_session


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


def test_timing_report_distinguishes_scheduler_from_disk_and_decode():
    rows = [
        {
            "sequence": 0, "scheduled_playout_mono_ns": 0,
            "decision_start_mono_ns": 1_000_000,
            "decode_end_mono_ns": 1_300_000,
            "wav_write_end_mono_ns": 30_000_000,
            "actual_playout_mono_ns": 30_000_000,
        },
        {
            "sequence": 1, "scheduled_playout_mono_ns": 20_000_000,
            "decision_start_mono_ns": 22_000_000,
            "decode_end_mono_ns": 22_200_000,
            "wav_write_end_mono_ns": 22_400_000,
            "actual_playout_mono_ns": 22_400_000,
        },
    ]
    result = _timing_report(rows)
    assert result["instrumented"]
    assert result["dispatch_max_ms"] == 2.0
    assert result["wav_write_max_ms"] == 28.7
    assert result["dispatch_over_20ms"] == 0
    assert result["worst_dispatch_frames"][0]["sequence"] == 1

    legacy = [{k: row[k] for k in (
        "sequence", "scheduled_playout_mono_ns", "actual_playout_mono_ns"
    )} for row in rows]
    result = _timing_report(legacy)
    assert not result["instrumented"]
    assert result["dispatch_max_ms"] == 30.0
    assert result["decode_max_ms"] is None
