import json
import time

from starvoice.cli import _with_clock_logging


def test_session_clock_log_is_automatically_created(tmp_path, monkeypatch):
    def fake_clock_snapshot(include_dish=True):
        return {"fake": True, "dish_requested": include_dish}

    monkeypatch.setattr("starvoice.clocklog.clock_snapshot", fake_clock_snapshot)

    output = tmp_path / "session"
    result = _with_clock_logging(
        output, 0.01, False,
        lambda capture_dir: (time.sleep(0.04), "completed")[1],
    )

    assert result == "completed"
    path = output / "clock.jsonl"
    assert path.is_file()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) >= 1
    assert all(row["dish_requested"] is False for row in rows)

    archive = json.loads((output / "archive-integrity.json").read_text())
    assert archive["schema_version"] == 1
    assert "clock.jsonl" in archive["files"]
