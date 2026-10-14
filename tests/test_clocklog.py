from starvoice.clocklog import run_clock_logger


def test_clock_logger_collects_without_external_ntp(tmp_path):
    out = tmp_path / "clocks.jsonl"
    count = run_clock_logger(out, duration_s=0.05, interval_s=0.02, include_dish=False)
    assert count >= 1
    import json
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == count
    assert all(row["mono_after_ns"] >= row["mono_before_ns"] for row in rows)
    assert all("timesync" in row and "dish_ntp_query" not in row for row in rows)
