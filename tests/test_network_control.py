import json
from pathlib import Path

import pytest

from starvoice.network_control import analyze_network_control_trace


def _write_trace(path: Path, interval_ms: float) -> None:
    rows = []
    interval_ns = int(interval_ms * 1_000_000)
    total = int(75_000 / interval_ms)
    for i in range(total):
        send_ns = i * interval_ns
        phase = (send_ns / 1e9) % 15.0
        rows.append(
            {
                "sequence": i,
                "send_ns": send_ns,
                "recv_ns": send_ns + 20_000_000,
                "rtt_ms": 80.0 if phase < interval_ms / 1000.0 else 20.0,
                "lost": False,
            }
        )
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


@pytest.mark.parametrize(
    ("interval_ms", "expected_packets"),
    [(10.0, 1500.0), (25.0, 600.0), (40.0, 375.0)],
)
def test_wall_clock_period_stays_fixed_while_packet_count_scales(
    tmp_path: Path,
    interval_ms: float,
    expected_packets: float,
) -> None:
    trace = tmp_path / f"trace-{interval_ms}.jsonl"
    _write_trace(trace, interval_ms)
    result = analyze_network_control_trace(
        trace,
        threshold_ms=50.0,
        min_period_s=14.9,
        max_period_s=15.1,
        step_s=0.001,
    )
    view = result["periodicity"]["rtt_only"]
    assert view["best_period_s"] == pytest.approx(15.0, abs=0.002)
    assert view["packets_per_period_effective"] == pytest.approx(
        expected_packets,
        rel=0.01,
    )
    assert view["period_coherence"] > 0.99
