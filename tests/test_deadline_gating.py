from starvoice.deadline_gating import _keep_by_feasibility


def test_feasibility_gate_uses_repair_delay_budget():
    assert _keep_by_feasibility(
        estimate_rtt_ms=40.0,
        playout_ms=50.0,
        frame_ms=20.0,
        rtt_fraction=0.5,
    )
    assert not _keep_by_feasibility(
        estimate_rtt_ms=80.0,
        playout_ms=50.0,
        frame_ms=20.0,
        rtt_fraction=0.5,
    )


def test_feasibility_gate_fails_open_without_rtt():
    assert _keep_by_feasibility(
        estimate_rtt_ms=None,
        playout_ms=40.0,
        frame_ms=20.0,
        rtt_fraction=0.5,
    )
