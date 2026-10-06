from starvoice.foresight import _top_risk_slots


def test_top_risk_slots_uses_risk_then_sequence():
    rows = [
        {"sequence": 0, "risk": 0.1},
        {"sequence": 1, "risk": 0.9},
        {"sequence": 2, "risk": 0.9},
        {"sequence": 3, "risk": 0.2},
    ]
    assert _top_risk_slots(rows, 2) == {1, 2}
    assert _top_risk_slots(rows, 3) == {1, 2, 3}
