from starvoice.foresight import _frontier_point


def test_global_frontier_uses_highest_risk_slots():
    slots = [
        {"trace_name": "t", "utterance_id": "u", "sequence": 0, "risk": 0.1, "opportunity": False},
        {"trace_name": "t", "utterance_id": "u", "sequence": 1, "risk": 0.9, "opportunity": True},
        {"trace_name": "t", "utterance_id": "u", "sequence": 2, "risk": 0.8, "opportunity": False},
        {"trace_name": "t", "utterance_id": "u", "sequence": 3, "risk": 0.7, "opportunity": True},
    ]
    point = _frontier_point(slots, opportunities=2, protected_frames=2)
    assert point["risk_ranked_hits"] == 1
    assert point["oracle_hits"] == 2
    assert point["random_expected_hits"] == 1.0
