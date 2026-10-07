from starvoice.observability import _condition_features


def test_condition_features_only_use_replies_arrived_by_send_time():
    trace = [
        {"send_ns": 0, "recv_ns": 30_000_000, "rtt_ms": 30.0, "lost": False},
        {"send_ns": 20_000_000, "recv_ns": 80_000_000, "rtt_ms": 60.0, "lost": False},
        {"send_ns": 40_000_000, "recv_ns": None, "rtt_ms": None, "lost": True},
    ]
    log = [
        {"sequence": 0, "send_ns": 0, "source_trace_index": 0, "risk": 0.1},
        {"sequence": 1, "send_ns": 20_000_000, "source_trace_index": 1, "risk": 0.2},
        {"sequence": 2, "send_ns": 40_000_000, "source_trace_index": 2, "risk": 0.3},
    ]

    feats = _condition_features(log, trace, (100.0,))

    assert feats[0]["last_rtt_ms"] == 0.0
    assert feats[1]["last_rtt_ms"] == 0.0
    assert feats[2]["last_rtt_ms"] == 30.0
    assert feats[2]["unanswered_100ms"] == 1.0
