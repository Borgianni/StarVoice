from starvoice.causal_fusion import _network_timely_labels


def test_network_timely_label_requires_loss_and_timely_next_packet():
    rows = [
        {"send_ns": 0, "recv_ns": None, "rtt_ms": None, "lost": True},
        {"send_ns": 20_000_000, "recv_ns": 40_000_000, "rtt_ms": 20.0, "lost": False},
        {"send_ns": 40_000_000, "recv_ns": None, "rtt_ms": None, "lost": True},
        {"send_ns": 60_000_000, "recv_ns": 160_000_000, "rtt_ms": 100.0, "lost": False},
    ]

    labels = _network_timely_labels(rows, playout_ms=40.0, rtt_fraction=0.5)
    assert labels[0] == 1
    assert labels[2] == 0
