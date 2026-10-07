from starvoice.deadline import timely_fec_opportunity_slots


def _trace(send_ms, rtt_ms=None, lost=False):
    return {
        "send_ns": int(send_ms * 1_000_000),
        "rtt_ms": rtt_ms,
        "lost": lost,
    }


def _packet(seq, send_ms, trace_idx, lost=False, next_lbrr=False):
    return {
        "sequence": seq,
        "send_ns": int(send_ms * 1_000_000),
        "source_trace_index": trace_idx,
        "lost": lost,
        "next_packet_has_lbrr": next_lbrr,
    }


def test_timely_opportunity_depends_on_playout_budget():
    trace = [
        _trace(0, lost=True),
        _trace(20, rtt_ms=20.0),
    ]
    packets = [
        _packet(0, 0, 0, lost=True, next_lbrr=True),
        _packet(1, 20, 1),
    ]

    assert timely_fec_opportunity_slots(
        packets, trace, playout_ms=20
    ) == set()
    assert timely_fec_opportunity_slots(
        packets, trace, playout_ms=40
    ) == {0}


def test_no_opportunity_without_realized_lbrr():
    trace = [
        _trace(0, lost=True),
        _trace(20, rtt_ms=20.0),
    ]
    packets = [
        _packet(0, 0, 0, lost=True, next_lbrr=False),
        _packet(1, 20, 1),
    ]

    assert timely_fec_opportunity_slots(
        packets, trace, playout_ms=40
    ) == set()
