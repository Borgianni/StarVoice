from starvoice.deadline import evaluate_deadlines


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


def test_lost_frame_recovered_only_if_following_lbrr_arrives_before_deadline():
    traces = [
        _trace(0, lost=True),
        _trace(20, rtt_ms=20.0),
    ]
    packets = [
        _packet(0, 0, 0, lost=True, next_lbrr=True),
        _packet(1, 20, 1),
    ]

    tight = evaluate_deadlines(packets, traces, playout_ms=20)
    assert tight["fec_on_time_recovered_frames"] == 0
    assert tight["final_deadline_misses"] == 1

    roomy = evaluate_deadlines(packets, traces, playout_ms=40)
    assert roomy["fec_on_time_recovered_frames"] == 1
    assert roomy["fec_on_time_recovered_from_loss"] == 1
    assert roomy["final_deadline_misses"] == 0


def test_late_primary_can_be_rescued_by_earlier_following_lbrr_arrival():
    traces = [
        _trace(0, rtt_ms=100.0),
        _trace(20, rtt_ms=20.0),
    ]
    packets = [
        _packet(0, 0, 0, next_lbrr=True),
        _packet(1, 20, 1),
    ]

    result = evaluate_deadlines(packets, traces, playout_ms=40)
    assert result["primary_late_frames"] == 1
    assert result["primary_deadline_failures"] == 1
    assert result["fec_on_time_recovered_from_late"] == 1
    assert result["final_deadline_misses"] == 0
