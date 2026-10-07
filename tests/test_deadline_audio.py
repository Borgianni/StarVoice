from starvoice.deadline_audio import _arrival_ns


def test_deadline_audio_arrival_uses_rtt_fraction():
    packet = {"send_ns": 100_000_000, "source_trace_index": 0}
    trace = [{"lost": False, "rtt_ms": 40.0}]
    assert _arrival_ns(packet, trace, 0.5) == 120_000_000


def test_deadline_audio_lost_packet_has_no_arrival():
    packet = {"send_ns": 100_000_000, "source_trace_index": 0}
    trace = [{"lost": True, "rtt_ms": None}]
    assert _arrival_ns(packet, trace, 0.5) is None
