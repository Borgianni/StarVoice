from starvoice.fec_cost import _window_starts


def test_window_starts_are_deterministic_and_span_utterance():
    starts = _window_starts(total_frames=100, window_frames=10, positions=6)
    assert starts == [0, 18, 36, 54, 72, 90]


def test_window_starts_short_utterance():
    assert _window_starts(total_frames=5, window_frames=10, positions=6) == [0]
