from starvoice.actuation import _schedule_for_targets


def test_schedule_for_targets_adds_only_preroll():
    assert _schedule_for_targets({5}, frames=10, preroll_frames=1) == {5}
    assert _schedule_for_targets({5}, frames=10, preroll_frames=3) == {3, 4, 5}


def test_schedule_for_targets_merges_overlaps():
    assert _schedule_for_targets({3, 4}, frames=10, preroll_frames=3) == {1, 2, 3, 4}
