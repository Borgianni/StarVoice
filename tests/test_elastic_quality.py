"""Matched source-time windows avoid post-hoc alignment or favorable cropping."""
import math
import struct
import wave

import pytest

from starvoice.elastic_quality import (
    HALF_WINDOW_FRAMES,
    eligible_centers,
    pcm_window,
    save_clip,
    stoi_score,
)
from starvoice.slackbank import ramp_targets, TRAIN_FRAMES


def test_rescued_and_negative_control_windows_are_deterministic():
    n, phase = 9600, 424
    budgets = ramp_targets(n, phase)
    rescued = [6424, 6425, 7174]
    windows = eligible_centers(n, phase, budgets, rescued)
    assert ("rescued", 6424) in windows
    assert ("rescued", 7174) in windows
    assert ("rescued", 6425) not in windows
    assert any(kind == "control" for kind, _ in windows)
    for category, center in windows:
        assert center - HALF_WINDOW_FRAMES >= TRAIN_FRAMES
        assert center + HALF_WINDOW_FRAMES < n
        assert budgets[center - HALF_WINDOW_FRAMES] == 60
        assert budgets[center + HALF_WINDOW_FRAMES] == 60
        if category == "control":
            assert not any(abs(center - rescue) <= HALF_WINDOW_FRAMES for rescue in rescued)
    assert windows == eligible_centers(n, phase, budgets, rescued)


def test_no_event_returns_no_matched_clips():
    assert eligible_centers(9000, None, [60] * 9000, []) == []


def test_pcm_window_is_sample_exact_and_rejects_overreach(tmp_path):
    audio = tmp_path / "test.wav"
    pcm = struct.pack("<960h", *range(960))
    save_clip(audio, pcm)
    assert pcm_window(audio, 10, 50) == pcm[20:120]
    with pytest.raises(ValueError, match="beyond"):
        pcm_window(audio, 900, 100)
    with wave.open(str(audio), "rb") as w:
        assert w.getframerate() == 48000
        assert w.getnframes() == 960


def test_stoi_identical_signal_is_near_one():
    # Deterministic voiced-and-harmonic speech-like signal, 6 seconds.
    samples = [
        round(9000 * math.sin(2 * math.pi * 220 * t / 48000)
              + 4000 * math.sin(2 * math.pi * 440 * t / 48000))
        for t in range(48000 * 6)
    ]
    pcm = struct.pack(f"<{len(samples)}h", *samples)
    score = stoi_score(pcm, pcm)
    assert score > 0.99
    with pytest.raises(ValueError, match="identical lengths"):
        stoi_score(pcm, pcm[:-2])
