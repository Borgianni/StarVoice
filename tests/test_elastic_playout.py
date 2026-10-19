"""Offline elastic audio: sample clock, causality and guardrails."""
import math
import struct
import wave

import pytest

from starvoice import elastic_playout as elastic
from starvoice.slackbank import TRAIN_FRAMES

SINE_PCM = struct.pack(
    "<960h",
    *(round(12000 * math.sin(j * math.tau * 200 / 48000)) for j in range(960)),
)


class FakeDecoder:
    def __init__(self, config):
        self.closed = False

    def decode(self, payload, fec=False):
        return SINE_PCM if payload is not None else b"\0" * 1920

    def close(self):
        self.closed = True


def test_frame_edits_change_actual_sample_count_not_sampling_rate():
    assert elastic.stretch_frame(SINE_PCM, 0) == SINE_PCM
    assert len(elastic.stretch_frame(SINE_PCM, 48)) == 1008 * 2
    assert len(elastic.stretch_frame(SINE_PCM, -48)) == 912 * 2
    for delta in (-48, 48):
        samples = struct.unpack(f"<{960 + delta}h", elastic.stretch_frame(SINE_PCM, delta))
        assert max(abs(x) for x in samples) <= 32767


def test_bad_frame_and_nonphysical_instantaneous_slack_rejected():
    with pytest.raises(ValueError, match="960"):
        elastic.stretch_frame(b"\0" * 100, 48)
    with pytest.raises(ValueError, match="1ms"):
        elastic.stretch_frame(SINE_PCM, 960)


def test_packet_selection_uses_only_timely_primary_or_following_fec():
    rx = {
        0: {"receive_mono_ns": 1_100_000_000},
        1: {"receive_mono_ns": 1_055_000_000, "has_lbrr": True},
    }
    assert elastic.select_packet(0, 1_060_000_000, rx) == ("fec", 1)
    rx[1]["has_lbrr"] = False
    assert elastic.select_packet(0, 1_060_000_000, rx) == ("plc", None)
    assert elastic.select_packet(0, 1_101_000_000, rx) == ("primary", 0)


def test_audio_length_exactly_realizes_budget_for_next_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(elastic, "OpusDecoder", FakeDecoder)
    n = TRAIN_FRAMES + 120
    arrivals = {
        seq: {"receive_mono_ns": 1_000_000_000 + seq * 20_000_000, "has_lbrr": True}
        for seq in range(n)
    }
    budgets = [60] * n
    for seq in range(6000, 6020):
        budgets[seq] = 60 + (seq - 5999)
    for seq in range(6020, 6041):
        budgets[seq] = 80
    for seq in range(6041, 6061):
        budgets[seq] = 80 - (seq - 6040)
    report = elastic.render_policy(
        tmp_path / "elastic.wav", [b"packet"] * n, arrivals, 0, budgets, 24000
    )
    assert report["plc"] == 0
    assert report["frames"] == n
    assert report["edited_frames"] == 40
    assert report["expanded_frames"] == 20
    assert report["contracted_frames"] == 20
    assert report["samples_written"] == n * 960
    assert report["scored_suffix"]["frames"] == 120
    with wave.open(str(tmp_path / "elastic.wav"), "rb") as audio:
        assert audio.getframerate() == 48000
        assert audio.getnframes() == n * 960
        assert audio.getsampwidth() == 2


def test_constant_budget_keeps_original_pcm_bit_exact(tmp_path, monkeypatch):
    monkeypatch.setattr(elastic, "OpusDecoder", FakeDecoder)
    n = TRAIN_FRAMES + 1
    arrivals = {
        seq: {"receive_mono_ns": 1_000_000_000 + seq * 20_000_000, "has_lbrr": False}
        for seq in range(n)
    }
    report = elastic.render_policy(
        tmp_path / "fixed.wav", [b"packet"] * n, arrivals, 0, [60] * n, 24000
    )
    assert report["edited_frames"] == 0
    assert report["primary"] == n
    assert report["samples_written"] == 960 * n
    with wave.open(str(tmp_path / "fixed.wav"), "rb") as audio:
        assert audio.readframes(1) == SINE_PCM[:2]
        assert audio.getnframes() == 960 * n


def test_rejects_buffer_instant_jump_before_writing_pcm(tmp_path, monkeypatch):
    monkeypatch.setattr(elastic, "OpusDecoder", FakeDecoder)
    n = 2
    arrivals = {
        seq: {"receive_mono_ns": 1_000_000_000 + seq * 20_000_000}
        for seq in range(n)
    }
    with pytest.raises(ValueError, match="1ms"):
        elastic.render_policy(
            tmp_path / "invalid.wav", [b"a", b"b"], arrivals, 0, [60, 80], 24000
        )
