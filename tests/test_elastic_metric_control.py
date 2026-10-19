"""Temporal-reference control: sound source edit clock and strict input guards."""
import math
import struct
import wave

import pytest

from starvoice.elastic_metric_control import clean_clock_warp
from starvoice.slackbank import BASE_MS


def build_wav(path, frames=120, source_remainder=0):
    count = (frames - 1) * 960 + 960 - source_remainder
    with wave.open(str(path), "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(48000)
        dst.writeframes(struct.pack(f"<{count}h", *(
            int(8000 * math.sin(i * math.tau * 200 / 48000)) for i in range(count)
        )))
    return count


def make_ramp(n=120):
    schedule = [BASE_MS] * n
    for seq in range(10, 30):
        schedule[seq] = 61 + seq - 10
    for seq in range(30, 50):
        schedule[seq] = 80
    for seq in range(50, 70):
        schedule[seq] = 79 - (seq - 50)
    return schedule


def test_source_warp_realizes_physical_clock_without_drift(tmp_path):
    source = tmp_path / "original.wav"
    destination = tmp_path / "source_warp.wav"
    build_wav(source)
    report = clean_clock_warp(source, destination, make_ramp())
    assert report["edited_frames"] == 40
    assert report["samples_written"] == report["expected_samples"] == 120 * 960
    with wave.open(str(destination), "rb") as wav:
        assert wav.getnframes() == 120 * 960
        assert wav.getframerate() == 48000
    assert destination.read_bytes() != source.read_bytes()


def test_unmodified_reference_is_pcm_bit_exact(tmp_path):
    source = tmp_path / "original.wav"
    destination = tmp_path / "copy.wav"
    build_wav(source)
    report = clean_clock_warp(source, destination, [BASE_MS] * 120)
    assert report["edited_frames"] == 0
    assert destination.read_bytes() == source.read_bytes()


def test_last_partial_source_frame_can_be_padded(tmp_path):
    source = tmp_path / "partial.wav"
    destination = tmp_path / "partial_warp.wav"
    build_wav(source, frames=120, source_remainder=37)
    report = clean_clock_warp(source, destination, [BASE_MS] * 120)
    assert report["unpadded_source_samples"] == 120 * 960 - 37
    assert report["samples_written"] == 120 * 960


def test_nonphysical_jump_rejected_without_creating_output(tmp_path):
    source = tmp_path / "source.wav"
    dest = tmp_path / "bad.wav"
    build_wav(source)
    bad = [BASE_MS] * 120
    bad[10] = 80
    with pytest.raises(ValueError, match="nonphysical"):
        clean_clock_warp(source, dest, bad)
    assert not dest.exists()


def test_output_refuses_overwrite(tmp_path):
    source = tmp_path / "source.wav"
    destination = tmp_path / "already.wav"
    build_wav(source)
    destination.write_bytes(b"not-a-wave")
    with pytest.raises(FileExistsError):
        clean_clock_warp(source, destination, [BASE_MS] * 120)
    assert destination.read_bytes() == b"not-a-wave"
