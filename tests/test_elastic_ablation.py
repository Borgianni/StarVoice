"""Offline factorial ablation: physical edit vs packet-decision rescue."""
import wave

import pytest

from starvoice import elastic_playout as playout
from starvoice.slackbank import TRAIN_FRAMES, ramp_targets


class DummyOpusDecoder:
    def __init__(self, cfg):
        pass

    def decode(self, packet, fec=False):
        return (b"\x10\x00" if packet else b"\x00\x00") * 960

    def close(self):
        pass


def test_factorial_ablation_isolates_rescue_and_stretch(tmp_path, monkeypatch):
    monkeypatch.setattr(playout, "OpusDecoder", DummyOpusDecoder)
    n = TRAIN_FRAMES + 340
    phase = 250
    missed = TRAIN_FRAMES + phase
    anchor = 1_000_000_000
    arrivals = {
        seq: {
            "receive_mono_ns": anchor + seq * 20_000_000
            + (70_000_000 if seq == missed else 0),
            "has_lbrr": False,
        }
        for seq in range(n)
    }
    packets = [b"dummy"] * n
    fixed = [60] * n
    ramp = ramp_targets(n, phase)
    assert ramp[missed] == 80 and ramp[-1] == 60

    args = (packets, arrivals, 0)
    fixed_report = playout.render_policy(tmp_path / "fixed.wav", *args, fixed, 24000)
    normal = playout.render_policy(tmp_path / "slackbank.wav", *args, ramp, 24000)
    stretched = playout.render_policy(
        tmp_path / "stretch-only.wav", *args, ramp, 24000,
        decision_budgets=fixed,
    )
    nonphysical = playout.render_policy(
        tmp_path / "rescue-only.wav", *args, fixed, 24000,
        decision_budgets=ramp,
    )
    assert fixed_report["scored_suffix"]["plc"] == 1
    assert normal["scored_suffix"]["plc"] == 0
    assert stretched["scored_suffix"]["plc"] == 1
    assert nonphysical["scored_suffix"]["plc"] == 0
    assert stretched["edited_frames"] == normal["edited_frames"] > 0
    assert nonphysical["edited_frames"] == fixed_report["edited_frames"] == 0
    assert stretched["pre_stretch_pcm_sha256"] == fixed_report["pre_stretch_pcm_sha256"]
    assert nonphysical["pre_stretch_pcm_sha256"] == normal["pre_stretch_pcm_sha256"]
    for name in ("fixed", "slackbank", "stretch-only", "rescue-only"):
        with wave.open(str(tmp_path / (name + ".wav")), "rb") as w:
            assert w.getnframes() == n * 960


def test_mismatched_diagnostic_budget_is_explicitly_validated(tmp_path, monkeypatch):
    monkeypatch.setattr(playout, "OpusDecoder", DummyOpusDecoder)
    packets = [b"packet"] * 3
    arrivals = {
        seq: {"receive_mono_ns": 1_000_000_000 + seq * 20_000_000}
        for seq in range(3)
    }
    with pytest.raises(ValueError, match="packet-decision budget"):
        playout.render_policy(
            tmp_path / "bad.wav", packets, arrivals, 0, [60] * 3, 24000,
            decision_budgets=[60, 80],
        )
    assert not (tmp_path / "bad.wav").exists()
    with pytest.raises(ValueError, match="physical buffer change"):
        playout.render_policy(
            tmp_path / "jump.wav", packets, arrivals, 0, [60, 80, 80], 24000,
        )
    assert not (tmp_path / "jump.wav").exists()
