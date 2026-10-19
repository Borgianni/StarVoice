"""Replay-audit guardrails; all tests are offline, without a network."""
import hashlib
import json
import wave
from pathlib import Path

import pytest

from starvoice import opus_replay_audit as audit


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def write_jsonl(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in data))


def sample_wav(path: Path, n=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as dest:
        dest.setframerate(48000)
        dest.setsampwidth(2)
        dest.setnchannels(1)
        dest.writeframes(b"\0" * (n * 960 * 2))


class DummyDecoder:
    def __init__(self, config):
        pass

    def decode(self, packet, fec=False):
        return b"\0" * 1920

    def close(self):
        pass


def synthetic_campaign(tmp_path: Path, monkeypatch, wrong_lengths=False):
    payloads = [b"abc", b"z"]
    calls = []

    def encode(wav_path, bitrate, loss):
        assert bitrate == 24000
        assert loss == 20
        calls.append(wav_path)
        return payloads

    monkeypatch.setattr(audit, "encode_continuous", encode)
    monkeypatch.setattr(audit, "OpusDecoder", DummyDecoder)
    monkeypatch.setattr(audit, "packet_has_fec", lambda packet: False)
    monkeypatch.setattr(audit, "libopus_version", lambda: "test-libopus")
    campaign = tmp_path / "campaign"
    wav = campaign / "workloads" / "segment-01.wav"
    sample_wav(wav)
    content_hash = audit.sha256(wav)
    manifest = {
        "status": "complete", "sender_policy": "always-fec", "injected_drops": 0,
        "codec": {"bitrate": 24000, "expected_loss_percent": 20},
        "segments": [{"index": 0, "path": str(wav), "sha256": content_hash}],
        "blocks": [],
    }
    for i in range(9):
        name = f"block-{i + 1:02d}"
        root = campaign / name
        manifest["blocks"].append({
            "block": name, "workload_segment": 1, "workload_sha256": content_hash,
        })
        write_json(root / "sender" / "summary.json", {
            "frames": 2, "policy": "always-fec", "injected_drops": 0,
            "opus_bytes": 4, "udp_payload_bytes": 4 + 2 * audit.WIRE_OVERHEAD,
        })
        write_jsonl(root / "sender" / "sender.jsonl", [{
            "sequence": seq, "opus_bytes": len(payload) + int(wrong_lengths and seq == 0),
            "udp_payload_bytes": len(payload) + audit.WIRE_OVERHEAD,
            "has_lbrr": False, "fec_enabled": True,
        } for seq, payload in enumerate(payloads)])
        write_json(root / "receiver" / "summary.json", {
            "frames_expected": 2, "primary_frames": 2, "fec_frames": 0, "plc_frames": 0,
        })
        write_jsonl(root / "receiver" / "frame-decisions.jsonl", [
            {"sequence": seq, "decision": "primary", "source_packet_sequence": seq}
            for seq in range(2)
        ])
        sample_wav(root / "receiver" / "received.wav")
    write_json(campaign / "campaign-manifest.json", manifest)
    return campaign, calls


def test_campaign_reports_passes_without_claiming_opus_byte_identity(tmp_path, monkeypatch):
    campaign, calls = synthetic_campaign(tmp_path, monkeypatch)
    result = audit.audit_campaign(campaign)
    assert result["all_replay_checks_passed"]
    assert result["blocks_metadata_consistent"] == 9
    assert result["blocks_pcm_reproduced"] == 9
    assert result["original_packet_byte_identity_proven"] is False
    assert len(calls) == 1  # one encoder reset per distinct continuous source segment


def test_sender_mismatch_is_detected(tmp_path, monkeypatch):
    campaign, _ = synthetic_campaign(tmp_path, monkeypatch, wrong_lengths=True)
    result = audit.audit_campaign(campaign)
    assert result["all_replay_checks_passed"] is False
    assert result["total_sender_metadata_mismatches"] == 9
    assert result["blocks_metadata_consistent"] == 0
    assert result["blocks_pcm_reproduced"] == 9


def test_missing_or_incomplete_campaign_fails_closed(tmp_path, monkeypatch):
    campaign, _ = synthetic_campaign(tmp_path, monkeypatch)
    manifest_path = campaign / "campaign-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "running"
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="complete nine-block campaign"):
        audit.audit_campaign(campaign)


def test_wrong_source_sha_fails_closed(tmp_path, monkeypatch):
    campaign, _ = synthetic_campaign(tmp_path, monkeypatch)
    manifest_path = campaign / "campaign-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["blocks"][0]["workload_sha256"] = hashlib.sha256(b"fake").hexdigest()
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="SHA mismatch"):
        audit.audit_campaign(campaign)
