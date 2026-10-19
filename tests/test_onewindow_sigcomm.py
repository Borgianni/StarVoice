"""Unit checks for the one-window field campaign (no remote access required)."""
import importlib.util
import json
import wave
from collections import Counter
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_onewindow_sigcomm.py"
spec = importlib.util.spec_from_file_location("run_onewindow_sigcomm", SCRIPT)
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def test_nine_block_design_is_counterbalanced():
    assert len(campaign.ORDER) == 9
    assert [Counter(campaign.ORDER[i:i + 3]) for i in (0, 3, 6)] == [
        Counter({0: 1, 1: 1, 2: 1})
    ] * 3
    assert Counter(campaign.ORDER) == Counter({0: 3, 1: 3, 2: 3})


def test_manifest_to_three_exact_pcm_workloads(tmp_path: Path):
    deployment = tmp_path / "deployment"
    (deployment / "datasets" / "starvoice-eval-v1" / "wav").mkdir(parents=True)
    source = deployment / "datasets/starvoice-eval-v1"
    output = tmp_path / "output"
    output.mkdir()
    rows = []
    for i in range(6):
        name = f"utt{i}.wav"
        wav = source / "wav" / name
        with wave.open(str(wav), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(48000)
            f.writeframes((i + 1).to_bytes(2, "little") * (3 * 48000))
        rows.append({
            "utterance_id": f"utt{i}",
            "wav_48k_mono_pcm16": f"datasets/starvoice-eval-v1/wav/{name}",
            "wav_sha256": campaign.sha256(wav),
        })
    (source / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    segments = campaign.prepare_segments(deployment, output, segment_seconds=5)
    assert len(segments) == 3
    assert [len(x["items"]) for x in segments] == [2, 2, 2]
    assert [x["duration_s"] for x in segments] == [6.0, 6.0, 6.0]
    assert all(Path(x["path"]).is_file() for x in segments)
    for s in segments:
        with wave.open(str(s["path"]), "rb") as f:
            assert f.getnframes() == 6 * 48000
            assert (f.getframerate(), f.getnchannels(), f.getsampwidth()) == (
                48000, 1, 2
            )
        assert s["sha256"] == campaign.sha256(Path(s["path"]))


def test_segment_refuses_modified_source(tmp_path: Path):
    deployment = tmp_path / "deployment"
    root = deployment / "datasets" / "starvoice-eval-v1"
    root.mkdir(parents=True)
    (root / "manifest.jsonl").write_text(json.dumps({
        "utterance_id": "1",
        "wav_48k_mono_pcm16": "missing.wav",
        "wav_sha256": "invalid",
    }) + "\n")
    out = tmp_path / "out"
    out.mkdir()
    try:
        campaign.prepare_segments(deployment, out, segment_seconds=1)
    except ValueError as ex:
        assert "missing or SHA mismatch" in str(ex)
    else:
        assert False, "missing source should be rejected"
