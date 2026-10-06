import json
import wave
from pathlib import Path

from starvoice.replay import replay_fixed_schedule_recovery_only


def _write_wav(path: Path, frames: int = 8) -> None:
    samples = 960 * frames
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(b"\0\0" * samples)


def _write_trace(path: Path, frames: int = 8) -> None:
    rows = []
    for i in range(frames):
        rows.append(
            {
                "sequence": i,
                "send_ns": i * 20_000_000,
                "recv_ns": None if i == 2 else i * 20_000_000 + 10_000_000,
                "rtt_ms": None if i == 2 else 10.0,
                "lost": i == 2,
            }
        )
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_fixed_schedule_replay_runs(tmp_path: Path) -> None:
    wav = tmp_path / "x.wav"
    trace = tmp_path / "trace.jsonl"
    _write_wav(wav)
    _write_trace(trace)
    result = replay_fixed_schedule_recovery_only(
        input_wav=wav,
        trace=trace,
        schedule={3},
    )
    assert result["frames"] == 8
    assert result["network_lost_frames"] == 1
    assert result["fec_protected_frames"] == 1
