import hashlib
import json
from pathlib import Path

import pytest

from starvoice.session_storage import archive_session, run_session_storage


def test_archive_session_hashes_files_and_publishes_atomically(tmp_path):
    ram = tmp_path / "ram"
    ram.mkdir()
    destination = tmp_path / "results" / "session"
    source = ram / "capture"
    source.mkdir()
    (source / "nested").mkdir()
    (source / "nested" / "trace.jsonl").write_text('{"sequence":0}\n')
    (source / "audio.wav").write_bytes(b"wav-example")

    archive_session(source, destination)

    manifest = json.loads((destination / "archive-integrity.json").read_text())
    assert manifest["files"]["audio.wav"]["sha256"] == hashlib.sha256(
        b"wav-example"
    ).hexdigest()
    assert "nested/trace.jsonl" in manifest["files"]
    assert (source / "audio.wav").exists()  # Archive alone never erases original.

    with pytest.raises(FileExistsError):
        archive_session(source, destination)


def test_ram_storage_archives_after_operation_and_cleans_stage(tmp_path):
    ram_root = tmp_path / "ram"
    ram_root.mkdir()
    destination = tmp_path / "results" / "session"

    def run(staged: Path):
        assert staged.parent == ram_root
        (staged / "audio.wav").write_bytes(b"ok")
        assert not destination.exists()
        return 23

    assert run_session_storage(destination, run, ram_root=ram_root) == 23
    assert (destination / "audio.wav").read_bytes() == b"ok"
    assert (destination / "archive-integrity.json").exists()
    assert list(ram_root.iterdir()) == []


def test_ram_storage_archives_partial_logs_after_session_exception(tmp_path):
    ram_root = tmp_path / "ram"
    ram_root.mkdir()
    destination = tmp_path / "results" / "partial"

    def fails(staged):
        (staged / "partial.jsonl").write_text('{"sample":1}\n')
        raise ValueError("network failed")

    with pytest.raises(ValueError, match="network failed"):
        run_session_storage(destination, fails, ram_root=ram_root)
    assert (destination / "partial.jsonl").exists()
    assert list(ram_root.iterdir()) == []


def test_failed_archive_keeps_ram_source(tmp_path, monkeypatch):
    ram_root = tmp_path / "ram"
    ram_root.mkdir()
    destination = tmp_path / "results" / "unavailable"
    stages = []

    def run(staged):
        stages.append(staged)
        (staged / "sender.jsonl").write_text("{}\n")

    def fail_copy(_source, _destination):
        raise OSError("disk full")

    monkeypatch.setattr("starvoice.session_storage.archive_session", fail_copy)
    with pytest.raises(RuntimeError, match="original session retained"):
        run_session_storage(destination, run, ram_root=ram_root)

    assert stages[0].exists()
    assert (stages[0] / "sender.jsonl").read_text() == "{}\n"
    assert not destination.exists()


def test_direct_storage_mode_avoids_ram(tmp_path):
    out = tmp_path / "direct"
    assert run_session_storage(out, lambda p: p.name, mode="direct") == "direct"
    assert out.is_dir()
    assert not (out / "archive-integrity.json").exists()
