"""Stage causal SV02 session logs in tmpfs; verify archival after capture."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def archive_session(source: Path, destination: Path) -> None:
    """Copy to a hidden sibling, verify every file, then publish atomically.

    Do not delete source on any failure. Destination must not exist beforehand.
    """
    if destination.exists():
        raise FileExistsError(f"session output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.partial-", dir=destination.parent
    ))
    try:
        shutil.copytree(source, partial, dirs_exist_ok=True, symlinks=False)
        entries = {}
        for file in sorted(source.rglob("*")):
            if file.is_symlink():
                raise ValueError(f"refusing symlink in session archive: {file}")
            if not file.is_file():
                continue
            relative = file.relative_to(source)
            copied = partial / relative
            digest = _hash(file)
            if not copied.is_file() or _hash(copied) != digest:
                raise IOError(f"archive checksum mismatch: {relative}")
            entries[str(relative)] = {"bytes": file.stat().st_size, "sha256": digest}
            _fsync_file(copied)
        manifest = {
            "schema_version": 1,
            "archival_method": "tmpfs-to-disk-verified-atomic-directory-rename",
            "files": entries,
        }
        (partial / "archive-integrity.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _fsync_file(partial / "archive-integrity.json")
        # Persist directories and metadata on the destination filesystem
        # before publishing and deleting the tmpfs originals.
        for directory in sorted(
            (p for p in partial.rglob("*") if p.is_dir()),
            key=lambda p: len(p.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
        _fsync_directory(partial)
        if destination.exists():
            raise FileExistsError(f"session output appeared during archive: {destination}")
        os.rename(partial, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def run_session_storage(
    destination: Path,
    operation: Callable[[Path], T],
    mode: str = "ram",
    ram_root: Path = Path("/dev/shm"),
) -> T:
    """Stage in RAM by default; archive even partial failed sessions.

    If archiving fails, the original RAM directory is retained and its path
    reported. Any unclean shutdown before archival remains a durability risk.
    """
    if mode not in ("ram", "direct"):
        raise ValueError(f"unsupported session storage mode: {mode}")
    if destination.exists():
        raise FileExistsError(f"session output already exists: {destination}")
    if mode == "direct":
        destination.mkdir(parents=True, exist_ok=False)
        return operation(destination)
    if not ram_root.is_dir():
        raise FileNotFoundError(f"RAM storage unavailable: {ram_root}")
    staged = Path(tempfile.mkdtemp(prefix="starvoice-session-", dir=ram_root))
    result = None
    operation_error: BaseException | None = None
    try:
        result = operation(staged)
    except BaseException as exc:
        operation_error = exc

    try:
        archive_session(staged, destination)
    except BaseException as exc:
        raise RuntimeError(
            f"Archiving failed; original session retained at {staged}; "
            f"intended output was {destination}"
        ) from exc
    shutil.rmtree(staged)
    if operation_error is not None:
        raise operation_error
    return result
