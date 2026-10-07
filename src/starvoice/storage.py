from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def environment_snapshot() -> dict[str, Any]:
    return {
        "captured_at_utc": utc_now_iso(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "pid": os.getpid(),
        "git_commit": git_commit(),
    }


def new_run_dir(root: Path, site: str, label: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rid = uuid.uuid4().hex[:12]
    run_dir = root / f"{stamp}_{site}_{label}_{rid}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


class JsonlWriter:
    def __init__(self, path: Path):
        self._f = path.open("a", encoding="utf-8", buffering=1)

    def write(self, obj: Any) -> None:
        if is_dataclass(obj):
            obj = asdict(obj)
        self._f.write(json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n")

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def dump_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
