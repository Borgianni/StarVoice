from __future__ import annotations

import hashlib
import json
import os
import platform
import queue
import threading
import time
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



class AsyncJsonlWriter:
    """Offload blocking JSON serialization and file I/O from a real-time loop.

    Each file has one worker, preserving row order. A bounded queue prevents
    unlimited memory growth; overflow or worker failure is explicit, never
    silently dropped. close() waits for all queued rows and reports I/O errors.
    """

    def __init__(self, path: Path, max_pending: int = 8192):
        self._queue: queue.Queue[Any | None] = queue.Queue(maxsize=max_pending)
        self._closed = False
        self._failure: BaseException | None = None
        self._max_pending = 0
        self._max_sink_write_ns = 0
        self._thread = threading.Thread(
            target=self._consume, args=(path,),
            name=f"jsonl-{path.name}", daemon=True,
        )
        self._thread.start()

    def _consume(self, path: Path) -> None:
        try:
            with JsonlWriter(path) as sink:
                while True:
                    row = self._queue.get()
                    if row is None:
                        break
                    started_ns = time.monotonic_ns()
                    sink.write(row)
                    self._max_sink_write_ns = max(
                        self._max_sink_write_ns, time.monotonic_ns() - started_ns
                    )
        except BaseException as exc:
            self._failure = exc

    def write(self, obj: Any) -> None:
        if self._closed:
            raise RuntimeError("cannot write to closed async JSONL writer")
        if self._failure is not None:
            raise RuntimeError("async JSONL worker failed") from self._failure
        try:
            self._queue.put_nowait(obj)
        except queue.Full as exc:
            raise RuntimeError("async JSONL queue full: refusing silent log loss") from exc
        self._max_pending = max(self._max_pending, self._queue.qsize())

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._thread.is_alive() and self._failure is None:
                self._queue.put(None)
            self._thread.join()
        if self._failure is not None:
            raise RuntimeError("async JSONL worker failed") from self._failure

    @property
    def stats(self) -> dict[str, float | int]:
        return {
            "max_queue_depth": self._max_pending,
            "max_sink_write_ms": round(self._max_sink_write_ns / 1e6, 3),
        }


def dump_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
