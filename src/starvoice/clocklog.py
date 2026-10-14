"""Best-effort host clock diagnostics. NTP offsets are estimates, not OWD truth."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from .storage import JsonlWriter, utc_now_iso


def _query(command: list[str], timeout: float = 4.0) -> dict:
    try:
        proc = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False,
        )
        return {
            "command": command, "returncode": proc.returncode,
            "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": str(exc)}


def clock_snapshot(include_dish: bool = True) -> dict:
    mono_before_ns = time.monotonic_ns()
    realtime_ns = time.time_ns()
    mono_after_ns = time.monotonic_ns()
    row = {
        "utc": utc_now_iso(),
        "realtime_ns": realtime_ns,
        "mono_before_ns": mono_before_ns,
        "mono_after_ns": mono_after_ns,
        "timesync": _query(["timedatectl", "timesync-status"]),
        "sync_state": _query(["timedatectl", "show", "-p", "NTPSynchronized", "--value"]),
    }
    if include_dish:
        row["dish_ntp_query"] = _query(["ntpdate", "-q", "192.168.100.1"])
    return row


def run_clock_logger(
    output: Path, duration_s: float, interval_s: float = 10.0,
    include_dish: bool = True,
) -> int:
    if duration_s <= 0 or interval_s <= 0:
        raise ValueError("duration and interval must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    samples = 0
    with JsonlWriter(output) as writer:
        while True:
            now = time.monotonic()
            if now - start >= duration_s and samples:
                break
            writer.write(clock_snapshot(include_dish))
            samples += 1
            next_at = start + samples * interval_s
            if next_at >= start + duration_s:
                break
            time.sleep(max(0.0, next_at - time.monotonic()))
    return samples
