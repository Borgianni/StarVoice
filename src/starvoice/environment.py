from __future__ import annotations

import ctypes.util
import shutil
import socket
import sys
from dataclasses import dataclass, asdict


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def run_doctor() -> list[Check]:
    checks: list[Check] = []
    checks.append(Check("python>=3.11", sys.version_info >= (3, 11), sys.version.split()[0]))
    opus = ctypes.util.find_library("opus")
    checks.append(Check("libopus", bool(opus), opus or "not found"))
    checks.append(Check("git", shutil.which("git") is not None, shutil.which("git") or "not found"))
    checks.append(
        Check(
            "monotonic_clock",
            hasattr(__import__("time"), "monotonic_ns"),
            "time.monotonic_ns",
        )
    )
    try:
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM).close()
        checks.append(Check("udp_socket", True, "available"))
    except OSError as e:
        checks.append(Check("udp_socket", False, str(e)))
    return checks


def checks_as_dict(checks: list[Check]) -> list[dict]:
    return [asdict(c) for c in checks]
