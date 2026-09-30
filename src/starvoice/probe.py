from __future__ import annotations

import math
import select
import socket
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from .protocol import KIND_PROBE, Packet, decode_packet, encode_packet
from .storage import JsonlWriter


@dataclass
class ProbeResult:
    sequence: int
    send_ns: int
    recv_ns: int | None
    rtt_ms: float | None
    lost: bool


def parse_target(target: str) -> tuple[str, int]:
    host, sep, port = target.rpartition(":")
    if not sep or not host:
        raise ValueError("target must be HOST:PORT")
    return host, int(port)


def run_probe(
    target: str,
    duration_s: float,
    interval_ms: float,
    timeout_ms: float,
    output: Path | None = None,
) -> list[ProbeResult]:
    host, port = parse_target(target)
    addr = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_DGRAM)[0][4]
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    interval_ns = int(interval_ms * 1_000_000)
    timeout_ns = int(timeout_ms * 1_000_000)
    start = time.monotonic_ns()
    stop = start + int(duration_s * 1_000_000_000)
    next_send = start
    seq = 0
    outstanding: dict[int, int] = {}
    results: list[ProbeResult] = []
    writer = JsonlWriter(output) if output else None

    try:
        while time.monotonic_ns() < stop or outstanding:
            now = time.monotonic_ns()
            while now >= next_send and next_send < stop:
                packet = Packet(KIND_PROBE, 0, seq, now)
                sock.sendto(encode_packet(packet), addr)
                outstanding[seq] = now
                seq += 1
                next_send += interval_ns
                now = time.monotonic_ns()

            readable, _, _ = select.select([sock], [], [], min(interval_ms / 1000.0, 0.01))
            if readable:
                try:
                    data, _ = sock.recvfrom(65535)
                    recv_ns = time.monotonic_ns()
                    packet = decode_packet(data)
                    sent = outstanding.pop(packet.sequence, None)
                    if sent is not None:
                        row = ProbeResult(
                            packet.sequence,
                            sent,
                            recv_ns,
                            (recv_ns - sent) / 1_000_000,
                            False,
                        )
                        results.append(row)
                        if writer:
                            writer.write(asdict(row))
                except (BlockingIOError, ValueError):
                    pass

            now = time.monotonic_ns()
            expired = [s for s, sent in outstanding.items() if now - sent >= timeout_ns]
            for s in expired:
                sent = outstanding.pop(s)
                row = ProbeResult(s, sent, None, None, True)
                results.append(row)
                if writer:
                    writer.write(asdict(row))
    finally:
        if writer:
            writer.close()
        sock.close()

    results.sort(key=lambda r: r.sequence)
    return results


def summarize(results: list[ProbeResult]) -> dict:
    received = [r for r in results if not r.lost and r.rtt_ms is not None]
    rtts = [r.rtt_ms for r in received if r.rtt_ms is not None]
    sent = len(results)
    lost = sent - len(received)
    ordered = sorted(rtts)

    def percentile(p: float) -> float | None:
        if not ordered:
            return None
        i = (len(ordered) - 1) * p
        lo, hi = math.floor(i), math.ceil(i)
        if lo == hi:
            return ordered[lo]
        return ordered[lo] * (hi - i) + ordered[hi] * (i - lo)

    return {
        "packets": sent,
        "received": len(received),
        "lost": lost,
        "loss_rate": (lost / sent) if sent else None,
        "rtt_ms_median": statistics.median(rtts) if rtts else None,
        "rtt_ms_p95": percentile(0.95),
        "rtt_ms_mean": statistics.mean(rtts) if rtts else None,
        "rtt_ms_stdev": statistics.pstdev(rtts) if len(rtts) >= 2 else None,
    }
