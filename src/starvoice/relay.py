from __future__ import annotations

import json
import socket
import time
from pathlib import Path

from .protocol import decode_packet
from .storage import JsonlWriter, utc_now_iso


def run_relay(bind: str, port: int, log_path: Path | None = None) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((bind, port))
    writer = JsonlWriter(log_path) if log_path else None
    print(f"StarVoice relay listening on udp://{bind}:{port}", flush=True)
    try:
        while True:
            data, addr = sock.recvfrom(65535)
            recv_ns = time.monotonic_ns()
            try:
                p = decode_packet(data)
                record = {
                    "event": "relay_rx",
                    "utc": utc_now_iso(),
                    "recv_ns": recv_ns,
                    "peer_ip": addr[0],
                    "peer_port": addr[1],
                    "kind": p.kind,
                    "sequence": p.sequence,
                    "sender_send_ns": p.send_ns,
                    "bytes": len(data),
                }
                if writer:
                    writer.write(record)
            except ValueError:
                if writer:
                    writer.write({
                        "event": "relay_invalid",
                        "utc": utc_now_iso(),
                        "recv_ns": recv_ns,
                        "peer_ip": addr[0],
                        "bytes": len(data),
                    })
            sock.sendto(data, addr)
    except KeyboardInterrupt:
        pass
    finally:
        if writer:
            writer.close()
        sock.close()
