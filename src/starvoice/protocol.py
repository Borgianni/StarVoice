from __future__ import annotations

import struct
from dataclasses import dataclass

MAGIC = b"SV01"
VERSION = 1
KIND_PROBE = 1
KIND_OPUS = 2

# magic, version, kind, flags, sequence, sender monotonic ns, payload length
_HEADER = struct.Struct("!4sBBHIQH")
HEADER_SIZE = _HEADER.size


@dataclass(frozen=True)
class Packet:
    kind: int
    flags: int
    sequence: int
    send_ns: int
    payload: bytes = b""


def encode_packet(packet: Packet) -> bytes:
    if not 0 <= packet.sequence <= 0xFFFFFFFF:
        raise ValueError("sequence out of range")
    if len(packet.payload) > 0xFFFF:
        raise ValueError("payload too large")
    return _HEADER.pack(
        MAGIC,
        VERSION,
        packet.kind,
        packet.flags,
        packet.sequence,
        packet.send_ns,
        len(packet.payload),
    ) + packet.payload


def decode_packet(data: bytes) -> Packet:
    if len(data) < HEADER_SIZE:
        raise ValueError("short packet")
    magic, version, kind, flags, sequence, send_ns, payload_len = _HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ValueError("bad magic")
    if version != VERSION:
        raise ValueError(f"unsupported protocol version {version}")
    payload = data[HEADER_SIZE:]
    if len(payload) != payload_len:
        raise ValueError("payload length mismatch")
    return Packet(kind=kind, flags=flags, sequence=sequence, send_ns=send_ns, payload=payload)
