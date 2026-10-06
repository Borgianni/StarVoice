from __future__ import annotations

import random
import select
import socket
import time
import wave
from dataclasses import dataclass, asdict
from pathlib import Path

from .opus import OpusConfig, OpusDecoder, OpusEncoder, packet_has_fec
from .predictor import PhaseModel
from .probe import parse_target
from .protocol import KIND_OPUS, Packet, decode_packet, encode_packet
from .storage import JsonlWriter


FLAG_FEC = 1
FLAG_DRED = 2


@dataclass
class SpeechFrame:
    sequence: int
    send_ns: int
    recv_ns: int | None
    bytes: int
    received: bool
    fec_enabled: bool
    risk: float
    policy: str


def _policy_fec(
    policy: str,
    risk: float,
    rng: random.Random,
    random_duty_cycle: float,
) -> bool:
    if policy == "plain":
        return False
    if policy == "always-fec":
        return True
    if policy == "predictive-fec":
        return risk >= 0.5
    if policy == "random-fec":
        return rng.random() < random_duty_cycle
    raise ValueError(f"unknown policy: {policy}")


def speech_loop(
    input_wav: Path,
    output_wav: Path,
    target: str,
    policy: str,
    predictor: PhaseModel | None,
    bitrate: int,
    expected_loss_percent: int,
    random_duty_cycle: float,
    seed: int,
    packet_log: Path | None = None,
    playout_wait_ms: float = 120.0,
) -> dict:
    config = OpusConfig(bitrate=bitrate)
    enc, dec = OpusEncoder(config), OpusDecoder(config)
    rng = random.Random(seed)
    host, port = parse_target(target)
    addr = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_DGRAM)[0][4]
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    frame_bytes = config.frame_samples * config.channels * 2
    writer = JsonlWriter(packet_log) if packet_log else None

    encoded: dict[int, tuple[bytes, int, bool, float]] = {}
    received: dict[int, bytes] = {}
    rows: list[SpeechFrame] = []

    try:
        with wave.open(str(input_wav), "rb") as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (
                config.sample_rate,
                config.channels,
                2,
            ):
                raise ValueError("input WAV must be 48 kHz, mono, signed PCM16")
            seq = 0
            start_ns = time.monotonic_ns()
            while True:
                pcm = w.readframes(config.frame_samples)
                if not pcm:
                    break
                if len(pcm) < frame_bytes:
                    pcm += b"\0" * (frame_bytes - len(pcm))
                deadline = start_ns + seq * config.frame_ms * 1_000_000
                while time.monotonic_ns() < deadline:
                    time.sleep(0.0005)
                send_ns = time.monotonic_ns()
                risk = predictor.risk(send_ns) if predictor else 0.0
                fec = _policy_fec(policy, risk, rng, random_duty_cycle)
                enc.set_fec(fec, expected_loss_percent)
                payload = enc.encode(pcm)
                flags = FLAG_FEC if fec else 0
                packet = encode_packet(Packet(KIND_OPUS, flags, seq, send_ns, payload))
                sock.sendto(packet, addr)
                encoded[seq] = (payload, send_ns, fec, risk)

                # Drain immediate echoes without perturbing pacing materially.
                while True:
                    ready, _, _ = select.select([sock], [], [], 0)
                    if not ready:
                        break
                    data, _ = sock.recvfrom(65535)
                    try:
                        p = decode_packet(data)
                        if p.kind == KIND_OPUS:
                            received[p.sequence] = p.payload
                    except ValueError:
                        pass
                seq += 1

        end_wait = time.monotonic() + playout_wait_ms / 1000.0
        while time.monotonic() < end_wait:
            ready, _, _ = select.select([sock], [], [], 0.01)
            if ready:
                data, _ = sock.recvfrom(65535)
                try:
                    p = decode_packet(data)
                    if p.kind == KIND_OPUS:
                        received[p.sequence] = p.payload
                except ValueError:
                    pass

        with wave.open(str(output_wav), "wb") as out:
            out.setnchannels(config.channels)
            out.setsampwidth(2)
            out.setframerate(config.sample_rate)
            total = len(encoded)
            recovered_fec = 0
            lost = 0
            for seq in range(total):
                payload, send_ns, fec_enabled, risk = encoded[seq]
                rx_payload = received.get(seq)
                if rx_payload is not None:
                    pcm = dec.decode(rx_payload, fec=False)
                    recv_ns = None
                    ok = True
                else:
                    lost += 1
                    # Opus in-band FEC for frame N is carried in packet N+1.
                    next_payload = received.get(seq + 1)
                    next_has_fec = (
                        next_payload is not None
                        and packet_has_fec(next_payload)
                    )
                    if next_payload is not None and next_has_fec:
                        try:
                            pcm = dec.decode(next_payload, fec=True)
                            recovered_fec += 1
                        except RuntimeError:
                            pcm = dec.decode(None, fec=False)
                    else:
                        pcm = dec.decode(None, fec=False)
                    recv_ns = None
                    ok = False
                out.writeframes(pcm)
                row = SpeechFrame(
                    seq, send_ns, recv_ns, len(payload), ok, fec_enabled, risk, policy
                )
                rows.append(row)
                if writer:
                    writer.write(asdict(row))

        total_bytes = sum(r.bytes for r in rows)
        protected = sum(1 for r in rows if r.fec_enabled)
        return {
            "frames": len(rows),
            "network_lost_frames": lost,
            "network_loss_rate": lost / len(rows) if rows else None,
            "fec_recovery_attempt_successes": recovered_fec,
            "encoded_bytes": total_bytes,
            "fec_duty_cycle": protected / len(rows) if rows else None,
            "policy": policy,
            "bitrate": bitrate,
        }
    finally:
        if writer:
            writer.close()
        sock.close()
        enc.close()
        dec.close()
