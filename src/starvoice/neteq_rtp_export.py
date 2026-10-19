"""Export audited SV02 observed packet arrivals for **native** WebRTC NetEQ.

Produces an rtptools-style .rtp dump for WebRTC's own neteq_rtpplay, NOT a
home-made NetEQ approximation. Reconstructed continuous Always-FEC Opus bytes
replace originals that were never retained; P0 audited exact original PCM
decisions and per-packet metadata. Preserve their limitation in every report.

RTP timestamp and sequence number derive from SOURCE frame order. Packet
records appear in observed RECEIVE order, not RTP sequence order. Arrival
offsets are receiver-local USERSPACE times rounded down to integer ms, as
required by rtpdump; their sub-ms detail is irrecoverably lost.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

from .opus import packet_has_fec
from .opus_replay_audit import encode_continuous, jsonl, libopus_version, sha256
from .slackbank import arrival_data

HEADER = struct.Struct("!IIII")  # start_seconds, start_useconds, IPv4, port/pad compacted below
PACKET = struct.Struct("!HHI")  # stored length, original RTP length, arrival offset ms
RTP = struct.Struct("!BBHII")  # version/flags, payload type, sequence, timestamp, SSRC
DUMP_LINE = b"#!rtpplay1.0 127.0.0.1/5004\n"
OPUS_PT = 111
RTP_CLOCK = 48000
FRAME_SAMPLES = 960
SSRC = 0x53564F32  # "SVO2"


def convert_arrivals(
    received: dict[int, dict], packets: list[bytes], output: Path,
    *, first_seq: int | None = None,
) -> dict:
    """Write one RTPdump from reconstructed Opus and observed receive timing."""
    if output.exists():
        raise FileExistsError("refusing to replace existing NetEQ input")
    if not received:
        raise ValueError("no original receiver arrivals")
    if not packets:
        raise ValueError("no Opus stream")
    for seq in received:
        if not 0 <= seq < len(packets):
            raise ValueError("receiver sequence beyond reconstructed Opus stream")
    ordered = sorted(received.items(), key=lambda p: (int(p[1]["receive_mono_ns"]), p[0]))
    first_receive = int(ordered[0][1]["receive_mono_ns"])
    if first_seq is not None and ordered[0][0] != first_seq:
        raise ValueError("original first receiver sequence mismatch")
    if first_receive < 0:
        raise ValueError("invalid receive monotonic clock")
    for seq, item in ordered:
        if "has_lbrr" in item and bool(item["has_lbrr"]) != packet_has_fec(packets[seq]):
            raise ValueError(f"Opus LBRR replay mismatch at RTP source sequence {seq}")
        if "opus_bytes" in item and int(item["opus_bytes"]) != len(packets[seq]):
            raise ValueError(f"Opus replay length mismatch at RTP source sequence {seq}")

    output.parent.mkdir(parents=True, exist_ok=True)
    last_offset = 0
    total_bytes = 0
    with output.open("xb") as stream:
        stream.write(DUMP_LINE)
        # Standard rtpdump global header, network byte order:
        # timestamp_sec=0, timestamp_usec=0, source_ipv4=127.0.0.1,
        # source_port=5004, padding=0. Not wall clock / sender clock.
        stream.write(struct.pack("!IIIHH", 0, 0, 0x7F000001, 5004, 0))
        for seq, item in ordered:
            rel_ns = int(item["receive_mono_ns"]) - first_receive
            if rel_ns < 0:
                raise AssertionError("receive records out of time order")
            offset_ms = rel_ns // 1_000_000
            if offset_ms > 0xFFFFFFFF:
                raise ValueError("rtpdump offset exceeds 32 bits")
            opus = packets[seq]
            rtp = RTP.pack(0x80, OPUS_PT, seq & 0xFFFF,
                           (seq * FRAME_SAMPLES) & 0xFFFFFFFF, SSRC) + opus
            if len(rtp) + PACKET.size > 65535:
                raise ValueError("rtpdump packet too long")
            stream.write(PACKET.pack(len(rtp) + PACKET.size, len(rtp), offset_ms))
            stream.write(rtp)
            last_offset = offset_ms
            total_bytes += len(rtp)
    return {
        "format": "rtpdump",
        "rtp_payload_type": OPUS_PT,
        "ssrc": SSRC,
        "rtp_clock_hz": RTP_CLOCK,
        "frames_per_packet": 1,
        "samples_per_frame": FRAME_SAMPLES,
        "arrival_clock": "receiver_userspace_monotonic_relative_to_first_observed_audio",
        "arrival_offset_quantization": "floor_to_integer_milliseconds_0_to_less_than_1ms_error",
        "first_received_sequence": ordered[0][0],
        "observed_rtp_packets": len(ordered),
        "source_frames": len(packets),
        "unobserved_rtp_packets": len(packets) - len(ordered),
        "last_recorded_arrival_offset_ms": last_offset,
        "rtp_packet_bytes_total": total_bytes,
        "rtpdump_sha256": sha256(output),
        "original_opus_packet_bytes_retained": False,
        "exact_original_packet_byte_identity_unproven": True,
        "claims": [
            "Input is for a genuine external native NetEQ runner, not an internal NetEQ implementation.",
            "Dump contains packets actually observed by receiver; unobserved frames are absent.",
            "RTP headers are synthetic; original transport is custom SV02 UDP, NOT RTP.",
            "Original 20ms continuous Opus source reconstructed under audited libopus.",
            "Receiver timestamps are userspace; rounded to 1ms, not kernel timestamps.",
            "NetEQ default playout target is NOT automatically latency-matched to fixed60.",
        ],
    }


def export_block(campaign: Path, block_name: str, audit_path: Path,
                 output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite NetEQ output directory")
    manifest = json.loads((campaign / "campaign-manifest.json").read_text())
    audit = json.loads(audit_path.read_text())
    if (manifest.get("status") != "complete" or len(manifest.get("blocks", [])) != 9
            or not audit.get("all_replay_checks_passed")
            or Path(audit["campaign"]).resolve() != campaign.resolve()
            or audit.get("local_libopus_version") != libopus_version()):
        raise ValueError("matching complete P0 Opus fidelity campaign audit required")
    block = next((item for item in manifest["blocks"] if item["block"] == block_name), None)
    audit_block = next((item for item in audit["blocks"] if item["block"] == block_name), None)
    if (block is None or audit_block is None
            or not audit_block["metadata_match"]
            or not audit_block["recorded_audio_reproduced"]):
        raise ValueError("block does not have valid P0 replay audit")
    segment_id = int(block["workload_segment"])
    segment = next(
        s for s in manifest["segments"] if int(s["index"]) + 1 == segment_id
    )
    source = campaign / "workloads" / f"segment-{segment_id:02d}.wav"
    if sha256(source) != segment["sha256"] or block["workload_sha256"] != segment["sha256"]:
        raise ValueError("source speech SHA mismatch")
    packets = encode_continuous(
        source, int(manifest["codec"]["bitrate"]),
        int(manifest["codec"]["expected_loss_percent"]),
    )
    n, first_seq, received = arrival_data(campaign / block_name)
    if n != len(packets):
        raise ValueError("source and receiver frame count mismatch")
    sender_path = campaign / block_name / "sender" / "sender.jsonl"
    sender_rows = list(jsonl(sender_path))
    if len(sender_rows) != len(packets):
        raise ValueError("full stream sender log size mismatch")
    for seq, row in enumerate(sender_rows):
        if (int(row["sequence"]) != seq or int(row["opus_bytes"]) != len(packets[seq])
                or bool(row["has_lbrr"]) != packet_has_fec(packets[seq])):
            raise ValueError(f"full sender stream re-encode mismatch at {seq}")
    output_dir.mkdir(parents=True)
    out_file = output_dir / (block_name + ".rtp")
    result = convert_arrivals(received, packets, out_file, first_seq=first_seq)
    result.update({
        "campaign": str(campaign),
        "block": block_name,
        "source_wav_sha256": sha256(source),
        "libopus_version": libopus_version(),
        "input_reconstruction_audit": str(audit_path),
        "rtpdump_path": str(out_file),
        "external_native_neteq_executed": False,
        "native_neteq_delay_and_audio_quality_evaluated": False,
        "neteq_rtpplay": "https://webrtc.googlesource.com/src/+/refs/heads/main/modules/audio_coding/neteq/tools/",
    })
    (output_dir / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--block", default="block-05")
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = export_block(args.campaign_dir, args.block, args.audit_report, args.output_dir)
    print(f"{result['block']}: observed RTP packets={result['observed_rtp_packets']} "
          f"unobserved={result['unobserved_rtp_packets']} "
          f"last_arrival_offset_ms={result['last_recorded_arrival_offset_ms']}")
    print(f"NetEQ RTP dump: {result['rtpdump_path']}")
    print("Native NetEQ has NOT run. Do not compare its default playout delay with 60ms uncalibrated.")


if __name__ == "__main__":
    main()
