"""Audit feasibility of offline Opus re-encoding against retained SV02 evidence.

The archived session does NOT contain original encoded packet bodies or hashes.
Matching packet lengths, LBRR flags, and the decoded WAV is evidence of a
faithful replay, but NEVER proof of byte-identical original Opus payloads.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import math
import wave
from pathlib import Path

from .opus import OpusConfig, OpusDecoder, OpusEncoder, packet_has_fec

SAMPLE_RATE = 48000
FRAME_SAMPLES = 960
WIRE_OVERHEAD = 35  # struct.calcsize("!4sB16sIQH")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def libopus_version() -> str:
    name = ctypes.util.find_library("opus")
    if not name:
        raise RuntimeError("libopus is unavailable")
    lib = ctypes.CDLL(name)
    lib.opus_get_version_string.restype = ctypes.c_char_p
    return lib.opus_get_version_string().decode("utf-8", errors="replace")


def encode_continuous(wav_path: Path, bitrate: int, loss_percent: int) -> list[bytes]:
    """Replicate the actual sender's continuous 20ms Always-FEC encoder state."""
    config = OpusConfig(bitrate=bitrate)
    encoder = OpusEncoder(config)
    packets = []
    try:
        with wave.open(str(wav_path), "rb") as source:
            if (source.getframerate(), source.getnchannels(), source.getsampwidth(),
                source.getcomptype()) != (SAMPLE_RATE, 1, 2, "NONE"):
                raise ValueError(f"expected 48kHz mono PCM16 WAV: {wav_path}")
            expected = math.ceil(source.getnframes() / FRAME_SAMPLES)
            for _ in range(expected):
                pcm = source.readframes(FRAME_SAMPLES)
                pcm = pcm.ljust(FRAME_SAMPLES * 2, b"\0")
                encoder.set_fec(True, loss_percent)
                packets.append(encoder.encode(pcm))
    finally:
        encoder.close()
    return packets


def compare_sender(sender_dir: Path, packets: list[bytes]) -> dict:
    """Check all sender metadata, retaining only bounded mismatch examples."""
    summary = json.loads((sender_dir / "summary.json").read_text())
    if summary.get("policy") != "always-fec" or summary.get("injected_drops") != 0:
        raise ValueError("expected natural zero-injection always-fec sender")
    if int(summary["frames"]) != len(packets):
        raise ValueError("frame count differs between corpus and sender")

    counts = {"rows": 0, "length_mismatches": 0, "lbrr_mismatches": 0,
              "wire_length_mismatches": 0, "unexpected_fec_off": 0,
              "sequence_errors": 0, "examples": []}
    for seq, record in enumerate(jsonl(sender_dir / "sender.jsonl")):
        if seq >= len(packets):
            raise ValueError("too many sender JSONL rows")
        if int(record["sequence"]) != seq:
            counts["sequence_errors"] += 1
        actual = packets[seq]
        if int(record["opus_bytes"]) != len(actual):
            counts["length_mismatches"] += 1
            if len(counts["examples"]) < 5:
                counts["examples"].append({"seq": seq, "issue": "length",
                    "observed": record["opus_bytes"], "reencoded": len(actual)})
        if bool(record["has_lbrr"]) != packet_has_fec(actual):
            counts["lbrr_mismatches"] += 1
            if len(counts["examples"]) < 5:
                counts["examples"].append({"seq": seq, "issue": "lbrr"})
        if int(record["udp_payload_bytes"]) != WIRE_OVERHEAD + len(actual):
            counts["wire_length_mismatches"] += 1
        if not bool(record["fec_enabled"]):
            counts["unexpected_fec_off"] += 1
        counts["rows"] += 1
    if counts["rows"] != len(packets):
        raise ValueError("sender JSONL rows missing")
    counts["metadata_consistent"] = not any(
        counts[k] for k in ("length_mismatches", "lbrr_mismatches",
                            "wire_length_mismatches", "unexpected_fec_off",
                            "sequence_errors")
    )
    counts["sum_opus_bytes_match"] = sum(len(p) for p in packets) == int(summary["opus_bytes"])
    counts["sum_wire_bytes_match"] = sum(len(p) + WIRE_OVERHEAD for p in packets) == int(summary["udp_payload_bytes"])
    counts["metadata_consistent"] &= counts["sum_opus_bytes_match"] and counts["sum_wire_bytes_match"]
    return counts


def pcm_sha256(wav_path: Path) -> tuple[int, str]:
    with wave.open(str(wav_path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (48000, 1, 2):
            raise ValueError("recorded receiver PCM format mismatch")
        frames = wav.getnframes()
        digest = hashlib.sha256()
        while True:
            data = wav.readframes(8192)
            if not data:
                break
            digest.update(data)
    return frames, digest.hexdigest()


def replay_recorded_decisions(receiver_dir: Path, packets: list[bytes], bitrate: int) -> dict:
    """Statefully decode recorded Primary/FEC/PLC choices with re-encoded bytes.

    Verifies reproducibility of original audio, NOT an alternative schedule.
    """
    summary = json.loads((receiver_dir / "summary.json").read_text())
    if int(summary["frames_expected"]) != len(packets):
        raise ValueError("receiver frame count differs from reconstructed packets")
    decoder = OpusDecoder(OpusConfig(bitrate=bitrate))
    pcm_digest = hashlib.sha256()
    counts = {"rows": 0, "primary": 0, "fec": 0, "plc": 0,
              "decision_sequence_errors": 0, "source_sequence_errors": 0}
    try:
        for seq, record in enumerate(jsonl(receiver_dir / "frame-decisions.jsonl")):
            if seq >= len(packets):
                raise ValueError("extra receiver decision row")
            if int(record["sequence"]) != seq:
                counts["decision_sequence_errors"] += 1
            decision = record["decision"]
            source_seq = record.get("source_packet_sequence")
            if decision == "primary":
                expected_source = seq
                pcm = decoder.decode(packets[seq], fec=False)
            elif decision == "fec":
                expected_source = seq + 1
                if expected_source >= len(packets):
                    raise ValueError("FEC carrier is beyond the source stream")
                pcm = decoder.decode(packets[expected_source], fec=True)
            elif decision == "plc":
                expected_source = None
                pcm = decoder.decode(None, fec=False)
            else:
                raise ValueError(f"unknown receiver decision: {decision}")
            if source_seq != expected_source:
                counts["source_sequence_errors"] += 1
            if len(pcm) != FRAME_SAMPLES * 2:
                raise ValueError(f"decoded PCM frame {seq} length is {len(pcm)}")
            pcm_digest.update(pcm)
            counts[decision] += 1
            counts["rows"] += 1
    finally:
        decoder.close()
    if counts["rows"] != len(packets):
        raise ValueError("receiver decision rows missing")
    recorded_samples, recorded_hash = pcm_sha256(receiver_dir / "received.wav")
    counts["recorded_samples"] = recorded_samples
    counts["redecoded_pcm_sha256"] = pcm_digest.hexdigest()
    counts["recorded_pcm_sha256"] = recorded_hash
    counts["pcm_identical"] = (recorded_samples == len(packets) * FRAME_SAMPLES
                               and recorded_hash == counts["redecoded_pcm_sha256"])
    counts["decision_counts_match"] = all(
        counts[key] == int(summary[key + "_frames"])
        for key in ("primary", "fec", "plc")
    ) and not (counts["decision_sequence_errors"] or counts["source_sequence_errors"])
    return counts


def audit_campaign(campaign_dir: Path) -> dict:
    manifest = json.loads((campaign_dir / "campaign-manifest.json").read_text())
    if manifest.get("status") != "complete" or len(manifest.get("blocks", [])) != 9:
        raise ValueError("expected a complete nine-block campaign")
    if manifest.get("sender_policy") != "always-fec" or manifest.get("injected_drops") != 0:
        raise ValueError("campaign must be natural Always-FEC")
    bitrate = int(manifest["codec"]["bitrate"])
    loss = int(manifest["codec"]["expected_loss_percent"])
    segments = {int(s["index"]) + 1: s for s in manifest["segments"]}
    packets_by_segment = {}
    report = {"schema_version": 1, "status": "metadata_and_redecoded_audio_audit",
              "original_encoded_packet_bytes_retained": False,
              "original_packet_byte_identity_proven": False,
              "cannot_infer_exact_original_opus_bytes": True,
              "local_libopus_version": libopus_version(), "campaign": str(campaign_dir),
              "blocks": []}
    for block in manifest["blocks"]:
        segment_id = int(block["workload_segment"])
        info = segments[segment_id]
        wav_path = Path(info["path"])
        if not wav_path.is_file():
            wav_path = campaign_dir / "workloads" / f"segment-{segment_id:02d}.wav"
        if sha256(wav_path) != info["sha256"] or info["sha256"] != block["workload_sha256"]:
            raise ValueError(f"workload WAV SHA mismatch on segment {segment_id}")
        if segment_id not in packets_by_segment:
            packets_by_segment[segment_id] = encode_continuous(wav_path, bitrate, loss)
        packets = packets_by_segment[segment_id]
        root = campaign_dir / block["block"]
        sender = compare_sender(root / "sender", packets)
        receiver = replay_recorded_decisions(root / "receiver", packets, bitrate)
        report["blocks"].append({
            "block": block["block"], "segment": segment_id,
            "sender": sender, "receiver": receiver,
            "metadata_match": sender["metadata_consistent"],
            "recorded_audio_reproduced": receiver["pcm_identical"] and receiver["decision_counts_match"],
        })
    report["total_sender_metadata_mismatches"] = sum(
        sum(row["sender"][key] for key in (
            "length_mismatches", "lbrr_mismatches", "wire_length_mismatches",
            "unexpected_fec_off", "sequence_errors"))
        for row in report["blocks"]
    )
    report["blocks_metadata_consistent"] = sum(b["metadata_match"] for b in report["blocks"])
    report["blocks_pcm_reproduced"] = sum(b["recorded_audio_reproduced"] for b in report["blocks"])
    report["all_replay_checks_passed"] = (
        report["blocks_metadata_consistent"] == len(report["blocks"])
        and report["blocks_pcm_reproduced"] == len(report["blocks"])
    )
    report["limitations"] = [
        "Original Opus packet bytes/hashes were NOT retained; cannot claim Opus byte identity.",
        "Encoding differences across NUC/Oracle libopus versions are possible.",
        "Matching stored WAV PCM verifies only the recorded decision schedule.",
        "Does not implement elastic playout, real playback, NetEQ or QoE.",
    ]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory exists; refusing to overwrite")
    result = audit_campaign(args.campaign_dir)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Libopus: {result['local_libopus_version']}")
    for block in result["blocks"]:
        print(f"{block['block']}: metadata={'PASS' if block['metadata_match'] else 'FAIL'} "
              f"recorded_PCM={'PASS' if block['recorded_audio_reproduced'] else 'FAIL'}")
    print("IMPORTANT: original Opus packet bytes were not retained; byte identity unproven.")
    print(f"Report saved to {args.output_dir / 'report.json'}")
    if not result["all_replay_checks_passed"]:
        raise SystemExit("Replay fidelity gate NOT MET; inspect report before audio QoE claims")


if __name__ == "__main__":
    main()
