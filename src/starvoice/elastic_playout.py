"""Offline, sample-clock-realized SlackBank renderer over reconstructed SV02 Opus.

This is a causal-policy, fixed-arrival audio *prototype*, NOT a live NetEQ
comparison or a certified real-time sink. Each 20ms Opus frame is decoded
statefully and rendered to a 48kHz PCM WAV. To implement the planned per-frame
playout start deadline, frame i outputs 960+48*(B[i+1]-B[i]) samples, i.e. its
length shifts the START of frame i+1. Modifications use a local overlap-add
edit, not pitch-changing resampling. Acoustic quality remains to be measured.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import struct
import time
import wave
from pathlib import Path

from .opus import OpusConfig, OpusDecoder
from .opus_replay_audit import encode_continuous, libopus_version, pcm_sha256, sha256
from .slackbank import BASE_MS, FRAME_MS, TRAIN_FRAMES, arrival_data, calibrate_phase, evaluate, ramp_targets

SAMPLES_PER_FRAME = 960
SAMPLES_PER_MS = 48
OVERLAP = 240  # 5ms crossfade internal to each decoded frame
SAMPLE_RATE = 48000


def stretch_frame(pcm: bytes, delta_samples: int) -> bytes:
    """Pitch-preserving *local overlap-add* time edit of a decoded 20ms frame.

    Uses ONLY this decoded frame's samples; no future PCM, rate conversion or
    speaker timing assumption. This is a basic two-grain overlap-add prototype,
    not production WSOLA and not validated for perceptual transparency.
    """
    if len(pcm) != SAMPLES_PER_FRAME * 2:
        raise ValueError(f"expected 960 mono PCM16 samples (20ms), got {len(pcm)} bytes")
    if delta_samples not in (-SAMPLES_PER_MS, 0, SAMPLES_PER_MS):
        raise ValueError("only +/-1ms output changes per 20ms frame are supported")
    if delta_samples == 0:
        return pcm

    samples = struct.unpack("<960h", pcm)
    best_cut = 0
    best_cost = None
    # The two overlapped grains differ by exactly 1ms. Find an edit region
    # whose sample structure aligns well to minimize the audible splice.
    for cut in range(312, 505, 24):
        cost = sum(
            (samples[cut + j] - samples[cut - delta_samples + j]) ** 2
            for j in range(0, OVERLAP, 8)
        )
        if best_cost is None or cost < best_cost:
            best_cost, best_cut = cost, cut
    left_end = best_cut
    right_start = best_cut - delta_samples
    fade = tuple(
        int(round(
            (samples[left_end + j] * (OVERLAP - 1 - j)
             + samples[right_start + j] * j) / (OVERLAP - 1)
        ))
        for j in range(OVERLAP)
    )
    output = samples[:left_end] + fade + samples[right_start + OVERLAP:]
    if len(output) != SAMPLES_PER_FRAME + delta_samples:
        raise AssertionError("PCM expansion/contraction length mismatch")
    return struct.pack(f"<{len(output)}h", *output)


def select_packet(seq: int, deadline_ns: int, arrivals: dict[int, dict]):
    primary = arrivals.get(seq)
    if primary is not None and int(primary["receive_mono_ns"]) <= deadline_ns:
        return "primary", seq
    carrier = arrivals.get(seq + 1)
    if (carrier is not None
            and bool(carrier.get("has_lbrr", False))
            and int(carrier["receive_mono_ns"]) <= deadline_ns):
        return "fec", seq + 1
    return "plc", None


def render_policy(
    out_wav: Path,
    packets: list[bytes],
    arrivals: dict[int, dict],
    first_seq: int,
    budgets: list[int],
    bitrate: int,
) -> dict:
    """Decode fixed receiver-local arrivals, write actual elastic PCM WAV.

    All potential byte changes are on the *previous* frame so that the
    first sample of frame seq occurs exactly at the planned D_seq modulo the
    receiver-local anchor; no future packet is read by the edit itself.
    """
    n = len(packets)
    if len(budgets) != n or any(not BASE_MS <= v <= 80 for v in budgets):
        raise ValueError("expected a complete <=80ms playout-budget schedule")
    if first_seq not in arrivals:
        raise ValueError("first receiver packet is missing")
    first_ns = int(arrivals[first_seq]["receive_mono_ns"])
    decoder = OpusDecoder(OpusConfig(bitrate=bitrate))
    counts = {"primary": 0, "fec": 0, "plc": 0, "frames": 0,
              "edited_frames": 0, "expanded_frames": 0, "contracted_frames": 0}
    suffix = {"primary": 0, "fec": 0, "plc": 0, "frames": 0}
    pcm_digest = hashlib.sha256()
    written_samples = 0
    times_ns = []
    try:
        with wave.open(str(out_wav), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            for seq in range(n):
                due_ns = first_ns + (seq - first_seq) * FRAME_MS * 1_000_000 + budgets[seq] * 1_000_000
                # Frames before first packet are startup-uncertain and always PLC.
                decision, packet_seq = (
                    select_packet(seq, due_ns, arrivals)
                    if seq >= first_seq else ("plc", None)
                )
                if decision == "primary":
                    pcm = decoder.decode(packets[packet_seq], fec=False)
                elif decision == "fec":
                    pcm = decoder.decode(packets[packet_seq], fec=True)
                else:
                    pcm = decoder.decode(None, fec=False)
                if len(pcm) != SAMPLES_PER_FRAME * 2:
                    raise AssertionError("Opus decoder did not produce 960 samples")
                pcm_digest.update(pcm)

                # Physical, sample-count-exact position where frame seq begins.
                planned_sample_start = (seq * SAMPLES_PER_FRAME
                                        + (budgets[seq] - budgets[0]) * SAMPLES_PER_MS)
                if written_samples != planned_sample_start:
                    raise AssertionError(f"playout clock drift at frame {seq}")
                delta = ((budgets[seq + 1] - budgets[seq]) * SAMPLES_PER_MS
                         if seq + 1 < n else 0)
                start = time.perf_counter_ns()
                stretched = stretch_frame(pcm, delta)
                if delta:
                    times_ns.append(time.perf_counter_ns() - start)
                    counts["edited_frames"] += 1
                    counts["expanded_frames" if delta > 0 else "contracted_frames"] += 1
                wav.writeframesraw(stretched)
                written_samples += len(stretched) // 2
                counts[decision] += 1
                counts["frames"] += 1
                if seq >= max(TRAIN_FRAMES, first_seq):
                    suffix[decision] += 1
                    suffix["frames"] += 1
    finally:
        decoder.close()
    expected_samples = (n * SAMPLES_PER_FRAME
                        + (budgets[-1] - budgets[0]) * SAMPLES_PER_MS)
    if written_samples != expected_samples:
        raise AssertionError("final sample clock does not match deadline schedule")
    counts.update({
        "samples_written": written_samples,
        "expected_samples": expected_samples,
        "first_playout_receiver_relative_mono_ns": (
            first_ns - first_seq * 20_000_000 + budgets[0] * 1_000_000),
        "pre_stretch_pcm_sha256": pcm_digest.hexdigest(),
        "scored_suffix": suffix,
        "edit_time_p50_ms": (round(statistics.median(times_ns) / 1e6, 4) if times_ns else 0),
        "edit_time_max_ms": (round(max(times_ns) / 1e6, 4) if times_ns else 0),
    })
    return counts


def render_block(campaign_dir: Path, block_name: str, audit_report: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError("output directory already exists; refusing to overwrite")
    manifest = json.loads((campaign_dir / "campaign-manifest.json").read_text())
    audit = json.loads(audit_report.read_text())
    if manifest.get("status") != "complete" or len(manifest.get("blocks", [])) != 9:
        raise ValueError("requires audited complete nine-block campaign")
    if (not audit.get("all_replay_checks_passed")
            or Path(audit["campaign"]).resolve() != campaign_dir.resolve()
            or audit.get("local_libopus_version") != libopus_version()):
        raise ValueError("matching P0 Opus replay-fidelity gate is mandatory")
    audit_block = next((r for r in audit["blocks"] if r["block"] == block_name), None)
    if not audit_block or not (audit_block["metadata_match"]
                               and audit_block["recorded_audio_reproduced"]):
        raise ValueError("block did not pass the P0 replay-fidelity gate")
    record = next((r for r in manifest["blocks"] if r["block"] == block_name), None)
    if record is None:
        raise ValueError("block was not part of the original campaign")
    segment_id = int(record["workload_segment"])
    src = campaign_dir / "workloads" / f"segment-{segment_id:02d}.wav"
    segment = next(s for s in manifest["segments"] if int(s["index"]) + 1 == segment_id)
    if sha256(src) != segment["sha256"] or segment["sha256"] != record["workload_sha256"]:
        raise ValueError("source WAV hash mismatch")
    bitrate = int(manifest["codec"]["bitrate"])
    packets = encode_continuous(src, bitrate, int(manifest["codec"]["expected_loss_percent"]))
    block = campaign_dir / block_name
    n, first_seq, arrivals = arrival_data(block)
    if n != len(packets):
        raise ValueError("source and receiver frame count disagree")
    phase, support = calibrate_phase(arrivals)
    strategies = {
        "fixed60": [BASE_MS] * n,
        "slackbank": ramp_targets(n, phase),
    }
    output_dir.mkdir(parents=True)
    reports = {}
    for name, budgets in strategies.items():
        predicted = evaluate(n, first_seq, arrivals, budgets)
        audio = render_policy(output_dir / f"{name}.wav", packets, arrivals, first_seq, budgets, bitrate)
        scored = audio["scored_suffix"]
        if any(scored[key] != predicted[key] for key in ("primary", "fec", "plc", "frames")):
            raise AssertionError("PCM decoder event counts mismatch idealized model")
        if name == "fixed60":
            recorded_samples, recorded_hash = pcm_sha256(block / "receiver" / "received.wav")
            if (recorded_samples != n * SAMPLES_PER_FRAME
                    or recorded_hash != audio["pre_stretch_pcm_sha256"]):
                raise AssertionError("fixed60 PCM fails to reproduce actual receiver audio")
            audio["recorded_pcm_exact_match"] = True
        else:
            audio["recorded_pcm_exact_match"] = None
        reports[name] = {"modeled_suffix": predicted, "audio": audio}

    out = {"schema_version": 1, "block": block_name, "phase_training_prefix_frames": TRAIN_FRAMES,
           "phase_estimate": phase, "phase_supporting_cycles": support,
           "bitrate": bitrate, "libopus_version": libopus_version(),
           "output_wavs_are_real_pcm16_48khz": True, "sample_clock_verified": True,
           "frame_edit": "local two-grain 5ms overlap-add +/-48 samples, NOT production WSOLA",
           "policies": reports,
           "limitations": [
               "Offline reconstructive rendering; no physical speaker/sink timing or real-time scheduler",
               "Fixed recorded arrivals; no native NetEQ; no DNSMOS/STOI/WER or listener scores",
               "No original Opus payload bytes; only archived metadata and PCM reproducibility certified",
               "Time-scaling artifacts and 20ms decode processing times must be evaluated",
               "Within-block first 120s calibration is excluded from the reported policy comparison",
           ]}
    (output_dir / "summary.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--block", default="block-05")
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = render_block(args.campaign_dir, args.block, args.audit_report, args.output_dir)
    for name, values in result["policies"].items():
        print(f"{name}: scored_PLC={values['modeled_suffix']['plc']} "
              f"edited_frames={values['audio']['edited_frames']} "
              f"PCM_samples={values['audio']['samples_written']}")
    print("Fixed60 PCM matches original received WAV; SlackBank WAV physically sample-length edited.")
    print("NOT real NetEQ or a live receiver; audio quality and latency still unverified.")
    print(f"Artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
