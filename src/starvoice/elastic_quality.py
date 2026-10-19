"""Matched, event-centered audible clips and diagnostic STOI for elastic audio.

Scores are *within-block diagnostic* signals, NOT a NetEQ comparison or
validated perceptual superiority. Each 6-second window starts and finishes
at zero additional playout slack so reference, fixed60 and SlackBank WAVs
have identical durations without post-hoc time alignment or cropping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import wave
from pathlib import Path

from .elastic_playout import select_packet
from .slackbank import TRAIN_FRAMES, arrival_data, ramp_targets

FRAME_SAMPLES = 960
RATE = 48000
HALF_WINDOW_FRAMES = 150  # six seconds in total, with 3s before and after
MIN_CENTER_SPACING = 160


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pcm_window(path: Path, start: int, frames: int) -> bytes:
    if start < 0 or frames <= 0:
        raise ValueError("invalid PCM window")
    with wave.open(str(path), "rb") as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (RATE, 1, 2):
            raise ValueError("requires 48kHz mono PCM16 WAV")
        if start + frames > source.getnframes():
            raise ValueError("clip extends beyond original WAV")
        source.setpos(start)
        result = source.readframes(frames)
    if len(result) != frames * 2:
        raise ValueError("short audio clip")
    return result


def save_clip(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as dest:
        dest.setnchannels(1)
        dest.setsampwidth(2)
        dest.setframerate(RATE)
        dest.writeframes(pcm)


def eligible_centers(
    n: int, phase: int | None, budgets: list[int],
    rescues: list[int], cap_controls: int = 24,
) -> list[tuple[str, int]]:
    """Deterministic, non-overlapping event and negative-control clip centers.

    Controls are selected at scheduled playout ramp peaks in periods with no
    Primary/PLC rescue within the entire clip window. No favorable cherry-pick.
    """
    if len(budgets) != n:
        raise ValueError("budget length mismatch")
    def valid(center: int) -> bool:
        lo, hi = center - HALF_WINDOW_FRAMES, center + HALF_WINDOW_FRAMES
        return (lo >= TRAIN_FRAMES and hi < n and budgets[lo] == budgets[hi] == 60)

    centers: list[tuple[str, int]] = []
    for seq in rescues:
        if not valid(seq):
            continue
        if any(abs(seq - other) < MIN_CENTER_SPACING for _, other in centers):
            continue
        centers.append(("rescued", seq))
    if phase is not None:
        for event in range(phase, n, 750):
            if not valid(event):
                continue
            if any(abs(event - rescued) <= HALF_WINDOW_FRAMES for rescued in rescues):
                continue
            if any(abs(event - existing) < MIN_CENTER_SPACING for _, existing in centers):
                continue
            if not any(b != 60 for b in budgets[event - 25:event + 26]):
                continue
            centers.append(("control", event))
            if sum(kind == "control" for kind, _ in centers) >= cap_controls:
                break
    return sorted(centers, key=lambda x: x[1])


def stoi_score(clean: bytes, candidate: bytes) -> float:
    """Unadjusted, 6s window-aligned STOI diagnostic at 16kHz.

    Dynamic time-warping inside the clip can bias alignment-sensitive STOI;
    report this limitation. No post-hoc variable alignment is applied.
    """
    import numpy as np
    from scipy.signal import resample_poly
    from pystoi import stoi

    if len(clean) != len(candidate):
        raise ValueError("matched windows must have identical lengths")
    ref = np.frombuffer(clean, dtype="<i2").astype(np.float64) / 32768.0
    deg = np.frombuffer(candidate, dtype="<i2").astype(np.float64) / 32768.0
    return float(stoi(resample_poly(ref, 1, 3), resample_poly(deg, 1, 3),
                      16000, extended=False))


def analyze(
    campaign_dir: Path, block: str, elastic_dir: Path, output_dir: Path,
) -> dict:
    if output_dir.exists():
        raise FileExistsError("output directory exists; will not overwrite")
    manifest = json.loads((campaign_dir / "campaign-manifest.json").read_text())
    render = json.loads((elastic_dir / "summary.json").read_text())
    if manifest["status"] != "complete" or render["block"] != block:
        raise ValueError("render and completed campaign block mismatch")
    if not render["sample_clock_verified"] or not render["policies"]["fixed60"]["audio"]["recorded_pcm_exact_match"]:
        raise ValueError("requires P1 sample-clock verification and original PCM replay PASS")
    wavs = {
        "source": campaign_dir / "workloads" / f"segment-{next(int(r['workload_segment']) for r in manifest['blocks'] if r['block'] == block):02d}.wav",
        "fixed60": elastic_dir / "fixed60.wav",
        "slackbank": elastic_dir / "slackbank.wav",
    }
    segment_id = next(int(r["workload_segment"]) for r in manifest["blocks"] if r["block"] == block)
    segment = next(r for r in manifest["segments"] if int(r["index"]) + 1 == segment_id)
    from .opus_replay_audit import sha256 as verify_sha
    if verify_sha(wavs["source"]) != segment["sha256"]:
        raise ValueError("source corpus WAV hash mismatch")
    n, first_seq, arrivals = arrival_data(campaign_dir / block)
    phase = render["phase_estimate"]
    budgets = ramp_targets(n, phase)
    if n != render["policies"]["slackbank"]["audio"]["frames"]:
        raise ValueError("render count mismatch")
    for kind in ("fixed60", "slackbank"):
        if render["policies"][kind]["audio"]["samples_written"] != n * FRAME_SAMPLES:
            raise ValueError("this 6s clip tool requires zero final accumulated slack")
    rescued: list[int] = []
    for seq in range(max(TRAIN_FRAMES, first_seq), n):
        due = int(arrivals[first_seq]["receive_mono_ns"]) + (seq - first_seq) * 20_000_000
        initial = select_packet(seq, due + 60_000_000, arrivals)
        shifted = select_packet(seq, due + budgets[seq] * 1_000_000, arrivals)
        if initial[0] == "plc" and shifted[0] != "plc":
            rescued.append(seq)
    predicted = render["policies"]["fixed60"]["modeled_suffix"]["plc"] - render["policies"]["slackbank"]["modeled_suffix"]["plc"]
    if len(rescued) != predicted:
        raise AssertionError(f"actual rescue count differs from renderer: {len(rescued)} != {predicted}")
    centers = eligible_centers(n, phase, budgets, rescued)
    if not centers:
        raise ValueError("no evaluable rescue or changed-audio control window")
    output_dir.mkdir(parents=True)
    details = []
    for ordinal, (category, center) in enumerate(centers, 1):
        lo, hi = center - HALF_WINDOW_FRAMES, center + HALF_WINDOW_FRAMES
        if budgets[lo] != 60 or budgets[hi] != 60:
            raise AssertionError("unmatched boundary playout slack")
        sample_start = lo * FRAME_SAMPLES
        sample_length = (hi - lo) * FRAME_SAMPLES
        clips = {
            name: pcm_window(path, sample_start, sample_length)
            for name, path in wavs.items()
        }
        scores = {
            method: stoi_score(clips["source"], clips[method])
            for method in ("fixed60", "slackbank")
        }
        label = f"{ordinal:02d}-{category}-seq{center}"
        for name, pcm in clips.items():
            save_clip(output_dir / f"{label}-{name}.wav", pcm)
        details.append({
            "category": category, "center_sequence": center,
            "start_sequence": lo, "end_sequence_exclusive": hi,
            "rescued_frames_in_window": sum(lo <= x < hi for x in rescued),
            "modified_frames_in_window": sum(budgets[seq + 1] != budgets[seq] for seq in range(lo, hi)),
            "stoi_fixed60": scores["fixed60"],
            "stoi_slackbank": scores["slackbank"],
            "delta_stoi": scores["slackbank"] - scores["fixed60"],
        })
    summary = {"schema_version": 1, "block": block, "phase": phase,
               "total_rescued_suffix_frames": len(rescued),
               "clip_length_seconds": 6,
               "window_count": len(details), "windows": details,
               "wav_sha256": {name: sha256(path) for name, path in wavs.items()},
               "group_means": {}, "limitations": [
                   "Diagnostic event-window STOI, not native NetEQ or listener MOS.",
                   "STOI assumes time-aligned signals; internal +/-20ms local warps can bias scores.",
                   "Windows chosen by fixed deterministic rules on observed rescue and ramp cycles.",
                   "6s windows do not give independent samples; overlapping temporal context is possible.",
                   "No absolute mouth-to-ear latency, no live audio sink, no generalized claims.",
               ]}
    for category in ("rescued", "control"):
        records = [r for r in details if r["category"] == category]
        summary["group_means"][category] = {
            "count": len(records),
            "mean_delta_stoi": statistics.mean(r["delta_stoi"] for r in records) if records else None,
            "mean_fixed60_stoi": statistics.mean(r["stoi_fixed60"] for r in records) if records else None,
            "mean_slackbank_stoi": statistics.mean(r["stoi_slackbank"] for r in records) if records else None,
        }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--block", default="block-05")
    parser.add_argument("--elastic-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.campaign_dir, args.block, args.elastic_dir, args.output_dir)
    print(f"block={report['block']} rescued_suffix_frames={report['total_rescued_suffix_frames']}")
    for category, values in report["group_means"].items():
        print(f"{category}: windows={values['count']} mean_delta_STOI={values['mean_delta_stoi']}")
    print(f"Matched 6-second triples: {args.output_dir}; scores are diagnostic, not NetEQ.")


if __name__ == "__main__":
    main()
