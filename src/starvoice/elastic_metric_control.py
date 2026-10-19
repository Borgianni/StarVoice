"""Temporal-reference falsification control for SlackBank acoustic STOI.

The program time-scales ORIGINAL CLEAN SOURCE PCM according to the exact
same 60..80ms sample-clock schedule used in the reconstructed SlackBank WAV.
This CLEAN-WARPED reference contains NO simulated packet loss; comparing it
with clean unmodified speech exposes metric sensitivity AND algorithmic audio
artifacts in the time-scaling operator. Using that SAME warped reference with
a warped decoded signal cancels part of the temporal misalignment but may also
cancel shared artifacts, so it is an explicitly diagnostic secondary metric,
NOT a standard STOI speech-intelligibility claim or MOS estimate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import wave
from pathlib import Path

from .elastic_playout import SAMPLES_PER_FRAME, SAMPLE_RATE, stretch_frame
from .elastic_quality import HALF_WINDOW_FRAMES, pcm_window, stoi_score
from .opus_replay_audit import sha256
from .slackbank import BASE_MS, arrival_data, ramp_targets

NAMES = ("fixed60", "stretch_only", "rescue_only_nonphysical", "slackbank")


def clean_clock_warp(source: Path, output: Path, budgets: list[int]) -> dict:
    """Create sample-exact 48kHz clean audio under the *physical* buffer ramp."""
    n = len(budgets)
    if n <= 0 or budgets[0] != BASE_MS or any(not BASE_MS <= b <= 80 for b in budgets):
        raise ValueError("requires a nonempty receiver budget 60..80ms starting at 60ms")
    if any(abs(b - a) > 1 for a, b in zip(budgets, budgets[1:])):
        raise ValueError("nonphysical buffer jump: >1ms per frame")
    if output.exists():
        raise FileExistsError("will not overwrite clean-warp reference")
    with wave.open(str(source), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(),
            wav.getcomptype()) != (SAMPLE_RATE, 1, 2, "NONE"):
            raise ValueError("clean reference must be mono 48kHz PCM16 WAV")
        original_frames = wav.getnframes()
        if (original_frames + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME != n:
            raise ValueError("clean corpus frame count mismatch")
        written = 0
        edits = 0
        with wave.open(str(output), "wb") as dest:
            dest.setframerate(SAMPLE_RATE)
            dest.setsampwidth(2)
            dest.setnchannels(1)
            for seq in range(n):
                pcm = wav.readframes(SAMPLES_PER_FRAME)
                if not pcm:
                    raise ValueError("unexpected early EOF")
                if len(pcm) > SAMPLES_PER_FRAME * 2:
                    raise AssertionError("oversize WAV frame")
                pcm = pcm.ljust(SAMPLES_PER_FRAME * 2, b"\0")
                sample_start = seq * SAMPLES_PER_FRAME + (budgets[seq] - BASE_MS) * 48
                if written != sample_start:
                    raise AssertionError(f"source-warp sample clock drift at seq {seq}")
                delta = (budgets[seq + 1] - budgets[seq]) * 48 if seq + 1 < n else 0
                edited = stretch_frame(pcm, delta)
                dest.writeframesraw(edited)
                written += len(edited) // 2
                edits += int(delta != 0)
    expected = n * SAMPLES_PER_FRAME + (budgets[-1] - BASE_MS) * 48
    if written != expected:
        raise AssertionError("clean-warp output sample count incorrect")
    return {"samples_written": written, "expected_samples": expected,
            "edited_frames": edits, "unpadded_source_samples": original_frames,
            "output_sha256": sha256(output)}


def metric_control(campaign_dir: Path, block: str, elastic_dir: Path,
                   ablation_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError("output directory exists; refusing to overwrite")
    ablation = json.loads((ablation_dir / "summary.json").read_text())
    rendered = json.loads((elastic_dir / "summary.json").read_text())
    manifest = json.loads((campaign_dir / "campaign-manifest.json").read_text())
    if (manifest.get("status") != "complete" or ablation["block"] != block
            or rendered["block"] != block or not rendered["sample_clock_verified"]
            or not rendered["policies"]["fixed60"]["audio"]["recorded_pcm_exact_match"]):
        raise ValueError("requires P0/P1 validated matching campaign block")
    rec = next((row for row in manifest["blocks"] if row["block"] == block), None)
    if rec is None:
        raise ValueError("unknown block")
    segment_id = int(rec["workload_segment"])
    src = campaign_dir / "workloads" / f"segment-{segment_id:02d}.wav"
    seg = next(row for row in manifest["segments"] if int(row["index"]) + 1 == segment_id)
    if sha256(src) != seg["sha256"] or seg["sha256"] != rec["workload_sha256"]:
        raise ValueError("original clean WAV SHA mismatch")
    n, _, arrivals = arrival_data(campaign_dir / block)
    budgets = ramp_targets(n, rendered["phase_estimate"])
    if budgets[-1] != BASE_MS:
        raise ValueError("end-of-session extra slack must be zero")
    paths = {
        "source": src,
        "fixed60": elastic_dir / "fixed60.wav",
        "slackbank": elastic_dir / "slackbank.wav",
        "stretch_only": ablation_dir / "stretch_only.wav",
        "rescue_only_nonphysical": ablation_dir / "rescue_only_NONPHYSICAL.wav",
        "source_warped": output_dir / "source_warped.wav",
    }
    for name in NAMES:
        if not paths[name].is_file():
            raise FileNotFoundError(f"missing prior P1 audio: {name}")
    if len(ablation.get("windows", [])) < 1:
        raise ValueError("ablation provides no deterministic matched windows")
    output_dir.mkdir(parents=True)
    clean_report = clean_clock_warp(src, paths["source_warped"], budgets)
    if clean_report["samples_written"] != n * SAMPLES_PER_FRAME:
        raise AssertionError("source reference failed to return to zero slack")
    records = []
    for row in ablation["windows"]:
        kind, center = row["kind"], int(row["center_sequence"])
        lo, hi = center - HALF_WINDOW_FRAMES, center + HALF_WINDOW_FRAMES
        if not (0 <= lo < hi < n and budgets[lo] == budgets[hi] == BASE_MS):
            raise ValueError("window boundaries are not unwarped source-aligned")
        samples_at = lo * SAMPLES_PER_FRAME
        samples_len = (hi - lo) * SAMPLES_PER_FRAME
        clips = {name: pcm_window(path, samples_at, samples_len)
                 for name, path in paths.items()}
        fixed_raw = stoi_score(clips["source"], clips["fixed60"])
        clean_self = stoi_score(clips["source"], clips["source"])
        clean_warp_raw = stoi_score(clips["source"], clips["source_warped"])
        raw_slack = stoi_score(clips["source"], clips["slackbank"])
        raw_stretch = stoi_score(clips["source"], clips["stretch_only"])
        warp_slack = stoi_score(clips["source_warped"], clips["slackbank"])
        warp_stretch = stoi_score(clips["source_warped"], clips["stretch_only"])
        records.append({
            "kind": kind, "center_sequence": center,
            "original_stoi_delta_slack_minus_fixed": raw_slack - fixed_raw,
            "reference_self_stoi": clean_self,
            "clean_warped_vs_clean_delta": clean_warp_raw - clean_self,
            "clean_warped_vs_clean_stoi": clean_warp_raw,
            "raw_reference_stretch_only_minus_fixed60": raw_stretch - fixed_raw,
            "warped_reference_slackbank_stoi": warp_slack,
            "warped_reference_stretch_only_stoi": warp_stretch,
            "warped_reference_rescue_gain": warp_slack - warp_stretch,
            "source_window_hashes": {
                "unwarped": hashlib.sha256(clips["source"]).hexdigest(),
                "warped": hashlib.sha256(clips["source_warped"]).hexdigest(),
            },
        })
    by_kind = {}
    cols = ("original_stoi_delta_slack_minus_fixed",
            "clean_warped_vs_clean_delta",
            "raw_reference_stretch_only_minus_fixed60",
            "warped_reference_rescue_gain")
    for kind in ("rescued", "control"):
        subset = [r for r in records if r["kind"] == kind]
        by_kind[kind] = {
            "count": len(subset),
            **{f"mean_{k}": statistics.mean(r[k] for r in subset) if subset else None
               for k in cols},
        }
    result = {
        "schema_version": 1, "block": block, "frames": n,
        "source_warped": clean_report, "groups": by_kind, "windows": records,
        "limitations": [
            "Original clean source warped by SAME primitive time-scaler; this control combines edit artifacts AND temporal-metric alignment.",
            "Warped-reference STOI can artificially mask common stretch artifacts and is DIAGNOSTIC only, not standard listener QoE.",
            "Event windows within a single 10min block are correlated; no independent sessions.",
            "Still offline with actual fixed packet arrivals, not native NetEQ, real-time sink or WER.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign-dir", required=True, type=Path)
    p.add_argument("--block", default="block-05")
    p.add_argument("--elastic-dir", required=True, type=Path)
    p.add_argument("--ablation-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()
    result = metric_control(
        args.campaign_dir, args.block, args.elastic_dir,
        args.ablation_dir, args.output_dir)
    for kind, group in result["groups"].items():
        print(f"{kind}: windows={group['count']}")
        for key, val in group.items():
            if key.startswith("mean_"):
                print(f"  {key}={val:+.6f}" if val is not None else f"  {key}=NA")
    print("This is a diagnostic matched-temporal-reference control, NOT QoE evidence.")
    print(f"Saved {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
