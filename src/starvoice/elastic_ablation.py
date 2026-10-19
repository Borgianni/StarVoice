"""Four-arm, block-local diagnostic to separate PLC rescue from audio edit cost.

Do NOT interpret the nonphysical "rescue_only" arm as a deployable playout
policy. It moves packet-use deadlines forward while leaving output duration
unchanged, violating the playout-time constraint. It exists ONLY to isolate
codec/PLC restoration from time-scale modification. This is NOT NetEQ/QoE.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .elastic_playout import (
    SAMPLES_PER_FRAME, render_policy, select_packet,
)
from .elastic_quality import (
    HALF_WINDOW_FRAMES, eligible_centers, pcm_window, stoi_score,
)
from .opus_replay_audit import encode_continuous, libopus_version, sha256
from .slackbank import (
    BASE_MS, FRAME_MS, TRAIN_FRAMES, arrival_data, ramp_targets,
)


def _same_suffix_counts(a: dict, b: dict) -> bool:
    return all(a[key] == b[key] for key in ("primary", "fec", "plc", "frames"))


def analyze(
    campaign_dir: Path, block_name: str, audit_report: Path,
    elastic_dir: Path, output_dir: Path,
) -> dict:
    """Produce two decompositional WAVs and score four matched strategies."""
    if output_dir.exists():
        raise FileExistsError("output directory exists: never overwrite a prior analysis")
    manifest = json.loads((campaign_dir / "campaign-manifest.json").read_text())
    audit = json.loads(audit_report.read_text())
    original = json.loads((elastic_dir / "summary.json").read_text())
    if manifest.get("status") != "complete" or len(manifest.get("blocks", [])) != 9:
        raise ValueError("requires an intact complete natural campaign")
    if (not audit.get("all_replay_checks_passed")
            or Path(audit["campaign"]).resolve() != campaign_dir.resolve()
            or audit.get("local_libopus_version") != libopus_version()):
        raise ValueError("P0 full-campaign Opus fidelity audit is required")
    audit_block = next((r for r in audit["blocks"] if r["block"] == block_name), None)
    if (not audit_block or not audit_block["metadata_match"]
            or not audit_block["recorded_audio_reproduced"]):
        raise ValueError("block is not P0-replay-audited")
    if original["block"] != block_name or not original["sample_clock_verified"]:
        raise ValueError("P1 baseline block/clock mismatch")
    baseline = original["policies"]["fixed60"]
    normal = original["policies"]["slackbank"]
    if not baseline["audio"]["recorded_pcm_exact_match"]:
        raise ValueError("P1 baseline is not PCM-identical to receiver")
    record = next((r for r in manifest["blocks"] if r["block"] == block_name), None)
    if record is None:
        raise ValueError("unknown block")
    segment_id = int(record["workload_segment"])
    src = campaign_dir / "workloads" / f"segment-{segment_id:02d}.wav"
    segment = next(r for r in manifest["segments"] if int(r["index"]) + 1 == segment_id)
    if sha256(src) != segment["sha256"] or segment["sha256"] != record["workload_sha256"]:
        raise ValueError("source WAV SHA mismatch")
    n, first_seq, arrivals = arrival_data(campaign_dir / block_name)
    packets = encode_continuous(
        src, int(manifest["codec"]["bitrate"]),
        int(manifest["codec"]["expected_loss_percent"]),
    )
    if n != len(packets):
        raise ValueError("corpus and receiver frame count mismatch")
    phase = original["phase_estimate"]
    ramp = ramp_targets(n, phase)
    fixed = [BASE_MS] * n
    if (ramp[-1] != BASE_MS
            or normal["audio"]["samples_written"] != n * SAMPLES_PER_FRAME
            or baseline["audio"]["samples_written"] != n * SAMPLES_PER_FRAME):
        raise ValueError("audio did not return to zero added slack at session end")

    paths = {
        "source": src,
        "fixed60": elastic_dir / "fixed60.wav",
        "slackbank": elastic_dir / "slackbank.wav",
        "stretch_only": output_dir / "stretch_only.wav",
        "rescue_only_nonphysical": output_dir / "rescue_only_NONPHYSICAL.wav",
    }
    if not all(p.is_file() for p in (paths["fixed60"], paths["slackbank"])):
        raise FileNotFoundError("original P1 rendered WAVs missing")
    # No measurements are written until inputs have been validated.
    output_dir.mkdir(parents=True)
    stretch_only = render_policy(
        paths["stretch_only"], packets, arrivals, first_seq, ramp,
        int(manifest["codec"]["bitrate"]), decision_budgets=fixed,
    )
    rescue_only = render_policy(
        paths["rescue_only_nonphysical"], packets, arrivals, first_seq, fixed,
        int(manifest["codec"]["bitrate"]), decision_budgets=ramp,
    )
    if (not _same_suffix_counts(stretch_only["scored_suffix"], baseline["modeled_suffix"])
            or not _same_suffix_counts(rescue_only["scored_suffix"], normal["modeled_suffix"])):
        raise AssertionError("diagnostic decoding decisions no longer match reference strategies")
    if (stretch_only["pre_stretch_pcm_sha256"]
            != baseline["audio"]["pre_stretch_pcm_sha256"]):
        raise AssertionError("stretch-only decoded audio does not match fixed60 before edits")
    if (rescue_only["pre_stretch_pcm_sha256"]
            != normal["audio"]["pre_stretch_pcm_sha256"]):
        raise AssertionError("rescue-only decoded audio does not match SlackBank before edits")
    if (stretch_only["edited_frames"] != normal["audio"]["edited_frames"]
            or rescue_only["edited_frames"] != 0
            or stretch_only["samples_written"] != n * SAMPLES_PER_FRAME
            or rescue_only["samples_written"] != n * SAMPLES_PER_FRAME):
        raise AssertionError("diagnostic physical sample clocks inconsistent")

    anchor = int(arrivals[first_seq]["receive_mono_ns"])
    rescues = []
    for seq in range(max(first_seq, TRAIN_FRAMES), n):
        at = anchor + (seq - first_seq) * FRAME_MS * 1_000_000
        if (select_packet(seq, at + 60_000_000, arrivals)[0] == "plc"
                and select_packet(seq, at + ramp[seq] * 1_000_000, arrivals)[0] != "plc"):
            rescues.append(seq)
    if len(rescues) != baseline["modeled_suffix"]["plc"] - normal["modeled_suffix"]["plc"]:
        raise AssertionError("rescued-event accounting mismatch")
    centers = eligible_centers(n, phase, ramp, rescues)
    if not centers:
        raise ValueError("no clips satisfy deterministic source-time window criteria")

    details = []
    for category, center in centers:
        lo = center - HALF_WINDOW_FRAMES
        hi = center + HALF_WINDOW_FRAMES
        if ramp[lo] != BASE_MS or ramp[hi] != BASE_MS:
            raise AssertionError("source-aligned window has nonzero edge slack")
        start = lo * SAMPLES_PER_FRAME
        length = (hi - lo) * SAMPLES_PER_FRAME
        reference = pcm_window(paths["source"], start, length)
        metrics = {
            key: stoi_score(reference, pcm_window(paths[key], start, length))
            for key in ("fixed60", "stretch_only", "rescue_only_nonphysical", "slackbank")
        }
        details.append({
            "kind": category, "center_sequence": center,
            "rescues_in_window": sum(lo <= seq < hi for seq in rescues),
            "edited_frames_in_window": sum(ramp[i+1] != ramp[i] for i in range(lo, hi)),
            "stoi": metrics,
            "stretch_cost_fixed_decisions": metrics["stretch_only"] - metrics["fixed60"],
            "rescue_gain_nonphysical_no_edits": (
                metrics["rescue_only_nonphysical"] - metrics["fixed60"]
            ),
            "rescue_gain_with_edits": metrics["slackbank"] - metrics["stretch_only"],
            "stretch_cost_rescued_decisions": (
                metrics["slackbank"] - metrics["rescue_only_nonphysical"]
            ),
            "combined_vs_fixed60": metrics["slackbank"] - metrics["fixed60"],
        })
    groups = {}
    fields = (
        "stretch_cost_fixed_decisions", "rescue_gain_nonphysical_no_edits",
        "rescue_gain_with_edits", "stretch_cost_rescued_decisions",
        "combined_vs_fixed60",
    )
    for category in ("rescued", "control"):
        selection = [row for row in details if row["kind"] == category]
        groups[category] = {"windows": len(selection), **{
            "mean_" + field: (
                statistics.mean(row[field] for row in selection) if selection else None
            ) for field in fields
        }}
    out = {
        "schema_version": 1, "block": block_name, "calibration_excluded_frames": TRAIN_FRAMES,
        "scored_suffix_plc": {
            "fixed60": baseline["modeled_suffix"]["plc"],
            "slackbank": normal["modeled_suffix"]["plc"],
        },
        "rescued_suffix_frames": len(rescues),
        "groups": groups,
        "windows": details,
        "diagnostic_renderers": {
            "stretch_only": stretch_only,
            "rescue_only_NONPHYSICAL": rescue_only,
        },
        "wav_paths": {key: str(value) for key, value in paths.items()},
        "limitations": [
            "All four conditions share archived receive timestamps and reconstructed Opus packets.",
            "Rescue-only moves packet-use deadlines without corresponding speaker delay: NONPHYSICAL oracle, NOT a deployable receiver.",
            "Stretch-only intentionally disregards packet arrivals enabled by its physical extra playout slack: diagnostic ablation, not optimized receiver.",
            "Original fixed60 and SlackBank WAVs were produced by an earlier verified PCM renderer.",
            "STOI is frame-aligned at zero-edge-slack boundaries but may penalize harmless internal time warps; no listening or NetEQ comparison.",
            "Within-block event windows are correlated, not independent experimental replicates.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--block", default="block-05")
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument("--elastic-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(
        args.campaign_dir, args.block, args.audit_report,
        args.elastic_dir, args.output_dir,
    )
    print(f"block={report['block']} rescued_suffix_frames={report['rescued_suffix_frames']}")
    for category, row in report["groups"].items():
        print(f"{category}: windows={row['windows']}")
        for key, value in row.items():
            if key.startswith("mean_"):
                print(f"  {key}={value:+.6f}" if value is not None else f"  {key}=NA")
    print("rescue_only_NONPHYSICAL is NOT a valid player; STOI is diagnostic.")
    print(f"Saved {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
