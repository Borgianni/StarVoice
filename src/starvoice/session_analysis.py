"""Analyze SV02 endpoint logs without mixing monotonic clocks across hosts."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def analyze_session(root: Path) -> dict:
    tx, rx = root / "sender", root / "receiver"
    summary_tx = json.loads((tx / "summary.json").read_text())
    summary_rx = json.loads((rx / "summary.json").read_text())
    sent = _rows(tx / "sender.jsonl")
    received = _rows(rx / "receiver.jsonl")
    decisions = _rows(rx / "frame-decisions.jsonl")
    clocks_tx, clocks_rx = _rows(tx / "clock.jsonl"), _rows(rx / "clock.jsonl")
    errors = []
    count = summary_rx["frames_expected"]
    if summary_tx["session_id"] != summary_rx["session_id"]:
        errors.append("session_id_mismatch")
    if summary_tx["frames"] != count or summary_rx["frames_decoded"] != count:
        errors.append("frame_count_mismatch")
    if [r["sequence"] for r in sent] != list(range(count)):
        errors.append("sender_sequence_mismatch")
    if [r["sequence"] for r in decisions] != list(range(count)):
        errors.append("decision_sequence_mismatch")

    first_rx = {}
    for row in received:
        first_rx.setdefault(row["sequence"], row)

    counts = Counter()
    scheduler_max_ms = 0.0
    for row in decisions:
        seq = row["sequence"]
        mode = row["decision"]
        counts[mode] += 1
        deadline = row["scheduled_playout_mono_ns"]
        scheduler_max_ms = max(
            scheduler_max_ms, (row["actual_playout_mono_ns"] - deadline) / 1e6,
        )
        source = row["source_packet_sequence"]
        if mode == "plc":
            if source is not None:
                errors.append(f"plc_source_{seq}")
            continue
        if mode not in ("primary", "fec"):
            errors.append(f"unknown_decision_{seq}")
            continue
        expected_source = seq if mode == "primary" else seq + 1
        if source != expected_source:
            errors.append(f"wrong_source_{seq}")
        pkt = first_rx.get(expected_source)
        if pkt is None or pkt["receive_mono_ns"] > deadline:
            errors.append(f"causality_{seq}")
        if mode == "fec" and pkt is not None and not pkt["has_lbrr"]:
            errors.append(f"no_lbrr_{seq}")

    for mode, key in (
        ("primary", "primary_frames"), ("fec", "fec_frames"), ("plc", "plc_frames")
    ):
        if counts[mode] != summary_rx[key]:
            errors.append(f"summary_{mode}")
    if len(decisions) != count or sum(counts.values()) != count:
        errors.append("decision_count")
    if not clocks_tx or not clocks_rx:
        errors.append("missing_clock_samples")
    return {
        "session_id": summary_tx["session_id"],
        "valid": not errors, "errors": errors,
        "frames": count,
        "primary": counts["primary"], "fec": counts["fec"], "plc": counts["plc"],
        "scheduler_max_lateness_ms": round(scheduler_max_ms, 3),
        "sender_clock_samples": len(clocks_tx),
        "receiver_clock_samples": len(clocks_rx),
        "observed_unique_packets": len(first_rx),
        "injected_drops": sum(bool(r.get("injected_drop")) for r in sent),
        "endpoint": "receiver-local playout; absolute OWD not estimated",
    }
