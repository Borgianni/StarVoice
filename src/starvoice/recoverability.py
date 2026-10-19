"""Exploratory receiver-relative Opus LBRR opportunity analysis.

This is NOT an absolute OWD measurement, a successful-decode oracle, or a
counterfactual estimate of turning FEC on/off. It holds observed encoded bytes
and receiver arrival timestamps fixed while varying the local playout budget.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

FRAME_NS = 20_000_000
DEFAULT_BUDGETS = (40, 50, 60, 80, 100)


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze(session_dir: Path, budgets: tuple[int, ...] = DEFAULT_BUDGETS) -> tuple[dict, list[dict]]:
    """Return a per-session feasibility summary and per-failed-frame event rows."""
    rx_dir = session_dir / "receiver"
    tx_dir = session_dir / "sender"
    rx_summary_path = rx_dir / "summary.json"
    rx_log_path = rx_dir / "receiver.jsonl"
    decisions_path = rx_dir / "frame-decisions.jsonl"
    tx_summary_path = tx_dir / "summary.json"

    summary = json.loads(rx_summary_path.read_text(encoding="utf-8"))
    tx_summary = json.loads(tx_summary_path.read_text(encoding="utf-8"))
    if summary["session_id"] != tx_summary["session_id"]:
        raise ValueError("sender/receiver session ID mismatch")
    n = int(summary["frames_expected"])
    if int(summary["frames_decoded"]) != n or int(tx_summary["frames"]) != n:
        raise ValueError("incomplete sender/receiver frame accounting")
    if summary.get("endpoint") != "receiver_relative_playout":
        raise ValueError("unexpected endpoint (expected receiver_relative_playout)")
    if not budgets or any(b < 0 or b > 5000 for b in budgets):
        raise ValueError("playout budgets must be in range 0..5000 ms")

    packets = _load_jsonl(rx_log_path)
    decisions = _load_jsonl(decisions_path)
    if not packets or not decisions:
        raise ValueError("empty receiver packet or decision log")
    if [r["sequence"] for r in decisions] != list(range(n)):
        raise ValueError("decision sequences do not cover 0..N-1 in order")

    arrivals: dict[int, dict] = {}
    for row in packets:
        seq = int(row["sequence"])
        if seq not in arrivals or int(row["receive_mono_ns"]) < int(arrivals[seq]["receive_mono_ns"]):
            arrivals[seq] = row
    first_seq, first = min(arrivals.items(), key=lambda pair: int(pair[1]["receive_mono_ns"]))
    first_ns = int(first["receive_mono_ns"])
    if first_seq >= n:
        raise ValueError("first received AUDIO packet outside session")

    events: list[dict] = []
    results: list[dict] = []
    for budget in budgets:
        counts: Counter[str] = Counter()
        disagreements: list[dict] = []
        for seq in range(first_seq, n):
            deadline = first_ns + (seq - first_seq) * FRAME_NS + budget * 1_000_000
            primary = arrivals.get(seq)
            carrier = arrivals.get(seq + 1)
            primary_timely = primary is not None and int(primary["receive_mono_ns"]) <= deadline
            carrier_timely = carrier is not None and int(carrier["receive_mono_ns"]) <= deadline
            carrier_lbrr = carrier is not None and bool(carrier.get("has_lbrr", False))
            eligible = (not primary_timely) and carrier_timely and carrier_lbrr
            modeled = "primary" if primary_timely else ("fec" if eligible else "plc")
            counts["frames_analyzed"] += 1
            counts[f"modeled_{modeled}"] += 1
            if primary_timely:
                continue
            counts["primary_failures"] += 1
            kind = "primary_not_observed" if primary is None else "primary_receiver_observed_late"
            counts[kind] += 1
            if carrier is None:
                carrier_kind = "carrier_not_observed"
            elif not carrier_timely:
                carrier_kind = "carrier_receiver_observed_late"
            elif not carrier_lbrr:
                carrier_kind = "carrier_timely_without_lbrr"
            else:
                carrier_kind = "carrier_timely_with_lbrr"
            counts[carrier_kind] += 1
            counts["eligible_lbrr_upper_bound"] += int(eligible)
            events.append({
                "budget_ms": budget,
                "sequence": seq,
                "sequence_mod_750": seq % 750,
                "primary_failure_kind": kind,
                "primary_arrival_minus_deadline_ms": "" if primary is None else round((int(primary["receive_mono_ns"]) - deadline) / 1e6, 3),
                "carrier_sequence": seq + 1,
                "carrier_kind": carrier_kind,
                "carrier_has_lbrr": int(carrier_lbrr),
                "carrier_arrival_minus_deadline_ms": "" if carrier is None else round((int(carrier["receive_mono_ns"]) - deadline) / 1e6, 3),
                "lbrr_eligible_upper_bound": int(eligible),
                "actual_decision_at_recorded_buffer": decisions[seq]["decision"] if budget == float(summary["buffer_ms"]) else "",
            })

        if budget == float(summary["buffer_ms"]):
            for seq in range(first_seq, n):
                deadline = first_ns + (seq - first_seq) * FRAME_NS + budget * 1_000_000
                primary = arrivals.get(seq)
                carrier = arrivals.get(seq + 1)
                p = primary is not None and int(primary["receive_mono_ns"]) <= deadline
                f = not p and carrier is not None and bool(carrier.get("has_lbrr", False)) and int(carrier["receive_mono_ns"]) <= deadline
                modeled = "primary" if p else ("fec" if f else "plc")
                actual = decisions[seq]["decision"]
                if actual != modeled:
                    disagreements.append({"sequence": seq, "modeled": modeled, "actual": actual})
        results.append({
            "buffer_ms": budget,
            "counts": dict(sorted(counts.items())),
            "observed_decision_disagreements_at_recorded_buffer": disagreements,
            "lbrr_eligible_fraction_of_failures": (
                counts["eligible_lbrr_upper_bound"] / counts["primary_failures"]
                if counts["primary_failures"] else None
            ),
        })

    report = {
        "schema_version": 1,
        "status": "exploratory_receiver_relative_fixed_packets_only",
        "session_id": summary["session_id"],
        "frames_expected": n,
        "frames_startup_excluded": first_seq,
        "injected_drops": tx_summary.get("injected_drops"),
        "sender_policy": tx_summary.get("policy"),
        "recorded_buffer_ms": summary["buffer_ms"],
        "frame_duration_ms": 20,
        "input_sha256": {
            "receiver_summary": _sha256(rx_summary_path),
            "receiver_packets": _sha256(rx_log_path),
            "frame_decisions": _sha256(decisions_path),
            "sender_summary": _sha256(tx_summary_path),
        },
        "caveats": [
            "Receiver-relative deadline anchored at first received AUDIO packet, not absolute sender-to-playout latency or OWD.",
            "Receiver timestamp is userspace receive time; kernel/network arrival may differ.",
            "LBRR eligibility is a necessary opportunity check, not proof that counterfactual decode would succeed.",
            "Different buffers replay fixed observed packet arrivals and encoded payloads; they do not simulate changed codec or network behavior.",
            "One network session and its frames do not provide independent statistical replication.",
        ],
        "budgets": results,
    }
    return report, events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output-dir already exists; refusing to overwrite existing analysis")
    report, events = analyze(args.session_dir, tuple(args.budgets))
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if events:
        with (args.output_dir / "failure-events.csv").open("w", newline="", encoding="utf-8") as dst:
            writer = csv.DictWriter(dst, fieldnames=list(events[0]))
            writer.writeheader()
            writer.writerows(events)
    for item in report["budgets"]:
        count = item["counts"]
        print(f"buffer={item['buffer_ms']}ms primary_fail={count.get('primary_failures', 0)} "
              f"eligible_LBRR={count.get('eligible_lbrr_upper_bound', 0)} "
              f"observed_disagreements={len(item['observed_decision_disagreements_at_recorded_buffer'])}")
    print(f"Wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
