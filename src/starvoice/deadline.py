from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


DEFAULT_PLAYOUT_MS = (20, 30, 40, 50, 60, 80, 100, 120, 150, 200)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _trace_rows(path: Path) -> list[dict]:
    rows = _read_jsonl(path)
    rows.sort(key=lambda r: int(r["send_ns"]))
    return rows



def _estimated_arrival_ns(
    packet_row: dict,
    trace_rows: list[dict],
    rtt_fraction: float,
) -> int | None:
    idx = int(packet_row["source_trace_index"])
    if idx < 0 or idx >= len(trace_rows):
        raise ValueError(f"trace index out of range: {idx}")
    tr = trace_rows[idx]
    if bool(tr.get("lost")) or tr.get("rtt_ms") is None:
        return None
    send_ns = int(packet_row["send_ns"])
    one_way_ns = int(round(float(tr["rtt_ms"]) * rtt_fraction * 1_000_000))
    return send_ns + one_way_ns


def timely_fec_opportunity_slots(
    packet_rows: list[dict],
    trace_rows: list[dict],
    playout_ms: float,
    rtt_fraction: float = 0.5,
    frame_ms: float = 20.0,
) -> set[int]:
    """Return source-frame slots that FEC can recover before playout.

    The packet log should come from the protection policy whose realized LBRR is
    being used to define opportunities, typically Always-FEC for an upper bound.
    A slot is an opportunity only when its primary misses playout and the
    following packet contains actual LBRR and arrives by that source frame's
    deadline under the explicit RTT-fraction delay model.
    """
    if playout_ms < 0:
        raise ValueError("playout_ms must be non-negative")
    if not 0.0 <= rtt_fraction <= 1.0:
        raise ValueError("rtt_fraction must be in [0, 1]")
    if frame_ms <= 0:
        raise ValueError("frame_ms must be positive")
    if not packet_rows:
        return set()

    rows = sorted(packet_rows, key=lambda r: int(r["sequence"]))
    for expected, row in enumerate(rows):
        if int(row["sequence"]) != expected:
            raise ValueError("packet log sequence must be contiguous from zero")

    first_send_ns = int(rows[0]["send_ns"])
    frame_ns = int(round(frame_ms * 1_000_000))
    playout_ns = int(round(playout_ms * 1_000_000))
    arrivals = [
        _estimated_arrival_ns(row, trace_rows, rtt_fraction)
        for row in rows
    ]

    opportunities: set[int] = set()
    for i, row in enumerate(rows):
        deadline_ns = first_send_ns + i * frame_ns + playout_ns
        primary_arrival = arrivals[i]
        primary_on_time = (
            primary_arrival is not None and primary_arrival <= deadline_ns
        )
        if primary_on_time or i + 1 >= len(rows):
            continue
        if not bool(row.get("next_packet_has_lbrr")):
            continue
        next_arrival = arrivals[i + 1]
        if next_arrival is not None and next_arrival <= deadline_ns:
            opportunities.add(i)

    return opportunities

def evaluate_deadlines(
    packet_rows: list[dict],
    trace_rows: list[dict],
    playout_ms: float,
    rtt_fraction: float = 0.5,
    frame_ms: float = 20.0,
) -> dict:
    """Evaluate whether each audio frame is useful by its playout deadline.

    RTT is converted to an explicitly approximate one-way delay via
    one_way_delay = rtt_fraction * RTT. The default 0.5 assumes symmetric paths.
    FEC can rescue an unusable primary frame only when the following packet both
    contains LBRR and itself arrives before the lost frame's playout deadline.
    """
    if playout_ms < 0:
        raise ValueError("playout_ms must be non-negative")
    if not 0.0 <= rtt_fraction <= 1.0:
        raise ValueError("rtt_fraction must be in [0, 1]")
    if frame_ms <= 0:
        raise ValueError("frame_ms must be positive")
    if not packet_rows:
        return {
            "frames": 0,
            "network_lost_frames": 0,
            "primary_late_frames": 0,
            "primary_deadline_failures": 0,
            "fec_on_time_recovered_frames": 0,
            "fec_on_time_recovered_from_loss": 0,
            "fec_on_time_recovered_from_late": 0,
            "final_deadline_misses": 0,
            "final_deadline_miss_rate": None,
        }

    rows = sorted(packet_rows, key=lambda r: int(r["sequence"]))
    for expected, row in enumerate(rows):
        if int(row["sequence"]) != expected:
            raise ValueError("packet log sequence must be contiguous from zero")

    first_send_ns = int(rows[0]["send_ns"])
    frame_ns = int(round(frame_ms * 1_000_000))
    playout_ns = int(round(playout_ms * 1_000_000))

    network_lost = 0
    primary_late = 0
    primary_failures = 0
    fec_recovered = 0
    fec_from_loss = 0
    fec_from_late = 0

    def arrival_ns(packet_row: dict) -> int | None:
        idx = int(packet_row["source_trace_index"])
        if idx < 0 or idx >= len(trace_rows):
            raise ValueError(f"trace index out of range: {idx}")
        tr = trace_rows[idx]
        if bool(tr.get("lost")) or tr.get("rtt_ms") is None:
            return None
        send_ns = int(packet_row["send_ns"])
        one_way_ns = int(round(float(tr["rtt_ms"]) * rtt_fraction * 1_000_000))
        return send_ns + one_way_ns

    arrivals = [arrival_ns(r) for r in rows]

    for i, row in enumerate(rows):
        deadline_ns = first_send_ns + i * frame_ns + playout_ns
        lost = bool(row.get("lost"))
        if lost:
            network_lost += 1

        primary_arrival = arrivals[i]
        primary_on_time = primary_arrival is not None and primary_arrival <= deadline_ns
        if not lost and primary_arrival is not None and not primary_on_time:
            primary_late += 1

        if primary_on_time:
            continue

        primary_failures += 1
        fec_on_time = False
        if i + 1 < len(rows) and bool(row.get("next_packet_has_lbrr")):
            next_arrival = arrivals[i + 1]
            fec_on_time = next_arrival is not None and next_arrival <= deadline_ns

        if fec_on_time:
            fec_recovered += 1
            if lost:
                fec_from_loss += 1
            else:
                fec_from_late += 1

    final_misses = primary_failures - fec_recovered
    frames = len(rows)
    return {
        "frames": frames,
        "network_lost_frames": network_lost,
        "primary_late_frames": primary_late,
        "primary_deadline_failures": primary_failures,
        "fec_on_time_recovered_frames": fec_recovered,
        "fec_on_time_recovered_from_loss": fec_from_loss,
        "fec_on_time_recovered_from_late": fec_from_late,
        "final_deadline_misses": final_misses,
        "final_deadline_miss_rate": final_misses / frames if frames else None,
    }


def _packet_log_for_result(results_path: Path, row: dict) -> Path:
    root = results_path.parent
    trace_stem = Path(row["trace"]).stem
    return (
        root
        / "packet-logs"
        / trace_stem
        / row["policy"]
        / f"{row['utterance_id']}.jsonl"
    )


def run_deadline_analysis(
    result_sets: list[Path],
    output: Path,
    playout_ms: tuple[float, ...] = DEFAULT_PLAYOUT_MS,
    rtt_fraction: float = 0.5,
    frame_ms: float = 20.0,
) -> dict:
    if not result_sets:
        raise ValueError("at least one results JSONL is required")
    if not playout_ms:
        raise ValueError("at least one playout delay is required")

    trace_cache: dict[Path, list[dict]] = {}
    aggregates: dict[tuple[str, float], dict[str, int | float]] = defaultdict(
        lambda: {
            "conditions": 0,
            "frames": 0,
            "network_lost_frames": 0,
            "primary_late_frames": 0,
            "primary_deadline_failures": 0,
            "fec_on_time_recovered_frames": 0,
            "fec_on_time_recovered_from_loss": 0,
            "fec_on_time_recovered_from_late": 0,
            "fec_protected_frames": 0,
        }
    )

    for results_path in result_sets:
        for row in _read_jsonl(results_path):
            policy = str(row["policy"])
            trace = Path(row["trace"])
            if trace not in trace_cache:
                trace_cache[trace] = _trace_rows(trace)

            packet_log = _packet_log_for_result(results_path, row)
            if not packet_log.exists():
                raise FileNotFoundError(f"packet log not found: {packet_log}")
            packets = _read_jsonl(packet_log)

            for delay in playout_ms:
                metrics = evaluate_deadlines(
                    packets,
                    trace_cache[trace],
                    playout_ms=float(delay),
                    rtt_fraction=rtt_fraction,
                    frame_ms=frame_ms,
                )
                a = aggregates[(policy, float(delay))]
                a["conditions"] += 1
                for key in (
                    "frames",
                    "network_lost_frames",
                    "primary_late_frames",
                    "primary_deadline_failures",
                    "fec_on_time_recovered_frames",
                    "fec_on_time_recovered_from_loss",
                    "fec_on_time_recovered_from_late",
                ):
                    a[key] += int(metrics[key])
                a["fec_protected_frames"] += int(row.get("fec_protected_frames", 0))

    curve = []
    for (policy, delay), a in sorted(aggregates.items(), key=lambda x: (x[0][1], x[0][0])):
        frames = int(a["frames"])
        primary_failures = int(a["primary_deadline_failures"])
        recovered = int(a["fec_on_time_recovered_frames"])
        protected = int(a["fec_protected_frames"])
        final_misses = primary_failures - recovered
        curve.append(
            {
                "policy": policy,
                "playout_ms": delay,
                **a,
                "fec_duty_cycle": protected / frames if frames else None,
                "fec_recovery_fraction_of_primary_deadline_failures": (
                    recovered / primary_failures if primary_failures else None
                ),
                "final_deadline_misses": final_misses,
                "final_deadline_miss_rate": final_misses / frames if frames else None,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "experiment": "deadline-aware real-time voice replay analysis",
        "result_sets": [str(p) for p in result_sets],
        "frame_ms": frame_ms,
        "playout_ms": list(playout_ms),
        "one_way_delay_model": f"{rtt_fraction:g} * RTT",
        "rtt_fraction": rtt_fraction,
        "caveat": (
            "Trace RTT is not synchronized one-way delay. RTT/2 is an explicit "
            "symmetry proxy; conclusions should be checked across playout budgets "
            "and, later, against one-way measurements if available."
        ),
        "fec_semantics": (
            "A primary frame missing its deadline is recoverable only if the "
            "following packet contains actual LBRR and its estimated arrival is "
            "no later than the primary frame's playout deadline."
        ),
        "curve": curve,
    }
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
