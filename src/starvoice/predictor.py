from __future__ import annotations

import cmath
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PhaseModel:
    period_s: float
    boundary_offset_s: float
    risk_half_width_s: float
    score_threshold_ms: float

    def phase_s(self, monotonic_ns: int) -> float:
        t = monotonic_ns / 1e9
        return (t - self.boundary_offset_s) % self.period_s

    def risk(self, monotonic_ns: int) -> float:
        p = self.phase_s(monotonic_ns)
        distance = min(p, self.period_s - p)
        if distance >= self.risk_half_width_s:
            return 0.0
        return 1.0 - distance / self.risk_half_width_s

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.__dict__, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path) -> "PhaseModel":
        return cls(**json.loads(path.read_text()))


def _read_probe(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r["send_ns"]))
    return rows


def _event_times(
    rows: list[dict],
    threshold_ms: float = 50.0,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> list[float]:
    t0 = int(rows[0]["send_ns"]) / 1e9
    times: list[float] = []
    for r in rows:
        t = int(r["send_ns"]) / 1e9 - t0
        if t < start_s:
            continue
        if end_s is not None and t >= end_s:
            continue
        if r.get("lost") or (
            r.get("rtt_ms") is not None and float(r["rtt_ms"]) >= threshold_ms
        ):
            times.append(t)
    return times


def _collapse_bursts(times: list[float], gap_s: float = 0.5) -> list[float]:
    if not times:
        return []
    bursts = [times[0]]
    for t in times[1:]:
        if t - bursts[-1] >= gap_s:
            bursts.append(t)
    return bursts


def _coherence(times: list[float], period_s: float) -> float:
    if not times:
        return 0.0
    z = sum(cmath.exp(2j * math.pi * t / period_s) for t in times) / len(times)
    return abs(z)


def search_period(
    trace_path: Path,
    min_period_s: float = 10.0,
    max_period_s: float = 20.0,
    step_s: float = 0.001,
    threshold_ms: float = 50.0,
) -> tuple[float, float]:
    rows = _read_probe(trace_path)
    bursts = _collapse_bursts(_event_times(rows, threshold_ms=threshold_ms))
    if not bursts:
        raise ValueError("no disruption bursts found")
    best_period = min_period_s
    best_score = -1.0
    p = min_period_s
    while p <= max_period_s + step_s / 2:
        score = _coherence(bursts, p)
        if score > best_score:
            best_period, best_score = p, score
        p += step_s
    return best_period, best_score


def estimate_robust_phase(
    trace_path: Path,
    period_s: float,
    warmup_s: float = 120.0,
    threshold_ms: float = 50.0,
    tolerance_s: float = 0.2,
) -> tuple[float, int, int]:
    rows = _read_probe(trace_path)
    bursts = _collapse_bursts(
        _event_times(rows, threshold_ms=threshold_ms, start_s=0.0, end_s=warmup_s)
    )
    if not bursts:
        raise ValueError("no warm-up disruption bursts found")
    phases = [t % period_s for t in bursts]

    def circ_dist(a: float, b: float) -> float:
        d = (a - b) % period_s
        if d > period_s / 2:
            d -= period_s
        return d

    best_phase = phases[0]
    best_count = -1
    best_error = float("inf")
    for candidate in phases:
        errors = [abs(circ_dist(p, candidate)) for p in phases]
        inside = [e for e in errors if e <= tolerance_s]
        count = len(inside)
        error = sum(inside) / count if count else float("inf")
        if count > best_count or (count == best_count and error < best_error):
            best_phase = candidate
            best_count = count
            best_error = error

    cluster = [p for p in phases if abs(circ_dist(p, best_phase)) <= tolerance_s]
    offsets = [circ_dist(p, best_phase) for p in cluster]
    phase = (best_phase + sum(offsets) / len(offsets)) % period_s
    return phase, len(cluster), len(phases)


def phase_model_from_warmup(
    trace_path: Path,
    period_s: float,
    warmup_s: float = 120.0,
    threshold_ms: float = 50.0,
    tolerance_s: float = 0.2,
    risk_half_width_s: float = 0.2,
) -> tuple[PhaseModel, dict]:
    rows = _read_probe(trace_path)
    phase, inliers, total = estimate_robust_phase(
        trace_path,
        period_s=period_s,
        warmup_s=warmup_s,
        threshold_ms=threshold_ms,
        tolerance_s=tolerance_s,
    )
    t0 = int(rows[0]["send_ns"]) / 1e9
    boundary = t0 + phase
    model = PhaseModel(
        period_s=period_s,
        boundary_offset_s=boundary,
        risk_half_width_s=risk_half_width_s,
        score_threshold_ms=threshold_ms,
    )
    meta = {
        "phase_s": phase,
        "warmup_s": warmup_s,
        "phase_inliers": inliers,
        "warmup_bursts": total,
    }
    return model, meta


def calibrate_phase(
    trace_path: Path,
    period_s: float = 15.0,
    bins: int = 150,
    risk_half_width_s: float = 0.35,
) -> PhaseModel:
    rows = _read_probe(trace_path)
    if len(rows) < 20:
        raise ValueError("need at least 20 probe observations")

    valid_rtts = sorted(float(r["rtt_ms"]) for r in rows if r.get("rtt_ms") is not None)
    if not valid_rtts:
        raise ValueError("trace contains no received probes")
    q_index = max(0, min(len(valid_rtts) - 1, int(0.9 * (len(valid_rtts) - 1))))
    threshold = valid_rtts[q_index]

    bad_times = [
        int(r["send_ns"]) / 1e9
        for r in rows
        if r.get("lost") or (r.get("rtt_ms") is not None and float(r["rtt_ms"]) >= threshold)
    ]
    if not bad_times:
        raise ValueError("no bad events found")

    step = period_s / bins
    best_offset = 0.0
    best_score = -1.0
    sigma = max(risk_half_width_s, step)
    for i in range(bins):
        offset = i * step
        score = 0.0
        for t in bad_times:
            phase = (t - offset) % period_s
            d = min(phase, period_s - phase)
            score += math.exp(-0.5 * (d / sigma) ** 2)
        if score > best_score:
            best_score = score
            best_offset = offset

    first_t = int(rows[0]["send_ns"]) / 1e9
    k = round((first_t - best_offset) / period_s)
    boundary = best_offset + k * period_s
    return PhaseModel(
        period_s=period_s,
        boundary_offset_s=boundary,
        risk_half_width_s=risk_half_width_s,
        score_threshold_ms=threshold,
    )
