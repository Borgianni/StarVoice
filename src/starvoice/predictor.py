from __future__ import annotations

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
    return rows


def calibrate_phase(
    trace_path: Path,
    period_s: float = 15.0,
    bins: int = 150,
    risk_half_width_s: float = 0.35,
) -> PhaseModel:
    """Estimate a recurring disruption boundary from probe RTT/loss observations.

    The estimator searches phase offsets and maximizes concentration of bad events
    near phase zero. It is deliberately transparent and deterministic; it is a P0
    baseline, not a claim that Starlink always has an exact 15 s clock.
    """
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

    # Absolute monotonic alignment: choose an equivalent boundary near first sample.
    first_t = int(rows[0]["send_ns"]) / 1e9
    k = round((first_t - best_offset) / period_s)
    boundary = best_offset + k * period_s
    return PhaseModel(
        period_s=period_s,
        boundary_offset_s=boundary,
        risk_half_width_s=risk_half_width_s,
        score_threshold_ms=threshold,
    )
