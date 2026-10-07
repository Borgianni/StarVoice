from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .counterfactual import _packet_log_from_output
from .deadline import timely_fec_opportunity_slots
from .foresight import _sorted_log
from .predictor import phase_model_from_warmup, search_period


FEATURES = (
    "phase_risk",
    "last_rtt_ms",
    "last_rtt_age_ms",
    "rtt_max_100ms",
    "rtt_mean_100ms",
    "unanswered_100ms",
    "rtt_max_250ms",
    "rtt_mean_250ms",
    "unanswered_250ms",
    "rtt_max_500ms",
    "rtt_mean_500ms",
    "unanswered_500ms",
)


@dataclass
class LinearRiskModel:
    feature_names: tuple[str, ...]
    means: list[float]
    scales: list[float]
    weights: list[float]
    intercept: float

    def score(self, features: dict[str, float]) -> float:
        z = [
            (float(features[name]) - mean) / scale
            for name, mean, scale in zip(
                self.feature_names, self.means, self.scales
            )
        ]
        value = self.intercept + sum(w * x for w, x in zip(self.weights, z))
        if value >= 0:
            e = math.exp(-value)
            return 1.0 / (1.0 + e)
        e = math.exp(value)
        return e / (1.0 + e)

    def as_dict(self) -> dict:
        return {
            "feature_names": list(self.feature_names),
            "means": self.means,
            "scales": self.scales,
            "weights": self.weights,
            "intercept": self.intercept,
        }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _trace_rows(path: Path) -> list[dict]:
    rows = _read_jsonl(path)
    rows.sort(key=lambda r: int(r["send_ns"]))
    return rows


def _causal_features_for_indices(
    trace_rows: list[dict],
    indices_and_send: list[tuple[int, int]],
    phase_model,
) -> list[dict[str, float]]:
    received = sorted(
        (
            (int(r["recv_ns"]), float(r["rtt_ms"]))
            for r in trace_rows
            if r.get("recv_ns") is not None and r.get("rtt_ms") is not None
        ),
        key=lambda x: x[0],
    )
    recv_index = 0
    observed: deque[tuple[int, float]] = deque()
    max_window_ns = 500_000_000
    result: list[dict[str, float]] = []

    for trace_index, now in indices_and_send:
        while recv_index < len(received) and received[recv_index][0] <= now:
            observed.append(received[recv_index])
            recv_index += 1
        while observed and observed[0][0] < now - max_window_ns:
            observed.popleft()

        feats: dict[str, float] = {
            "phase_risk": float(phase_model.risk(now)),
            "last_rtt_ms": observed[-1][1] if observed else 0.0,
            "last_rtt_age_ms": (
                (now - observed[-1][0]) / 1_000_000 if observed else 1000.0
            ),
        }

        for window_ms in (100, 250, 500):
            cutoff = now - window_ms * 1_000_000
            vals = [v for t, v in observed if t >= cutoff]
            feats[f"rtt_max_{window_ms}ms"] = max(vals) if vals else 0.0
            feats[f"rtt_mean_{window_ms}ms"] = (
                sum(vals) / len(vals) if vals else 0.0
            )

            unanswered = 0
            j = trace_index - 1
            while j >= 0 and int(trace_rows[j]["send_ns"]) >= cutoff:
                recv_ns = trace_rows[j].get("recv_ns")
                if recv_ns is None or int(recv_ns) > now:
                    unanswered += 1
                j -= 1
            feats[f"unanswered_{window_ms}ms"] = float(unanswered)

        result.append(feats)

    return result


def _network_timely_labels(
    trace_rows: list[dict],
    playout_ms: float,
    rtt_fraction: float,
) -> list[int]:
    labels = [0] * len(trace_rows)
    playout_ns = int(round(playout_ms * 1_000_000))
    for i in range(len(trace_rows) - 1):
        current = trace_rows[i]
        nxt = trace_rows[i + 1]
        if not bool(current.get("lost")):
            continue
        if bool(nxt.get("lost")) or nxt.get("rtt_ms") is None:
            continue
        deadline_ns = int(current["send_ns"]) + playout_ns
        next_arrival = int(nxt["send_ns"]) + int(
            round(float(nxt["rtt_ms"]) * rtt_fraction * 1_000_000)
        )
        if next_arrival <= deadline_ns:
            labels[i] = 1
    return labels


def _fit_logistic(
    xs: list[list[float]],
    ys: list[int],
    iterations: int = 400,
    learning_rate: float = 0.05,
    l2: float = 0.05,
) -> LinearRiskModel:
    if not xs or len(xs) != len(ys):
        raise ValueError("invalid training examples")
    positives = sum(ys)
    negatives = len(ys) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("training labels need both classes")

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "analyze-causal-fusion requires numpy; install the analysis extras"
        ) from exc

    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales = np.maximum(scales, 1e-6)
    z = (x - means) / scales

    weights = np.zeros(z.shape[1], dtype=np.float64)
    intercept = 0.0
    positive_weight = negatives / positives
    sample_weights = np.where(y > 0.5, positive_weight, 1.0)
    weight_sum = float(sample_weights.sum())

    for _ in range(iterations):
        linear = intercept + z @ weights
        p = np.empty_like(linear)
        pos = linear >= 0
        p[pos] = 1.0 / (1.0 + np.exp(-linear[pos]))
        exp_linear = np.exp(linear[~pos])
        p[~pos] = exp_linear / (1.0 + exp_linear)

        error = (p - y) * sample_weights
        grad_b = float(error.sum() / weight_sum)
        grad_w = (z.T @ error) / weight_sum + l2 * weights

        intercept -= learning_rate * grad_b
        weights -= learning_rate * grad_w

    return LinearRiskModel(
        feature_names=FEATURES,
        means=[float(v) for v in means],
        scales=[float(v) for v in scales],
        weights=[float(v) for v in weights],
        intercept=float(intercept),
    )

def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def run_causal_fusion_experiment(
    calibration_trace: Path,
    benchmark_results: Path,
    output: Path,
    playout_ms: float = 60.0,
    rtt_fraction: float = 0.5,
    warmup_s: float = 120.0,
    risk_half_width_s: float = 0.2,
    threshold_ms: float = 50.0,
) -> dict:
    """Train only on calibration trace; evaluate zero-shot on frozen benchmark traces."""
    calibration_rows = _trace_rows(calibration_trace)
    period_s, coherence = search_period(
        calibration_trace,
        threshold_ms=threshold_ms,
    )
    calibration_phase, _ = phase_model_from_warmup(
        calibration_trace,
        period_s=period_s,
        warmup_s=warmup_s,
        threshold_ms=threshold_ms,
        risk_half_width_s=risk_half_width_s,
    )

    warmup_end = int(calibration_rows[0]["send_ns"] + warmup_s * 1e9)
    train_indices = [
        (i, int(r["send_ns"]))
        for i, r in enumerate(calibration_rows)
        if int(r["send_ns"]) >= warmup_end
    ]
    train_features = _causal_features_for_indices(
        calibration_rows, train_indices, calibration_phase
    )
    all_labels = _network_timely_labels(
        calibration_rows,
        playout_ms=playout_ms,
        rtt_fraction=rtt_fraction,
    )
    train_labels = [all_labels[i] for i, _ in train_indices]
    train_x = [[f[name] for name in FEATURES] for f in train_features]
    model = _fit_logistic(train_x, train_labels)

    benchmark = _read_jsonl(benchmark_results)
    by_condition: dict[tuple[str, str], dict[str, dict]] = {}
    for row in benchmark:
        key = (row["trace_name"], row["utterance_id"])
        by_condition.setdefault(key, {})[row["policy"]] = row

    trace_cache: dict[Path, list[dict]] = {}
    phase_cache = {}
    slots: list[dict] = []
    current_budget = 0
    timely_opportunities = 0

    for key, policies in sorted(by_condition.items()):
        pred = policies.get("predictive-fec")
        always = policies.get("always-fec")
        if pred is None or always is None:
            continue

        trace = Path(pred["trace"])
        if trace not in trace_cache:
            trace_cache[trace] = _trace_rows(trace)
            phase_cache[trace], _ = phase_model_from_warmup(
                trace,
                period_s=period_s,
                warmup_s=warmup_s,
                threshold_ms=threshold_ms,
                risk_half_width_s=risk_half_width_s,
            )

        trace_rows = trace_cache[trace]
        phase_model = phase_cache[trace]
        pred_log = _sorted_log(_packet_log_from_output(pred["output_wav"]))
        always_log = _sorted_log(_packet_log_from_output(always["output_wav"]))
        opportunities = timely_fec_opportunity_slots(
            always_log,
            trace_rows,
            playout_ms=playout_ms,
            rtt_fraction=rtt_fraction,
        )
        indices = [
            (int(r["source_trace_index"]), int(r["send_ns"]))
            for r in pred_log
        ]
        features = _causal_features_for_indices(trace_rows, indices, phase_model)

        for row, feats in zip(pred_log, features):
            seq = int(row["sequence"])
            slots.append(
                {
                    "trace_name": key[0],
                    "utterance_id": key[1],
                    "sequence": seq,
                    "opportunity": seq in opportunities,
                    "phase_risk": float(row.get("risk", 0.0)),
                    "fused_risk": model.score(feats),
                }
            )

        current_budget += sum(bool(r.get("fec_enabled")) for r in pred_log)
        timely_opportunities += len(opportunities)

    def topk(feature: str) -> dict:
        ranked = sorted(
            slots,
            key=lambda x: (
                -float(x[feature]),
                str(x["trace_name"]),
                str(x["utterance_id"]),
                int(x["sequence"]),
            ),
        )
        selected = ranked[:current_budget]
        hits = sum(bool(x["opportunity"]) for x in selected)
        return {
            "feature": feature,
            "protected_frames": len(selected),
            "hits": hits,
            "recall": hits / timely_opportunities if timely_opportunities else None,
        }

    phase = topk("phase_risk")
    fused = topk("fused_risk")
    random_expected = (
        timely_opportunities * current_budget / len(slots) if slots else 0.0
    )

    weights = [
        {"feature": name, "weight": weight}
        for name, weight in zip(model.feature_names, model.weights)
    ]
    weights.sort(key=lambda x: -abs(float(x["weight"])))

    summary = {
        "schema_version": 1,
        "experiment": "calibration-trained causal feature fusion, frozen-trace evaluation",
        "calibration_trace": str(calibration_trace),
        "benchmark_results": str(benchmark_results),
        "training": {
            "warmup_s": warmup_s,
            "examples": len(train_labels),
            "positive_timely_opportunities": sum(train_labels),
            "positive_rate": sum(train_labels) / len(train_labels),
            "period_s": period_s,
            "period_coherence": coherence,
        },
        "evaluation": {
            "playout_ms": playout_ms,
            "rtt_fraction": rtt_fraction,
            "frames": len(slots),
            "timely_opportunities": timely_opportunities,
            "protected_frames": current_budget,
            "effective_duty_cycle": current_budget / len(slots) if slots else None,
            "random_expected_hits": random_expected,
            "phase_only": phase,
            "fused": fused,
            "absolute_hit_gain": fused["hits"] - phase["hits"],
        },
        "model": {
            **model.as_dict(),
            "weights_by_absolute_magnitude": weights,
        },
        "causality": (
            "Weights and normalization are fit only on the calibration trace after "
            "its warm-up. Frozen validation traces contribute no labels or fitted "
            "parameters. Test-trace phase is estimated only from each trace's "
            "warm-up, matching the deployable phase baseline."
        ),
        "target": (
            "Network timely-FEC opportunity: current packet is lost and the next "
            "received packet's RTT-derived arrival is within the current frame's "
            "playout deadline. Codec actuation is evaluated separately."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
