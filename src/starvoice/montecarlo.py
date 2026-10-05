from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .replay import replay_random_recovery_only


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _quantile(values: list[int], q: float) -> float:
    if not values:
        raise ValueError("empty values")
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(xs[lo])
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def _condition_seed(master_seed: int, mc_index: int, trace_name: str, utterance_id: str) -> int:
    raw = f"{master_seed}|{mc_index}|{trace_name}|{utterance_id}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def run_random_fec_monte_carlo(
    benchmark_results: Path,
    manifest: Path,
    output: Path,
    repetitions: int = 100,
    seed: int = 2027,
    bitrate: int = 24000,
    expected_loss_percent: int = 20,
) -> dict:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    output.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(benchmark_results)
    corpus = {r["utterance_id"]: r for r in _read_jsonl(manifest)}

    predictive = [r for r in rows if r.get("policy") == "predictive-fec"]
    if not predictive:
        raise ValueError("benchmark results contain no predictive-fec rows")

    predictive_recovered = sum(int(r["fec_recovered_frames"]) for r in predictive)
    total_losses = sum(int(r["network_lost_frames"]) for r in predictive)
    total_protected = sum(int(r["fec_protected_frames"]) for r in predictive)
    total_frames = sum(int(r["frames"]) for r in predictive)

    per_rep_path = output / "random-fec-monte-carlo.jsonl"
    recovered_values: list[int] = []

    with per_rep_path.open("w", encoding="utf-8") as f:
        for rep in range(repetitions):
            recovered = 0
            losses = 0
            protected = 0
            frames = 0

            for row in predictive:
                utt = row["utterance_id"]
                item = corpus.get(utt)
                if item is None:
                    raise ValueError(f"utterance missing from manifest: {utt}")

                trace = Path(row["trace"])
                condition_seed = _condition_seed(
                    seed,
                    rep,
                    row["trace_name"],
                    utt,
                )
                result = replay_random_recovery_only(
                    input_wav=Path(item["wav_48k_mono_pcm16"]),
                    trace=trace,
                    protected_frames=int(row["fec_protected_frames"]),
                    seed=condition_seed,
                    trace_offset_frames=int(row["trace_offset_frames"]),
                    bitrate=bitrate,
                    expected_loss_percent=expected_loss_percent,
                )
                recovered += int(result["fec_recovered_frames"])
                losses += int(result["network_lost_frames"])
                protected += int(result["fec_protected_frames"])
                frames += int(result["frames"])

            if losses != total_losses:
                raise RuntimeError(
                    f"loss-mask drift in repetition {rep}: {losses} != {total_losses}"
                )
            if protected != total_protected:
                raise RuntimeError(
                    f"budget drift in repetition {rep}: {protected} != {total_protected}"
                )
            if frames != total_frames:
                raise RuntimeError(
                    f"frame-count drift in repetition {rep}: {frames} != {total_frames}"
                )

            recovered_values.append(recovered)
            rec = {
                "repetition": rep,
                "master_seed": seed,
                "network_lost_frames": losses,
                "fec_recovered_frames": recovered,
                "fec_recovery_rate": recovered / losses if losses else None,
                "fec_protected_frames": protected,
                "frames": frames,
                "fec_duty_cycle": protected / frames if frames else None,
            }
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            f.flush()

    ge_predictive = sum(v >= predictive_recovered for v in recovered_values)
    p_empirical = (ge_predictive + 1) / (repetitions + 1)

    mean = sum(recovered_values) / len(recovered_values)
    var = sum((x - mean) ** 2 for x in recovered_values) / max(1, len(recovered_values) - 1)

    summary = {
        "schema_version": 1,
        "benchmark_results": str(benchmark_results),
        "manifest": str(manifest),
        "repetitions": repetitions,
        "master_seed": seed,
        "total_losses": total_losses,
        "total_frames": total_frames,
        "equal_budget_protected_frames": total_protected,
        "equal_budget_duty_cycle": total_protected / total_frames if total_frames else None,
        "predictive_recovered_frames": predictive_recovered,
        "predictive_recovery_rate": (
            predictive_recovered / total_losses if total_losses else None
        ),
        "random_recovered_mean": mean,
        "random_recovered_std": math.sqrt(var),
        "random_recovered_min": min(recovered_values),
        "random_recovered_q025": _quantile(recovered_values, 0.025),
        "random_recovered_median": _quantile(recovered_values, 0.5),
        "random_recovered_q975": _quantile(recovered_values, 0.975),
        "random_recovered_max": max(recovered_values),
        "random_ge_predictive_count": ge_predictive,
        "empirical_p_random_ge_predictive": p_empirical,
        "per_repetition_jsonl": str(per_rep_path),
        "interpretation": (
            "One-sided Monte Carlo baseline: probability estimated under equal-budget "
            "random placement of obtaining at least the predictive recovery count."
        ),
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
