from __future__ import annotations

import json
import statistics
import wave
from importlib.metadata import version
from pathlib import Path

import numpy as np


POLICIES = ("plain", "always-fec", "random-fec", "predictive-fec", "reactive-fec")
PRIMARY_COMPARISONS = (
    ("predictive-fec", "random-fec"),
    ("predictive-fec", "reactive-fec"),
    ("predictive-fec", "plain"),
    ("always-fec", "predictive-fec"),
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _read_pcm16_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1:
            raise ValueError(f"STOI input must be mono: {path}")
        if w.getsampwidth() != 2:
            raise ValueError(f"STOI input must be PCM16: {path}")
        rate = w.getframerate()
        data = w.readframes(w.getnframes())
    samples = np.frombuffer(data, dtype="<i2").astype(np.float64) / 32768.0
    return samples, rate


def _stoi(reference: Path, degraded: Path) -> tuple[float, int, int]:
    try:
        from pystoi import stoi
    except ImportError as exc:
        raise RuntimeError(
            "pystoi is required for STOI evaluation; install StarVoice analysis extras"
        ) from exc

    clean, clean_rate = _read_pcm16_mono(reference)
    test, test_rate = _read_pcm16_mono(degraded)
    if clean_rate != test_rate:
        raise ValueError(
            f"sample-rate mismatch for {degraded}: {test_rate} != reference {clean_rate}"
        )
    if len(test) < len(clean):
        raise ValueError(
            f"degraded audio is shorter than reference: {degraded} "
            f"({len(test)} < {len(clean)} samples)"
        )

    # replay_speech pads the final Opus frame to 20 ms. STOI requires equal
    # lengths, so remove only this deterministic tail padding.
    padding_samples = len(test) - len(clean)
    test = test[: len(clean)]

    score = float(stoi(clean, test, clean_rate, extended=False))
    return score, len(clean), padding_samples


def _policy_summary(rows: list[dict]) -> dict:
    scores = [float(r["stoi"]) for r in rows]
    return {
        "conditions": len(scores),
        "mean": statistics.fmean(scores) if scores else None,
        "median": statistics.median(scores) if scores else None,
        "min": min(scores) if scores else None,
        "max": max(scores) if scores else None,
    }


def _paired_summary(rows: list[dict], left: str, right: str) -> dict:
    by_key: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        key = (row["trace_name"], row["utterance_id"])
        by_key.setdefault(key, {})[row["policy"]] = float(row["stoi"])

    deltas: list[float] = []
    missing = 0
    for values in by_key.values():
        if left not in values or right not in values:
            missing += 1
            continue
        deltas.append(values[left] - values[right])

    eps = 1e-12
    return {
        "left": left,
        "right": right,
        "pairs": len(deltas),
        "missing_pairs": missing,
        "mean_delta": statistics.fmean(deltas) if deltas else None,
        "median_delta": statistics.median(deltas) if deltas else None,
        "left_better": sum(d > eps for d in deltas),
        "ties": sum(abs(d) <= eps for d in deltas),
        "left_worse": sum(d < -eps for d in deltas),
    }


def run_stoi_evaluation(
    benchmark_results: Path,
    reactive_results: Path,
    manifest: Path,
    output: Path,
    limit: int | None = None,
) -> dict:
    """Evaluate classical STOI on the frozen paired StarVoice replay outputs.

    Two analysis sets are produced by construction:
      * all: every frozen (trace, utterance) condition;
      * loss_exposed: conditions with >=1 network loss, independent of policy.
    """
    benchmark = _read_jsonl(benchmark_results)
    reactive = _read_jsonl(reactive_results)
    corpus = {r["utterance_id"]: r for r in _read_jsonl(manifest)}

    canonical = [r for r in benchmark if r.get("policy") == "predictive-fec"]
    if not canonical:
        raise ValueError("benchmark contains no predictive-fec rows")

    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        allowed = {
            (r["trace_name"], r["utterance_id"])
            for r in canonical[:limit]
        }
        benchmark = [
            r for r in benchmark
            if (r["trace_name"], r["utterance_id"]) in allowed
        ]
        reactive = [
            r for r in reactive
            if (r["trace_name"], r["utterance_id"]) in allowed
        ]
        canonical = canonical[:limit]

    expected_keys = {
        (r["trace_name"], r["utterance_id"]): r
        for r in canonical
    }

    by_key_policy: dict[tuple[str, str, str], dict] = {}
    for row in benchmark + reactive:
        key = (row["trace_name"], row["utterance_id"], row["policy"])
        if key in by_key_policy:
            raise ValueError(f"duplicate condition/policy row: {key}")
        by_key_policy[key] = row

    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.jsonl"
    result_rows: list[dict] = []

    with results_path.open("w", encoding="utf-8") as f:
        for (trace_name, utterance_id), base in sorted(expected_keys.items()):
            item = corpus.get(utterance_id)
            if item is None:
                raise ValueError(f"utterance missing from manifest: {utterance_id}")
            reference = Path(item["wav_48k_mono_pcm16"])
            loss_count = int(base["network_lost_frames"])

            for policy in POLICIES:
                key = (trace_name, utterance_id, policy)
                row = by_key_policy.get(key)
                if row is None:
                    raise ValueError(f"missing policy row: {key}")
                if int(row["network_lost_frames"]) != loss_count:
                    raise RuntimeError(
                        f"loss-mask drift for {trace_name}/{utterance_id}/{policy}"
                    )

                degraded = Path(row["output_wav"])
                score, reference_samples, padding_samples = _stoi(reference, degraded)
                out = {
                    "trace_name": trace_name,
                    "utterance_id": utterance_id,
                    "speaker_id": base["speaker_id"],
                    "policy": policy,
                    "reference_wav": str(reference),
                    "degraded_wav": str(degraded),
                    "network_lost_frames": loss_count,
                    "loss_exposed": loss_count > 0,
                    "fec_recovered_frames": int(row["fec_recovered_frames"]),
                    "fec_protected_frames": int(row["fec_protected_frames"]),
                    "stoi": score,
                    "reference_samples": reference_samples,
                    "trimmed_padding_samples": padding_samples,
                }
                f.write(json.dumps(out, sort_keys=True) + "\n")
                result_rows.append(out)

    subsets = {
        "all": result_rows,
        "loss_exposed": [r for r in result_rows if r["loss_exposed"]],
    }

    summary_subsets: dict[str, dict] = {}
    for subset_name, subset_rows in subsets.items():
        policies = {
            policy: _policy_summary([r for r in subset_rows if r["policy"] == policy])
            for policy in POLICIES
        }
        comparisons = {
            f"{left}_minus_{right}": _paired_summary(subset_rows, left, right)
            for left, right in PRIMARY_COMPARISONS
        }
        condition_count = len(
            {(r["trace_name"], r["utterance_id"]) for r in subset_rows}
        )
        summary_subsets[subset_name] = {
            "conditions": condition_count,
            "policies": policies,
            "paired_comparisons": comparisons,
        }

    max_padding = max((int(r["trimmed_padding_samples"]) for r in result_rows), default=0)
    summary = {
        "schema_version": 1,
        "metric": "STOI",
        "implementation": "pystoi",
        "pystoi_version": version("pystoi"),
        "numpy_version": np.__version__,
        "extended": False,
        "benchmark_results": str(benchmark_results),
        "reactive_results": str(reactive_results),
        "manifest": str(manifest),
        "conditions": len(expected_keys),
        "policies": list(POLICIES),
        "results_jsonl": str(results_path),
        "max_trimmed_tail_padding_samples": max_padding,
        "analysis_sets": summary_subsets,
        "statistical_note": (
            "Paired descriptive summaries only. No iid significance test is reported "
            "because conditions reuse a small number of network traces; clustered "
            "inference is deferred until multi-run/multi-site data are available."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
