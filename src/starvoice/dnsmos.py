from __future__ import annotations

import hashlib
import json
import statistics
import urllib.request
from importlib.metadata import version
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf


POLICIES = ("plain", "always-fec", "random-fec", "predictive-fec", "reactive-fec")
PRIMARY_COMPARISONS = (
    ("predictive-fec", "random-fec"),
    ("predictive-fec", "reactive-fec"),
    ("predictive-fec", "plain"),
    ("always-fec", "predictive-fec"),
)

DNSMOS_MODEL_URL = (
    "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/"
    "DNSMOS/DNSMOS/sig_bak_ovr.onnx"
)
DNSMOS_MODEL_SHA256 = "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd"
DNSMOS_SAMPLE_RATE = 16000
DNSMOS_INPUT_LENGTH_S = 9.01


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_model(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(DNSMOS_MODEL_URL, tmp)
        tmp.replace(path)
    actual = _sha256(path)
    if actual != DNSMOS_MODEL_SHA256:
        raise RuntimeError(
            f"DNSMOS model SHA256 mismatch: {actual} != {DNSMOS_MODEL_SHA256}"
        )
    return path


def _polyfit(sig: float, bak: float, ovr: float) -> tuple[float, float, float]:
    # Official non-personalized DNSMOS P.835 polynomial calibration.
    p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
    p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
    p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
    return float(p_sig(sig)), float(p_bak(bak)), float(p_ovr(ovr))


class DNSMOSScorer:
    def __init__(self, model_path: Path) -> None:
        self.model_path = _ensure_model(model_path)
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

    def score(self, path: Path) -> dict:
        audio, input_sr = sf.read(str(path), always_2d=False)
        if np.asarray(audio).ndim != 1:
            raise ValueError(f"DNSMOS input must be mono: {path}")
        audio = np.asarray(audio, dtype=np.float32)
        if input_sr != DNSMOS_SAMPLE_RATE:
            audio = librosa.resample(
                audio,
                orig_sr=input_sr,
                target_sr=DNSMOS_SAMPLE_RATE,
            ).astype(np.float32, copy=False)

        actual_len = len(audio)
        required = int(DNSMOS_INPUT_LENGTH_S * DNSMOS_SAMPLE_RATE)
        if actual_len == 0:
            raise ValueError(f"empty audio: {path}")
        while len(audio) < required:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / DNSMOS_SAMPLE_RATE) - DNSMOS_INPUT_LENGTH_S) + 1
        raw_sig: list[float] = []
        raw_bak: list[float] = []
        raw_ovr: list[float] = []
        sigs: list[float] = []
        baks: list[float] = []
        ovrs: list[float] = []

        for idx in range(num_hops):
            start = idx * DNSMOS_SAMPLE_RATE
            segment = audio[start : start + required]
            if len(segment) < required:
                continue
            features = segment.astype(np.float32, copy=False)[np.newaxis, :]
            output = self.session.run(None, {"input_1": features})[0][0]
            sig_raw, bak_raw, ovr_raw = map(float, output)
            sig, bak, ovr = _polyfit(sig_raw, bak_raw, ovr_raw)
            raw_sig.append(sig_raw)
            raw_bak.append(bak_raw)
            raw_ovr.append(ovr_raw)
            sigs.append(sig)
            baks.append(bak)
            ovrs.append(ovr)

        if not ovrs:
            raise RuntimeError(f"DNSMOS produced no windows for {path}")

        return {
            "duration_s": actual_len / DNSMOS_SAMPLE_RATE,
            "num_hops": len(ovrs),
            "sig": statistics.fmean(sigs),
            "bak": statistics.fmean(baks),
            "ovrl": statistics.fmean(ovrs),
            "sig_raw": statistics.fmean(raw_sig),
            "bak_raw": statistics.fmean(raw_bak),
            "ovrl_raw": statistics.fmean(raw_ovr),
        }


def _summary(values: list[float]) -> dict:
    return {
        "conditions": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _paired(rows: list[dict], field: str, left: str, right: str) -> dict:
    by_condition: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        key = (row["trace_name"], row["utterance_id"])
        by_condition.setdefault(key, {})[row["policy"]] = float(row[field])

    values = []
    missing = 0
    for policies in by_condition.values():
        if left not in policies or right not in policies:
            missing += 1
            continue
        values.append(policies[left] - policies[right])

    eps = 1e-12
    return {
        "left": left,
        "right": right,
        "pairs": len(values),
        "missing_pairs": missing,
        "mean_delta": statistics.fmean(values) if values else None,
        "median_delta": statistics.median(values) if values else None,
        "left_better": sum(v > eps for v in values),
        "ties": sum(abs(v) <= eps for v in values),
        "left_worse": sum(v < -eps for v in values),
    }


def run_dnsmos_evaluation(
    benchmark_results: Path,
    reactive_results: Path,
    counterfactual_results: Path,
    output: Path,
    model_path: Path,
    limit: int | None = None,
) -> dict:
    """Evaluate official DNSMOS P.835 on actual and codec-only frozen audio."""
    benchmark = _read_jsonl(benchmark_results)
    reactive = _read_jsonl(reactive_results)
    counterfactual = _read_jsonl(counterfactual_results)

    canonical = [r for r in benchmark if r.get("policy") == "predictive-fec"]
    if not canonical:
        raise ValueError("benchmark contains no predictive-fec rows")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        canonical = canonical[:limit]

    expected = {(r["trace_name"], r["utterance_id"]): r for r in canonical}
    allowed = set(expected)

    actual_by_key: dict[tuple[str, str, str], dict] = {}
    for row in benchmark + reactive:
        condition = (row["trace_name"], row["utterance_id"])
        if condition not in allowed:
            continue
        key = (*condition, row["policy"])
        if key in actual_by_key:
            raise ValueError(f"duplicate actual row: {key}")
        actual_by_key[key] = row

    codec_by_key: dict[tuple[str, str, str], dict] = {}
    for row in counterfactual:
        condition = (row["trace_name"], row["utterance_id"])
        if condition not in allowed:
            continue
        key = (*condition, row["policy"])
        if key in codec_by_key:
            raise ValueError(f"duplicate codec-only row: {key}")
        codec_by_key[key] = row

    output.mkdir(parents=True, exist_ok=True)
    scorer = DNSMOSScorer(model_path)
    cache_path = output / "score-cache.jsonl"
    cache: dict[str, dict] = {}
    if cache_path.exists():
        for row in _read_jsonl(cache_path):
            cache[row["audio_sha256"]] = row

    def cached_score(path: Path) -> dict:
        audio_sha = _sha256(path)
        if audio_sha in cache:
            return cache[audio_sha]["score"]
        score = scorer.score(path)
        record = {
            "audio_sha256": audio_sha,
            "audio_path": str(path),
            "score": score,
        }
        with cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        cache[audio_sha] = record
        return score

    results_path = output / "results.jsonl"
    result_rows: list[dict] = []
    with results_path.open("w", encoding="utf-8") as f:
        for (trace_name, utterance_id), base in sorted(expected.items()):
            loss_count = int(base["network_lost_frames"])
            for policy in POLICIES:
                key = (trace_name, utterance_id, policy)
                actual = actual_by_key.get(key)
                codec = codec_by_key.get(key)
                if actual is None:
                    raise ValueError(f"missing actual row: {key}")
                if codec is None:
                    raise ValueError(f"missing codec-only row: {key}")
                if int(actual["network_lost_frames"]) != loss_count:
                    raise RuntimeError(f"loss-mask drift for {key}")

                actual_score = cached_score(Path(actual["output_wav"]))
                codec_score = cached_score(Path(codec["counterfactual_wav"]))
                out = {
                    "trace_name": trace_name,
                    "utterance_id": utterance_id,
                    "speaker_id": base["speaker_id"],
                    "policy": policy,
                    "network_lost_frames": loss_count,
                    "loss_exposed": loss_count > 0,
                    "fec_recovered_frames": int(actual["fec_recovered_frames"]),
                    "fec_protected_frames": int(actual["fec_protected_frames"]),
                    "actual_wav": actual["output_wav"],
                    "codec_only_wav": codec["counterfactual_wav"],
                }
                for metric in ("sig", "bak", "ovrl"):
                    out[f"actual_{metric}"] = actual_score[metric]
                    out[f"codec_only_{metric}"] = codec_score[metric]
                    out[f"network_delta_{metric}"] = (
                        actual_score[metric] - codec_score[metric]
                    )
                f.write(json.dumps(out, sort_keys=True) + "\n")
                f.flush()
                result_rows.append(out)

    analysis: dict[str, dict] = {}
    subsets = {
        "all": result_rows,
        "loss_exposed": [r for r in result_rows if r["loss_exposed"]],
    }
    for subset_name, subset_rows in subsets.items():
        policies: dict[str, dict] = {}
        for policy in POLICIES:
            rr = [r for r in subset_rows if r["policy"] == policy]
            policies[policy] = {
                "actual_ovrl": _summary([float(r["actual_ovrl"]) for r in rr]),
                "codec_only_ovrl": _summary([float(r["codec_only_ovrl"]) for r in rr]),
                "network_delta_ovrl": _summary(
                    [float(r["network_delta_ovrl"]) for r in rr]
                ),
                "actual_sig": _summary([float(r["actual_sig"]) for r in rr]),
                "codec_only_sig": _summary([float(r["codec_only_sig"]) for r in rr]),
                "network_delta_sig": _summary(
                    [float(r["network_delta_sig"]) for r in rr]
                ),
            }

        paired = {}
        for left, right in PRIMARY_COMPARISONS:
            for field in (
                "actual_ovrl",
                "codec_only_ovrl",
                "network_delta_ovrl",
                "actual_sig",
                "codec_only_sig",
                "network_delta_sig",
            ):
                paired[f"{field}_{left}_minus_{right}"] = _paired(
                    subset_rows, field, left, right
                )
        analysis[subset_name] = {
            "conditions": len(
                {(r["trace_name"], r["utterance_id"]) for r in subset_rows}
            ),
            "policies": policies,
            "paired_comparisons": paired,
        }

    summary = {
        "schema_version": 1,
        "metric": "DNSMOS P.835",
        "primary_metric": "OVRL",
        "secondary_metric": "SIG",
        "benchmark_results": str(benchmark_results),
        "reactive_results": str(reactive_results),
        "counterfactual_results": str(counterfactual_results),
        "conditions": len(expected),
        "policies": list(POLICIES),
        "results_jsonl": str(results_path),
        "score_cache_jsonl": str(cache_path),
        "model": {
            "path": str(scorer.model_path),
            "source_url": DNSMOS_MODEL_URL,
            "sha256": _sha256(scorer.model_path),
        },
        "implementation": {
            "sample_rate": DNSMOS_SAMPLE_RATE,
            "input_length_s": DNSMOS_INPUT_LENGTH_S,
            "personalized_mos": False,
            "onnxruntime_version": version("onnxruntime"),
            "librosa_version": version("librosa"),
            "soundfile_version": version("soundfile"),
            "numpy_version": np.__version__,
            "provider": "CPUExecutionProvider",
        },
        "analysis_sets": analysis,
        "interpretation": (
            "Higher DNSMOS is better. network_delta = actual - codec_only; "
            "more negative values indicate more quality loss attributable to "
            "the recorded network condition under the same FEC schedule."
        ),
        "statistical_note": (
            "Paired descriptive summaries only. No iid significance test is reported "
            "because conditions reuse a small number of network traces; clustered "
            "inference is deferred until multi-run/multi-site data are available."
        ),
        "citation": (
            "Reddy et al., DNSMOS P.835: A non-intrusive perceptual objective "
            "speech quality metric to evaluate noise suppressors, ICASSP 2022."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
