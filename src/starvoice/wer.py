from __future__ import annotations

import hashlib
import json
import statistics
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from huggingface_hub import snapshot_download
from jiwer import process_words
from whisper_normalization import EnglishTextNormalizer


POLICIES = ("plain", "always-fec", "random-fec", "predictive-fec", "reactive-fec")
MODEL_ID = "Systran/faster-whisper-small.en"
MODEL_REVISION = "4e49ce629e3fa4c3da596c602b212cb026910443"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _score(reference: str, hypothesis: str) -> dict:
    result = process_words(reference, hypothesis)
    reference_words = result.hits + result.substitutions + result.deletions
    errors = result.substitutions + result.deletions + result.insertions
    return {
        "wer": float(result.wer),
        "reference_words": int(reference_words),
        "errors": int(errors),
        "substitutions": int(result.substitutions),
        "deletions": int(result.deletions),
        "insertions": int(result.insertions),
        "hits": int(result.hits),
    }


@dataclass(frozen=True)
class ASRConfig:
    model_id: str
    model_revision: str
    device: str
    compute_type: str
    cpu_threads: int
    beam_size: int

    def cache_namespace(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


class ASRRunner:
    def __init__(
        self,
        output: Path,
        model_cache: Path,
        config: ASRConfig,
    ) -> None:
        from faster_whisper import WhisperModel

        self.output = output
        self.config = config
        self.normalizer = EnglishTextNormalizer()
        self.cache_path = output / "asr-cache.jsonl"
        self.cache: dict[str, dict] = {}
        self.transcriptions_done = 0

        if self.cache_path.exists():
            for row in _read_jsonl(self.cache_path):
                self.cache[row["cache_key"]] = row

        snapshot = Path(
            snapshot_download(
                repo_id=config.model_id,
                revision=config.model_revision,
                cache_dir=str(model_cache),
            )
        )
        self.model_snapshot = snapshot
        self.model_hashes = {
            name: _sha256(snapshot / name)
            for name in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt")
            if (snapshot / name).exists()
        }
        self.model = WhisperModel(
            str(snapshot),
            device=config.device,
            compute_type=config.compute_type,
            cpu_threads=config.cpu_threads,
            num_workers=1,
        )

    def normalize(self, text: str) -> str:
        return " ".join(self.normalizer(text).split())

    def transcribe(self, path: Path) -> dict:
        audio_sha = _sha256(path)
        cache_key = f"{self.config.cache_namespace()}:{audio_sha}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        segments, info = self.model.transcribe(
            str(path),
            language="en",
            task="transcribe",
            beam_size=self.config.beam_size,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
            word_timestamps=False,
        )
        raw = "".join(segment.text for segment in segments).strip()
        row = {
            "cache_key": cache_key,
            "audio_sha256": audio_sha,
            "audio_path": str(path),
            "raw_text": raw,
            "normalized_text": self.normalize(raw),
            "language": info.language,
        }
        self.output.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        self.cache[cache_key] = row
        self.transcriptions_done += 1
        if self.transcriptions_done % 25 == 0:
            print(
                f"ASR: {self.transcriptions_done} new unique audio files transcribed "
                f"({len(self.cache)} cached total)",
                file=sys.stderr,
                flush=True,
            )
        return row


def _metric_from_row(row: dict, prefix: str) -> dict:
    return {
        "wer": float(row[f"{prefix}_wer"]),
        "reference_words": int(row[f"{prefix}_reference_words"]),
        "errors": int(row[f"{prefix}_errors"]),
        "substitutions": int(row[f"{prefix}_substitutions"]),
        "deletions": int(row[f"{prefix}_deletions"]),
        "insertions": int(row[f"{prefix}_insertions"]),
    }


def _aggregate_metric(rows: list[dict], prefix: str) -> dict:
    values = [_metric_from_row(r, prefix) for r in rows]
    reference_words = sum(v["reference_words"] for v in values)
    errors = sum(v["errors"] for v in values)
    substitutions = sum(v["substitutions"] for v in values)
    deletions = sum(v["deletions"] for v in values)
    insertions = sum(v["insertions"] for v in values)
    wers = [v["wer"] for v in values]
    return {
        "conditions": len(values),
        "reference_words": reference_words,
        "errors": errors,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "micro_wer": errors / reference_words if reference_words else None,
        "mean_condition_wer": statistics.fmean(wers) if wers else None,
        "median_condition_wer": statistics.median(wers) if wers else None,
    }


def _network_excess_summary(rows: list[dict]) -> dict:
    ref_words = sum(int(r["actual_reference_words"]) for r in rows)
    actual_errors = sum(int(r["actual_errors"]) for r in rows)
    codec_errors = sum(int(r["codec_only_errors"]) for r in rows)
    deltas = [float(r["network_excess_wer"]) for r in rows]
    return {
        "conditions": len(rows),
        "reference_words": ref_words,
        "actual_errors": actual_errors,
        "codec_only_errors": codec_errors,
        "excess_errors": actual_errors - codec_errors,
        "micro_excess_wer": (actual_errors - codec_errors) / ref_words if ref_words else None,
        "mean_condition_excess_wer": statistics.fmean(deltas) if deltas else None,
        "median_condition_excess_wer": statistics.median(deltas) if deltas else None,
    }


def _paired(rows: list[dict], field: str, left: str, right: str) -> dict:
    by_condition: dict[tuple[str, str], dict[str, dict]] = {}
    for row in rows:
        key = (row["trace_name"], row["utterance_id"])
        by_condition.setdefault(key, {})[row["policy"]] = row

    deltas: list[float] = []
    error_deltas: list[int] = []
    missing = 0
    for values in by_condition.values():
        if left not in values or right not in values:
            missing += 1
            continue
        l = values[left]
        r = values[right]
        deltas.append(float(l[field]) - float(r[field]))
        if field == "actual_wer":
            error_deltas.append(int(l["actual_errors"]) - int(r["actual_errors"]))
        elif field == "codec_only_wer":
            error_deltas.append(int(l["codec_only_errors"]) - int(r["codec_only_errors"]))
        elif field == "network_excess_wer":
            lerr = int(l["actual_errors"]) - int(l["codec_only_errors"])
            rerr = int(r["actual_errors"]) - int(r["codec_only_errors"])
            error_deltas.append(lerr - rerr)

    eps = 1e-12
    return {
        "left": left,
        "right": right,
        "pairs": len(deltas),
        "missing_pairs": missing,
        "mean_delta": statistics.fmean(deltas) if deltas else None,
        "median_delta": statistics.median(deltas) if deltas else None,
        "left_better": sum(d < -eps for d in deltas),
        "ties": sum(abs(d) <= eps for d in deltas),
        "left_worse": sum(d > eps for d in deltas),
        "total_error_delta": sum(error_deltas),
    }


def run_wer_evaluation(
    benchmark_results: Path,
    reactive_results: Path,
    counterfactual_results: Path,
    manifest: Path,
    output: Path,
    model_cache: Path,
    device: str = "cpu",
    compute_type: str = "float32",
    cpu_threads: int = 4,
    beam_size: int = 5,
    limit: int | None = None,
) -> dict:
    """Run paired ASR/WER evaluation on actual and codec-only outputs.

    Actual outputs contain the recorded network losses. Counterfactual outputs
    preserve each exact FEC schedule but remove network loss. Their WER
    difference therefore isolates the lexical impairment attributable to the
    recorded network condition under each policy.
    """
    output.mkdir(parents=True, exist_ok=True)
    benchmark = _read_jsonl(benchmark_results)
    reactive = _read_jsonl(reactive_results)
    counterfactual = _read_jsonl(counterfactual_results)
    corpus = {r["utterance_id"]: r for r in _read_jsonl(manifest)}

    canonical = [r for r in benchmark if r.get("policy") == "predictive-fec"]
    if not canonical:
        raise ValueError("benchmark contains no predictive-fec rows")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        canonical = canonical[:limit]

    expected = {
        (r["trace_name"], r["utterance_id"]): r
        for r in canonical
    }
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
            raise ValueError(f"duplicate counterfactual row: {key}")
        codec_by_key[key] = row

    config = ASRConfig(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        beam_size=beam_size,
    )
    runner = ASRRunner(output=output, model_cache=model_cache, config=config)

    clean_cache: dict[str, tuple[dict, dict, str]] = {}
    result_rows: list[dict] = []
    results_path = output / "results.jsonl"

    with results_path.open("w", encoding="utf-8") as f:
        for (trace_name, utterance_id), base in sorted(expected.items()):
            item = corpus.get(utterance_id)
            if item is None:
                raise ValueError(f"utterance missing from manifest: {utterance_id}")

            transcript_raw = item["transcript"]
            transcript_norm = runner.normalize(transcript_raw)
            if not transcript_norm:
                raise ValueError(f"empty normalized reference transcript: {utterance_id}")

            if utterance_id not in clean_cache:
                clean_asr = runner.transcribe(Path(item["wav_48k_mono_pcm16"]))
                clean_score = _score(transcript_norm, clean_asr["normalized_text"])
                clean_cache[utterance_id] = (clean_asr, clean_score, transcript_norm)
            clean_asr, clean_score, _ = clean_cache[utterance_id]

            loss_count = int(base["network_lost_frames"])

            for policy in POLICIES:
                key = (trace_name, utterance_id, policy)
                actual = actual_by_key.get(key)
                codec = codec_by_key.get(key)
                if actual is None:
                    raise ValueError(f"missing actual row: {key}")
                if codec is None:
                    raise ValueError(f"missing counterfactual row: {key}")
                if int(actual["network_lost_frames"]) != loss_count:
                    raise RuntimeError(f"loss-mask drift for {key}")

                actual_asr = runner.transcribe(Path(actual["output_wav"]))
                codec_asr = runner.transcribe(Path(codec["counterfactual_wav"]))
                actual_score = _score(transcript_norm, actual_asr["normalized_text"])
                codec_score = _score(transcript_norm, codec_asr["normalized_text"])

                out = {
                    "trace_name": trace_name,
                    "utterance_id": utterance_id,
                    "speaker_id": base["speaker_id"],
                    "policy": policy,
                    "network_lost_frames": loss_count,
                    "loss_exposed": loss_count > 0,
                    "reference_text": transcript_raw,
                    "reference_text_normalized": transcript_norm,
                    "clean_asr_text": clean_asr["raw_text"],
                    "clean_asr_text_normalized": clean_asr["normalized_text"],
                    "actual_asr_text": actual_asr["raw_text"],
                    "actual_asr_text_normalized": actual_asr["normalized_text"],
                    "codec_only_asr_text": codec_asr["raw_text"],
                    "codec_only_asr_text_normalized": codec_asr["normalized_text"],
                    "actual_wav": actual["output_wav"],
                    "codec_only_wav": codec["counterfactual_wav"],
                    "clean_wav": item["wav_48k_mono_pcm16"],
                    **{f"clean_{k}": v for k, v in clean_score.items()},
                    **{f"actual_{k}": v for k, v in actual_score.items()},
                    **{f"codec_only_{k}": v for k, v in codec_score.items()},
                    "network_excess_wer": actual_score["wer"] - codec_score["wer"],
                    "codec_excess_vs_clean_wer": codec_score["wer"] - clean_score["wer"],
                    "actual_excess_vs_clean_wer": actual_score["wer"] - clean_score["wer"],
                }
                f.write(json.dumps(out, sort_keys=True) + "\n")
                f.flush()
                result_rows.append(out)

    analysis_sets: dict[str, dict] = {}
    subsets = {
        "all": result_rows,
        "loss_exposed": [r for r in result_rows if r["loss_exposed"]],
    }
    comparisons = (
        ("predictive-fec", "plain"),
        ("predictive-fec", "random-fec"),
        ("predictive-fec", "reactive-fec"),
        ("always-fec", "predictive-fec"),
    )

    for subset_name, subset_rows in subsets.items():
        policy_summary = {}
        for policy in POLICIES:
            rr = [r for r in subset_rows if r["policy"] == policy]
            policy_summary[policy] = {
                "actual": _aggregate_metric(rr, "actual"),
                "codec_only": _aggregate_metric(rr, "codec_only"),
                "network_excess": _network_excess_summary(rr),
            }

        paired = {}
        for left, right in comparisons:
            paired[f"actual_{left}_minus_{right}"] = _paired(
                subset_rows, "actual_wer", left, right
            )
            paired[f"codec_only_{left}_minus_{right}"] = _paired(
                subset_rows, "codec_only_wer", left, right
            )
            paired[f"network_excess_{left}_minus_{right}"] = _paired(
                subset_rows, "network_excess_wer", left, right
            )

        analysis_sets[subset_name] = {
            "conditions": len(
                {(r["trace_name"], r["utterance_id"]) for r in subset_rows}
            ),
            "policies": policy_summary,
            "paired_comparisons": paired,
        }

    clean_unique = {}
    for row in result_rows:
        clean_unique[row["utterance_id"]] = {
            "clean_wer": row["clean_wer"],
            "clean_reference_words": row["clean_reference_words"],
            "clean_errors": row["clean_errors"],
            "clean_substitutions": row["clean_substitutions"],
            "clean_deletions": row["clean_deletions"],
            "clean_insertions": row["clean_insertions"],
        }
    clean_rows = list(clean_unique.values())
    clean_ref_words = sum(int(r["clean_reference_words"]) for r in clean_rows)
    clean_errors = sum(int(r["clean_errors"]) for r in clean_rows)

    summary = {
        "schema_version": 1,
        "metric": "WER",
        "benchmark_results": str(benchmark_results),
        "reactive_results": str(reactive_results),
        "counterfactual_results": str(counterfactual_results),
        "manifest": str(manifest),
        "conditions": len(expected),
        "policies": list(POLICIES),
        "results_jsonl": str(results_path),
        "asr_cache_jsonl": str(runner.cache_path),
        "asr": {
            **config.__dict__,
            "faster_whisper_version": version("faster-whisper"),
            "ctranslate2_version": version("ctranslate2"),
            "jiwer_version": version("jiwer"),
            "whisper_normalization_version": version("whisper-normalization"),
            "model_file_sha256": runner.model_hashes,
            "language": "en",
            "task": "transcribe",
            "temperature": 0.0,
            "condition_on_previous_text": False,
            "vad_filter": False,
            "num_workers": 1,
        },
        "normalization": "OpenAI Whisper EnglishTextNormalizer via whisper-normalization",
        "clean_source_asr": {
            "unique_utterances": len(clean_rows),
            "reference_words": clean_ref_words,
            "errors": clean_errors,
            "micro_wer": clean_errors / clean_ref_words if clean_ref_words else None,
            "mean_utterance_wer": (
                statistics.fmean(float(r["clean_wer"]) for r in clean_rows)
                if clean_rows
                else None
            ),
        },
        "analysis_sets": analysis_sets,
        "interpretation": (
            "Lower WER is better. codec_only uses the exact frozen FEC schedule with "
            "network loss removed. network_excess_wer = actual_wer - codec_only_wer; "
            "lower values mean less lexical damage attributable to the recorded network."
        ),
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
