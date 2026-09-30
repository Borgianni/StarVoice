from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .probe import run_probe, summarize
from .predictor import PhaseModel
from .speech import speech_loop
from .storage import dump_json, environment_snapshot, new_run_dir, sha256_file


@dataclass
class CampaignResult:
    campaign_dir: Path
    runs: list[Path]


def load_campaign(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("campaign YAML must contain a mapping")
    if data.get("schema_version") != 1:
        raise ValueError("unsupported campaign schema_version")
    return data


def execute_campaign(
    campaign_path: Path,
    target: str,
    site: str,
    output_root: Path,
) -> CampaignResult:
    cfg = load_campaign(campaign_path)
    seed = int(cfg.get("seed", 1))
    rng = random.Random(seed)
    campaign_dir = new_run_dir(output_root, site, cfg.get("name", "campaign"))
    (campaign_dir / "campaign.source.yaml").write_text(campaign_path.read_text())
    dump_json(campaign_dir / "environment.json", environment_snapshot())

    conditions = copy.deepcopy(cfg.get("conditions", []))
    repeats = int(cfg.get("repeats", 1))
    order: list[dict[str, Any]] = []
    for block in range(repeats):
        block_conditions = copy.deepcopy(conditions)
        rng.shuffle(block_conditions)
        for c in block_conditions:
            c["_block"] = block
            order.append(c)
    dump_json(campaign_dir / "randomization.json", {"seed": seed, "order": order})

    predictor = None
    predictor_path = cfg.get("predictor_model")
    if predictor_path:
        predictor = PhaseModel.load((campaign_path.parent / predictor_path).resolve())

    runs: list[Path] = []
    for i, condition in enumerate(order):
        label = f"{i:03d}_{condition['name']}"
        run_dir = campaign_dir / label
        run_dir.mkdir()
        resolved = {
            "campaign": cfg.get("name"),
            "site": site,
            "target": target,
            "seed": seed,
            "condition": condition,
        }
        (run_dir / "manifest.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=True))

        kind = condition.get("kind", "probe")
        if kind == "probe":
            rows = run_probe(
                target=target,
                duration_s=float(condition.get("duration_s", 60)),
                interval_ms=float(condition.get("interval_ms", 20)),
                timeout_ms=float(condition.get("timeout_ms", 1000)),
                output=run_dir / "network.jsonl",
            )
            summary = summarize(rows)
        elif kind == "speech":
            wav = (campaign_path.parent / condition["input_wav"]).resolve()
            summary = speech_loop(
                input_wav=wav,
                output_wav=run_dir / "received.wav",
                target=target,
                policy=condition["policy"],
                predictor=predictor,
                bitrate=int(condition.get("bitrate", 24000)),
                expected_loss_percent=int(condition.get("expected_loss_percent", 20)),
                random_duty_cycle=float(condition.get("random_duty_cycle", 0.1)),
                seed=seed + i,
                packet_log=run_dir / "packets.jsonl",
            )
            summary["input_wav_sha256"] = sha256_file(wav)
        else:
            raise ValueError(f"unknown condition kind: {kind}")

        summary.update({"site": site, "condition": condition["name"], "block": condition["_block"]})
        dump_json(run_dir / "summary.json", summary)
        runs.append(run_dir)

    dump_json(
        campaign_dir / "summary.json",
        {"site": site, "campaign": cfg.get("name"), "runs": [r.name for r in runs]},
    )
    return CampaignResult(campaign_dir, runs)
