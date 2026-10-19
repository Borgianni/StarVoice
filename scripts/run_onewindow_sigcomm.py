#!/usr/bin/env python3
"""One-command, <=114-minute exploratory natural Starlink speech campaign.

Run on the NUC from the separate research worktree. Nine ~10-minute SV02
sessions are one contiguous measurement *campaign*, NOT nine independent days
or sites. Always-FEC captures an observed LBRR carrier feasibility ceiling.
All actual policy comparisons must be paired offline on held-out packet traces.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import subprocess
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

from starvoice.recoverability import analyze

DEFAULT_SHA = "95dae64cdb6d95f87ecf767a8dee8dc4438f67c8"
ORDER = (0, 1, 2, 1, 2, 0, 2, 0, 1)  # Latin-square balance over three epochs
MAX_WALL_SECONDS = 114 * 60


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def prepare_segments(deployment: Path, out: Path, segment_seconds: int = 600) -> list[dict]:
    """Split frozen 30-minute LibriSpeech manifest into 3 complete-utterance WAVs."""
    manifest = deployment / "datasets/starvoice-eval-v1/manifest.jsonl"
    rows = [json.loads(x) for x in manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rows:
        raise ValueError("empty LibriSpeech manifest")
    segments = [{"index": i, "items": [], "samples": 0} for i in range(3)]
    cumulative = 0
    segment = 0
    for row in rows:
        src = deployment / row["wav_48k_mono_pcm16"]
        if not src.is_file() or sha256(src) != row["wav_sha256"]:
            raise ValueError(f"source missing or SHA mismatch: {src}")
        with wave.open(str(src), "rb") as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(),
                wav.getcomptype()) != (48000, 1, 2, "NONE"):
                raise ValueError(f"not 48k/mono/PCM16: {src}")
            n = wav.getnframes()
        if segment < 2 and cumulative >= (segment + 1) * segment_seconds * 48000:
            segment += 1
        segments[segment]["items"].append({"id": row["utterance_id"], "wav": str(src),
                                           "sha256": row["wav_sha256"], "samples": n})
        segments[segment]["samples"] += n
        cumulative += n

    for item in segments:
        if not item["items"]:
            raise ValueError("empty corpus segment")
        wav_out = out / f"segment-{item['index'] + 1:02d}.wav"
        with wave.open(str(wav_out), "wb") as dest:
            dest.setnchannels(1)
            dest.setsampwidth(2)
            dest.setframerate(48000)
            for source in item["items"]:
                with wave.open(source["wav"], "rb") as wav:
                    while True:
                        chunk = wav.readframes(8192)
                        if not chunk:
                            break
                        dest.writeframesraw(chunk)
        item["path"] = str(wav_out)
        item["duration_s"] = item["samples"] / 48000
        item["sha256"] = sha256(wav_out)
    return segments


def call(cmd: list[str], *, cwd: Path | None = None, timeout: float = 30) -> str:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                       check=False)
    if p.returncode:
        raise RuntimeError(f"command returned {p.returncode}: {cmd!r}\n{p.stdout}\n{p.stderr}")
    return p.stdout.strip()


def ssh_prefix(args) -> list[str]:
    return ["ssh", "-i", str(args.ssh_key), "-o", "IdentitiesOnly=yes",
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
            "-o", "ServerAliveInterval=30", args.oracle_host]


def remote(args, command: str, timeout: float = 30) -> str:
    return call(ssh_prefix(args) + [command], timeout=timeout)


def capacity_kib(args) -> tuple[int, int]:
    """Oracle tmpfs and available system memory (both KiB)."""
    cmd = ("awk '/^MemAvailable:/ {print $2}' /proc/meminfo; "
           "df -Pk /dev/shm | awk 'NR==2 {print $4}'")
    output = remote(args, cmd).splitlines()
    if len(output) != 2:
        raise ValueError("unexpected Oracle capacity output")
    return int(output[0]), int(output[1])


def verify_preflight(args) -> tuple[str, str]:
    if not args.ssh_key.is_file():
        raise FileNotFoundError(f"NUC SSH key missing: {args.ssh_key}")
    deployment = args.deployment_root.resolve()
    expected = args.expected_sha
    local_sha = call(["git", "rev-parse", "HEAD"], cwd=deployment)
    oracle_sha = remote(args, "cd ~/StarVoice && git rev-parse HEAD")
    if local_sha != expected or oracle_sha != expected:
        raise RuntimeError(
            f"deployed phase2 code mismatch: expected={expected}, "
            f"NUC={local_sha}, Oracle={oracle_sha}")
    remote(args, "cd ~/StarVoice && . .venv/bin/activate && command -v starvoice && command -v timeout")
    avail, tmpfs = capacity_kib(args)
    print(f"Preflight: deployed code={expected[:9]}; Oracle MemAvailable={avail//1024} MiB; "
          f"tmpfs free={tmpfs//1024} MiB", flush=True)
    if avail < 200 * 1024 or tmpfs < 200 * 1024:
        raise RuntimeError("Oracle memory/tmpfs under 200 MiB before session")
    return local_sha, oracle_sha


def acquire(args, campaign: Path, block_index: int, segment: dict, seconds: float) -> dict:
    prefix = f"block-{block_index + 1:02d}"
    block = campaign / prefix
    block.mkdir()
    remote_output = f"results/{campaign.name}/{prefix}/receiver"
    local_sender = block / "sender"
    target = args.oracle_target

    command = (
        "cd ~/StarVoice && . .venv/bin/activate && exec "
        f"timeout --signal=INT --kill-after=10s {math.ceil(seconds + 150)} "
        "starvoice receive-speech --bind 0.0.0.0 --port 5005 "
        "--playout-buffer-ms 60 --idle-timeout-s 600 --clock-interval 10 "
        f"--storage-mode ram --output-dir {shlex.quote(remote_output)}"
    )
    started = datetime.now(timezone.utc).isoformat()
    rxlog = block / "receiver-process.log"
    with rxlog.open("w", encoding="utf-8") as logfile:
        receiver = subprocess.Popen(ssh_prefix(args) + [command],
                                    stdout=logfile, stderr=subprocess.STDOUT)
        try:
            time.sleep(2.5)
            if receiver.poll() is not None:
                raise RuntimeError(f"Oracle receiver failed to start; inspect {rxlog}")
            sender_cmd = [
                str(args.deployment_root / ".venv/bin/starvoice"), "send-speech",
                "--target", target, "--input", str(Path(segment["path"]).resolve()),
                "--output-dir", str(local_sender), "--policy", "always-fec",
                "--bitrate", "24000", "--expected-loss", "20",
                "--clock-interval", "10", "--storage-mode", "ram",
            ]
            with (block / "sender-process.log").open("w", encoding="utf-8") as sender_log:
                tx = subprocess.run(sender_cmd, cwd=args.deployment_root,
                                    stdout=sender_log, stderr=subprocess.STDOUT,
                                    timeout=seconds + 150, check=False)
            if tx.returncode:
                raise RuntimeError(f"sender exit {tx.returncode}; check {block/'sender-process.log'}")
            rc = receiver.wait(timeout=90)
            if rc:
                raise RuntimeError(f"Oracle receiver exit {rc}; check {rxlog}")
        finally:
            if receiver.poll() is None:
                receiver.terminate()
                try:
                    receiver.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    receiver.kill()
                    receiver.wait(timeout=5)

    # No receiver output on the NUC unless the remote archival fully completed.
    scp = [
        "scp", "-i", str(args.ssh_key), "-o", "IdentitiesOnly=yes",
        "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", "-r",
        f"{args.oracle_host}:StarVoice/{remote_output}", str(block),
    ]
    with (block / "scp.log").open("w", encoding="utf-8") as scplog:
        copied = subprocess.run(scp, stdout=scplog, stderr=subprocess.STDOUT,
                                timeout=180, check=False)
    if copied.returncode:
        raise RuntimeError(f"receiver copy exit {copied.returncode}; inspect {block/'scp.log'}")

    with (block / "audit.json").open("w", encoding="utf-8") as audit_out:
        audited = subprocess.run([str(args.deployment_root / ".venv/bin/starvoice"),
                                  "analyze-session", "--session-dir", str(block)],
                                 cwd=args.deployment_root,
                                 capture_output=True, text=True, timeout=90, check=False)
        audit_out.write(audited.stdout)
        if audited.returncode:
            audit_out.write(audited.stderr)
            raise RuntimeError(f"auditor failed on {prefix}")
    audit = json.loads(audited.stdout)
    if not audit["valid"] or audit["errors"]:
        raise RuntimeError(f"causal audit failed on {prefix}: {audit['errors']}")

    feasibility, _events = analyze(block, (40, 50, 60, 80, 100))
    atomic_json(block / "recoverability.json", feasibility)
    recorded = next(b for b in feasibility["budgets"] if b["buffer_ms"] == 60)
    if recorded["observed_decision_disagreements_at_recorded_buffer"]:
        raise RuntimeError(f"recoverability audit mismatch on {prefix}")

    result = {
        "block": prefix,
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "workload_segment": segment["index"] + 1,
        "workload_sha256": segment["sha256"],
        "duration_s": seconds,
        "session_id": audit["session_id"],
        "frames": audit["frames"],
        "primary": audit["primary"],
        "fec": audit["fec"],
        "plc": audit["plc"],
        "injected_drops": audit["injected_drops"],
        "dispatch_p99_ms": audit["timing"].get("dispatch_p99_ms"),
        "dispatch_max_ms": audit["timing"].get("dispatch_max_ms"),
        "primary_failures_60ms": recorded["counts"].get("primary_failures", 0),
        "timely_real_lbrr_opportunities_60ms":
            recorded["counts"].get("eligible_lbrr_upper_bound", 0),
        "source_hashes": feasibility["input_sha256"],
    }
    atomic_json(block / "run-summary.json", result)
    return result


def run(args) -> None:
    args.deployment_root = args.deployment_root.expanduser().resolve()
    args.ssh_key = args.ssh_key.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    local_sha, oracle_sha = verify_preflight(args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    campaign = args.output_root / f"sigcomm-onewindow-{stamp}"
    campaign.mkdir(parents=True, exist_ok=False)
    segments_dir = campaign / "workloads"
    segments_dir.mkdir()
    segments = prepare_segments(args.deployment_root, segments_dir)
    planned = sum(segments[i]["duration_s"] for i in ORDER)
    print(f"Campaign {campaign}: {len(ORDER)} blocks, {planned/60:.1f} minutes of voice "
          f"(plus transfer/audit overhead), hard start gate <114 minutes", flush=True)
    if planned > 104 * 60:
        raise RuntimeError("workload too long to fit safe 114-minute campaign limit")
    route = call(["ip", "route", "get", args.oracle_target.rsplit(":", 1)[0]])
    manifest = {
        "schema_version": 1, "purpose": "exploratory_onewindow_recoverability_feasibility",
        "nuc_route_to_oracle_at_start": route,
        "single_window_not_independent_days": True,
        "limits": ["one site", "one day", "receiver-relative deadline",
                   "no direct comparison between simultaneously observed alternative policies"],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "deployed_nuc_sha": local_sha, "deployed_oracle_sha": oracle_sha,
        "research_sha": call(["git", "rev-parse", "HEAD"],
                             cwd=Path(__file__).resolve().parents[2]),
        "sender_policy": "always-fec", "injected_drops": 0,
        "codec": {"frame_ms": 20, "bitrate": 24000, "expected_loss_percent": 20},
        "buffer_recorded_ms": 60, "sensitivity_buffers_ms": [40, 50, 60, 80, 100],
        "order": list(ORDER), "segments": segments,
        "blocks": [], "status": "running",
    }
    atomic_json(campaign / "campaign-manifest.json", manifest)
    clock0 = time.monotonic()
    try:
        for index, segment_id in enumerate(ORDER):
            sec = segments[segment_id]["duration_s"]
            if time.monotonic() - clock0 + sec + 120 > MAX_WALL_SECONDS:
                raise TimeoutError("114-minute campaign limit: refusing new block")
            avail, tmpfs = capacity_kib(args)
            if avail < 200 * 1024 or tmpfs < 200 * 1024:
                raise RuntimeError(
                    f"Oracle resources too low before block {index+1}: "
                    f"RAM available={avail/1024:.1f} MiB, tmpfs={tmpfs/1024:.1f} MiB")
            print(f"[{index+1}/9] segment={segment_id+1} ({sec/60:.2f}m) "
                  f"oracle RAM={avail//1024}MiB tmpfs={tmpfs//1024}MiB", flush=True)
            result = acquire(args, campaign, index, segments[segment_id], sec)
            manifest["blocks"].append(result)
            atomic_json(campaign / "campaign-manifest.json", manifest)
            print(f"[{index+1}/9] PASS frames={result['frames']} "
                  f"primary_fail_60ms={result['primary_failures_60ms']} "
                  f"timely_LBRR={result['timely_real_lbrr_opportunities_60ms']} "
                  f"FEC={result['fec']} PLC={result['plc']}", flush=True)
    except BaseException as exc:
        manifest["status"] = "stopped_with_error"
        manifest["error"] = repr(exc)
        atomic_json(campaign / "campaign-manifest.json", manifest)
        print(f"STOP: {exc}; completed block artifacts retained at {campaign}", flush=True)
        raise
    manifest["status"] = "complete"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(campaign / "campaign-manifest.json", manifest)
    columns = ["block", "workload_segment", "frames", "primary", "fec", "plc",
               "primary_failures_60ms", "timely_real_lbrr_opportunities_60ms",
               "dispatch_p99_ms", "dispatch_max_ms", "session_id"]
    with (campaign / "blocks.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for block in manifest["blocks"]:
            writer.writerow({k: block.get(k) for k in columns})
    print(f"CAMPAIGN COMPLETE: {campaign}", flush=True)
    print(f"Block results: {campaign/'blocks.csv'}", flush=True)
    print("WARNING: nine contiguous blocks are not independent days or external-site validation.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-root", type=Path, default=Path("../StarVoice"))
    parser.add_argument("--output-root", type=Path, default=Path("../StarVoice/results"))
    parser.add_argument("--ssh-key", type=Path, default=Path("~/.ssh/starvoice_nuc_oracle"))
    parser.add_argument("--oracle-host", default="ubuntu@89.168.29.239")
    parser.add_argument("--oracle-target", default="89.168.29.239:5005")
    parser.add_argument("--expected-sha", default=DEFAULT_SHA)
    parser.add_argument("--run", action="store_true", help="launch the full campaign")
    args = parser.parse_args()
    if not args.run:
        parser.error("use --run to launch the campaign (it lasts approximately 95 minutes)")
    run(args)


if __name__ == "__main__":
    main()
