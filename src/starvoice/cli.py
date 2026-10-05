from __future__ import annotations

import argparse
import json
from pathlib import Path

from .campaign import execute_campaign
from .dataset import prepare_librispeech
from .environment import run_doctor
from .evaluation import run_codec_benchmark
from .predictor import PhaseModel, calibrate_phase
from .probe import run_probe, summarize
from .relay import run_relay
from .replay import replay_speech
from .report import build_report
from .speech import speech_loop


POLICIES = ["plain", "always-fec", "random-fec", "predictive-fec"]


def _add_policy_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--policy", choices=POLICIES, required=True)
    p.add_argument("--predictor", type=Path)
    p.add_argument("--bitrate", type=int, default=24000)
    p.add_argument("--expected-loss", type=int, default=20)
    p.add_argument("--random-duty-cycle", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--packet-log", type=Path)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="starvoice")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor")

    r = sub.add_parser("relay")
    r.add_argument("--bind", default="0.0.0.0")
    r.add_argument("--port", type=int, default=5005)
    r.add_argument("--log", type=Path)

    q = sub.add_parser("probe")
    q.add_argument("--target", required=True)
    q.add_argument("--duration", type=float, default=60)
    q.add_argument("--interval-ms", type=float, default=20)
    q.add_argument("--timeout-ms", type=float, default=1000)
    q.add_argument("--output", type=Path)

    c = sub.add_parser("calibrate")
    c.add_argument("trace", type=Path)
    c.add_argument("--period", type=float, default=15.0)
    c.add_argument("--risk-half-width", type=float, default=0.35)
    c.add_argument("--output", type=Path, default=Path("phase-model.json"))

    ds = sub.add_parser("dataset")
    ds_sub = ds.add_subparsers(dest="dataset_cmd", required=True)
    lp = ds_sub.add_parser("prepare-librispeech")
    lp.add_argument("--root", type=Path, required=True)
    lp.add_argument("--output", type=Path, required=True)
    lp.add_argument("--target-minutes", type=float, default=30.0)
    lp.add_argument("--seed", type=int, default=2027)
    lp.add_argument("--max-per-speaker", type=int, default=20)

    s = sub.add_parser("speech-loop")
    s.add_argument("--input", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--target", required=True)
    _add_policy_args(s)

    bench = sub.add_parser("benchmark-codec")
    bench.add_argument("--manifest", type=Path, required=True)
    bench.add_argument("--calibration-trace", type=Path, required=True)
    bench.add_argument("--validation-trace", type=Path, action="append", required=True)
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--seed", type=int, default=2027)
    bench.add_argument("--warmup-s", type=float, default=120.0)
    bench.add_argument("--threshold-ms", type=float, default=50.0)
    bench.add_argument("--risk-half-width-s", type=float, default=0.2)
    bench.add_argument("--bitrate", type=int, default=24000)
    bench.add_argument("--expected-loss", type=int, default=20)
    bench.add_argument("--limit", type=int)

    replay = sub.add_parser("replay-speech")
    replay.add_argument("--input", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--trace", type=Path, required=True)
    _add_policy_args(replay)

    run = sub.add_parser("run")
    run.add_argument("campaign", type=Path)
    run.add_argument("--target", required=True)
    run.add_argument("--site", required=True)
    run.add_argument("--output-root", type=Path, default=Path("runs"))

    rep = sub.add_parser("report")
    rep.add_argument("run_dir", type=Path)

    return p


def _predictor(path: Path | None) -> PhaseModel | None:
    return PhaseModel.load(path) if path else None


def main() -> None:
    args = _parser().parse_args()
    if args.cmd == "doctor":
        checks = run_doctor()
        for c in checks:
            print(f"{'OK' if c.ok else 'FAIL':4} {c.name:20} {c.detail}")
        raise SystemExit(0 if all(c.ok for c in checks) else 2)

    if args.cmd == "relay":
        run_relay(args.bind, args.port, args.log)
        return

    if args.cmd == "probe":
        rows = run_probe(args.target, args.duration, args.interval_ms, args.timeout_ms, args.output)
        print(json.dumps(summarize(rows), indent=2, sort_keys=True))
        return

    if args.cmd == "calibrate":
        model = calibrate_phase(args.trace, args.period, risk_half_width_s=args.risk_half_width)
        model.save(args.output)
        print(json.dumps(model.__dict__, indent=2, sort_keys=True))
        return

    if args.cmd == "dataset" and args.dataset_cmd == "prepare-librispeech":
        result = prepare_librispeech(
            root=args.root,
            output=args.output,
            target_minutes=args.target_minutes,
            seed=args.seed,
            max_per_speaker=args.max_per_speaker,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.cmd == "benchmark-codec":
        result = run_codec_benchmark(
            manifest=args.manifest,
            calibration_trace=args.calibration_trace,
            validation_traces=args.validation_trace,
            output=args.output,
            seed=args.seed,
            warmup_s=args.warmup_s,
            threshold_ms=args.threshold_ms,
            risk_half_width_s=args.risk_half_width_s,
            bitrate=args.bitrate,
            expected_loss_percent=args.expected_loss,
            limit=args.limit,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.cmd == "speech-loop":
        result = speech_loop(
            args.input,
            args.output,
            args.target,
            args.policy,
            _predictor(args.predictor),
            args.bitrate,
            args.expected_loss,
            args.random_duty_cycle,
            args.seed,
            args.packet_log,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.cmd == "replay-speech":
        result = replay_speech(
            input_wav=args.input,
            output_wav=args.output,
            trace=args.trace,
            policy=args.policy,
            predictor=_predictor(args.predictor),
            bitrate=args.bitrate,
            expected_loss_percent=args.expected_loss,
            random_duty_cycle=args.random_duty_cycle,
            seed=args.seed,
            packet_log=args.packet_log,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.cmd == "run":
        result = execute_campaign(args.campaign, args.target, args.site, args.output_root)
        print(result.campaign_dir)
        return

    if args.cmd == "report":
        print(build_report(args.run_dir))
        return


if __name__ == "__main__":
    main()
