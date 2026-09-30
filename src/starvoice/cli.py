from __future__ import annotations

import argparse
import json
from pathlib import Path

from .campaign import execute_campaign
from .environment import run_doctor
from .predictor import calibrate_phase
from .probe import run_probe, summarize
from .relay import run_relay
from .report import build_report
from .speech import speech_loop


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

    s = sub.add_parser("speech-loop")
    s.add_argument("--input", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--target", required=True)
    s.add_argument("--policy", choices=["plain", "always-fec", "random-fec", "predictive-fec"], required=True)
    s.add_argument("--predictor", type=Path)
    s.add_argument("--bitrate", type=int, default=24000)
    s.add_argument("--expected-loss", type=int, default=20)
    s.add_argument("--random-duty-cycle", type=float, default=0.1)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--packet-log", type=Path)

    run = sub.add_parser("run")
    run.add_argument("campaign", type=Path)
    run.add_argument("--target", required=True)
    run.add_argument("--site", required=True)
    run.add_argument("--output-root", type=Path, default=Path("runs"))

    rep = sub.add_parser("report")
    rep.add_argument("run_dir", type=Path)

    return p


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

    if args.cmd == "speech-loop":
        predictor = None
        if args.predictor:
            from .predictor import PhaseModel
            predictor = PhaseModel.load(args.predictor)
        result = speech_loop(
            args.input,
            args.output,
            args.target,
            args.policy,
            predictor,
            args.bitrate,
            args.expected_loss,
            args.random_duty_cycle,
            args.seed,
            args.packet_log,
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
