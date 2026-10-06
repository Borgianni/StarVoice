#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

from starvoice.opus import (
    OpusConfig,
    OpusEncoder,
    packet_has_fec_compat,
    packet_has_fec_native,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Validate StarVoice's compatibility LBRR detector against native libopus."
    )
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--native-lib", type=Path, required=True)
    p.add_argument("--max-packets", type=int, default=20000)
    p.add_argument("--bitrate", type=int, default=24000)
    p.add_argument("--expected-loss", type=int, default=20)
    p.add_argument("--block-frames", type=int, default=50)
    p.add_argument("--output", type=Path)
    args = p.parse_args()

    if args.max_packets <= 0:
        raise SystemExit("--max-packets must be positive")
    if args.block_frames <= 0:
        raise SystemExit("--block-frames must be positive")
    if not args.native_lib.exists():
        raise SystemExit(f"native libopus not found: {args.native_lib}")

    cfg = OpusConfig(bitrate=args.bitrate)
    enc = OpusEncoder(cfg)
    frame_bytes = cfg.frame_samples * cfg.channels * 2

    total = 0
    native_positive = 0
    native_negative = 0
    compat_recent_mismatches = 0
    compat_system_mismatches = 0
    examples: list[dict] = []

    try:
        for item in _read_jsonl(args.manifest):
            wav_path = Path(item["wav_48k_mono_pcm16"])
            with wave.open(str(wav_path), "rb") as w:
                if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (
                    cfg.sample_rate,
                    cfg.channels,
                    2,
                ):
                    raise RuntimeError(f"unexpected WAV format: {wav_path}")

                while total < args.max_packets:
                    pcm = w.readframes(cfg.frame_samples)
                    if not pcm:
                        break
                    if len(pcm) < frame_bytes:
                        pcm += b"\0" * (frame_bytes - len(pcm))

                    # Long deterministic on/off blocks exercise both states while
                    # preserving realistic stateful encoder transitions.
                    fec_enabled = ((total // args.block_frames) % 2) == 1
                    enc.set_fec(fec_enabled, args.expected_loss)
                    payload = enc.encode(pcm)

                    native = packet_has_fec_native(payload, args.native_lib)
                    compat_recent = packet_has_fec_compat(payload, args.native_lib)
                    compat_system = packet_has_fec_compat(payload)

                    native_positive += int(native)
                    native_negative += int(not native)

                    if compat_recent != native:
                        compat_recent_mismatches += 1
                        if len(examples) < 20:
                            examples.append(
                                {
                                    "packet_index": total,
                                    "utterance_id": item["utterance_id"],
                                    "fec_enabled": fec_enabled,
                                    "packet_bytes": len(payload),
                                    "native": native,
                                    "compat_recent": compat_recent,
                                    "compat_system": compat_system,
                                    "mismatch": "compat_recent",
                                }
                            )

                    if compat_system != native:
                        compat_system_mismatches += 1
                        if len(examples) < 20:
                            examples.append(
                                {
                                    "packet_index": total,
                                    "utterance_id": item["utterance_id"],
                                    "fec_enabled": fec_enabled,
                                    "packet_bytes": len(payload),
                                    "native": native,
                                    "compat_recent": compat_recent,
                                    "compat_system": compat_system,
                                    "mismatch": "compat_system",
                                }
                            )

                    total += 1

                if total >= args.max_packets:
                    break
            if total >= args.max_packets:
                break
    finally:
        enc.close()

    result = {
        "schema_version": 1,
        "manifest": str(args.manifest),
        "native_lib": str(args.native_lib),
        "packets_tested": total,
        "bitrate": args.bitrate,
        "expected_loss_percent": args.expected_loss,
        "fec_toggle_block_frames": args.block_frames,
        "native_lbrr_positive": native_positive,
        "native_lbrr_negative": native_negative,
        "compat_recent_mismatches": compat_recent_mismatches,
        "compat_system_mismatches": compat_system_mismatches,
        "pass": (
            total > 0
            and native_positive > 0
            and native_negative > 0
            and compat_recent_mismatches == 0
            and compat_system_mismatches == 0
        ),
        "mismatch_examples": examples,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 2)


if __name__ == "__main__":
    main()
