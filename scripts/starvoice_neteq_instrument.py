#!/usr/bin/env python3
"""STARVOICE research-only NetEQ pin-checked instrumentation and block-01 pilot.

Patches ONLY ~/StarVoice-sigcomm-native/webrtc/src (independent scratch checkout),
saves exact pristine sources/binary, builds incrementally, and writes new results.
No changes to ~/StarVoice deployment, prior closeout, capture, or Oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import statistics
import subprocess
import wave
from pathlib import Path

PIN = "0c0e991273c90fc000527a53fac9ba96ee22078b"
BASE_BINARY_SHA = "0d2a086cf7472d97ba645e40d5b55d5a959dadaa77bda38ef6a6355a3b2847b2"
ROOT = "modules/audio_coding/neteq/tools"
BLOBS = {
    "neteq_test.cc": "67cd892f85490a14b0fd3b1dbfabc982f2e96375",
    "neteq_test_factory.cc": "f74d2110899790abb3dccbe747859090531c5cb7",
    "neteq_test_factory.h": "c7564fa718b99eb34f84d6bedd887f6a992c1753",
    "neteq_rtpplay.cc": "a05e1b360e8a24cedac68564cc0e7a18a43b1265",
}
MARK = "STARVOICE_NETEQ_INSTRUMENT_V1"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def blob_sha(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def once(s, original, replacement, name):
    n = s.count(original)
    if n != 1:
        raise RuntimeError(f"Unsafe patch {name}: expected one source anchor, found {n}")
    return s.replace(original, replacement)


def planned_patches(originals):
    p = dict(originals)
    old = "    int max_nr_packets_in_buffer = default_max_nr_packets_in_buffer();"
    p["neteq_test_factory.h"] = once(p["neteq_test_factory.h"], old,
        old + "\n    // " + MARK + ": research controls, NOT absolute playout delay.\n"
        "    int starvoice_min_delay_ms = 0;\n"
        "    int starvoice_max_delay_ms = 0;", "factory.h")
    old = "  neteq_config.enable_fast_accelerate = config.enable_fast_accelerate;"
    p["neteq_test_factory.cc"] = once(p["neteq_test_factory.cc"], old,
        old + "\n  // " + MARK + ": internal NetEQ target controls only.\n"
        "  neteq_config.min_delay_ms = config.starvoice_min_delay_ms;\n"
        "  neteq_config.max_delay_ms = config.starvoice_max_delay_ms;", "factory.cc")
    old = ('ABSL_FLAG(bool,\n          enable_fast_accelerate,\n          false,\n'
           '          "Enables jitter buffer fast accelerate");')
    p["neteq_rtpplay.cc"] = once(p["neteq_rtpplay.cc"], old,
        old + "\n// " + MARK + ": research-only NetEQ config flags.\n"
        'ABSL_FLAG(int, starvoice_min_delay_ms, 0,\n'
        '          "Internal NetEQ minimum delay ms; NOT playout OWD.");\n'
        'ABSL_FLAG(int, starvoice_max_delay_ms, 0,\n'
        '          "Internal NetEQ maximum delay ms (0=default); NOT playout OWD.");',
        "rtpplay flags")
    old = "  config.enable_fast_accelerate = absl::GetFlag(FLAGS_enable_fast_accelerate);"
    p["neteq_rtpplay.cc"] = once(p["neteq_rtpplay.cc"], old,
        old + "\n  // " + MARK + ": record externally chosen bounds.\n"
        "  config.starvoice_min_delay_ms = absl::GetFlag(FLAGS_starvoice_min_delay_ms);\n"
        "  config.starvoice_max_delay_ms = absl::GetFlag(FLAGS_starvoice_max_delay_ms);\n"
        "  RTC_CHECK_GE(config.starvoice_min_delay_ms, 0);\n"
        "  RTC_CHECK_LE(config.starvoice_min_delay_ms, 500);\n"
        "  RTC_CHECK_GE(config.starvoice_max_delay_ms, 0);\n"
        "  RTC_CHECK_LE(config.starvoice_max_delay_ms, 500);\n"
        "  RTC_CHECK(config.starvoice_max_delay_ms == 0 ||\n"
        "            config.starvoice_min_delay_ms <= config.starvoice_max_delay_ms);",
        "rtpplay config")
    old = '                   << current_state_.current_delay_ms << std::endl;'
    extension = r'''
        // STARVOICE_NETEQ_INSTRUMENT_V1: delivered last-sample RTP clock.
        const auto starvoice_ts = neteq_->GetPlayoutTimestamp();
        *text_log_ << "SVPLAY wallclock_ms=" << time_now_ms
                   << " timestamp_valid=" << (starvoice_ts.has_value() ? 1 : 0)
                   << " rtp_last_sample=" << starvoice_ts.value_or(0)
                   << " output_samples=" << out_frame.samples_per_channel_
                   << " sample_rate_hz=" << out_frame.sample_rate_hz_
                   << " filtered_delay_ms=" << neteq_->FilteredCurrentDelayMs()
                   << " target_delay_ms=" << neteq_->TargetDelayMs()
                   << " base_min_delay_ms=" << neteq_->GetBaseMinimumDelayMs()
                   << " plc=" << (plc ? 1 : 0)
                   << " concealed_samples_total=" << lifetime_stats.concealed_samples
                   << std::endl;'''
    p["neteq_test.cc"] = once(p["neteq_test.cc"], old, old + extension, "test.cc log")
    if any(MARK not in value for value in p.values()):
        raise AssertionError("One of the source files lacks the instrumentation marker")
    return p


def install(webrtc, output):
    commit = subprocess.check_output(["git", "-C", str(webrtc), "rev-parse", "HEAD"], text=True).strip()
    if commit != PIN:
        raise RuntimeError(f"WebRTC commit mismatch: {commit}; expected {PIN}")
    sources = {k: webrtc / ROOT / k for k in BLOBS}
    current = {k: v.read_bytes() for k, v in sources.items()}
    backup_dir = output / "pristine_sources"
    executable = webrtc / "out/StarVoice/neteq_rtpplay"
    pristine_bin = executable.with_name("neteq_rtpplay.starvoice_pristine")

    # Guard the executable BEFORE modifying any source.
    if pristine_bin.exists():
        if digest(pristine_bin) != BASE_BINARY_SHA:
            raise RuntimeError("Previously saved original NetEQ binary has wrong SHA")
    elif digest(executable) != BASE_BINARY_SHA:
        raise RuntimeError("Expected original native NetEQ executable SHA differs; abort")
    else:
        shutil.copy2(executable, pristine_bin)

    fully_patched = all(MARK.encode() in data for data in current.values())
    partially_patched = any(MARK.encode() in data for data in current.values())
    if fully_patched:
        if not backup_dir.is_dir():
            raise RuntimeError("Existing modified checkout lacks verified pristine source backups")
        originals = {k: (backup_dir / k).read_bytes() for k in BLOBS}
    elif partially_patched:
        raise RuntimeError("Partially modified WebRTC checkout; refusing unsafe recovery")
    else:
        originals = current
    for k, data in originals.items():
        if blob_sha(data) != BLOBS[k]:
            raise RuntimeError(f"WebRTC pinned source differs at {k}; do not overwrite")
    patched = planned_patches({k: data.decode() for k, data in originals.items()})
    if fully_patched:
        if any(current[k] != patched[k].encode() for k in BLOBS):
            raise RuntimeError("Existing WebRTC instrumentation differs from this exact script")
    else:
        backup_dir.mkdir(parents=True, exist_ok=True)
        for k, data in originals.items():
            bp = backup_dir / k
            if bp.exists() and bp.read_bytes() != data:
                raise RuntimeError(f"Pristine source backup conflict: {bp}")
            bp.write_bytes(data)
        for k, data in patched.items():
            tmp = sources[k].with_name(k + ".starvoice_pending")
            tmp.write_text(data)
            tmp.replace(sources[k])
        print("Pinned WebRTC source instrumented safely; pristine backups retained.")
    return executable, pristine_bin


def read_wave_sha(path):
    with wave.open(str(path), "rb") as w:
        params = (w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth())
        h = hashlib.sha256()
        while True:
            chunk = w.readframes(16384)
            if not chunk:
                break
            h.update(chunk)
        return params, h.hexdigest()


def parse_playout(log):
    first_ts = None
    valid_post_training = []
    invalid = 0
    total = 0
    plc = 0
    p = re.compile(r"(\w+)=(-?\d+)")
    packet = re.compile(r"^Packet\s+- wallclock:\s*\d+.*timestamp:\s*(\d+)")
    with log.open() as f:
        for line in f:
            if first_ts is None:
                m = packet.search(line)
                if m:
                    first_ts = int(m.group(1))
            if not line.startswith("SVPLAY "):
                continue
            total += 1
            x = {k: int(v) for k, v in p.findall(line)}
            plc += x["plc"]
            if not x["timestamp_valid"] or first_ts is None:
                invalid += 1
                continue
            source_ms = ((x["rtp_last_sample"] - first_ts) & 0xffffffff) / 48.
            if source_ms < 120000:  # 120 seconds training, identical to fixed60.
                continue
            if x["sample_rate_hz"] < 1:
                raise RuntimeError("Invalid decoded sample rate")
            end_ms = x["wallclock_ms"] + x["output_samples"] * 1000 / x["sample_rate_hz"]
            d = end_ms - source_ms
            valid_post_training.append((d, x["filtered_delay_ms"], x["target_delay_ms"]))
    if not valid_post_training:
        raise RuntimeError(f"No usable post-training timestamps in {log}")
    v = sorted(x[0] for x in valid_post_training)
    q = lambda t: v[round((len(v) - 1) * t)]
    return {
        "total_getaudio_callbacks": total,
        "invalid_playout_timestamps": invalid,
        "valid_post_training_samples": len(v),
        "receiver_relative_offset_approx_mean_ms": round(statistics.fmean(v), 4),
        "receiver_relative_offset_approx_p95_ms": round(q(.95), 4),
        "receiver_relative_offset_approx_max_ms": round(max(v), 4),
        "receiver_relative_offset_approx_min_ms": round(min(v), 4),
        "above_80_ms_callbacks": sum(d > 80 for d in v),
        "below_0_ms_callbacks": sum(d < 0 for d in v),
        "internal_filtered_mean_ms": round(statistics.fmean(x[1] for x in valid_post_training), 4),
        "internal_target_mean_ms": round(statistics.fmean(x[2] for x in valid_post_training), 4),
        "plc_getaudio_callbacks_full_run": plc,
        "definition": "GetAudio simulated output end ms - last delivered RTP sample timestamp relative to first packet, receiver-local. Diagnostic only; not measured mouth-to-ear.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--webrtc", type=Path, required=True)
    p.add_argument("--closeout", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    webrtc, closeout, output = (x.expanduser().resolve() for x in (a.webrtc, a.closeout, a.output))
    if any(output == root or output.is_relative_to(root) for root in (webrtc, closeout)):
        p.error("Output must not reside inside original inputs or WebRTC checkout")
    output.mkdir(parents=True, exist_ok=True)
    exe, pristine = install(webrtc, output)
    build_log = output / "native_incremental_build.log"
    print("Building instrumented native NetEQ; original binary preserved:", pristine, flush=True)
    with build_log.open("w") as f:
        r = subprocess.run(["ninja", "-C", str(webrtc / "out/StarVoice"), "neteq_rtpplay"],
                           stdout=f, stderr=subprocess.STDOUT, check=False)
    if r.returncode:
        raise RuntimeError(f"Native incremental build failed: {build_log}")
    if digest(exe) == BASE_BINARY_SHA:
        raise RuntimeError("Unexpected unchanged native executable: instrumentation not linked")
    rtp = closeout / "block-01/rtpdump/block-01.rtp"
    reference = closeout / "block-01/native/neteq.wav"
    if not rtp.is_file() or not reference.is_file():
        raise FileNotFoundError(f"Need existing audited block-01 RTP and WAV: {rtp} {reference}")
    reports = []
    for label, low, high in (("default", 0, 0), ("target60_cap80", 60, 80)):
        folder = output / label
        folder.mkdir(exist_ok=True)
        wav = folder / "neteq.wav"
        prefix = folder / "run"
        log = folder / "run.text_log.txt"
        if not (wav.is_file() and log.is_file()):
            cmd = [str(exe), f"--starvoice_min_delay_ms={low}",
                   f"--starvoice_max_delay_ms={high}", "--textlog=true",
                   f"--output_files_base_name={prefix}", str(rtp), str(wav)]
            print("Replaying pinned native NetEQ configuration", label, flush=True)
            with (folder / "stdout.txt").open("w") as f:
                proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
            if proc.returncode:
                raise RuntimeError(f"NetEQ replay failed: {folder / 'stdout.txt'}")
        metrics = parse_playout(log)
        metrics.update({"variant": label, "configured_internal_min_ms": low,
                        "configured_internal_max_ms": high, "neteq_binary_sha256": digest(exe)})
        if label == "default":
            if read_wave_sha(reference) != read_wave_sha(wav):
                raise RuntimeError("Instrumented default PCM differs from original native NetEQ baseline; do not compare")
            metrics["original_native_pcm_identical"] = True
        reports.append(metrics)
        print(label, "measured approximate receiver-relative mean/p95/max (ms):",
              metrics["receiver_relative_offset_approx_mean_ms"],
              metrics["receiver_relative_offset_approx_p95_ms"],
              metrics["receiver_relative_offset_approx_max_ms"],
              "; callbacks above 80ms:", metrics["above_80_ms_callbacks"], flush=True)
    summary = {
        "status": "native_neteq_timeline_diagnostic_block01",
        "webrtc_commit": PIN, "pristine_neteq_binary_sha256": digest(pristine),
        "original_starvoice_and_closeout_untouched": True,
        "comparability": "Diagnostic only: config bounds are INTERNAL; receiver-relative delay and QoE equal-cost not established",
        "configurations": reports,
    }
    result = output / "block01_neteq_timeline.json"
    result.write_text(json.dumps(summary, indent=2) + "\n")
    print("Saved:", result, flush=True)


if __name__ == "__main__":
    main()
