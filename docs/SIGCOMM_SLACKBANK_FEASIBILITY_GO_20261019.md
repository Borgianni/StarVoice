# SlackBank: idealized prospective feasibility GO (2026-10-19)

## Input and evaluation boundary

Nine contiguous Always-FEC Starlink NUC -> Oracle blocks collected in ONE 90.6-minute, single-site/day natural acquisition window; 271,851 total 20ms speech frames. Source: locally archived `results/sigcomm-onewindow-20261019T092009Z`, with analysis output `results/slackbank-feasibility-v1`. User provided CLI output on 2026-10-19. Research branch code SHA: `99eb9aaad829a535bbd0d34935ac8f333ca394c6`. NUC unit tests: **7 passed in 0.10s**.

SlackBank excludes the first 120 seconds (6,000 frames) of EACH of nine blocks from scoring; these frames calibrate a *fixed* disturbance phase from only earlier receiver-local within-host timestamp increments and sender pacing intervals. Remaining frames scored using original receiver packet arrival timestamps, existing actual Always-FEC metadata, 20ms speech frames. Targets vary by <=1ms per frame and between 60 and 80ms. This is a causal split with fixed arrivals but NOT a full prospective running estimator, physically synthesized audio, or a live experiment.

| Model | PLC in scored suffixes | mean extra modeled receiver-relative playout budget (ms) |
|---|---:|---:|
| fixed60 | 82 | 0.000 |
| fixed80 | 9 | 20.000 |
| fixed100 | 2 | 40.000 |
| causal_phase_ramp_60_80 | 15 | 0.668 |
| wrong_phase_ramp_60_80 | 82 | 0.667 |

**Feasibility gate: PASS, constrained to this model and dataset**. Versus fixed60, smooth causal phase-aligned ramp models 67 fewer PLC (81.7% of 82), with 0.668ms mean extra receiver-local buffer and 80ms peak. Fixed80 avoids 73 of these 82, so smooth ramp obtains 67/73=91.8% of fixed80's achievable benefit at ~30x less mean *modeled buffer increment* (0.668ms vs 20ms). Wrong-phase control spends the same average budget without reducing PLC. This provides an event-alignment *sanity check*; it does NOT establish confidence intervals across independent sites/days or superiority to adaptive NetEQ. Note scoring denominator excludes training; don't compare 82 directly with full-window fixed60 96 without explaining exclusion.

## Distinctions mandatory for all claims

- Buffer is receiver-relative and anchored to first received audio; not absolute OWD or mouth-to-ear latency.
- The average buffer increment is computed from an idealized deadline profile, not actual speaker latency. A real sink needs sample-clock-consistent 48kHz audio time-scale modification; length-changing audio processing incurs pitch/transient/quality costs.
- This test holds observed arrival times/Opus packet metadata fixed, and does NOT re-encode source for an altered sender policy. It is not an end-to-end controller or experimentally observed waveform QoE win.
- The first 120s training splits are within each block; nine blocks in one acquisition window are *correlated*, not independent replications. Fixed 15s phase can drift under other paths / load.
- Existing WebRTC NetEQ implements jitter-buffer adaptation and time stretching; do not frame adaptive jitter buffering as novel. Compare **real**, pinned NetEQ and its tuning, not an approximation.

## Actual implementation gate: codec and physical audio

1. **Recorded-byte reproducibility gate**: Current `realtime.py` sender/receiver JSONL records packet byte LENGTHS and `has_lbrr`, not Opus packet bodies. Source corpus WAV segments are in campaign workloads. Re-encode complete original Always-FEC streams with the same libopus version and initial encoder state; verify frame-by-frame actual sender `opus_bytes`, `has_lbrr`, and decoded fixed60 WAV/decision accounting against recorded receiver outputs. Length+FEC flags alone DO NOT prove bit-identical packets; absent retained raw packet hashes/bytes, report fidelity as approximate and avoid claiming exact replay. Preserve decoder state and distinguish sender packet availability from packet-use deadline.
2. Implement a physical sample-count-consistent, pitch-preserving time-scaling sink. Per 20ms/960-sample source frame, a +/-1ms slack change means an effective output increment of +/-48 samples at 48kHz. Correctly account for accumulated real audio samples, decoder lookahead and processing time; audit that no deadline is effectively moved without producing corresponding audio time scaling. Expose latency, stretched fraction, acoustic artifacts and buffer occupancy.
3. Compare same arrivals/content and matched quality-latency-byte budgets against fixed60, fixed80, a real pinned WebRTC NetEQ build, a phase-only ramp (without codec-aware decision), a wrong-phase control and a noncausal oracle labeled as such.
4. Score actual output speech (time-aligned STOI, DNSMOS, and WER as secondary, suitable baselines and limits). Report *paired per-block* metrics, temporal dependence, average AND peak receiver buffer, true wire bytes, CPU overhead, failures from phase drift. No stronger generalization claim without independent dates or site.
5. Add actual prospective per-frame recoverability utility/codec switch **only after** the physical sink gate. Holding sender Always-FEC across current tests does not demonstrate any joint codec-and-playout novelty.

**GO only to build and falsify real playout**. Stop or reframe the novelty if equal-cost NetEQ and real audio decoding eliminate gains. Do not relaunch natural acquisition as part of this gate. Frozen phase2-confirmatory code and preregistration remain unchanged.
