# Architecture

StarVoice deliberately separates transport measurement, prediction, codec policy, and analysis.

## Components

### Agent
Runs behind the access network under test. It transmits either high-rate probe packets or Opus speech frames and records monotonic timestamps and policy state.

### Relay
A minimal UDP echo relay on a stable wired/cloud host. It does not make adaptation decisions. Keeping the relay policy-free prevents server logic from becoming a hidden experimental variable.

### Predictor
The first predictor (`PhaseModel`) is intentionally transparent: it estimates a recurring disruption boundary from calibration data and emits a continuous risk score around that boundary. Later predictors must implement the same conceptual interface so policies can be compared without changing transport code.

### Policy
Current policies:
- `plain`
- `always-fec`
- `random-fec`
- `predictive-fec`

The randomized policy exists as an equal-budget control: a predictive policy must outperform spending comparable redundancy at unrelated times, not merely outperform an unprotected stream.

### Campaign runner
Campaign manifests specify conditions and a seed. Within each repeated block the condition order is randomized and the resolved order is written to disk before execution.

## Time model

Latency is measured with `CLOCK_MONOTONIC` through Python's `time.monotonic_ns()`. Wall-clock time is metadata only. No one-way-delay claim should be made from unsynchronized hosts; the default probe reports RTT.

## Packet protocol

All experiment packets use a small versioned binary header containing:
- magic/version
- packet kind
- flags
- sequence number
- sender monotonic timestamp
- payload length

Protocol changes require a version bump.

## Codec

The initial implementation calls system libopus directly through `ctypes`; it does not depend on a Python codec wrapper. Current experiments target 48 kHz mono PCM16 and 20 ms frames. Opus in-band FEC is controlled with the official encoder CTLs. DRED capability detection is exposed in the codec layer but DRED decoding/evaluation is intentionally not claimed until a complete receive path is implemented.

## Scientific non-goals

- A fixed 15-second period is **not** assumed to be universally true. It is a hypothesis/initial model parameter that must be estimated and validated independently per study.
- Dish telemetry is not required for the core predictor.
- ASR quality is not used as a substitute for packet-level ground truth.
