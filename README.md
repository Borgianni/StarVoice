# StarVoice

Reproducible predictive speech protection experiments over LEO satellite networks.

This repository contains the StarVoice research system and SatRTC-Bench experimentation toolkit.

> Status: active research prototype. The default branch is intended to stay runnable; experiments are described by versioned campaign manifests.

## Scientific goals

StarVoice asks whether short-horizon network-risk prediction can be used to proactively control speech error-resilience mechanisms such as Opus in-band FEC and, where supported, DRED.

The project is designed around four principles:

1. **Reproducibility** — every run records its configuration, software/runtime metadata and random seed.
2. **Fair baselines** — predictive policies are compared against plain, reactive, always-on and equal-budget randomized protection.
3. **Multi-site validation** — the same campaign manifest can be executed at geographically separated Starlink terminals.
4. **Separation of measurement and inference** — raw packet/network observations are retained so analyses can be repeated without re-running the network experiment.

## Quick start

```bash
git clone https://github.com/Borgianni/StarVoice.git
cd StarVoice

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,analysis]"

starvoice doctor
starvoice relay --bind 0.0.0.0 --port 5005
# on the Starlink-connected host:
starvoice probe --target RELAY_IP:5005 --duration 60
starvoice run campaigns/smoke.yaml --target RELAY_IP:5005 --site pisa-01
```

See `docs/EXPERIMENTS.md` for the experimental protocol and `docs/ARCHITECTURE.md` for the system design.

## Commands

- `starvoice doctor` — environment and dependency checks.
- `starvoice relay` — deterministic UDP echo/measurement relay.
- `starvoice probe` — high-rate RTT/loss/jitter probe with machine-readable output.
- `starvoice calibrate` — estimate periodic network phase from a probe trace.
- `starvoice run` — execute a randomized, versioned campaign.
- `starvoice report` — summarize one run/campaign and generate publication-oriented tables/figures.

## Data layout

Each run is self-describing:

```text
runs/<run-id>/
  manifest.resolved.yaml
  environment.json
  packets.jsonl
  network.jsonl
  events.jsonl
  summary.json
```

Parquet export is produced by the analysis extra when `pyarrow` is installed.

## License

Apache-2.0.
