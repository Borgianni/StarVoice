# Experimental protocol

This document is normative for the main StarVoice study.

## 1. Roles

For each site:
1. one host is connected through the Starlink router/access path being studied;
2. one relay is placed on a stable non-Starlink Internet connection;
3. the site runs the same Git commit and campaign manifest.

Record the exact Git commit, OS/kernel, Python version and site label.

## 2. Calibration is separate from evaluation

Never estimate predictor parameters from the same time interval used to report final policy performance.

Recommended split:
- calibration/training days;
- optional tuning day;
- frozen test days;
- geographically distinct sites as external validation.

`starvoice calibrate` implements the transparent P0 phase model. Its period is a configurable model assumption, not ground truth.

## 3. Randomized blocked design

Each campaign repeat is a block. StarVoice randomizes condition order inside each block from a stored seed. Do not run all baseline trials in the morning and all predictive trials later.

Recommended minimum condition set:
- plain Opus;
- always-FEC;
- random-FEC with duty cycle matched to predictive protection;
- predictive-FEC.

Add a reactive baseline before making claims against state-of-practice congestion/loss response.

## 4. Equal-budget comparison

The strongest causal comparison fixes protection budget. Estimate predictive FEC duty cycle on training/calibration data, freeze it, then configure random-FEC to spend approximately the same fraction of protected frames on held-out runs.

Report:
- speech quality;
- encoded byte overhead;
- protected-frame duty cycle;
- network loss;
- recovery behavior;
- latency/playout assumptions.

## 5. Directionality

Do not merge uplink, downlink and round-trip results under one label. The current speech loop is a loopback experiment and therefore measures the combined path. Add separate one-way experiments before making direction-specific claims.

## 6. Workload

Use a fixed, redistributable speech corpus with transcripts and a committed manifest containing SHA-256 hashes. The repository intentionally does not vendor third-party speech datasets.

Current speech input contract:
- WAV
- 48 kHz
- mono
- signed PCM16

Randomize workload order; store the seed and file hash.

## 7. Repetition

A run containing only a handful of periodic events is exploratory. For final results, collect many independent blocks across:
- multiple hours of day;
- multiple days;
- multiple sites.

Treat blocks/days/sites—not individual packets—as the primary units for confidence intervals.

## 8. Statistics

Before looking at final outcomes, define:
- primary endpoint;
- primary policy comparison;
- exclusion criteria;
- bootstrap/mixed-effects procedure;
- significance level if hypothesis testing is used.

Do not report millions of packets as millions of independent samples.

## 9. Speech metrics

Network metrics are always retained separately from speech metrics.

Planned speech metrics:
- STOI or an appropriate intelligibility metric;
- DNSMOS or another validated perceptual proxy;
- ASR WER with an exactly pinned model/version/decoding configuration.

When using ASR, report degradation relative to a clean/locally encoded reference so ASR model errors are not attributed to the network.

## 10. Reproducibility checklist

Before accepting a run:
- doctor passes;
- target/relay identity recorded;
- commit hash present;
- campaign seed stored;
- workload SHA-256 stored;
- monotonic timestamps used;
- condition order stored before execution;
- no predictor fitting on test outcomes;
- raw logs retained.

## 11. Current limitations

Version 0.1 provides the runnable substrate, FEC policies, phase calibration and randomized campaigns. It does **not yet** claim:
- DRED recovery;
- WebRTC-equivalent jitter-buffer behavior;
- synchronized one-way delay;
- Starlink dish telemetry ingestion;
- ASR/STOI/DNSMOS scoring.

Those are explicit next milestones, not silently approximated features.
