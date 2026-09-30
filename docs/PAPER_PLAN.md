# Paper plan

## Working title

**StarVoice: Predictive Speech Protection over Operational LEO Satellite Networks**

## Core claim to test

Short-horizon network-risk information can be exposed to a real-time speech codec early enough to place redundancy where it is most valuable, improving speech robustness at a fixed redundancy/latency budget.

This is a hypothesis. The repository must make it possible to falsify it cleanly.

## Research questions

### RQ1 — Temporal structure
Are packet-delay/loss events affecting speech delivery temporally structured, and are the structures stable across time and sites?

### RQ2 — Predictability
At what horizons (50, 100, 250, 500, 1000 ms) can damaging events be predicted on held-out data?

### RQ3 — Utility
At equal redundancy budget, does predictive Opus protection outperform randomized placement and reactive protection?

### RQ4 — Generalization
Does a policy/model frozen at one site retain value on geographically separated Starlink terminals without re-fitting?

### RQ5 — Mechanism
Which signals (phase, RTT history, loss history, optional terminal telemetry) contribute useful predictive information?

## Required baselines before submission

1. Plain Opus
2. Always FEC
3. Equal-budget Random FEC
4. Reactive FEC
5. Predictive phase FEC
6. Predictive learned FEC
7. Oracle upper bound in offline replay

DRED is an extension unless its full receive path is validated early enough.

## Required evaluations

- live operational Starlink experiments;
- controlled trace replay;
- calibration/test temporal split;
- external-site zero-shot validation;
- predictive-horizon curves;
- equal-budget quality/overhead curves;
- ablations;
- block/day/site-aware uncertainty.

## Primary figures

1. Network event probability versus estimated phase.
2. Prediction quality versus horizon.
3. Speech quality versus redundancy overhead.
4. Equal-budget policy comparison.
5. Cross-site generalization.
6. Ablation study.
7. Live versus replay consistency.

## Stop conditions

Do not add semantic-importance models, VAD gating, WebRTC integration or large neural predictors until the primary equal-budget predictive-FEC result is established on held-out data.
