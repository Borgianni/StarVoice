# STARVOICE — SIGCOMM mechanism hypothesis (research track, 2026-10-15)

**Status: OPEN HYPOTHESIS / NOT CONFIRMATORY.** This note lives on `research-sigcomm-deadline-codec` to prevent post-hoc changes to the frozen `phase2-confirmatory` policy and preregistration.

## 1. The problem is not simply a predictable 15-second handover

Previous LEO systems work already exploits reconfiguration: LeoCC (SIGCOMM 2025) controls congestion using endpoint reconfiguration detection; *Dynamic Pacing for Real-time Satellite Traffic* (2025) adapts sender queues using handover forecasts. SIGCOMM 2026's *Dissecting the StarLink* reports 15-second resets of queue/allocation dynamics, head-drop, active queue management and shared-flow loss. The claim 'predict risk and switch on FEC' is therefore not sufficiently differentiated by itself.

References:
- LeoCC: https://conferences.sigcomm.org/sigcomm/2025/program/papers-info/
- Dynamic Pacing: https://arxiv.org/abs/2507.09798
- Dissecting StarLink: https://doi.org/10.1145/3789240.3829162
- WebRTC codec FEC semantics: https://www.rfc-editor.org/rfc/rfc8854.html

## 2. Falsifiable mechanism hypothesis

**A sender can improve deadline-constrained speech quality or useful recovery per protected byte by predicting *recoverability*, not merely packet-loss probability, and choosing protection based on both delivery-time feasibility and the codec's intrinsic quality cost.**

For a speech frame i whose primary may be lost/late, standard Opus LBRR only helps if:
1. the encoder actually places recovery data for i into carrier packet i+1;
2. i+1 itself arrives and is usable before the playback deadline of i;
3. the resulting recovered audio provides more quality utility than protection's self-distortion and byte/latency cost.

This can fail under correlated bursts and queue-induced lateness even when loss risk is high.

Define, for an available action a, a **prospective** reward (not an outcome-time oracle):
`U_i(a) = E[Q_audio(a) - Q_audio(no_protection) | features observed by decision time] - lambda * E[extra bytes] - mu * E[extra playout delay]`.
The model must explicitly represent deadline feasibility, codec schedule distortion and carrier survival. 'Always-FEC' is a high-cost baseline, not an oracle. A true oracle upper bound is noncausal and labeled so.

### Candidate decisions
- `off`: no extra protection.
- `opus-lbrr`: in-band next-packet recovery, when predicted useful.
- `burst-capable protection`: add only **after** complete, validated DRED or alternative redundancy receive path. Do not claim DRED is implemented in SV02 today.
- Adaptive buffering can be a *separate latency-constrained ablation*, never called a free gain.

The initial clean proof-of-concept should first test **off versus LBRR with a deadline-feasibility and codec-cost gate**, not simultaneously introduce DRED, speech VAD, WebRTC and a neural predictor.

## 3. Scientific split: frozen Phase II vs new mechanism

### Track A — confirm / potentially reject the existing phase model
- Keep `docs/preregistration-confirmatory-v1.md` and the existing phase-only predictor frozen.
- Per-session primary: timely FEC recoveries vs **exact equal-protected-frame** Random, 60-ms *specified* primary endpoint and its timing uncertainty.
- Include Plain, Reactive, Always-FEC and noncausal oracle as appropriate.
- Do not use the 2026-10-15 first cross-host 3000/3000 pilot to claim any FEC benefit or perceive 60-ms end-to-end latency.

### Track B — novel deadline-and-codec-aware mechanism, exploratory until preregistered
- Features must be observable causally **before** the packet whose encoder settings are changed (history of RTT, receiver feedback, estimated phase/risk, queue signals if available).
- Distinguish prediction of primary failure, carrier failure (i+1), deadline feasibility and burst length.
- Use held-out days and external terminals; never tune on evaluation sessions.
- Evaluate the net audio-quality effect, not merely count of FEC protected packets or nominal FEC enable settings.

## 4. Gates and success criteria

### G1: real-time transport baseline — technically passed (partial)
- SV02 20-ms frames, causal receiver-local playout, real LBRR unit test, async logs and RAM archival.
- 2026-10-15 cross-host pilot 002: same session ID, 3000 sent/received/decoded, zero observed loss/deadline miss/FEC recovery/PLC; START/END worked.
- **G1 is not an end-to-end <60 ms QoE demonstration.** The 60-ms buffer starts at the receiver's first audio arrival; cross-host clock offset not calibrated to bound OWD.

### G2: transport+experiment validity under impairments — OPEN
- Verify the NUC's actual WAN path traverses the intended Starlink terminal, not another uplink.
- Preserve both endpoint logs/clock state and run the causal auditor.
- Exercise isolated drops, adjacent burst losses, delay/reordering and late carrier frames; injected and organic loss must be separate.
- Obtain 3–5 new, independent pilot sessions including null-loss observations to estimate event yield and required confirmatory session count.
- See https://github.com/Borgianni/StarVoice/issues/3.

### G3: Phase II replication/generalization — OPEN
- Frozen model, frozen session-level primary statistic, sessions as clustering units, equal-budget random, multi-day and multi-site external terminal(s).
- Record the hypothesis as falsified if predictive placement has no held-out session-level benefit.
- Report perception as neutral if confidence intervals do not support gains.

### G4: mechanism innovation — OPEN
- Design leakage-proof offline counterfactual/trace replay at several receiver buffer sizes; show actual LBRR availability and carrier-arrival deadlines, *not merely network drop flags*.
- Perform codec-only vs network-only decomposition (critical in previous STOI experiments).
- Implement one causal real-time controller and benchmark compute/memory/timing overhead on the NUC.
- Validate a quality-vs-network-bytes-vs-latency Pareto frontier against equal-budget Random, Reactive, Always, naive risk-only, phase-only and oracle.
- Repeat on at least one external Starlink terminal; compare against wired control and multiple network/load conditions.

## 5. What the existing evidence means

The Phase I audit covers two Pisa traces (137 replay loss instances but only 89 distinct positions, repeated across conditions). At the old 60-ms RTT-derived *proxy* deadline, there were 139 primary deadline failures, and 45 timely predictive recoveries versus 81 Always-FEC recoveries. These figures are **exploratory** and cannot be treated as independent evidence or as the new SV02 receiver-local endpoint.

In a 94-condition, loss-exposed exploratory STOI comparison, predictive-minus-random mean STOI was about +0.000099, i.e. extremely small; always-FEC was worse than predictive by about -0.008389. The codec-only counterfactual suggests protection placement itself can cause perceptual distortion, motivating the proposed codec-cost component. Neither result is a multi-session QoE claim.

Sources: `docs/G1_PHASE1_AUDIT.md`, `docs/G1_REALTIME_ENDPOINT_LIMITATIONS.md`, `docs/preregistration-confirmatory-v1.md`; preserved 2026-10-07 exploratory transcript.

## 6. Decision gates / kill criteria

- **No forecast horizon** useful after causal feature and deadline tests -> stop predictive claim; study robust/reactive receiver instead.
- **Protection cannot arrive by deadline** on natural losses -> do not propose more LBRR duty cycle; test a burst-capable mechanism or bounded delay/latency trade-off.
- **Improvement only at matched nominal duty cycle but not matched wire bytes** -> report as a cost/confounding issue, not an efficiency win.
- **Positive timely recoveries but neutral/negative DNSMOS/STOI** -> separate network-efficiency contribution from perceptual benefit.
- **Only works after tuning per test session/site** -> no generalization claim.

## 7. Immediate next actions

1. G2 physical path verification on NUC; save output in pilot manifest.
2. Copy Oracle pilot-002 receiver artifacts to the NUC, audit full session and archive.
3. Run a *new* deliberate-loss session (distinct output directories) and demonstrate where real LBRR succeeds/fails.
4. Only after the above, design randomized multi-session collection and compute power from pilot variance.
5. Keep new mechanism code exclusively on `research-sigcomm-deadline-codec` until independently evaluated.
