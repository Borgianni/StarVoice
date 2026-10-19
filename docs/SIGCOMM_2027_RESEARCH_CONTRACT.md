# STARVOICE SIGCOMM 2027: Research Contract v1

**Date:** 2026-10-19. **Status:** FROZEN RESEARCH DIRECTION, NOT FROZEN NEW EXPERIMENTAL PROTOCOL. **Branch:** `research-sigcomm-deadline-codec` only. This document is a strategic research decision, **not evidence that any new scientific contribution has already been demonstrated**. Do not edit the pre-existing `phase2-confirmatory` preregistration or predictor based on this document.

## 1. ONE paper thesis

**Protect useful audio, not predicted network losses.** On deadline-constrained LEO voice paths, packet-loss risk is a poor proxy for the **causal, deadline-feasible marginal utility** of codec protection. A good sender must account jointly for (i) primary unusability, (ii) survival and **timely** arrival of the redundancy carrier, (iii) actual stateful codec redundancy realization, (iv) protection-induced codec distortion, and (v) real wire-byte/latency cost. We will design and evaluate a low-complexity **recoverability-aware protection controller** that chooses when to enable Opus LBRR versus leave it off. This is a falsifiable hypothesis, not a claim of novelty by wording alone.

A pure sender-side switch cannot recover audio during an interval in which both primary and LBRR carrier miss their deadline. Such intervals are a **feasibility boundary**, not a controller failure that can be fixed by optimizing the same LBRR flag. Alternative actions (temporally diverse redundancy, path diversity, or bounded latency adaptation) are explicitly out of scope for the first minimal prototype; each would require a separately implemented and validated receive path / latency budget. If the feasibility ceiling is too low, report a negative result rather than making impossible QoE claims.

## 2. Exactly THREE candidate contributions

**C1. Deadline-aware recoverability characterization.** A reproducible decomposition of primary failures into: (a) primary missing vs receiver-observed late; (b) next-carrier missing/late/on-time; (c) LBRR requested vs actually emitted; (d) FEC decodable by the deadline; (e) net codec-only quality cost. Provide achievable recovery ceilings under *the explicitly defined endpoint*, burst length, time-of-day and multiple independent sessions/sites. Do NOT claim discovery of Starlink's known ~15 s reconfiguration period or the known single-frame Opus LBRR limitation.

**C2. Causal marginal-utility controller.** At encoding decision time t, estimate the conditional incremental value of enabling Opus LBRR from *past-only* network and codec state, not hindsight labels:
`Delta U_t = E[Q_playout(on) - Q_playout(off) | history_t] - lambda * Delta(wire bytes) - mu * Delta(latency)`.
Operationally use a **lightweight, bounded-memory, calibrated** predictor with separate components for primary failure, carrier deadline feasibility and codec realization/cost; a deliberately simple gate or table-based model is acceptable. Crucial: actual LBRR emission can be stateful and a current FEC flag is not equivalent to a carrier with LBRR. Estimate cost causally using a separate offline training split; do not peek at current encoded frame's future LBRR or receiver arrival. Varying codec state may change subsequent frames: evaluate exact encoded *schedules*, not independent isolated decisions.

**C3. End-to-end implementation and matched-cost external validation.** Demonstrate the system in the existing SV02 real-time NUC->Oracle architecture, then reproducible held-out trace replay and independent live sessions/days, including at least one **external Starlink terminal/site** and a wired control. Compare *paired* recovered audio quality, timely useful recoveries, protection efficiency and actual network bytes under consistent playout limits. This is a systems contribution only if gains generalize and are larger than measurement/model uncertainty.

## 3. Ground truth available on 2026-10-19 (exploratory only)

- Phase I: two Pisa network traces, 181,396 replayed audio frames but only **89 distinct underlying loss positions** behind 137 replay loss instances. At old **RTT/2 proxy** 60-ms endpoint: 45 Predictive and 81 Always timely recoveries; Reactive 13 at different duty cycle; Random equal-requested budget nearly zero in earlier replay. QoE improvement **not** established; codec schedule distortion can offset network recoveries. These are neither independent sessions nor the SV02 endpoint.
- G2: real LBRR can recover one deliberately dropped frame; adjacent 2-loss burst (8,9) yielded 1 PLC + 1 FEC; triple burst (9,10,11) yielded 3 PLC because two carriers dropped and the last lacked LBRR. Injected impairment proves mechanisms, **not** organic performance.
- Natural Starlink->Oracle 10-minute capacity pilot `bc7f9214-879f-40e0-887b-dbb286c2725e`: **30,045 frames**, 30,044 received unique packets, 30,020 Primary, 1 FEC, 24 PLC, **0 injected drops**, audit valid. One primary packet was not observed and was FEC-recovered; the 24 PLC primary packets were eventually receiver-observed late. In **21/24** PLC cases the *following* packet contained actual LBRR but **0/24 carriers** arrived by the corresponding receiver-local playout deadline. Thus that session has **only one timely LBRR opportunity among 25 primary failures**, already recovered. This is a single-session, post-hoc, receiver-relative observation, **not** a global bound, not a causal proof of network-side latency, not proof of an adaptive controller advantage, and not speaker-to-ear OWD.
- A ~15-second periodic receiver-arrival disruption appears in that pilot with near-regular sender spacing; **this periodicity is known prior work**, not claimed as a contribution or literal per-event satellite handover evidence.

## 4. Prior-work novelty boundary (must cite and inspect in the paper)

- SIGCOMM 2026 **Dissecting the StarLink** establishes 15-s reconfiguration/queuing, so timing periodicity is background: https://doi.org/10.1145/3789240.3829162
- SIGCOMM 2025 **LeoCC** uses LEO reconfiguration awareness for congestion control: https://doi.org/10.1145/3718958.3750491
- 2025 **Dynamic Pacing for Real-time Satellite Traffic** predicts handover for RTC pacing: https://arxiv.org/abs/2507.09798
- NSDI 2025 **Tooth** already performs QoE/redudancy-cost-aware per-frame FEC in cloud gaming: https://www.usenix.org/conference/nsdi25/presentation/an
- RFC 8854 describes Opus next-packet FEC and its inability to cover multiple consecutive lost packets: https://www.rfc-editor.org/rfc/rfc8854.html
- SIGCOMM 2025 **ACE** jointly adapts encoder/network pacing for RTC: https://doi.org/10.1145/3718958.3750520
Therefore *generic adaptive FEC*, *generic deadline-aware coding*, *15-s prediction*, *Opus LBRR limitations*, and *FEC overhead* are not novel individually. The potentially distinguishing contribution is **evidence and a deployable controller for *prospective codec-realized timely speech-recovery utility* with explicit counterfactual codec degradation, on independent LEO traces**, tested against strong scheduling controls. Full literature review remains required before writing a novelty claim.

## 5. Evaluation protocol (NO shortcuts)

1. **Measurement validity first:** finish G2 controlled late/reorder/startup tests; timestamp kernel or independently bounded clock uncertainty when claiming one-way latency; keep SV02 receiver-relative 60ms endpoint separate from Phase I RTT/2. Record sender pacing, receiver dispatch, archive hashes, audio content and config. The 1GB Oracle VM showed rare dispatch spikes; no unconditional exact OWD claim.
2. **Pilot planning:** 3-5 new *independent* network sessions across days/time strata, with zero-loss sessions retained. Workload: real LibriSpeech frozen utterance set and order, SHA256. Estimate usable event yield, between-session variance, RAM/tmpfs capacity, and power. Do not announce a final confirmatory N before observing pilot variance; plan for 80% power at two-sided alpha 0.05 according to the existing preregistration.
3. **Dataset separation:** separate initial calibration, training, validation and untouched final evaluation at the **session/day/site** level; identical network trace can be replayed against all policies, but replications of utterances over the same loss positions are not independent.
4. **Frozen confirmatory Track A:** keep existing phase-only model, frozen risk half-width 0.2s, Opus 20ms/24kbps/loss hint20, nominal 60ms primary and 40/50/80/100 secondary. Primary effect predictive minus exact **equal protected-frame count** Random, analyzed per independent session with cluster-bootstrap 95% CI; other comparisons secondary. Any SV02 endpoint change needs a **versioned prospective amendment before confirmatory outcome inspection**. Do not turn exploratory Track B results into preregistered A results.
5. **New exploratory Track B:** compare **Plain, Always, Reactive, frozen Phase-only, Random exact equal requested protection, Random block/window-matched schedules, proposed Recoverability-aware**, and labeled **noncausal Oracle upper bound**. Also evaluate actual wire-byte-matched controls, because equal requested protection does not imply equal bytes, and enforce encoder state/history consistency. Replay codec outputs with exact frozen schedules. Include per-frame costs and utility ablations: loss-only vs carrier-feasibility-only vs codec-cost-only vs joint controller.
6. **Outcomes:** timely FEC audio recovery and primary deadline failures with error-bound sensitivity; paired DNSMOS P.835 OVRL (perceptual), STOI, WER exploratory; original decoded audio vs codec-only vs network-only decomposition; actual UDP/wire bytes, latency budget, CPU/memory, dispatch p95/p99/max, feedback costs, failure bursts and calibration. Primary QoE improvement claim only with positive relevant clustered CI, otherwise report null.
7. **Invariance and fairness:** report held-out session/site outcomes, no selective exclusion of low-loss sessions, no post-hoc move of deadline, same audio and trace for pairings, explicit distinction between natural and injected losses, and pre-declared sensitivity for clock offset and path asymmetry.

## 6. Figures to earn, not draw prematurely

- F1: **Recoverability map**: probability of primary deadline failure, carrier deadline feasibility, actual LBRR realization and decodable FEC vs burst length, relative deadline and path condition; per-session bands.
- F2: **Accuracy/calibration** of *recoverability forecast*, not just loss forecast, at 50/100/250/500/1000-ms horizons, independent held-out sessions.
- F3: **Paired controller utility gain** vs strongest *equal-byte*, temporal-window-matched and phase-only controls, session-level dots and 95% cluster-bootstrap CIs.
- F4: **Quality–byte–latency Pareto frontier**: DNSMOS/STOI and timely recoveries vs transmitted bytes and playout budget.
- F5: **Generalization** across days, second terminal/site, wired control and risk regimes; include failures.
- Table: **system cost and reproducibility**: code commit, Opus version, receiver timestamp method, CPU/memory, dispatch tails, hashes and dataset stratification.

## 7. GO / NO-GO decision rule

- **GO** only if held-out, genuinely causal estimates identify a meaningful *recoverable* subset and the new controller improves net QoE or timely recovery efficiency **relative to frozen phase-only and strong window/byte-matched controls**, with session-level uncertainty compatible with a robust effect and cross-site replication. At minimum, quality must not be misrepresented if negative.
- **PIVOT** if Opus LBRR has too few deadline-feasible opportunities: stop claiming better Opus placement will fix correlated latency episodes; choose a *separately validated* intervention that can actually affect deadline feasibility (e.g., explicitly priced bounded latency adaptation or independently routed redundancy), or publish characterization/null results in a fitting venue.
- **STOP SIGCOMM-level systems claim** if improvement exists only against frame-wise sparse Random, only in reused same-site trace positions, only with OWD assumed RTT/2, only at a post-hoc deadline, or if no new independent session/site effects remain.

## 8. Execution ordering and scope freeze

**P0** Preserve existing G2 and natural-run artifacts; finish missing timestamp/late/reorder/startup correctness gates **without redoing completed single/drop/burst tests**.
**P1** Prepare reproducible independent acquisition format and 3-5 pilots; size confirmatory campaign.
**P2** Run Track A as preregistered (or prospectively amended for its endpoint), do not alter frozen policy.
**P3** Implement minimal Track B controller OFF vs Opus LBRR in the research branch, evaluate matched costs and decoder outcomes on disjoint data; add sophisticated redundancy only after a separate feasibility finding.
**P4** Cross-site held-out testing, publication-quality analysis and figures. No unrelated feature expansion (WebRTC/DRED/VAD/neural large models) before P3 evidence.

**One-sentence paper question:** *Can a causal sender predict the incremental, deadline-feasible AUDIO QUALITY value of Opus protection, rather than merely network impairment, and thereby outperform equal-cost protection baselines on real LEO links?*
