# One-window readout and hypothesis pivot

Date: 2026-10-19. Nine consecutive, same-site/day natural Starlink-to-Oracle Always-FEC blocks, 90.6 minutes / 271,851 20ms frames. This is exploratory calibration, NOT multi-day independent evidence.

| Receiver-relative buffer ms | Primary failures | Actual timely LBRR | PLC | Never observed Primary | Observed-late Primary |
|---:|---:|---:|---:|---:|---:|
| 40 | 585 | 4 | 581 | 7 | 578 |
| 50 | 282 | 4 | 278 | 7 | 275 |
| 60 | 100 | 4 | 96 | 7 | 93 |
| 80 | 15 | 4 | 11 | 7 | 8 |
| 100 | 8 | 4 | 4 | 7 | 1 |

**Decisive fixed-arrival inference:** At 60ms, all four on-time following LBRR carriers were used. Among 100 failed Primaries, 93 were eventually observed late and 7 were never observed: of those 7, 4 had timely LBRR and were recovered, and 3 next carriers were not observed. The dataset contains 0 timely following carriers without LBRR. Thus the four timely following-carrier opportunities were already fully exploited in this specific always-FEC stream. A different ON/OFF FEC schedule cannot recover the 96 PLC under the *same fixed arrivals*, though encoding changes can alter bytes and timings in a live network.

60->80ms fixed playout budget reduces modeled PLC from 96 to 11 (-88.5%), with +20ms nominal delay *throughout the entire session*. 60->100ms reduces 96 to 4 (-95.8%), with +40ms. Neither proves any real dynamic jitter buffer, speech-quality benefit, or absolute end-to-end delay.

## SINGLE NEW RESEARCH MECHANISM HYPOTHESIS (not a novelty claim)

**STARVOICE SlackBank**: use past-only LEO-path signals to predict a near-future disruption, progressively accumulate strictly bounded extra playout slack ahead of it via pitch-preserving audio time-scale expansion, then gradually pay back the delay via acceleration. Couple the playout decision to the *actual probability the next Opus LBRR carrier can arrive before that planned deadline*, while accounting for encoder-state, speech-quality and wire-byte effects. This is prospective joint playout/codec utility control, NOT just adaptive FEC or a 15-second phase detector.

**Physical feasibility:** frame i deadlines must obey D(i+1)-D(i) = 20ms + B(i+1)-B(i). Any time-varying buffer B(i) must be connected to actual audio pacing/time-stretching; instant +20ms per endangered frame is not valid. Initial idealized model: 60<=B(i)<=80ms; |B(i+1)-B(i)|<=1ms per 20ms frame (5% stretch, approximated); mean buffer no more than 62ms; outputs are modeling upper bounds until real time-scale modification is executed and scored.

**Novelty boundary:** WebRTC NetEQ already performs adaptive jitter buffering, time-stretching (accelerate/preemptive expand) and loss concealment; handover-aware transmission pacing and Opus LBRR limitations are prior art. NetEQ documentation: https://webrtc.googlesource.com/src/+/c8abb0469dfb1ea4c5a55a56e897bafebbfdaf8d/modules/audio_coding/neteq/g3doc/index.md. Any paper must compare against a REAL pinned NetEQ implementation at equal max AND average playout latency, speech quality, true wire bytes and compute cost, plus fixed60/fixed80, phase-only ramp, reactive, and noncausal oracle.

## Freeze order: do NOT ask user for another natural campaign now

1. Offline prospective *idealized* feasibility on existing 9 blocks, first 120 seconds per block used for strictly past-only local phase estimation and NOT scored. Evaluate fixed60, fixed80, fixed100, phase-only smooth 60..80 ramp, and if useful an oracle smooth upper bound. Report per-block PLC, FEC, mean/max playout, fraction of stretched frames. No tuning on held-out part.
2. If prospective 60..80 ramp fails to substantially beat fixed60 under <=2ms *mean* additional receiver-relative budget and <=80ms peak, STOP SlackBank. Otherwise implement REAL stretch/concealment and compare to native NetEQ.
3. Add stateful Opus codec ON/OFF policy with incremental QoE and byte cost, using whole-stream schedule re-encoding; evaluate against equal temporal-window and byte controls.
4. Evaluate paired real decoded audio DNSMOS/STOI; clustered results and independent site/day generalization are still necessary for external SIGCOMM claims. One 9-block window cannot replace them.

Do not change the frozen phase2-confirmatory preregistration; this is exploratory mechanism Track B.
