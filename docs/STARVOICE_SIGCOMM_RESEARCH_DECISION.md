# STARVOICE — scientific decision and falsifiable contribution, 20 October 2026

**Scope:** SIGCOMM 2027 research track only. Offline analysis of archived Starlink nine-block campaign and external published trace validation. **Not a paper-acceptance promise.** No fresh Starlink acquisition, no modifications to Oracle or the frozen `phase2-confirmatory` path.

## 1. What the data say (not the hoped-for story)

- 217,851 scored 20-ms frames = 4,357.02 s of evaluated speech. Fixed60 has 82 modelled PLC frames. Phase-only physical SlackBank has 15. It recovers 67 frames, **only 1.34 s or 0.0308% of the evaluated audio**.
- SlackBank edits 11,600 to 11,646 frames, depending on the exact scored and boundary scope, to borrow and repay up to 20 ms of playout slack. The original edit primitive modifies +/-48 samples with a 5-ms local overlap-add region.
- The selective full-ramp **future-knowing oracle** activates 65/290 cycles, retains all 67 recoveries, and performs 2,600 edits. Past-one-cycle gating gets only 22/67; past-two-cycle only 31/67. Therefore neither simple causal predictor qualifies. **65/290 is not a true global optimum:** a recovered frame may need less than the full +20 ms ramp.
- On selected windows DNSMOS OVRL mean is: fixed60 3.22571; fixed80 3.23054; real NetEQ 3.24597; SlackBank 3.20033. WER fixed60/NetEQ 0.03977, fixed80/SlackBank 0.04211. NetEQ 488 PLC GetAudio 10-ms callbacks do **not** compare directly with 20-ms frame PLC. NetEQ internal delay and receiver-relative deadlines have not been equalized.
- A single campaign cannot justify cross-day or cross-terminal inference, and reconstructed Opus packet bytes have not been proved bit identical.

**Consequence:** maximizing PLC reduction alone is not the right speech objective. A tiny amount of rescued content may not compensate for diffuse, continuous time-scale artifacts. This is a research hypothesis, not yet a causal measurement of acoustic penalty.

## 2. Precise paper-worthy *question*, not an invented novelty claim

> When deadlines are violated by correlated LEO queuing, what is the minimum **physical audio edit and playout-delay cost** needed to make speech decodable in time, and can a **causal receiver** approach that bound without paying for an audible distortion every predictable cycle?

Possible contribution: **deadline-feasible, distortion-budgeted recovery**, jointly modeling Opus primary/LBRR usability, physically realizable playout sample clocks and cost of audio edits. This could be scientifically distinct from a simple phase-tuned jitter buffer only if it proves a real **quality–latency frontier** improvement over comparable NetEQ and older time-scale / silence-aware methods. The model alone or speech-aware time scaling alone is **not** a novelty claim.

### Formal variables

- Source frames `i` are 20 ms, 48-kHz PCM 960 samples; `a_i` is receiver-local packet arrival time, not cross-host OWD.
- `D_i(60)` is the receiver-relative baseline deadline, anchored to the first receiver packet, and `F_i(b)` is true iff frame i is playable by `D_i(60)+b`, using its primary or **actual Opus-LBRR in i+1 arriving by i's deadline**.
- `b_i∈{0,…,20}` ms is the additional playout reserve above 60 ms. `u_i=b_{i+1}−b_i∈{−1,0,+1}` ms is physical sample-count edit `48u_i`, so `0≤b≤20` and `|u|≤1` per frame.
- `c_i(u_i)` estimates splice distortion using only current decoded PCM; initial thresholds are trained on the first 120 s of the same block (excluded from scored suffix). Neither future audio nor future packet outcomes can be used by a causal policy.

### Noncausal oracle — exact DP and bounded claims

For each frame the existing phase policy rescues, compute its **minimum necessary reserve**

\[
 r_i=\min\{b\in\{1,\ldots,20\}: F_i(b)=1\}.
\]

Find a physically feasible buffer trajectory with `b_i ≥ r_i` at these rescued frames, zero reserve at the start/end and halfway between 15-second events, and `|u|≤1`:

\[
\operatorname{lexmin}_{b_0,\ldots,b_N}
\left(\sum_{i\geq 6000}b_i,\;\sum_i |u_i|\right),
\quad\text{or}\quad
\operatorname{lexmin}\left(\sum_i |u_i|,\;\sum b_i\right).
\]

These are **two exact lexicographic resource bounds**, not a simultaneous global minimum of both objectives. Dynamic programming visits 21 reserve states and at most 3 transitions per frame, i.e. O(63N) time and O(21N) traceback storage. The oracle knows all actual arrivals and is therefore NOT deployable. Its constraints guarantee preservation of the original 67 rescues; additional unexpected timely recoveries may occur.

A true full-rate conversational-latency bound remains unknown, because receiver-local playout reserve is not mouth-to-ear delay.

### Causal comparison — fixed parameters before analyzing held-out audio

1. Estimate 15-second phase using only the 120-second training prefix (existing validated code).
2. Require `b_i ≥ b_i^phase` **at every frame**, guaranteeing the original phase-only modeled deadline recoveries remain possible. The trade-off is potentially increased mean buffering; report that honestly.
3. Permit early 1-ms additions in the **40 frames before** a predicted phase event, and delayed 1-ms repayments in the **45 frames after** it, up to the existing peak 80 ms. Classify an edit as cheap if current decoded PCM overlap mismatch is at/below the **25th percentile** of its direction's training costs. Otherwise enforce the target reserve by physical one-ms-per-frame fallback.
4. Render real continuous PCM using the existing local overlap-add primitive, not free packet decision shifts. Decoder decisions are made at the actual dynamically realized deadline, then the decoded current frame determines the next actuation.
5. Do not change encoding, bitrate, FEC duty cycle, or observed packet arrivals. The candidate isolates the **timing of distortion**, not a joint codec effect.

**Engineering gate frozen before seeing these results:** PLC no more than 15; peak 80 ms; mean added receiver-local buffer no more than 2 ms per block; paired DNSMOS candidate-minus-original-phase at least +0.03 **separately for event and control** windows; candidate at least fixed60 in both groups. A pass is **not** a SIGCOMM GO. A fail means do not tune these hyperparameters on the same evaluation suffix.

## 3. Required scientific figures / measurement outputs

1. **Opportunity ceiling:** histogram of `r_i` minimum needed slack (1..20 ms), per-block exact oracle minimal average budget and edit operations; overlay the old 65-cycle full-ramp oracle. Show what is inherently avoidable even with noncausal information.
2. **Mechanism:** causal physical reserve and PCM sample-clock traces, edited-frame counts, measured CPU, 60/80ms peak and mean receiver-local buffer. Separate information, actuation, and audio costs.
3. **Quality at the same speech positions:** 6-second matched windows for rescued and no-rescue controls, paired DNSMOS OVRL and diagnostic STOI. Report block-level heterogeneity. WER and ITU-T P.804/discontinuity listening remain necessary for a full audible-quality claim; objective metrics alone are insufficient.
4. **Actual NetEQ:** existing pinned native WebRTC binary and original rtpdump audio are genuine. Do **not** compare its 488 x10ms PLC callbacks with 20-ms modeled PLC. Implement fair equal-cost config and account for NetEQ's actual internal and peak delay before asserting superiority.
5. **Independent networking dimension without new private Starlink captures:** preregister a trace-subset selection and train/test-by-site/day split using public `Dissecting the StarLink` per-packet dataset, `LENS`, and/or the hardware-clocked 10-day Starlink OWD dataset. Packet-cadence/traffic mismatch must be disclosed. These are **external network traces**, not live voice QoE replications.

## 4. Prior art that forbids easy novelty claims

- WebRTC **NetEQ** already optimizes delay and dynamically accelerates / preemptively expands PCM: https://webrtc.googlesource.com/src/+/1331c1821cae1c19a330fd042b6e33829427f235/modules/audio_coding/neteq/g3doc/index.md
- **Silence-aware and intra-talkspurt adaptive playout** predate us, e.g. Zhang et al., *A GARCH-based adaptive playout delay algorithm for VoIP*, Computer Networks 2010: https://www.sciencedirect.com/science/article/abs/pii/S1389128610001726
- **Packet-based time-scale modification under an average delay bound** predates us (Liu et al., ICASSP 2001): https://doi.org/10.1109/ICASSP.2001.941206
- **Content-aware playout + transmission** and DP cross-layer multimedia optimization have precedent (Li et al., IEEE Transactions on Multimedia 2008): https://www.researchgate.net/publication/3424764_Content-Aware_Playout_and_Packet_Scheduling_for_Video_Streaming_Over_Wireless_Links
- Starlink 15-second queue-control resets are described by Cech, Mohan and Ott at SIGCOMM 2026: https://doi.org/10.1145/3789240.3829162
- Deep audio PLC/perceptual challenges include ITU-T P.804 listening, latency budgets and word recognition; use them as evaluation guidance: https://www.microsoft.com/en-us/research/academic-program/audio-deep-packet-loss-concealment-challenge-icassp-2024/

## 5. Kill criteria and deliverables

- If even the causal speech-aware controller cannot remove the audio penalty on held-out suffixes at reasonable latency, **reject this candidate**, don't grid-search it on the same nine blocks.
- If external traces do not exhibit deadline-rescuable bursts under relevant 20-ms audio packet pacing, **do not claim generality**.
- If real NetEQ at comparable average AND peak delay performs as well as the candidate, **do not claim system superiority**.
- If WER / listener discontinuity and paired quality disagree, **no headline based on PLC alone**.
- The one-command study outputs `FINAL_DECISION.md`, `FINAL_DECISION.json`, `paired_quality.csv`, per-block controller evidence, and original-intact checks. The exact DP bounds and causal policy have independent synthetic unit tests but are not yet validated on the NUC. All test conditions and thresholds were frozen before inspecting the new candidate's nine-block quality results.

**This is the next falsifiable engineering/research gate, not yet final validated publication code. A memorable paper must emerge from positive, independent evidence, not from adjectives in its title.**