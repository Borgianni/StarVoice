# STARVOICE — deadline–distortion duality: frozen research gate (2026-10-20)

**Status:** exploratory and falsifiable. Not a SIGCOMM contribution until measured and independently replicated. The original SV02 campaign, frozen phase2 policy, and Oracle remain unchanged.

## Evidence already observed

- Nine correlated 10-min natural blocks from one site/day; 217,851 post-training 20-ms frames.
- Fixed60 modeled PLC 82, phase-only SlackBank 15, fixed80 9. SlackBank rescues 67 frames = **1.34 seconds = ~0.0308%** of scored speech; the physical implementation performs ~11,600 temporal PCM edits.
- The previously reported NONCAUSAL full-ramp selector activates 65/290 events and needs 2,600 PCM edits to keep all 67 rescues. Causal previous-one-cycle and previous-two-cycle gates retain only 22 and 31 rescues respectively. Both FAIL the preregistered >=90% recovery / <=50% activation gate.
- DNSMOS OVRL on selected windows: fixed60 3.22571, fixed80 3.23054, NetEQ native 3.24597, phase SlackBank 3.20033. WER did not improve. **No evidence of QoE superiority**. Native NetEQ PLC GetAudio callback counts are not comparable with the 20-ms modeled PLC and its internal delay is not matched to the receiver-relative budget.

## What the earlier oracle did NOT establish

The selector using 65 event cycles is NOT the minimum required actuation cost: its activated event always borrows the full +20 ms. A rescued packet may require much less incremental slack.

For source frame i let F_i(b) denote actual primary-or-timely-real-Opus-LBRR decodability at receiver-relative baseline deadline D_i(60)+b. For each of the original 67 rescued frames, compute:

    r_i = min { b in [1,20] integer ms : F_i(b) is true }.

This is retrospective, explicitly NONCAUSAL.

## Exact physical oracle

Let extra playout reserve b_i ∈ {0,...,20}, physical time-edit u_i=b_(i+1)-b_i ∈ {-1,0,1}, and each step lengthen/shorten current 48kHz 20ms PCM by 48 samples. Constrain b_i >= r_i for each saved frame, return to 0 at the start/end and halfway between adjacent 15-second events to avoid the trivial always-high-buffer answer. Solve exactly, using 21-state/3-transition dynamic programming:

1. Lexicographically minimize (sum_i b_i, sum_i |u_i|).
2. Separately lexicographically minimize (sum_i |u_i|, sum_i b_i).

The solutions provide **two distinct network/actuation resource lower bounds** under their explicit physical constraints, NOT optimal speech QoE and NOT deployable controllers. Runtime O(63N), traceback O(21N).

## Frozen causal quality candidate

- 15-second disruption phase from prior 120-second training prefix, as in existing phase-only SlackBank.
- Budget b_i always **at least** the frozen causal phase-ramp reserve, to preserve its modeled deadline recoveries (at possible extra average latency).
- At most one ms per current 20ms decoded frame. Optional early reserve placements begin 40 frames before a phase event; repayments can continue 45 frames after. A candidate edit uses **only current decoded PCM** and is preferred when local splice mismatch is at/below the 25th-percentile threshold calibrated on the first 120 seconds of that block.
- Hard peak total budget 80ms, total added mean per block <=2ms. Full Opus input remains Always-FEC, no new wire-byte/codec claim. All changed PCM follows the sample-clock invariant.
- Fixed comparison windows: six seconds, original phase-scheduled rescued/control centers. Report paired DNSMOS OVRL and diagnostic STOI, per block and by event class.

**Predeclared engineering gate:** candidate retains <=15 modeled PLC, peak <=80ms, mean extra <=2ms; in BOTH rescued and no-rescue windows candidate-vs-original-phase DNSMOS difference >=+0.03 and candidate-vs-fixed60 >=0. Failing this rejects this single frozen candidate; no suffix retuning. Passing is not a SIGCOMM GO.

## Why this could matter, and why novelty is still OPEN

The opportunity is a recoverability benefit paid for by acoustic time modification and conversational buffering, *not* simply the existence of Starlink 15-second periodicity (SIGCOMM 2026), jitter-buffer adaptation (NetEQ), silence-based scaling (VoIP prior art since at least 2001), or generic multimedia delay–distortion optimization (prior art). Any novelty must reside in a provably differentiating deadline-feasible/actual-codec/perceptual resource model PLUS experimentally superior causal receiver.

The independent dimension can be checked without further private Starlink acquisition using public Starlink packet datasets, e.g. Cech–Mohan–Ott SIGCOMM'26 traces, LENS multi-node traces and Garcia–Sundberg synchronized OWD traces. Public network traces are **not** independent live Opus audio endpoints. Prefer externally withheld sites/days; report failure to generalize.

**Hard stop:** No quality advantage at comparable average+peak playout delay vs actually measured NetEQ; or no independent trace support; or no convincing listening/discontinuity/recognition gain -> no SIGCOMM superiority claim.

Relevant prior art and public data:
- https://webrtc.googlesource.com/src/+/1331c1821cae1c19a330fd042b6e33829427f235/modules/audio_coding/neteq/g3doc/index.md
- https://doi.org/10.1109/ICASSP.2001.941206
- https://www.sciencedirect.com/science/article/abs/pii/S1389128610001726
- https://doi.org/10.1145/3789240.3829162
- https://github.com/hendrikcech/sigcomm26-dissecting-starlink
- https://github.com/clarkzjw/LENS
- https://zenodo.org/records/16275284

A standalone research study program, source code and eleven local synthetic unit tests are furnished in the associated ChatGPT conversation. These are **not yet NUC-validated or committed as a source file to GitHub**.