# STARVOICE: one-window 90-minute field acquisition (2026-10-19)

**Status:** prospective exploratory campaign design. **Branch:** research-sigcomm-deadline-codec. Not the frozen Phase II confirmatory sample. **Hard operational maximum:** 114 minutes after the three workloads are prepared, plus startup/preflight, i.e. safely less than the user's two-hour limit under normal operation; the script refuses to start a block if the projected duration cannot fit. This run must NOT be described as independent multi-day replication.

## Why this ONE run is high value

In the first real 10-minute NUC->Oracle natural-path run (30,045 audio frames) there were 25 receiver-relative primary deadline failures, 24 PLC, 1 useful Opus LBRR recovery. All 24 PLC following packets missed their recovery deadline, even though 21 carried LBRR. Fixed recorded-arrival feasibility analysis at B=40/50/60/80/100ms found primary misses 77/52/25/4/2 and a single timely LBRR opportunity across all buffers. This is a potentially severe ceiling on the scientific value of changing only the Opus FEC flag.

One contiguous **~90.6-minute real-network campaign** supplies approximately nine times as much time exposure while varying speech content and observing many network impairment cycles. Split into 9 sessions of ~10min each to avoid the 1GB Oracle VM's bounded /dev/shm and increasing in-memory packet dictionary. Source corpus: the existing SHA-verified ~30.21-minute, 236-utterance LibriSpeech set, partitioned without cutting utterances into three contiguous WAV segments. Each segment occurs once per epoch, with the order:

| Epoch | Block1 | Block2 | Block3 |
|:--|:--|:--|:--|
| 1 | A | B | C |
| 2 | B | C | A |
| 3 | C | A | B |

Thus audio-content/time-of-campaign position is counterbalanced. One fixed real-time sender policy **Always-FEC** (Opus 20ms, 24kb/s, expected-loss hint20%), **NO** injected drops. This is used to characterize opportunities when in-band redundancy is offered broadly; it does *not* compare competing policies in identical live network conditions.

For each block, a single orchestrator on the NUC: confirms exact locked deployed Phase II SHA on both machines; checks Oracle tmpfs/RAM >=200MiB at block boundaries; starts Oracle receiver via SSH; sends from NUC Starlink; waits for verified archiving; copies full Oracle logs over SSH; runs the causal auditor; computes five-budget feasibility; saves unique directories and SHA-based provenance. On first error STOP; do not discard/overwrite past blocks. No modification to frozen deployed policy, receiver, Opus implementation or preregistration.

## Outputs

\`results/sigcomm-onewindow-YYYYMMDDTHHMMSSZ/\`:
- \`campaign-manifest.json\`: design, research/deployment commits, NUC route to Oracle, ordered SHA256 workload files, completed session IDs and status.
- \`workloads/segment-*.wav\`: the exact three 48k mono PCM16 inputs, SHA256 hashed.
- \`block-NN/sender\` and \`block-NN/receiver\`: raw two-host SV02 WAV, packets, decisions, clock logs, manifest, archive-integrity, metrics.
- \`block-NN/audit.json\`, \`block-NN/recoverability.json\`, \`block-NN/failure-events.csv\`: raw-audited per-session and five-buffer feasibility evidence.
- \`blocks.csv\`: all nine local-60ms runs and receiver dispatch diagnostics.
- \`frontier-by-block.csv\`: each of 9 blocks at B 40/50/60/80/100ms (45 paired budget rows).
- \`frontier-total.csv\`: **descriptive**, pooled totals of those same nine blocks; never treat pooled packet numbers as independent samples.

## Primary decision after THIS ONE experiment (not inferential SIGCOMM claims)

**F1: fraction of observed primary failures for which next carrier contained LBRR and was on time**, at fixed receiver-relative B=60ms, per block and in total. This is a carrier-feasibility *necessary condition* and LBRR upper bound under the observed always-FEC encoding, not a counterfactual measure of a different FEC schedule.

**F2: fixed-arrival buffer frontier and paid latency.** B=40/50/60/80/100 changes the receiver-relative budget, while holding packet arrival times and payloads fixed. Changes in Primary failures vs 60ms have an explicit +20ms/+40ms nominal buffering cost. This is **not** a deployable dynamic jitter buffer, end-to-end latency, or a QoE result.

**F3: burst/deadline correlation** from packet-arrival logs; do not claim a ~15s Starlink scheduling period is novel. Compare individual block event yields, conditional carrier feasibility, and scheduler outliers. Artifact/late distinctions remain conditional on user-space receiver timestamp quality until a kernel arrival study.

**GO** if there are sufficiently many naturally occurring deadline-feasible carrier opportunities to design and power a meaningful recoverability-aware Opus ON/OFF controller against equal-cost/window-matched baselines in held-out network conditions. An isolated single campaign is model-development/calibration evidence only.

**PIVOT** if the 60ms carrier feasibility fraction is consistently tiny across all blocks, as suggested by the initial 10-minute run. Do not spend months improving a phase-only switch that cannot recover dead carrier deadlines. Instead test a separately priced latency action or temporally diverse redundancy, ensuring that codec quality, average/max latency and actual extra bytes are treated as costs rather than free benefits. If neither action improves quality at bounded cost, report the negative mechanism finding.

## What is NOT established by one run

Nine contiguous blocks are not independent sites, days, or randomized live policy treatment arms. Replaying the exact network arrival timeline against re-encoded Opus policies gives useful paired *offline* control under a path-invariance approximation, but changing packet length or policy could affect network queues in reality. Accordingly, the original frozen Phase II preregistered session-level confidence interval across independent sessions cannot be replaced by an artificial N=9. This study selects and rejects mechanisms cheaply; eventual SIGCOMM generalization needs held-out data, ideally delegated to other terminals or future unattended monitoring.
