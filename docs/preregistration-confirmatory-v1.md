# StarVoice Confirmatory Study Preregistration v1

Status: **FROZEN before confirmatory data collection**

This document separates the completed exploratory phase from the next confirmatory
validation phase. Any deviation must be versioned before inspecting confirmatory
outcomes and reported explicitly.

## 1. Confirmatory question

Does the frozen StarVoice predictive-FEC policy place protection more efficiently
than uninformed or reactive protection for real-time voice, when usefulness is
defined by the playout deadline rather than packet delivery alone?

The confirmatory study does **not** test whether Starlink has a 15 s periodicity,
does **not** claim that periodicity is caused by handover, and does **not** treat
Always-FEC as an oracle.

## 2. Frozen system

The deployed candidate is the existing phase-based predictive FEC policy.

Frozen elements:
- Opus frame duration: 20 ms.
- Opus bitrate: 24 kb/s.
- Opus in-band FEC expected-loss setting: 20%.
- Predictor family: phase-only model already implemented in StarVoice.
- Period estimation procedure: existing calibration procedure.
- Risk half-width: 0.2 s.
- No causal-fusion model.
- No deadline-feasibility gate.
- No tuning on confirmatory sessions.

Any future model trained using confirmatory sessions belongs to a separate study
and requires session-level train/test separation.

## 3. Primary operating point

Primary playout budget: **60 ms**.

This value was selected after exploratory analysis and is therefore not itself a
discovery. It is frozen here before confirmatory collection.

Secondary playout budgets: 40, 50, 80, 100 ms.

Only the 60 ms result is confirmatory-primary. Secondary budgets are robustness
and mechanism analyses and must not replace the primary result post hoc.

## 4. Primary endpoint

For each independent network session, replay the frozen speech workload and
compute the number of **timely FEC recoveries** at 60 ms.

A timely FEC recovery is a lost source frame whose primary packet is unusable and
whose following packet:
1. contains actual Opus LBRR for that source frame; and
2. arrives by that source frame's playout deadline.

Primary comparison:
- StarVoice predictive FEC
- equal-budget Random-FEC

Random-FEC must receive the exact protected-frame count used by Predictive within
the same session/condition. Random uncertainty is estimated with schedule-only
Monte Carlo; a smaller set of actual Opus replay repetitions is retained as an
end-to-end validation check.

Primary effect:
- paired session-level difference in timely recoveries between Predictive and the
  expected equal-budget Random baseline.

Report additionally:
- FEC protected frames / duty cycle;
- timely recoveries per 1,000 protected frames.

## 5. Secondary comparisons

- Predictive vs causal Reactive-FEC.
- Predictive vs Plain Opus.
- Always-FEC ceiling at maximal protection cost.
- Noncausal placement upper bound, labeled explicitly as an upper bound and never
  as a deployable policy.

Reactive comparisons are cost-aware because Reactive is not constrained to the
same duty cycle. Report both timely recoveries and timely recoveries per 1,000
protected frames.

## 6. Delay measurement

Preferred confirmatory measurement is receiver-observed one-way arrival time using
bidirectional timestamps and continuously logged clock synchronization state.

A four-timestamp NTP-style exchange alone does **not** identify path asymmetry
separately from clock offset. Therefore:
- if an external timing reference such as GPS/PPS provides an empirical clock
  error bound, report OWD with that bound;
- otherwise treat one-way delay as uncertain and report explicit sensitivity /
  bounds rather than claiming exact OWD;
- clock offset and stability must be logged throughout each run, not only at
  startup.

The exploratory constant RTT-fraction model remains a sensitivity analysis only.

## 7. Statistical unit and inference

Independent network **session** is the primary statistical unit.

Primary uncertainty:
- cluster bootstrap over sessions, preserving all observations belonging to a
  sampled session.

Do not treat utterances, frames, or individual loss events as independent samples.

For the existing two-session exploratory dataset only, contiguous block bootstrap
may be reported as:
"conditional on these traces".
It is not evidence of cross-session generalization.

Report 95% confidence intervals for:
- Predictive minus equal-budget Random timely recoveries;
- Predictive minus Reactive timely recoveries;
- timely recoveries per 1,000 protected frames;
- QoE paired differences when evaluated.

## 8. Pilot and power analysis

Before the confirmatory campaign, collect **3-5 new pilot sessions** using the
final logging format.

Pilot objectives:
1. verify bidirectional timestamp and clock-state logging;
2. estimate between-session variance and loss-event yield;
3. inspect loss counts by session and predeclared strata;
4. determine confirmatory sample size.

Pilot sessions are not used to tune the frozen predictor.

Sample-size planning will use the session-level paired effect variance observed in
the pilot. Planning may use a parametric paired approximation or bootstrap
simulation, but final inference remains cluster-bootstrap based.

Target:
- 80% power;
- two-sided alpha 0.05;
- effect size used for planning must be recorded before confirmatory collection.

The resulting planned number of confirmatory sessions must be committed in a new
version of this document before those sessions are inspected.

## 9. Session stratification

Because loss is rare, sessions may be deliberately distributed across predeclared
operating strata, for example:
- time-of-day / load windows;
- weather classes when objectively recorded;
- stationary vs mobility conditions, if mobility is part of the intended scope.

Strata must be defined before outcome inspection. Do not selectively retain
high-loss sessions after collection.

Report the full distribution of:
- sent frames;
- losses;
- loss bursts;
- RTT / delay;
- timely-FEC opportunities
per session and stratum.

## 10. QoE

QoE is a secondary endpoint until the network effect generalizes across sessions.

Metrics:
- DNSMOS P.835 OVRL primary perceptual metric;
- STOI secondary;
- WER exploratory / secondary because sparse loss has shown low sensitivity.

All QoE uses deadline-aware replay.

Codec-only counterfactuals use the exact frozen FEC schedules to separate:
- codec/protection distortion;
- network impairment;
- net perceptual outcome.

The exploratory 60 ms result currently shows that Predictive reduces network
impairment relative to Random but that this benefit is largely offset by codec
schedule distortion. Therefore no confirmatory claim of perceived-quality
improvement is preregistered.

## 11. Claim thresholds

Minimum network claim:
- Predictive shows a positive session-level effect versus equal-budget Random at
  the 60 ms primary operating point, with a 95% cluster-bootstrap CI excluding
  zero.

Stronger systems claim:
- Predictive also improves protection efficiency relative to Reactive-FEC under
  the same deadline-aware evaluation.

QoE claim:
- only if the paired 95% cluster-bootstrap CI for the preregistered perceptual
  metric is positive versus the relevant baseline.

If QoE is neutral, the paper must say so and frame the contribution around
deadline-aware recovery efficiency rather than perceived-quality improvement.

## 12. Stop / interpretation rules

- Do not introduce new predictors or controllers after confirmatory collection
  begins.
- Do not move the primary playout budget after observing confirmatory results.
- Do not drop zero-loss or low-loss sessions because they weaken the effect.
- Do not relabel secondary analyses as primary.
- If the primary Predictive-vs-Random effect does not generalize, the central
  foresight claim is considered unsupported.
- If the network effect generalizes but QoE does not, report a network-efficiency
  result and a null QoE result.
