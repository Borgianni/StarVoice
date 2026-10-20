#!/usr/bin/env python3
"""STARVOICE: exact noncausal deadline/actuation oracle + a frozen causal audio controller.

READS archived SV02/closeout artifacts, creates entirely new output directory.
The oracle has future arrival information and IS NOT A DEPLOYABLE CONTROLLER.
This is a research falsification study, NOT a SIGCOMM quality claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import tempfile
import time
import wave
from collections import defaultdict, Counter
from pathlib import Path
from statistics import mean

BASE = 60
TOP = 80
FRAME_MS = 20
SAMPLES = 960
SAMPLE_RATE = 48000
TRAIN = 6000
PERIOD = 750
PRE = 40
POST = 45
QUANTILE = 0.25


def wave_pcm(path: Path, start_frame: int, frame_count: int) -> bytes:
    with wave.open(str(path), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (48000, 1, 2):
            raise ValueError(f"unexpected WAV format: {path}")
        if start_frame < 0 or start_frame + frame_count > w.getnframes():
            raise ValueError(f"WAV window outside file: {path}")
        w.setpos(start_frame)
        data = w.readframes(frame_count)
        if len(data) != frame_count * 2:
            raise IOError(f"short WAV: {path}")
        return data


def splice_cost(pcm: bytes, direction: int) -> float:
    """ONLY current decoded 20ms PCM; measure best overlap mismatch for +/-1ms.

    This is a selection proxy, not a validated perceptual metric.
    The renderer uses the same offline local overlap-add primitive as SlackBank.
    """
    if len(pcm) != SAMPLES * 2 or direction not in (-1, 1):
        raise ValueError("splice_cost requires 960 PCM16 samples and sign +/-1")
    x = struct.unpack('<960h', pcm)
    offset = direction * 48
    return float(min(
        sum((x[cut + j] - x[cut - offset + j]) ** 2
            for j in range(0, 240, 8))
        for cut in range(312, 505, 24)
    ))


def percentile(vals: list[float], q: float) -> float:
    if not vals or not 0 <= q <= 1:
        raise ValueError("invalid training quantile")
    s = sorted(vals)
    return s[int(math.floor(q * (len(s) - 1)))]


def calibrate_edit_cost(training_wav: Path) -> tuple[float, float]:
    plus, minus = [], []
    # Read ONLY past 120 seconds (6,000 frames); never inspect held-out speech.
    with wave.open(str(training_wav), 'rb') as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (48000, 1, 2):
            raise ValueError('bad training audio WAV')
        if w.getnframes() < TRAIN * SAMPLES:
            raise ValueError('training audio shorter than 120s')
        for _ in range(TRAIN):
            pcm = w.readframes(SAMPLES)
            plus.append(splice_cost(pcm, +1))
            minus.append(splice_cost(pcm, -1))
    return percentile(plus, QUANTILE), percentile(minus, QUANTILE)


class SpeechSlack:
    """Online one-frame-at-a-time controller; cost is computed from CURRENT PCM.

    For each known phase boundary e, choose extra +1ms edits in frames
    [e-40, e) and extra -1ms edits in frames [e+5, e+45).
    Force the original phase-only schedule as a lower bound at EVERY frame;
    therefore all of its deadline recoveries remain feasible. The cost is a
    small increase in average buffer delay, explicitly reported and bounded.
    """
    def __init__(self, phase: int, baseline: list[int], up: float, down: float):
        if not 0 <= phase < PERIOD:
            raise ValueError("unavailable causal phase")
        self.phase, self.ref = phase, baseline
        self.th_up, self.th_down = up, down
        self.budget = BASE
        self.n = len(baseline)
        self.t = 0
        self.cheap_expansions = 0
        self.cheap_contractions = 0
        self.forced_expansions = 0
        self.forced_contractions = 0

    def step(self, pcm: bytes) -> int:
        """Observe frame t's current decoded audio; return B[t+1]."""
        t = self.t
        if t >= self.n:
            raise ValueError('extra frame')
        target_next = self.ref[t + 1] if t + 1 < self.n else BASE
        b = self.budget
        if b < self.ref[t]:
            raise AssertionError('candidate fell behind phase-only floor')
        delta = 0
        previous_e = self.phase + ((t - self.phase) // PERIOD) * PERIOD
        next_e = previous_e + PERIOD
        pre_active = (t >= TRAIN and next_e - PRE <= t < next_e
                      and next_e + POST < self.n)
        post_active = (t >= TRAIN and previous_e + 5 <= t < previous_e + POST
                       and previous_e - PRE >= TRAIN)
        if pre_active and b < TOP:
            if splice_cost(pcm, +1) <= self.th_up:
                delta = +1
                self.cheap_expansions += 1
        if post_active and b > target_next:
            remaining = previous_e + POST - t
            if splice_cost(pcm, -1) <= self.th_down:
                delta = -1
                self.cheap_contractions += 1
            elif b - BASE >= remaining:
                delta = -1
                self.forced_contractions += 1
        # The phase-only lower envelope is a mandatory constraint; it prevents
        # retrospective removal of recoveries and never uses arrival outcomes.
        if b + delta < target_next:
            delta = +1
            self.forced_expansions += 1
        # Close any incomplete boundary episode instead of accumulating drift.
        if not (pre_active or post_active) and b + delta > target_next:
            delta = -1
            self.forced_contractions += 1
        new_b = b + delta
        if abs(delta) > 1 or not BASE <= new_b <= TOP or new_b < target_next:
            raise AssertionError(f'causal budget violation at seq={t}')
        self.budget = new_b
        self.t += 1
        return new_b


def packet_mode(seq: int, extra_ms: int, arrivals: dict, anchor_ns: int,
                first_seq: int) -> str:
    deadline = anchor_ns + (seq - first_seq) * 20_000_000 + (BASE + extra_ms) * 1_000_000
    primary = arrivals.get(seq)
    if primary is not None and int(primary['receive_mono_ns']) <= deadline:
        return 'primary'
    carrier = arrivals.get(seq + 1)
    if carrier is not None and bool(carrier.get('has_lbrr')) and int(carrier['receive_mono_ns']) <= deadline:
        return 'fec'
    return 'plc'


def required_rescues(n: int, first_seq: int, arrivals: dict,
                     phase_schedule: list[int]) -> dict[int, int]:
    """Oracle knows true future arrivals; return MINIMUM budget for each rescue.

    Importantly, using the *original* full 20ms ramp to label rescued frames
    does not imply that those packets each needed +20ms extra buffer.
    """
    anchor = int(arrivals[first_seq]['receive_mono_ns'])
    req = {}
    for i in range(max(first_seq, TRAIN), n):
        if packet_mode(i, 0, arrivals, anchor, first_seq) != 'plc':
            continue
        if packet_mode(i, phase_schedule[i] - BASE, arrivals, anchor, first_seq) == 'plc':
            continue
        budget = next((b for b in range(1, 21)
                       if packet_mode(i, b, arrivals, anchor, first_seq) != 'plc'), None)
        if budget is None or budget > phase_schedule[i] - BASE:
            raise AssertionError('phase-only rescue minimum budget inconsistent')
        req[i] = budget
    return req


def exact_oracle(n: int, phase: int, required: dict[int, int], mode: str) -> dict:
    """Exact noncausal DP over 21 physical buffer states and 1ms/frame slew.

    Every rescued frame is constrained to meet its MINIMUM observed deadline.
    Force 0 slack halfway between successive phase events to exclude the
    degenerate 'leave a large permanent jitter buffer all day' solution.
    Lexicographic objectives: (delay-area, #edits) or (#edits, delay-area).
    """
    if mode not in ('min_delay', 'min_edits'):
        raise ValueError(mode)
    start = TRAIN - 1
    middle_resets = {phase + k * PERIOD + PERIOD // 2
                     for k in range(-2, n // PERIOD + 2)
                     if start <= phase + k * PERIOD + PERIOD // 2 < n}
    resets = middle_resets | {start, n - 1}
    inf = (10**15, 10**15)
    dp = [inf] * 21
    dp[0] = (0, 0)
    # For every target boundary j, remember exact prev state (0..20) or 255.
    traceback = []
    for seq in range(start, n):
        next_dp = [inf] * 21
        prevs = bytearray([255] * 21)
        for b in range(21):
            if dp[b] == inf:
                continue
            if (seq in resets and b != 0) or b < required.get(seq, 0):
                continue
            # Source frame seq has reserve b. This is the actual playout budget.
            area = b if seq >= TRAIN else 0
            for delta in (-1, 0, 1):
                b2 = b + delta
                if not 0 <= b2 <= 20:
                    continue
                if (seq + 1 in resets and b2 != 0) or seq + 1 == n and b2 != 0:
                    continue
                add = (area, abs(delta)) if mode == 'min_delay' else (abs(delta), area)
                candidate = (dp[b][0] + add[0], dp[b][1] + add[1])
                if candidate < next_dp[b2]:
                    next_dp[b2] = candidate
                    prevs[b2] = b
        if min(next_dp) == inf:
            raise RuntimeError(f'no oracle-feasible trajectory at {seq} ({mode})')
        traceback.append(prevs)
        dp = next_dp
    if dp[0] == inf:
        raise AssertionError('zero end state unreachable')
    b = 0
    extras = [0] * n
    for seq in range(n - 1, start - 1, -1):
        p = traceback[seq - start][b]
        if p == 255:
            raise AssertionError('oracle traceback corrupted')
        extras[seq] = p
        b = p
    if b != 0:
        raise AssertionError('oracle must start at zero slack')
    for seq, need in required.items():
        if extras[seq] < need:
            raise AssertionError('oracle rescue constraint broken')
    edits = sum(abs(a - b) for a, b in zip(extras[start:], extras[start + 1:]))
    delay_area = sum(extras[TRAIN:])
    return {
        'mode': mode,
        'edits': edits,
        'mean_extra_ms': delay_area / (n - TRAIN),
        'peak_budget_ms': BASE + max(extras),
        'required_rescues': len(required),
        'budget_ms': [BASE + x for x in extras],
        'note': 'NONCAUSAL; true future packet arrivals used to constrain recovered frames',
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def render_causal(out_wav: Path, packets: list[bytes], arrivals: dict, first_seq: int,
                  phase: int, baseline: list[int], bitrate: int,
                  thresholds: tuple[float, float]) -> dict:
    from starvoice.opus import OpusConfig, OpusDecoder
    from starvoice.elastic_playout import stretch_frame
    n = len(packets)
    controller = SpeechSlack(phase, baseline, *thresholds)
    first_ns = int(arrivals[first_seq]['receive_mono_ns'])
    decoder = OpusDecoder(OpusConfig(bitrate=bitrate))
    budgets = []
    render_start_cpu = time.process_time()
    render_start_wall = time.perf_counter()
    modes = defaultdict(int)
    frames_edited = 0
    written = 0
    try:
        with wave.open(str(out_wav), 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(48000)
            for seq in range(n):
                b = controller.budget
                if b < baseline[seq]:
                    raise AssertionError(f'late budget floor violation at {seq}')
                due_ns = first_ns + (seq - first_seq) * 20_000_000 + b * 1_000_000
                from starvoice.elastic_playout import select_packet
                kind, packet_seq = (select_packet(seq, due_ns, arrivals)
                                    if seq >= first_seq else ('plc', None))
                if kind == 'primary':
                    pcm = decoder.decode(packets[packet_seq], fec=False)
                elif kind == 'fec':
                    pcm = decoder.decode(packets[packet_seq], fec=True)
                else:
                    pcm = decoder.decode(None, fec=False)
                if len(pcm) != SAMPLES * 2:
                    raise AssertionError('Opus decoder frame length changed')
                if written != seq * SAMPLES + (b - BASE) * 48:
                    raise AssertionError(f'physical 48kHz output clock drift at {seq}')
                next_b = controller.step(pcm)
                delta = (next_b - b) * 48 if seq + 1 < n else 0
                if delta:
                    frames_edited += 1
                if seq >= TRAIN:
                    modes[kind] += 1
                rendered = stretch_frame(pcm, delta)
                w.writeframesraw(rendered)
                written += len(rendered) // 2
                budgets.append(b)
    finally:
        decoder.close()
    if written != n * SAMPLES or budgets[-1] != BASE:
        raise AssertionError('physical output must return to zero accumulated slack')
    return {
        'budget_ms': budgets,
        'frames_edited': frames_edited,
        'scored_suffix_modes': dict(modes),
        'mean_extra_ms': sum(b - BASE for b in budgets[TRAIN:]) / (n - TRAIN),
        'max_budget_ms': max(budgets),
        'training_splice_threshold_up': thresholds[0],
        'training_splice_threshold_down': thresholds[1],
        'cheap_expansions': controller.cheap_expansions,
        'cheap_contractions': controller.cheap_contractions,
        'forced_expansions': controller.forced_expansions,
        'forced_contractions': controller.forced_contractions,
        'samples_written': written,
        'offline_cpu_seconds': time.process_time() - render_start_cpu,
        'offline_wall_seconds': time.perf_counter() - render_start_wall,
        'offline_cpu_us_per_frame': (time.process_time() - render_start_cpu) * 1e6 / n,
        'source_audio_lookahead': 0,
        'description': 'causal decoded-current-frame acoustic-cost selection; never uses future packet outcomes',
    }


def save_window(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)


def summary_stats(records: list[dict]) -> dict:
    by_kind = {}
    for kind in ('rescued', 'control'):
        subset = [r for r in records if r['kind'] == kind]
        by_kind[kind] = {
            'windows': len(subset),
            'paired_dnsmos_delta_vs_phase': (mean(r['candidate_ovrl'] - r['phase_ovrl'] for r in subset)
                                            if subset else None),
            'paired_dnsmos_delta_vs_fixed60': (mean(r['candidate_ovrl'] - r['fixed60_ovrl'] for r in subset)
                                              if subset else None),
            'paired_stoi_delta_vs_phase': (mean(r['candidate_stoi'] - r['phase_stoi'] for r in subset)
                                           if subset else None),
            'paired_stoi_delta_vs_fixed60': (mean(r['candidate_stoi'] - r['fixed60_stoi'] for r in subset)
                                             if subset else None),
        }
    return by_kind


def quality_block(block: str, src: Path, fixed_wav: Path, phase_wav: Path,
                  candidate_wav: Path, n: int, phase: int, ramp: list[int],
                  rescued: list[int], candidate_budget: list[int], scorer, temp_dir: Path) -> list[dict]:
    from starvoice.elastic_quality import eligible_centers, stoi_score
    centers = eligible_centers(n, phase, ramp, rescued)
    records = []
    for k, (kind, seq) in enumerate(centers):
        low, high = seq - 150, seq + 150
        if (ramp[low] != BASE or ramp[high] != BASE
                or candidate_budget[low] != BASE or candidate_budget[high] != BASE):
            raise AssertionError('6s comparison has nonzero edge reserve; metric invalid')
        start, samples = low * SAMPLES, (high - low) * SAMPLES
        ref = wave_pcm(src, start, samples)
        candidate = wave_pcm(candidate_wav, start, samples)
        fixed = wave_pcm(fixed_wav, start, samples)
        old = wave_pcm(phase_wav, start, samples)
        # Identical time coordinates at the window edges: candidate budget
        # must also return to 60 to prevent event leakage across windows.
        for name, data in (('fixed60', fixed), ('phase', old), ('candidate', candidate)):
            save_window(temp_dir / f'{name}.wav', data)
        quality = {name: scorer.score(temp_dir / f'{name}.wav')['ovrl']
                   for name in ('fixed60', 'phase', 'candidate')}
        records.append({
            'block': block, 'kind': kind, 'center_seq': seq,
            'candidate_ovrl': quality['candidate'],
            'phase_ovrl': quality['phase'],
            'fixed60_ovrl': quality['fixed60'],
            'candidate_stoi': stoi_score(ref, candidate),
            'phase_stoi': stoi_score(ref, old),
            'fixed60_stoi': stoi_score(ref, fixed),
        })
    return records


def run(args) -> None:
    research = args.research.expanduser().resolve()
    sys.path.insert(0, str(research / 'src'))
    from starvoice.slackbank import arrival_data, calibrate_phase, ramp_targets, evaluate
    from starvoice.opus_replay_audit import encode_continuous, sha256
    from starvoice.dnsmos import DNSMOSScorer

    campaign = args.campaign.expanduser().resolve()
    closeout = args.closeout.expanduser().resolve()
    output = args.output.expanduser().resolve()
    # Checkpoint-aware: prior complete blocks are immutable and verified by hash.
    # Only incomplete files inside this NEW output directory may be discarded.
    if (output / 'FINAL_DECISION.json').is_file():
        print('Previously completed study (no recomputation):', output / 'FINAL_DECISION.json')
        print((output / 'FINAL_DECISION.json').read_text()[:4000])
        return
    if (output in (campaign, closeout, research) or output.is_relative_to(campaign)
            or output.is_relative_to(closeout) or output.is_relative_to(research)):
        raise ValueError('refusing to write into original archives')
    manifest = json.loads((campaign / 'campaign-manifest.json').read_text())
    if manifest.get('status') != 'complete' or len(manifest.get('blocks', [])) != 9:
        raise ValueError('requires exactly nine completed archived blocks')
    prior_net = json.loads((closeout / 'network_native_summary.json').read_text())
    if prior_net['aggregate']['status'] != 'nine_blocks_audio_and_native_neteq_executed':
        raise ValueError('real native NetEQ baseline required')
    bitrate = int(manifest['codec']['bitrate'])
    expected_loss = int(manifest['codec']['expected_loss_percent'])
    model = closeout / 'models/dnsMOS/sig_bak_ovr.onnx'
    if not model.is_file():
        raise FileNotFoundError('need locally cached verified DNSMOS model; no download')
    scorer = DNSMOSScorer(model)
    output.mkdir(parents=True, exist_ok=True)
    script_hash = sha256_file(Path(__file__))
    all_network = []
    all_quality = []
    original_rescues = 0
    for i, record in enumerate(manifest['blocks'], 1):
        block = record['block']
        block_out = output / block
        completed = block_out / 'completed.json'
        if completed.is_file():
            previous = json.loads(completed.read_text())
            if previous.get('script_sha256') != script_hash:
                raise RuntimeError('existing checkpoint created by different code; choose a new output directory')
            prev_net = previous['network']
            wav_check = block_out / 'causal_speech_slack.wav'
            if (not wav_check.is_file() or sha256_file(wav_check) != prev_net['candidate_wav_sha256']):
                raise RuntimeError(f'checkpoint waveform hash mismatch: {block}')
            all_network.append(prev_net)
            all_quality.extend(previous['quality'])
            original_rescues += prev_net['rescues']
            print(f'[{i}/9] {block}: prior COMPLETE checkpoint verified', flush=True)
            continue
        if block_out.exists():
            raise RuntimeError(f'unrecognized incomplete directory: {block_out}; refusing overwrite')
        part = output / (block + '.incomplete')
        if part.exists():
            shutil.rmtree(part)  # Created only by this study; NO archived input touched.
        part.mkdir()
        block_out = part
        src = campaign / 'workloads' / f"segment-{int(record['workload_segment']):02d}.wav"
        if sha256(src) != record['workload_sha256']:
            raise AssertionError(f'original source SHA mismatch: {block}')
        fixed = closeout / block / 'render/fixed60.wav'
        old = closeout / block / 'render/slackbank.wav'
        for path in (fixed, old):
            if not path.is_file():
                raise FileNotFoundError(path)
        n, first_seq, arrivals = arrival_data(campaign / block)
        phase, support = calibrate_phase(arrivals)
        if phase is None or support < 4:
            raise AssertionError(f'phase calibration failed in training: {block}')
        baseline = ramp_targets(n, phase)
        expected_fixed = evaluate(n, first_seq, arrivals, [BASE] * n)
        expected_phase = evaluate(n, first_seq, arrivals, baseline)
        req = required_rescues(n, first_seq, arrivals, baseline)
        if len(req) != expected_fixed['plc'] - expected_phase['plc']:
            raise AssertionError(f'not all original modeled rescues accounted for in {block}')
        original_rescues += len(req)
        oracle_delay = exact_oracle(n, phase, req, 'min_delay')
        oracle_edits = exact_oracle(n, phase, req, 'min_edits')
        # Both exact solutions must respect the receiver-relative physical clock.
        for candidate in (oracle_delay, oracle_edits):
            stats = evaluate(n, first_seq, arrivals, candidate['budget_ms'])
            if stats['plc'] > expected_phase['plc']:
                raise AssertionError('oracle rescued-frame constraint broken')
            candidate['actual_modeled_plc'] = stats['plc']
            candidate['budget_ms_sha256'] = hashlib.sha256(bytes(b - BASE for b in candidate['budget_ms'])).hexdigest()
            del candidate['budget_ms']  # store aggregates and hash, not huge arrays
        thresholds = calibrate_edit_cost(fixed)
        packets = encode_continuous(src, bitrate, expected_loss)
        if n != len(packets):
            raise AssertionError('Opus reencoding frame count does not match receiver')
        wav = block_out / 'causal_speech_slack.wav'
        candidate = render_causal(wav, packets, arrivals, first_seq, phase, baseline, bitrate, thresholds)
        evaluated = evaluate(n, first_seq, arrivals, candidate['budget_ms'])
        if (candidate['scored_suffix_modes'].get('plc', 0) != evaluated['plc']
                or evaluated['plc'] > expected_phase['plc']
                or candidate['max_budget_ms'] > TOP
                or candidate['mean_extra_ms'] > 2.0):
            raise AssertionError(f'causal rendering failed PLC/delay guard: {block}')
        if candidate['budget_ms'][-1] != BASE:
            raise AssertionError('causal final slack nonzero')
        causal_budget_trace = candidate['budget_ms']
        budget_hash = hashlib.sha256(bytes(b - BASE for b in causal_budget_trace)).hexdigest()
        candidate.pop('budget_ms')
        candidate['budget_ms_sha256'] = budget_hash
        original_edits = sum(a != b for a, b in zip(baseline, baseline[1:]))
        if candidate['frames_edited'] != original_edits:
            raise AssertionError('unexpected changes to number of physical edit operations')
        network_row = {
            'block': block,
            'frames_scored': n - TRAIN,
            'plc_fixed60': expected_fixed['plc'],
            'plc_phase': expected_phase['plc'],
            'rescues': len(req),
            'minimum_required_slack_histogram_ms': {str(k): v for k, v in sorted(Counter(req.values()).items())},
            'old_edits': original_edits,
            'old_mean_extra_ms': expected_phase['average_extra_budget_ms'],
            'oracle_min_area': {k: v for k, v in oracle_delay.items() if k != 'note'},
            'oracle_min_edits': {k: v for k, v in oracle_edits.items() if k != 'note'},
            'causal': candidate,
            'causal_modeled_plc': evaluated['plc'],
            'source_sha256': sha256_file(src),
            'candidate_wav_sha256': sha256_file(wav),
            'native_neteq_already_executed': True,
            'webrtc_neteq_equal_delay_comparison_valid': False,
        }
        (block_out / 'network_and_controller.json').write_text(json.dumps(network_row, indent=2) + '\n')
        print(f'[{i}/9] {block}: PLC {expected_fixed["plc"]}->{expected_phase["plc"]}->{evaluated["plc"]}; '
              f'peak {candidate["max_budget_ms"]}ms; extra {candidate["mean_extra_ms"]:.3f}ms; '
              f'ORACLE min-area {oracle_delay["mean_extra_ms"]:.3f}ms, '
              f'min-edits {oracle_edits["edits"]}', flush=True)
        first_ns = int(arrivals[first_seq]['receive_mono_ns'])
        rescued = [seq for seq in range(max(first_seq, TRAIN), n)
                   if packet_mode(seq, 0, arrivals, first_ns, first_seq) == 'plc'
                   and packet_mode(seq, baseline[seq] - BASE, arrivals, first_ns, first_seq) != 'plc']
        with tempfile.TemporaryDirectory(prefix='svq-', dir=block_out) as tmp:
            windows = quality_block(block, src, fixed, old, wav, n, phase, baseline,
                                    rescued, causal_budget_trace, scorer, Path(tmp))
        (block_out / 'completed.json').write_text(json.dumps({
            'script_sha256': script_hash, 'network': network_row, 'quality': windows,
        }, indent=2) + '\n')
        part.rename(output / block)  # Atomic completion marker for restart safety.
        all_network.append(network_row)
        all_quality.extend(windows)
        print(f'      STOI + DNSMOS evaluated: {len(windows)} paired 6s event/control windows', flush=True)
    if (sum(row['plc_fixed60'] for row in all_network) != 82
            or sum(row['plc_phase'] for row in all_network) != 15
            or original_rescues != 67):
        raise AssertionError('frozen original nine-block accounting drifted')
    with (output / 'paired_quality.csv').open('w', newline='') as f:
        columns = list(all_quality[0])
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        writer.writerows(all_quality)
    by_block = {}
    for row in all_network:
        subset = [x for x in all_quality if x['block'] == row['block']]
        by_block[row['block']] = summary_stats(subset)
    required_hist = Counter()
    for x in all_network:
        required_hist.update({int(k): v for k, v in x['minimum_required_slack_histogram_ms'].items()})
    summary = {
        'schema': 1,
        'status': 'offline_nonconfirmatory_decisive_study_complete',
        'campaign': str(campaign),
        'pre_registered_parameters': {'phase_training_frames': TRAIN, 'phase_period_frames': PERIOD,
                                      'lookahead_pre_frames': PRE, 'release_post_frames': POST,
                                      'splice_cost_quantile': QUANTILE, 'fixed_opus_stream': True,
                                      'max_extra_mean_ms_per_block': 2.0,
                                      'max_playout_budget_ms': TOP},
        'nine_blocks_one_site_one_day': True,
        'total_frames_scored': sum(x['frames_scored'] for x in all_network),
        'fixed60_plc': 82, 'phase_plc': 15,
        'minimum_required_slack_histogram_ms': {str(k): v for k, v in sorted(required_hist.items())},
        'fraction_rescues_needing_no_more_than_10ms': (sum(v for k, v in required_hist.items() if k <= 10) / 67),
        'causal_plc': sum(x['causal_modeled_plc'] for x in all_network),
        'causal_mean_extra_ms': (sum(x['causal']['mean_extra_ms'] * x['frames_scored'] for x in all_network)
                                  / sum(x['frames_scored'] for x in all_network)),
        'causal_edits': sum(x['causal']['frames_edited'] for x in all_network),
        'phase_edits': sum(x['old_edits'] for x in all_network),
        'oracle_min_area_mean_extra_ms': (sum(x['oracle_min_area']['mean_extra_ms'] * x['frames_scored']
                                              for x in all_network) / sum(x['frames_scored'] for x in all_network)),
        'oracle_min_edits': sum(x['oracle_min_edits']['edits'] for x in all_network),
        'quality_window_count': len(all_quality),
        'quality_by_kind': summary_stats(all_quality),
        'quality_by_block': by_block,
        'native_neteq_sha256': prior_net['aggregate']['native_neteq_binary_sha256'],
        'original_audio_and_neteq_unmodified': True,
        'causal_new_audio_vs_neteq_equal_delay_proven': False,
        'original_wire_opus_bytes_retained': False,
        'candidate_wer_evaluated': False,
        'claim_status': 'NO_SIGCOMM_GO_UNLESS_LISTENER_QUALITY_AND_MATCHED_NETEQ_AND_EXTERNAL_TRACES_PASS',
        'limitations': [
            'Exact DP oracle uses future arrivals and is NONCAUSAL, not an implementation or QoE bound.',
            'Causal acoustic cost threshold is trained on the first 120s of this same block only.',
            'Six-second DNSMOS clips are repeated by DNSMOS scorer to fill its 9.01s input; diagnostic only.',
            'STOI is sensitive to temporal warping; do not treat as listening quality.',
            'Nine blocks are correlated one-day/site, not independent replications.',
            'Native NetEQ internal delay is not matched to 60/80ms receiver-relative budget.',
            'WER for new candidate, matched-delay NetEQ and listener subjective ratings not yet evaluated.',
            'Same fixed reconstructed Always-FEC packets are used; original wire-byte identity unproven.',
        ],
    }
    quality_gate = (summary['quality_by_kind']['rescued']['windows'] >= 1
                    and summary['quality_by_kind']['control']['windows'] >= 1
                    and all(summary['quality_by_kind'][k]['paired_dnsmos_delta_vs_phase'] >= .03
                            for k in ('rescued', 'control'))
                    and all(summary['quality_by_kind'][k]['paired_dnsmos_delta_vs_fixed60'] >= 0
                            for k in ('rescued', 'control'))
                    and summary['causal_plc'] <= summary['phase_plc']
                    and summary['causal_mean_extra_ms'] <= 2.0)
    summary['predeclared_local_engineering_quality_gate'] = bool(quality_gate)
    summary['gate_rule'] = ('paired DNSMOS candidate-phase >=+.03 AND candidate-fixed60 >=0 '
                            'separately on rescued and control clips; <=15 PLC; mean extra <=2ms')
    summary['independent_sigcomm_claim_allowed'] = False
    (output / 'FINAL_DECISION.json').write_text(json.dumps(summary, indent=2) + '\n')
    r, c = (summary['quality_by_kind'][k] for k in ('rescued', 'control'))
    lines = [
        '# STARVOICE — exact deadline oracle and causal speech-cost study', '',
        '**Not confirmatory; one correlated nine-block campaign.**', '',
        '## Outcome (120s training excluded per block)', '',
        '| Metric | Result |', '|---|---:|',
        f"| Fixed60 modeled PLC | {summary['fixed60_plc']} |",
        f"| Frozen phase SlackBank modeled PLC | {summary['phase_plc']} |",
        f"| Causal speech-cost PLC | {summary['causal_plc']} |",
        f"| Causal mean extra receiver-relative budget | {summary['causal_mean_extra_ms']:.5f} ms |",
        f"| Causal PCM edit operations | {summary['causal_edits']} |",
        f"| Frozen phase PCM edit operations | {summary['phase_edits']} |",
        f"| Exact noncausal min-delay oracle extra budget | {summary['oracle_min_area_mean_extra_ms']:.5f} ms |",
        f"| Exact noncausal min-edits oracle actions | {summary['oracle_min_edits']} |",
        f"| Rescued frames requiring <=10ms extra | {summary['fraction_rescues_needing_no_more_than_10ms']*100:.1f}% |",
        '', '## Paired six-second acoustic diagnostics', '',
        '| Subset | Windows | DNSMOS candidate − phase | DNSMOS candidate − fixed60 | STOI candidate − phase |',
        '|---|---:|---:|---:|---:|',
        f"| Rescued | {r['windows']} | {r['paired_dnsmos_delta_vs_phase']:+.5f} | {r['paired_dnsmos_delta_vs_fixed60']:+.5f} | {r['paired_stoi_delta_vs_phase']:+.5f} |",
        f"| Control | {c['windows']} | {c['paired_dnsmos_delta_vs_phase']:+.5f} | {c['paired_dnsmos_delta_vs_fixed60']:+.5f} | {c['paired_stoi_delta_vs_phase']:+.5f} |",
        '', f"**Predeclared local engineering gate:** {'PASS' if quality_gate else 'FAIL'}.",
        '', '## Scientific interpretation', '',
        'The DP oracle observes future arrivals and is a NONCAUSAL lower-cost benchmark, not a deployable algorithm.',
        'The causal policy uses training-only phase and acoustic thresholds and only the CURRENT decoded frame.',
        'Every candidate budget dominates the frozen phase-only budget at each frame; recovered frames are not removed.',
        'The causal candidate spends potentially more average playout budget to reposition edits; compare at a delay Pareto frontier.',
        'NetEQ was genuinely executed previously, but its internal delay is not equal-cost matched.',
        'STOI and DNSMOS are diagnostic; new WER and human perceptual study remain missing.',
        'One campaign does not establish independent-day/site generalization. **No SIGCOMM superiority claim.**',
        '', 'Detailed scores: `paired_quality.csv` and `FINAL_DECISION.json`.', '',
    ]
    (output / 'FINAL_DECISION.md').write_text('\n'.join(lines))
    print('\n=== FROZEN STUDY COMPLETE ===')
    print(f"PLC fixed60={summary['fixed60_plc']} phase={summary['phase_plc']} causal={summary['causal_plc']}")
    print(f"Causal mean extra buffer {summary['causal_mean_extra_ms']:.4f}ms, edits {summary['causal_edits']}")
    print(f"Oracle min edit count (noncausal): {summary['oracle_min_edits']}; min area {summary['oracle_min_area_mean_extra_ms']:.4f}ms")
    print('Paired diagnostic acoustic results:', json.dumps(summary['quality_by_kind'], indent=2))
    print('Scientific decision stays NO-GO absent quality/latency/independence evidence.')
    print('All new artifacts:', output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--closeout', type=Path, required=True)
    p.add_argument('--research', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    run(args)


if __name__ == '__main__':
    main()