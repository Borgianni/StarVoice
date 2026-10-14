# G1 Real-time Receiver: Endpoint, Timing, and Storage Limitations

Status: experimental engineering validation on localhost. This does **not**
authorize confirmatory Phase II claims or change the Phase I frozen definition.

## Timing and causal endpoint

- SV02 sends 20-ms, 48-kHz mono PCM frames encoded as Opus datagrams.
- The first received AUDIO frame establishes a receiver-local monotonic anchor.
  The nominal 60-ms jitter buffer is measured from receiver arrival, not
  sender capture or an independently verified one-way delay.
- The receiver checks primary packets and next-packet FEC/LBRR candidates
  against their scheduled receiver-local deadlines. Sender FEC enabled,
  LBRR actually present, and in-time FEC recovery are distinct outcomes.
- Sender and receiver monotonic clocks cannot be subtracted across hosts.
  Clock JSONL contains NTP diagnostics, not ground-truth one-way delay.
- Frames preceding the first AUDIO arrival are flagged startup_uncertain
  and must be excluded from confirmatory endpoint calculations.
- The receiver writes decoded WAV incrementally. There is no hardware speaker
  output and no measured speaker latency. Do not call the metric audible QoE.
- Phase I RTT/2 deadline estimates and SV02 receiver-buffer deadlines are
  different endpoints. Confirmatory comparison requires a preregistered revision.

## Logging and scheduling

- Packet arrival, scheduled playout, dispatch, Opus decode, WAV write,
  source packet and primary/FEC/PLC decisions are logged.
- AsyncJsonlWriter sends rows in order to bounded background queues.
  Queue overflow and worker errors fail explicitly.
- Localhost 60-s NUC tests (3,000 frames) found disk JSONL writes as long
  as 37 ms, correlated with dispatch spikes. RAM staging substantially
  reduced measured lateness. This is a local engineering observation, not
  independently replicated causal efficacy or a QoE finding.

## Session storage contract

- Both sender and receiver default to --storage-mode ram on Linux,
  staging separate sessions under /dev/shm.
- All writers stop before post-session archival. Files are copied into a
  hidden directory next to the requested --output-dir, SHA-256 verified,
  flushed to disk, then published via directory rename.
  Archive-integrity.json records hashes of captured files.
- On archive failure, the original RAM directory is retained and its path
  appears in the error. If the machine crashes or reboots before archival,
  RAM-staged data can be lost. Check tmpfs capacity for long sessions.
- --storage-mode direct is available when RAM is insufficient, but direct
  file I/O can perturb the experimental receiver.
- Output directories must be new and unique. Storage mode should be
  recorded as a measurement condition before any confirmatory run.

## Next gates

1. Validate the default automatic RAM-to-disk archival on the NUC.
2. Confirm endpoint definitions and startup exclusions.
3. Validate controlled drop and reorder, and loss/late/PLC accounting.
   Only then consider cross-machine Starlink pilots.
