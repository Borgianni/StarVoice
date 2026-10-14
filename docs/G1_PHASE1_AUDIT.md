# G1 Phase I Audit

## Event reuse

| Trace | Replay losses | Unique loss positions | Reused positions | Max reuse |
|---|---:|---:|---:|---:|
| Pisa-02 | 85 | 60 | 18 | 4 |
| Pisa-03 | 52 | 29 | 14 | 5 |
| Total | 137 | 89 | 32 | 5 |

There are 137 replayed loss instances but only 89 distinct
source-trace loss positions. These are not independent events.

## Deadline reconciliation at 60 ms

- Complete benchmark: 181,396 frames.
- Network loss instances: 137.
- Additional late primary frames: 2.
- Primary deadline failures: 139.
- Always-FEC timely recoveries: 81.
- Predictive timely recoveries: 45.
- Reactive timely recoveries: 13.

## Loss-exposed subset

| Trace | Frames | Primary failures | Predictive recoveries |
|---|---:|---:|---:|
| Pisa-02 | 26,211 | 85 | 24 |
| Pisa-03 | 23,343 | 52 | 21 |
| Total | 49,554 | 137 | 45 |

The two additional late-primary failures are outside the
loss-exposed subset.

## Interpretation

Results are exploratory, based on two same-site network traces.
The phase predictor uses local 120-second warm-up.
Replay conditions reuse underlying network events.
The 60 ms deadline uses an RTT-derived one-way delay proxy.
No cross-session generalization or perceptual QoE improvement
has been established.
