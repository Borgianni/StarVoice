# Multi-site instructions (Canada / external site)

The goal is to minimize operator degrees of freedom.

## Install

```bash
git clone https://github.com/Borgianni/StarVoice.git
cd StarVoice
git checkout <FROZEN_COMMIT>

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[analysis]"

starvoice doctor
```

Save the output.

## Relay connectivity smoke test

```bash
starvoice probe \
  --target <RELAY_HOST>:5005 \
  --duration 60 \
  --interval-ms 20 \
  --output canada-smoke.jsonl
```

Do not modify interval, timeout, codec bitrate, predictor thresholds or campaign ordering unless the shared protocol is formally revised for every site.

## Main campaign

After the campaign manifest and model are frozen:

```bash
starvoice run campaigns/mmsys27-template.yaml \
  --target <RELAY_HOST>:5005 \
  --site canada-01 \
  --output-root runs
```

Archive the complete generated campaign directory. Never send only summary tables; raw JSONL logs and resolved manifests are the scientific record.

## Transfer

Create a checksum before transfer:

```bash
tar -czf canada-01-run.tar.gz runs/<campaign-directory>
sha256sum canada-01-run.tar.gz > canada-01-run.tar.gz.sha256
```

Record any operational anomaly separately; do not delete inconvenient runs.
