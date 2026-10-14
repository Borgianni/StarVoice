#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
SRC=$(find results -maxdepth 1 -type d -name 'g1-stability-*' | sort | tail -1)
if [[ -z "$SRC" || ! -f "$SRC/input.wav" ]]; then
  echo "Missing results/g1-stability-*/input.wav; run G1 stability test first" >&2
  exit 1
fi

OUT="results/g1-ram-archive-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"
echo "SESSION: $OUT"

starvoice receive-speech \
  --bind 127.0.0.1 --port 55132 \
  --output-dir "$OUT/receiver" > "$OUT/receiver.log" 2>&1 &
RX_PID=$!
trap 'kill "$RX_PID" 2>/dev/null || true' EXIT

sleep 1
starvoice send-speech \
  --target 127.0.0.1:55132 \
  --input "$SRC/input.wav" --policy always-fec \
  --output-dir "$OUT/sender" > "$OUT/sender.log" 2>&1

wait "$RX_PID"
trap - EXIT

starvoice analyze-session --session-dir "$OUT" > "$OUT/analysis.json"

python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
report = json.loads((root / "analysis.json").read_text())
for role in ("sender", "receiver"):
    folder = root / role
    manifest = json.loads((folder / "archive-integrity.json").read_text())
    assert manifest["files"], f"{role}: empty archive manifest"
    assert (folder / "clock.jsonl").is_file()
    print(f"{role}: archived {len(manifest['files'])} verified files")

t = report["timing"]
print("valid:", report["valid"])
print("frames:", report["frames"])
print("dispatch_p99_ms:", t["dispatch_p99_ms"])
print("dispatch_max_ms:", t["dispatch_max_ms"])
print("dispatch_over_5ms:", t["dispatch_over_5ms"])
print("mode:", report["receiver_log_writer"]["mode"])
assert report["valid"]
assert report["frames"] == 3000
print("AUTOMATIC RAM ARCHIVE: OK")
PY
