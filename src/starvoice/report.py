from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


def _summaries(root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(root.rglob("summary.json")):
        if path == root / "summary.json":
            continue
        try:
            obj = json.loads(path.read_text())
            if "condition" in obj:
                rows.append(obj)
        except json.JSONDecodeError:
            pass
    return rows


def build_report(root: Path) -> Path:
    rows = _summaries(root)
    out = root / "report.csv"
    keys = sorted({k for row in rows for k in row.keys() if not isinstance(row[k], (dict, list))})
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("condition")), []).append(row)

    aggregate: dict[str, dict] = {}
    for condition, items in grouped.items():
        numeric_keys = {
            k
            for item in items
            for k, v in item.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        aggregate[condition] = {}
        for k in sorted(numeric_keys):
            vals = [float(i[k]) for i in items if isinstance(i.get(k), (int, float))]
            if vals:
                aggregate[condition][k] = {
                    "n": len(vals),
                    "mean": statistics.mean(vals),
                    "median": statistics.median(vals),
                }

    (root / "report.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    return out
