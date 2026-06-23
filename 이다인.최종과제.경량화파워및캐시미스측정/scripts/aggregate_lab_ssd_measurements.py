#!/usr/bin/env python3
"""Aggregate lab SSD measurement summary.json files into CSV reports."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment_folder",
        nargs="?",
        default="experiments/lab_ssd_measurement",
        type=Path,
        help="Folder to search recursively for summary.json files.",
    )
    return parser.parse_args()


def flatten_value(value: Any) -> Any:
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, list):
        return json.dumps(value)
    if value is None:
        return ""
    return value


def numeric(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_all_runs(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: flatten_value(row.get(key, "")) for key in keys})


def write_by_condition(path: Path, rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("mode", "")), str(row.get("expected_label", "")))].append(row)

    numeric_fields: list[str] = []
    seen = set()
    for row in rows:
        for key, value in row.items():
            if key in seen:
                continue
            if numeric(value) is not None:
                seen.add(key)
                numeric_fields.append(key)

    fieldnames = ["mode", "expected_label", "run_count"]
    for key in numeric_fields:
        fieldnames.append(f"{key}_mean")
        fieldnames.append(f"{key}_min")
        fieldnames.append(f"{key}_max")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (mode, expected_label), group_rows in sorted(groups.items()):
            output: dict[str, Any] = {
                "mode": mode,
                "expected_label": expected_label,
                "run_count": len(group_rows),
            }
            for key in numeric_fields:
                values = [numeric(row.get(key)) for row in group_rows]
                clean = [value for value in values if value is not None]
                if clean:
                    output[f"{key}_mean"] = statistics.fmean(clean)
                    output[f"{key}_min"] = min(clean)
                    output[f"{key}_max"] = max(clean)
            writer.writerow(output)


def main() -> int:
    args = parse_args()
    root = args.experiment_folder.resolve()
    summary_paths = sorted(root.rglob("summary.json"))
    if not summary_paths:
        raise SystemExit(f"No summary.json files found under {root}")

    rows = []
    for path in summary_paths:
        with path.open(encoding="utf-8") as handle:
            row = json.load(handle)
        row.setdefault("summary_path", str(path))
        rows.append(row)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_runs_path = root / f"summary_all_runs_{timestamp}.csv"
    by_condition_path = root / f"summary_by_condition_{timestamp}.csv"
    write_all_runs(all_runs_path, rows)
    write_by_condition(by_condition_path, rows)
    print(f"Read {len(rows)} summary files")
    print(f"Wrote {all_runs_path}")
    print(f"Wrote {by_condition_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
