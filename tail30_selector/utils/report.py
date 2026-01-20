"""Reporting utilities."""
from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Iterable


OUTPUT_DIR = Path("output")


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def write_csv(filename: str, rows: Iterable[dict]) -> Path:
    ensure_output_dir()
    path = OUTPUT_DIR / filename
    rows = list(rows)
    if not rows:
        path.write_text("")
        return path
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def to_dict(item) -> dict:
    if hasattr(item, "__dataclass_fields__"):
        return asdict(item)
    return dict(item)
