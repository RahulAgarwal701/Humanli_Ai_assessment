"""Portable, deterministic manifest and JSON report persistence."""

import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Sequence

from data.records import ManifestRow


def read_manifest(source: Path) -> tuple[ManifestRow, ...]:
    """Read the Phase 1 schema and reject malformed rows or duplicate paths."""
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["path", "label", "group_id", "split"]:
            raise ValueError("Manifest requires exactly path,label,group_id,split columns")
        rows: list[ManifestRow] = []
        paths: set[str] = set()
        for line, values in enumerate(reader, start=2):
            if None in values or any(not value for value in values.values()):
                raise ValueError(f"Malformed manifest row at line {line}")
            row = ManifestRow(**values)
            if row.split not in {"train", "dev", "test"}:
                raise ValueError(f"Unknown split at manifest line {line}: {row.split!r}")
            if row.path in paths:
                raise ValueError(f"Duplicate manifest path: {row.path}")
            paths.add(row.path)
            rows.append(row)
    if not rows:
        raise ValueError("Manifest contains no images")
    return tuple(rows)


def save_manifest(rows: Sequence[ManifestRow], destination: Path) -> None:
    """Write one row per image; paths are relative to the configured dataset root."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "label", "group_id", "split"])
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    temporary.replace(destination)


def save_json(report: dict[str, Any], destination: Path) -> None:
    """Write sorted JSON atomically; also supports config Path objects."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(destination)
