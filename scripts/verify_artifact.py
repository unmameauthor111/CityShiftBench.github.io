#!/usr/bin/env python3
"""Lightweight release checks for the anonymous CityShiftBench artifact."""

from __future__ import annotations

import csv
import json
import re
import sys
from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "croissant_cityshiftbench_scale122.json",
    "data/cityshiftbench_scale122_tile_targets.csv",
    "data/slices/paper_slice_registry_122.csv",
    "data/splits/target_shot_registry_scale122.csv",
    "results/fewshot_baseline_suite_scale122_summary.csv",
    "results/satca_scale122_local_sharded_k010_paired_significance.csv",
    "results/scale122_shortcut_baselines_summary.csv",
    "docs/DATA_CARD.md",
    "docs/BENCHMARK_CARD.md",
    "docs/LICENSES.md",
    "docs/REVIEWER_COMMANDS.md",
]

IDENTITY_PATTERNS = [
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    r"~[A-Z][A-Za-z]+_[A-Z][A-Za-z]+[0-9]+",
]

PATH_PATTERNS = [
    r"C:\\Users\\",
    r"One" + r"Drive\\Desktop",
    r"144\.214\.",
    "\u5a13",
]

TOKEN_PATTERNS = [
    "ghp" + r"_[A-Za-z0-9_]+",
    "github" + r"_pat_[A-Za-z0-9_]+",
    "hf" + r"_[A-Za-z0-9_]+",
    "sk" + r"-[A-Za-z0-9_]+",
    "AK" + r"IA[0-9A-Z]{16}",
    r"-----BEGIN (RSA |OPENSSH |EC )?" + "PRIVATE" + r" KEY-----",
]

TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".tex", ".txt", ".yml", ".yaml"}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_required(messages: list[str]) -> None:
    for rel_path in REQUIRED_FILES:
        if not (ROOT / rel_path).exists():
            messages.append(f"FAIL missing required file: {rel_path}")


def check_text_patterns(messages: list[str]) -> None:
    patterns = [*IDENTITY_PATTERNS, *PATH_PATTERNS, *TOKEN_PATTERNS]
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel == "scripts/verify_artifact.py":
            continue
        text = read_text(path)
        for pattern in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                messages.append(f"FAIL pattern `{pattern}` in {rel}")


def check_croissant(messages: list[str]) -> None:
    path = ROOT / "croissant_cityshiftbench_scale122.json"
    data = json.loads(read_text(path))
    if data.get("creator") != [{"@type": "Organization", "name": "Anonymous Authors"}]:
        messages.append("FAIL Croissant creator is not anonymous.")
    if not data.get("rai:hasSyntheticData") is False:
        messages.append("FAIL Croissant RAI synthetic-data field is missing or not false.")
    distributions = data.get("distribution", [])
    for item in distributions:
        name = item.get("name")
        expected = item.get("sha256")
        if not name or not expected:
            continue
        candidates = list(ROOT.rglob(name))
        if not candidates:
            messages.append(f"FAIL Croissant distribution missing local file: {name}")
            continue
        actual = file_sha256(candidates[0])
        if actual != expected:
            messages.append(f"FAIL Croissant sha256 mismatch for {name}: {actual} != {expected}")


def check_csv_shapes(messages: list[str]) -> None:
    target_path = ROOT / "data/cityshiftbench_scale122_tile_targets.csv"
    with target_path.open("r", encoding="utf-8", newline="") as handle:
        rows = sum(1 for _ in csv.reader(handle)) - 1
    if rows != 8359:
        messages.append(f"FAIL expected 8359 tile-target rows, found {rows}")

    paired_path = ROOT / "results/satca_scale122_local_sharded_k010_paired_significance.csv"
    with paired_path.open("r", encoding="utf-8", newline="") as handle:
        paired_rows = list(csv.DictReader(handle))
    targets = {row.get("target") for row in paired_rows}
    if targets != {"road_segments", "intersection_nodes"}:
        messages.append(f"FAIL paired SATCA file has unexpected targets: {sorted(targets)}")


def main() -> int:
    messages: list[str] = []
    check_required(messages)
    check_text_patterns(messages)
    check_croissant(messages)
    check_csv_shapes(messages)
    if messages:
        print("SUMMARY: failures={}".format(len(messages)))
        for message in messages:
            print(message)
        return 1
    print("SUMMARY: failures=0")
    print("Artifact structure, anonymity patterns, Croissant hashes, and core CSV shapes passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
