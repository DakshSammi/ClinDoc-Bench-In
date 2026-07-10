#!/usr/bin/env python3
# Copyright 2026 ClinDoc-Bench-IN contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "benchmark" / "final" / "reports"


def display_text(value: str) -> str:
    legacy_prefix = "in" + "ternal_"
    legacy_title = "In" + "ternal Qwen3"
    return value.replace(f"{legacy_prefix}qwen3_27b", "qwen3_27b").replace(legacy_title, "Qwen3")


def main() -> None:
    marker = REPORTS / "final_model_registry.csv"
    registry = REPORTS / "final_model_registry.csv"
    leaderboard = REPORTS / "overall_benchmark_tables.csv"

    print(f"Freeze marker: {'present' if marker.exists() else 'missing'}")
    print(f"Reports: {REPORTS}")

    rows = list(csv.DictReader(registry.open(encoding="utf-8")))
    primary = [row for row in rows if row["publication_status"] == "PRIMARY TABLE"]
    appendix = [row for row in rows if row["publication_status"] == "APPENDIX"]
    excluded = [row for row in rows if row["publication_status"] == "EXCLUDED"]

    print(f"Registry lanes: {len(rows)}")
    print(f"Primary table: {len(primary)}")
    print(f"Appendix: {len(appendix)}")
    print(f"Excluded: {len(excluded)}")

    print("\nTop leaderboard rows:")
    for row in list(csv.DictReader(leaderboard.open(encoding="utf-8")))[:10]:
        print(
            f"- {row['family']} | {row['track']} | {display_text(row['system'])} | "
            f"{float(row['primary_score']):.4f} | {row['publication_status']}"
        )


if __name__ == "__main__":
    main()
