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

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = """# Copyright 2026 ClinDoc-Bench-IN contributors
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
"""

INCLUDE_DIRS = ["agents", "configs", "models", "scripts", "src", "tests", "utils"]


def apply_header(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if "Licensed under the Apache License" in text[:1000]:
        return False
    if text.startswith("#!"):
        first, rest = text.split("\n", 1)
        new_text = f"{first}\n{HEADER}\n{rest}"
    else:
        new_text = f"{HEADER}\n{text}"
    path.write_text(new_text, encoding="utf-8")
    return True


def main() -> None:
    changed = []
    for dirname in INCLUDE_DIRS:
        for path in (ROOT / dirname).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            if apply_header(path):
                changed.append(path)
    print(f"Updated {len(changed)} Python files with Apache headers.")


if __name__ == "__main__":
    main()
