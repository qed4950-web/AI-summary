"""Initialize default workspace directories based on smart folder config."""
from __future__ import annotations

import json
import os
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[2] / "core" / "config" / "smart_folders.json"


def expand(path: str) -> Path:
    return Path(os.path.expanduser(path)).resolve()


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not CONFIG_PATH.exists():
        raise SystemExit(f"Config file not found: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as fp:
        entries = json.load(fp)

    created = []
    base_root = expand("~/Desktop/AI Summary")
    ensure_directory(base_root)

    for entry in entries:
        raw_path = entry.get("path")
        if not raw_path:
            continue
        target = expand(raw_path)
        ensure_directory(target)
        created.append(target)

    print("✅ Workspace initialized. Directories:")
    for target in created:
        print(f"  - {target}")


if __name__ == "__main__":
    main()
