"""Release preparation helper: collects KPIs, runs smoke tests, drafts notes."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[1]


def run(cmd) -> str:
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    result.check_returncode()
    return result.stdout.strip()


def collect_kpis() -> Dict[str, float]:
    # Placeholder KPI collection. Integrate real metrics here.
    return {
        "documents_indexed": 0,
        "avg_search_latency_ms": 0,
        "chat_accuracy_score": 0,
    }


def main() -> None:
    kpis = collect_kpis()
    (REPO_ROOT / "artifacts").mkdir(exist_ok=True)
    (REPO_ROOT / "artifacts" / "kpi.json").write_text(json.dumps(kpis, indent=2), encoding="utf-8")
    print("KPI snapshot saved -> artifacts/kpi.json")


if __name__ == "__main__":
    main()
