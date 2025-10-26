"""Helpers to invoke the Infopilot CLI from the desktop UI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
INFOPILOT_ENTRY = REPO_ROOT / "infopilot.py"


def _format_cmd(cmd: Sequence[str]) -> str:
    return " ".join(cmd)


def run_infopilot_cli(
    args: Sequence[str],
    *,
    log_callback: callable | None = None,
) -> None:
    """Run `python infopilot.py <args>` and stream stdout to `log_callback`."""

    if not INFOPILOT_ENTRY.exists():
        raise FileNotFoundError(f"CLI entrypoint를 찾을 수 없습니다: {INFOPILOT_ENTRY}")

    cmd: Sequence[str] = [sys.executable, str(INFOPILOT_ENTRY), *args]
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        if proc.stdout is not None:
            for line in proc.stdout:
                if log_callback:
                    log_callback(line.rstrip())
        return_code = proc.wait()
    finally:
        if proc.stdout:
            proc.stdout.close()
    if return_code != 0:
        raise RuntimeError(f"CLI 명령 실패(exit {return_code}): {_format_cmd(cmd)}")
