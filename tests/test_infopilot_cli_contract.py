from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from scripts.pipeline import infopilot

pytestmark = [pytest.mark.smoke, pytest.mark.integration]
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pipeline" / "infopilot.py"


def test_infopilot_help_exposes_core_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(infopilot.cli, ["--help"])
    assert result.exit_code == 0
    assert "scan" in result.output
    assert "chat" in result.output
    assert "schedule" in result.output
    assert "watch" in result.output


def test_infopilot_watch_help_exposes_expected_options() -> None:
    runner = CliRunner()
    result = runner.invoke(infopilot.cli, ["watch", "--help"])
    assert result.exit_code == 0
    assert "--output-root" in result.output
    assert "--target" in result.output
    assert "--policy" in result.output
    assert "--debounce" in result.output


@pytest.mark.skipif(not SCRIPT.exists(), reason="infopilot.py script missing")
def test_infopilot_run_shim_help_contract() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "run", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "InfoPilot CLI" in (proc.stdout or "")
    assert "chat" in (proc.stdout or "")
    assert "scan" in (proc.stdout or "")


def test_infopilot_chat_help_exposes_query_and_no_auto_train_contract() -> None:
    runner = CliRunner()
    result = runner.invoke(infopilot.cli, ["chat", "--help"])
    assert result.exit_code == 0
    assert "--query" in result.output
    assert "--no-auto-train" in result.output
