from __future__ import annotations

import importlib
import traceback


def _import_module(module_name: str, label: str) -> None:
    importlib.import_module(module_name)
    print(f"   - {label}: OK")


def main() -> int:
    print("Starting Comprehensive Integrity Check...")

    checks = [
        ("core.utils", "core.utils"),
        ("core.conversation.orchestrator", "core.conversation.orchestrator"),
        ("core.conversation.lnp_chat", "core.conversation.lnp_chat"),
        ("core.agents.web.agent", "core.agents.web.agent"),
        ("core.agents.document.agent", "core.agents.document.agent"),
        ("core.agents.meeting", "core.agents.meeting"),
        ("core.agents.photo", "core.agents.photo"),
        ("scripts.pipeline.infopilot_cli.chat", "scripts.pipeline.infopilot_cli.chat"),
        ("desktop_app.main", "desktop_app.main"),
    ]

    try:
        for index, (module_name, label) in enumerate(checks, start=1):
            print(f"{index}. Checking {label}...")
            _import_module(module_name, label)

        chat_module = importlib.import_module("scripts.pipeline.infopilot_cli.chat")
        if not hasattr(chat_module, "cmd_chat"):
            raise AttributeError("cmd_chat missing in chat.py")

        print("\nAll Modules Loaded Successfully. Syntax and Import Integrity Verified.")
        return 0
    except ImportError as exc:
        print(f"\nImportError: {exc}")
        return 1
    except SyntaxError as exc:
        print(f"\nSyntaxError: {exc}")
        return 1
    except NameError as exc:
        print(f"\nNameError: {exc}")
        return 1
    except Exception as exc:
        print(f"\nCritical Error: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
