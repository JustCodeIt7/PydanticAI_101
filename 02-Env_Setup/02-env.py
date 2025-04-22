"""
PydanticAI Masterclass – Video 2 Demo
------------------------------------
`verify_pydanticai_setup.py`

Purpose
▪ Ensure the current interpreter is Python ≥ 3.9  
▪ Import PydanticAI and print its version (plus Pydantic’s)  
▪ Optionally detect common extras (OpenAI, Google Gemini) and report their status
"""

from __future__ import annotations

import importlib
import platform
import sys
from types import ModuleType
from typing import Final

MIN_VERSION: Final = (3, 9)

# ── Helpers ──────────────────────────────────────────────────────────────────
def color(text: str, clr: str) -> str:  # simple ANSI colouring utility
    codes = {"green": "32", "red": "31", "yellow": "33", "cyan": "36"}
    return f"\033[{codes[clr]}m{text}\033[0m"


def status_line(label: str, ok: bool, detail: str = "") -> None:
    mark = color("✔" if ok else "✘", "green" if ok else "red")
    print(f"{mark} {label:<25} {detail}")


def safe_import(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(name)
    except Exception:  # noqa: BLE001
        return None


# ── 1. Python Version Check ─────────────────────────────────────────────────
print("\n🔍  Verifying environment...\n")

py_tuple = sys.version_info[:3]
py_str = platform.python_version()
version_ok = py_tuple >= MIN_VERSION
status_line("Python version", version_ok, py_str)
if not version_ok:
    sys.exit(
        color(
            "\n❌  Python ≥ 3.9 required. "
            "Create a new virtual env with `python3 -m venv .venv`.",
            "red",
        )
    )

# ── 2. Import PydanticAI & Pydantic ─────────────────────────────────────────
pydantic_ai = safe_import("pydantic_ai")
pydantic_mod = safe_import("pydantic")

status_line(
    "pydantic-ai import",
    pydantic_ai is not None,
    getattr(pydantic_ai, "__version__", "not found"),
)
status_line(
    "pydantic import",
    pydantic_mod is not None,
    getattr(pydantic_mod, "__version__", "not found"),
)

if pydantic_ai is None:
    sys.exit(
        color(
            "\n❌  PydanticAI not installed in this environment.\n"
            "Run `pip install pydantic-ai` (or `pydantic-ai-slim`) "
            "and re‑execute the script.",
            "red",
        )
    )

# ── 3. Check Popular Extras (optional) ──────────────────────────────────────
print("\n🔌  Optional provider extras:")

for provider, module_name in {
    "OpenAI": "openai",
    "Google GLa": "google.generativeai",
}.items():
    mod = safe_import(module_name)
    status_line(provider, mod is not None, getattr(mod, "__version__", "-"))

print(
    color(
        "\n✅  Environment looks good! "
        "You’re ready to start coding with PydanticAI. 🚀\n",
        "cyan",
    )
)
