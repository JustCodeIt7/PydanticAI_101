"""
PydanticAI Masterclass – Video 1 Demo
------------------------------------
“Welcome to PydanticAI!”

What this script does
1. Verifies the library is installed and prints its version.
2. Highlights the framework’s headline features.
3. Spins up a *local* “Hello‑World” agent with PydanticAI’s `TestModel`
   (no external LLM calls or API keys required).
"""

# ── 1. Imports ──────────────────────────────────────────────────────────────
from __future__ import annotations

import sys
from pprint import pprint

from pydantic_ai import Agent, __version__          # type: ignore
from pydantic_ai.models.test import TestModel      # zero‑cost mock LLM

# ── 2. Constants ────────────────────────────────────────────────────────────
FEATURES: list[str] = [
    "✅ Built‑in Pydantic validation ⇒ type‑safe, structured outputs",
    "✅ Model‑agnostic – switch providers with a single string",
    "✅ Python‑centric control flow; no YAML or custom DSL",
    "✅ Optional Dependency Injection & streaming support",
    "✅ Integrated observability via Pydantic Logfire",
    "✅ Pydantic Graph for complex, stateful workflows",
]

# ── 3. Helper Functions ────────────────────────────────────────────────────
def show_banner() -> None:
    print("=" * 60)
    print(f"  PydanticAI {__version__}  –  Quick Tour")
    print("=" * 60, end="\n\n")


def show_features() -> None:
    print("Key features at a glance:\n")
    for feat in FEATURES:
        print(" •", feat)
    print()


def demo_local_agent() -> None:
    """
    Create a minimal agent backed by `TestModel`, which simply echoes the prompt.
    This lets viewers run the script *offline* without API credentials.
    """
    agent = Agent(
        TestModel(),                       # swap in ‘openai:gpt-4o’ later
        system_prompt="Reply in one concise sentence.",
    )

    question = "What is PydanticAI in one sentence?"
    result = agent.run_sync(question)

    print("Demo agent output:")
    print(">", result.output, "\n")


# ── 4. Script Entry‑Point ──────────────────────────────────────────────────
def main() -> None:
    show_banner()
    show_features()
    demo_local_agent()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
