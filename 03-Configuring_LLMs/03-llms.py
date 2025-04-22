"""
PydanticAI Masterclass – Video 3 Demo
------------------------------------
`video03_configure_llms.py`

Prerequisites:
  ▸ pip install pydantic-ai python-dotenv
  ▸ create a .env file alongside this script:
        OPENAI_API_KEY="sk‑..."
        GOOGLE_API_KEY="AIza..."

Run:
  python video03_configure_llms.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic_ai import Agent


# ── 1. Load environment variables safely ──────────────────────────────────
ENV_PATH = Path(__file__).with_suffix(".env")
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)  # silently loads OPENAI_API_KEY / GOOGLE_API_KEY

# Validate keys (fail‑fast for nicer DX)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# if not OPENAI_API_KEY or not GOOGLE_API_KEY:
#     print(
#         "❌  Missing API keys! Add them to a `.env` file or your shell env:\n"
#         "    OPENAI_API_KEY=...\n"
#         "    GOOGLE_API_KEY=...\n"
#     )
#     sys.exit(1)


# ── 2. Configure Agents ───────────────────────────────────────────────────
agent_openai = Agent(
    "openai:gpt-4o",                 # provider:model
    system_prompt="You are a helpful assistant. Reply concisely.",
)

# agent_gemini = Agent(
#     "google-gla:gemini-1.5-flash",
#     system_prompt="You are a creative assistant. Reply concisely.",
# )

# Swap models at will, e.g.:
# agent_openai = Agent("groq:mixtral-8x7b-32768", system_prompt="Be helpful.")


# ── 3. Compare responses from both providers ──────────────────────────────
PROMPT = "Explain PydanticAI in one short sentence."

print("\n🗣  Prompt:", PROMPT, "\n")

# OpenAI
result_oa = agent_openai.run_sync(PROMPT)
print("🔵  OpenAI GPT‑4o  →", result_oa.output)

# Gemini
# result_gem = agent_gemini.run_sync(PROMPT)
# print("🟢  Gemini 1.5‑Flash →", result_gem.output)

print(
    "\n✔  Success! You can now switch providers by editing the "
    "`<provider>:<model>` string without changing any other code.\n"
)
