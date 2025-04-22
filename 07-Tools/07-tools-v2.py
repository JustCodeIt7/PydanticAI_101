"""
tools_demo.py  –  PydanticAI Masterclass · Video 7
--------------------------------------------------
Demonstrates how to register **tools** with @agent.tool so the LLM
can call real Python functions while reasoning.

Prerequisites
-------------
• pip install pydantic-ai
• Set an LLM provider key – e.g.  OPENAI_API_KEY="sk‑..."  – or switch
  the model string to your local Ollama instance ("ollama:llama3.2").

Run
---
python tools_demo.py
"""

from __future__ import annotations

import random
from decimal import Decimal

from pydantic_ai import Agent, RunContext  # Import RunContext

# ── 1. Create an Agent with a helpful system prompt ────────────────────────
agent = Agent(
    "openai:gpt-4o-mini",  # ← change to "ollama:llama3.2" if using Ollama
    system_prompt=(
        "You are a helpful assistant.\n"
        "When appropriate, you can call the provided tools to answer."
    ),
)

# ── 2. Define & register a *simple* tool (no parameters) ───────────────────
@agent.tool
def flip_coin(ctx: RunContext) -> str:  # Add ctx: RunContext
    """Flips a virtual coin and returns 'Heads' or 'Tails'."""
    result = random.choice(["Heads", "Tails"])
    print(f"[tool] flip_coin → {result}")
    return result


# ── 3. Define a tool WITH a parameter and proper docstring -----------------
@agent.tool
def get_stock_price(ctx: RunContext, symbol: str) -> float:  # Add ctx: RunContext
    """
    Fetches the current stock price for a given ticker symbol.

    Args:
        symbol: Stock ticker (e.g., 'AAPL').
    """
    # (Real code would call an API like Alpha Vantage / Finnhub etc.)
    # Note: The 'ctx' parameter is added here only as a workaround
    # for the schema generation error and is not actually used by this tool.
    mock_price = Decimal(random.uniform(50, 500)).quantize(Decimal("1.00"))
    print(f"[tool] get_stock_price({symbol}) → {mock_price}")
    return float(mock_price)


# ── 4. Try the tools via a synchronous run ---------------------------------
def run_examples() -> None:
    print("\n🪙  Asking the LLM to flip a coin …")
    res1 = agent.run_sync("Flip a coin for me.")
    print("Assistant:", res1.output)

    print("\n💹  Asking for a stock quote …")
    res2 = agent.run_sync("What's the current price of MSFT?")
    print("Assistant:", res2.output)


if __name__ == "__main__":
    run_examples()
