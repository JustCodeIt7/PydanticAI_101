# -*- coding: utf-8 -*-
"""
Tutorial: PydanticAI Agents
Estimated video length: 10–15 minutes
This script provides annotated examples demonstrating key PydanticAI Agent features.
"""

# Section 0: Imports and Setup
import asyncio
from datetime import date
from typing import TypedDict, List

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, ModelRetry, UnexpectedModelBehavior, capture_run_messages
from pydantic_ai.usage import UsageLimits
from pydantic_ai.models.gemini import GeminiModelSettings


# ========================================================================
# Section 1: Create a Basic Agent and Tool (Roulette Wheel Example)
# ========================================================================
# 1. Define a simple Agent that takes an `int` dependency and returns a `bool`.
roulette_agent = Agent(
    "gpt-4.1-nano",
    deps_type=int,
    output_type=bool,
    system_prompt=(
        "Use the `roulette_wheel` function to check if the given number is the winning square."
    ),
)

# 2. Register a tool on the agent for checking the winning square.
@roulette_agent.tool
async def roulette_wheel(ctx: RunContext[int], square: int) -> str:
    """Return 'winner' if square matches the secret number, else 'loser'."""
    return "winner" if square == ctx.deps else "loser"

# ========================================================================
# Section 2: Running Agents (Sync, Async, Streaming)
# ========================================================================

def demo_basic_runs(secret: int):
    # Synchronous run
    result_sync = roulette_agent.run_sync(
        f"I bet on square {secret}",
        deps=secret,
    )
    print("Sync output:", result_sync.output)

    # Asynchronous run via asyncio
    async def async_run():
        result = await roulette_agent.run(
            f"Try square {secret}", deps=secret
        )
        print("Async output:", result.output)
    asyncio.run(async_run())

    # Streaming run
    async def stream_run():
        async with roulette_agent.run_stream(
            "Will square 5 win?", deps=secret
        ) as stream:
            out = await stream.get_output()
            print("Streamed output:", out)
    asyncio.run(stream_run())


# ========================================================================
# Section 3: Iterating Over Agent Graph (Async For)
# ========================================================================
async def demo_iteration():
    numbers = []
    async with roulette_agent.iter(
        f"Test square {18}", deps=18
    ) as run:
        async for node in run:
            numbers.append(type(node).__name__)
    print("Executed nodes:", numbers)
    print("Final result:", run.result.output)


# ========================================================================
# Section 4: System Prompts vs. Instructions
# ========================================================================
# Create another agent to illustrate system_prompt vs instructions
instr_agent = Agent(
    "gpt-4.1-nano",
    deps_type=str,
    system_prompt="You are a friendly assistant.",
    instructions="Answer briefly."
)

@instr_agent.system_prompt
def dynamic_name(ctx: RunContext[str]) -> str:
    return f"Greeting for: {ctx.deps}"

@instr_agent.instructions
def dynamic_date() -> str:
    return f"Today's date: {date.today()}"


def demo_prompts():
    result = instr_agent.run_sync(
        "What is the date?", deps="Alice"
    )
    print("Prompts output:", result.output)


# ========================================================================
# Section 5: Usage Limits and Model Settings
# ========================================================================
# Agent with response token limit
limited_agent = Agent("gpt-4.1-nano")

@limited_agent.tool
def dummy_tool() -> str:
    return "dummy"


def demo_usage_limits():
    try:
        limited_agent.run_sync(
            "Answer in one word.",
            usage_limits=UsageLimits(response_tokens_limit=1)
        )
    except Exception as e:
        print("Usage limit exceeded:", e)

# Agent with custom Gemini settings
gemini_agent = Agent(
    "google-gla:gemini-1.5-flash",
    model_settings=GeminiModelSettings(
        temperature=0.0,
        gemini_safety_settings=[
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_LOW_AND_ABOVE"},
        ],
    ),
)

def demo_model_settings():
    try:
        gemini_agent.run_sync(
            "Write rude jokes.",
        )
    except UnexpectedModelBehavior as e:
        print("Model safety triggered:", e)


# ========================================================================
# Section 6: Reflection & Self-Correction (ModelRetry)
# ========================================================================

class NeverOutputType(TypedDict):
    never: str

retry_agent = Agent(
    "gpt-4.1-nano",
    deps_type=dict,
    output_type=NeverOutputType,
    retries=2,
)

@retry_agent.tool(retries=1)
def loop_tool() -> int:
    raise ModelRetry("Please try again.")


def demo_retries():
    try:
        retry_agent.run_sync("Trigger loop.")
    except Exception as e:
        print("Retry error:", e)


# ========================================================================
# Section 7: Conversation Across Runs
# ========================================================================
conv_agent = Agent("gpt-4.1-nano")

def demo_conversation():
    first = conv_agent.run_sync("Who discovered penicillin?")
    print("Q1:", first.output)
    second = conv_agent.run_sync(
        "When was it discovered?",
        message_history=first.new_messages()
    )
    print("Q2:", second.output)


# ========================================================================
# Main: Run all demos
# ========================================================================
if __name__ == "__main__":
    secret_number = 18
    print("--- Demo: Basic Runs ---")
    demo_basic_runs(secret_number)

    print("--- Demo: Iteration ---")
    asyncio.run(demo_iteration())

    print("--- Demo: Prompts & Instructions ---")
    demo_prompts()

    print("--- Demo: Usage Limits & Settings ---")
    demo_usage_limits()
    demo_model_settings()

    print("--- Demo: Retries & Reflection ---")
    demo_retries()

    print("--- Demo: Conversation Across Runs ---")
    demo_conversation()
