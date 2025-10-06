import asyncio
from datetime import date
from typing import TypedDict, List

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, ModelRetry, UnexpectedModelBehavior, capture_run_messages
from pydantic_ai.usage import UsageLimits
from pydantic_ai.models.gemini import GeminiModelSettings
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider


base_url = 'http://100.95.122.242:11434/v1'
model_name = 'qwen3:1.7b'
ollama_model = OpenAIModel(
    model_name=model_name,
    provider=OpenAIProvider(base_url=base_url)
)

# ========================================================================
# Section 0.5: Simplest Agent (Direct LLM Interaction)
# ========================================================================
# 1. Define an agent that simply passes input to the LLM and returns its output.
#    No tools, no complex dependencies, output_type is str.
simple_chat_agent = Agent(
    ollama_model,
    output_type=str,
    system_prompt="You are a helpful assistant.",
)

def demo_simple_chat():
    user_input = "Explain the concept of a Large Language Model in one sentence."
    result = simple_chat_agent.run_sync(user_input)
    print("Simple Chat Agent Output:", result.output)

# ========================================================================
# Section 1: Create a Basic Agent and Tool (Roulette Wheel Example)
# ========================================================================
# 1. Define a simple Agent that takes an `int` dependency and returns a `bool`.
agent = Agent(
    ollama_model,
    deps_type=int,
    output_type=bool,
    system_prompt=(
        "Use the `roulette_wheel` function to check if the given number is the winning square."
    ),
)

# 2. Register a tool on the agent for checking the winning square.
@agent.tool
async def roulette_wheel(ctx: RunContext[int], square: int) -> str:
    """Return 'winner' if square matches the secret number, else 'loser'."""
    return "winner" if square == ctx.deps else "loser"

# ========================================================================
# Section 2: Running Agents (Sync, Async, Streaming)
# ========================================================================

def demo_basic_runs(secret: int):
    # Synchronous run
    result_sync = agent.run_sync(
        f"I bet on square {secret}",
        deps=secret,
    )
    print("Sync result:", result_sync.output)

    # Asynchronous run via asyncio
    async def async_run():
        result = await agent.run(
            f"Try square {secret}", deps=secret
        )
        print("Async result:", result.output) # Changed to print result.output for consistency
    asyncio.run(async_run())

    # Streaming run
    async def stream_run():
        async with agent.run_stream(
            f"Will square {secret} win?", deps=secret
        ) as stream:
            out = await stream.get_output()
            print("Streamed output value:", out)
    asyncio.run(stream_run())

# ========================================================================
# Section 3: Iterating Over Agent Graph (Async For)
# ========================================================================
async def demo_iteration():
    numbers = []
    async with agent.iter(
        f"Test square {18}", deps=18
    ) as run:
        async for node in run:
            numbers.append(type(node).__name__)
    print("Executed nodes:", numbers)
    print("Final result:", run.result.output)


# ========================================================================
# Section 4: System Prompts vs. Instructions
# ========================================================================
instr_agent = Agent(
    ollama_model,
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
limited_agent = Agent(ollama_model)

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

ollama_custom_agent = Agent(
    ollama_model,
)

def demo_model_settings():
    try:
        ollama_custom_agent.run_sync(
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
    ollama_model,
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
conv_agent = Agent(ollama_model)

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

    print("--- Demo: Simplest Chat Agent ---")
    demo_simple_chat()

    print("\n--- Demo: Basic Runs ---")
    demo_basic_runs(secret_number)

    print("\n--- Demo: Iteration ---")
    asyncio.run(demo_iteration())

    print("\n--- Demo: Prompts & Instructions ---")
    demo_prompts()

    print("\n--- Demo: Usage Limits & Settings ---")
    demo_usage_limits()
    demo_model_settings()

    print("\n--- Demo: Retries & Reflection ---")
    demo_retries()

    print("\n--- Demo: Conversation Across Runs ---")
    demo_conversation()