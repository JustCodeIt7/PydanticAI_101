"""
agent_lifecycle_ollama.py  –  PydanticAI Masterclass · Video 6
-----------------------------------------------------------------
Showcases the three execution modes (`run_sync`, async `run`, and
`run_stream`) using a **locally‑hosted Ollama** model (`llama3.2`).

Prerequisites
-------------
1. Install and start Ollama:  https://ollama.com  
2. Pull the model once:       ollama pull llama3:latest   (or any tag)  
3. pip install pydantic-ai
4. (Optional) ensure the Ollama host is discoverable, e.g.:
      export OLLAMA_BASE_URL="http://localhost:11434"   # default port

Run
---
python agent_lifecycle_ollama.py
"""

from __future__ import annotations

import asyncio

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# ---------------------------------------------------------------------------
# 1. Instantiate the Agent (one line)
# ---------------------------------------------------------------------------

model = OpenAIModel(
    'llama3.2',
    provider=OpenAIProvider(
        base_url='http://localhost:11434/v1',
        
    ),
)
agent = Agent(model)
# ---------------------------------------------------------------------------
# 2. Synchronous example – run_sync
# ---------------------------------------------------------------------------
# def sync_example() -> None:
#     print("\n--- run_sync() example ---")
#     result = agent.run_sync("Explain PydanticAI in one sentence.")
#     print("Result:", result.output)


# ---------------------------------------------------------------------------
# 3. Asynchronous example – await run()
# ---------------------------------------------------------------------------
async def async_example() -> None:
    print("\n--- async run() example ---")
    result = await agent.run("What is the capital of France?")
    print("Result:", result.output)


# ---------------------------------------------------------------------------
# 4. Streaming example – run_stream()
# ---------------------------------------------------------------------------
async def streaming_example() -> None:
    print("\n--- run_stream() example ---")
    query = "Tell me a short bedtime story set on Mars."
    async with agent.run_stream(query) as response:
        print(await response.get_output())

    print("\n--- end of stream ---\n")


# ---------------------------------------------------------------------------
# 5. Entry‑point
# ---------------------------------------------------------------------------
async def main() -> None:
    # sync_example()
    await async_example()
    await streaming_example()


if __name__ == "__main__":
    asyncio.run(main())
