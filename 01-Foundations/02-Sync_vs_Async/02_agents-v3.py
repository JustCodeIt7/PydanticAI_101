# Comprehensive Python script demonstrating Pydantic AI agent running methods for YouTube tutorial
# Focuses on agent run methods: run(), run_sync(), run_stream(), run_stream_events(), iter()
# Uses basic examples with sync/async execution, error handling, console output

import asyncio
from datetime import date

from pydantic_ai import Agent, RunContext
from rich import print
# Section 1: Agent Definition
# Define a simple weather agent with a tool for demonstration


weather_agent = Agent(
    "openai:gpt-4o-mini",  # Requires OpenAI API key in environment
    system_prompt="Provide weather forecasts using tools.",
    model_settings={"temperature": 0.2},
)

# Section 2: Tool Implementation
# Basic tool for weather queries


@weather_agent.tool
async def get_weather(ctx: RunContext, location: str, query_date: date) -> str:
    """Mock tool to simulate weather fetch."""
    return f"Sunny in {location} on {query_date}."


# Section 3: Execution Examples
# Demonstrate each run method with basic prompts and console output


async def demo_run_async():
    """Demonstrate agent.run() - async function returning RunResult."""
    print("=== agent.run() Example (Async) ===")
    result = await weather_agent.run("Weather in Paris tomorrow?")
    print(f"Result: {result.output}")


def demo_run_sync():
    """Demonstrate agent.run_sync() - sync function returning RunResult."""
    print("=== agent.run_sync() Example (Sync) ===")
    result = weather_agent.run_sync("Weather in London today?")
    print(f"Result: {result.output}")


async def demo_run_stream():
    """Demonstrate agent.run_stream() - async context manager for streaming text."""
    print("=== agent.run_stream() Example (Async Stream) ===")
    async with weather_agent.run_stream("Weather in Tokyo next week?") as response:
        async for text in response.stream_text():
            print(text, end="")
    print("\nStream complete.")


async def demo_run_stream_events():
    """Demonstrate agent.run_stream_events() - async iterable of events and final result."""
    print("=== agent.run_stream_events() Example (Async Events) ===")
    events = []
    async for event in weather_agent.run_stream_events("Weather in Berlin?"):
        events.append(event)
        if hasattr(event, "result"):
            print(f"Final Result: {event.result.output}")
    print(f"Total Events: {len(events)}")


async def demo_iter():
    """Demonstrate agent.iter() - context manager for iterating over graph nodes."""
    print("=== agent.iter() Example (Async Iteration) ===")
    async with weather_agent.iter("Weather in Rome?") as agent_run:
        nodes = []
        async for node in agent_run:
            nodes.append(node)
        print(f"Final Output: {agent_run.result.output}")
        print(f"Nodes Processed: {len(nodes)}")


# Section 4: Main Function
# Orchestrates all demos, running sync and async appropriately


def main():
    """Main function to run all agent run method demonstrations."""
    print("Starting Pydantic AI Agent Run Methods Tutorial Script\n")

    # Sync demo
    demo_run_sync()
    print()

    # Async demos
    asyncio.run(demo_run_async())
    print()
    asyncio.run(demo_run_stream())
    print()
    asyncio.run(demo_run_stream_events())
    print()
    asyncio.run(demo_iter())
    print()

    print("Tutorial script completed.")


if __name__ == "__main__":
    main()
