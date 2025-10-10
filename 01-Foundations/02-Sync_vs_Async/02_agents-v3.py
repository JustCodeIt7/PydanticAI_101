import asyncio
from datetime import date

from pydantic_ai import Agent, RunContext
from rich import print

################################ Agent Definition & Configuration ################################

# Define a simple agent for weather forecasting
weather_agent = Agent(
    "openai:gpt-4o-mini",  # Specify the LLM, requiring an OpenAI API key
    system_prompt="Provide weather forecasts using tools.",  # Define the agent's core instruction
    model_settings={"temperature": 0.2},  # Set a low temperature for more deterministic outputs
)

################################ Tool Implementation ################################


# Register a function as a tool the agent can use
@weather_agent.tool
async def get_weather(ctx: RunContext, location: str, query_date: date) -> str:
    # Simulate fetching weather data from an external API
    return f"Sunny in {location} on {query_date}."


################################ Agent Execution Demonstrations ################################


# Demonstrate the standard async `run()` method
async def demo_run_async():
    """Demonstrate agent.run(), which returns the final result asynchronously."""
    print("=== agent.run() Example (Async) ===")
    # Execute the agent and wait for the complete `RunResult`
    result = await weather_agent.run("Weather in Paris tomorrow?")
    print(f"Result: {result.output}")


# Demonstrate the synchronous `run_sync()` method
def demo_run_sync():
    """Demonstrate agent.run_sync() for use in synchronous code."""
    print("=== agent.run_sync() Example (Sync) ===")
    # Execute the agent and block until the final `RunResult` is available
    result = weather_agent.run_sync("Weather in London today?")
    print(f"Result: {result.output}")


# Demonstrate streaming text output with `run_stream()`
async def demo_run_stream():
    """Demonstrate agent.run_stream() for handling real-time text output."""
    print("=== agent.run_stream() Example (Async Stream) ===")
    # Use a context manager to handle the streaming response
    async with weather_agent.run_stream("Weather in Tokyo next week?") as response:
        # Iterate over incoming text chunks as they are generated
        async for text in response.stream_text():
            print(text, end="")
    print("\nStream complete.")


# Demonstrate processing structured events with `run_stream_events()`
async def demo_run_stream_events():
    """Demonstrate agent.run_stream_events() for detailed, structured updates."""
    print("=== agent.run_stream_events() Example (Async Events) ===")
    events = []
    # Iterate over each event generated during the agent's execution
    async for event in weather_agent.run_stream_events("Weather in Berlin?"):
        events.append(event)
        # Check if the event contains the final result and print it
        if hasattr(event, "result") and hasattr(event.result, "output"):
            print(f"Final Result: {event.result.output}")
    print(f"Total Events: {len(events)}")


# Demonstrate iterating over the agent's internal steps with `iter()`
async def demo_iter():
    """Demonstrate agent.iter() to inspect the agent's thought process."""
    print("=== agent.iter() Example (Async Iteration) ===")
    # Use a context manager to access the agent's execution graph
    async with weather_agent.iter("Weather in Rome?") as agent_run:
        nodes = []
        # Iterate through each internal processing node (e.g., tool call, LLM response)
        async for node in agent_run:
            nodes.append(node)
        # Access the final output after the iteration is complete
        print(f"Final Output: {agent_run.result.output}")
        print(f"Nodes Processed: {len(nodes)}")


################################ Main Execution Block ################################


# Orchestrate all demonstration functions
def main():
    """Main function to run all agent run method demonstrations."""
    print("Starting Pydantic AI Agent Run Methods Tutorial Script\n")

    # Run the synchronous demonstration
    demo_run_sync()
    print()

    # Run all asynchronous demonstrations
    asyncio.run(demo_run_async())
    print()
    asyncio.run(demo_run_stream())
    print()
    asyncio.run(demo_run_stream_events())
    print()
    asyncio.run(demo_iter())
    print()

    print("Tutorial script completed.")


# Ensure the script runs only when executed directly
if __name__ == "__main__":
    main()
