import asyncio
import os
from datetime import date

# load environment variables from a .env file if present
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.providers.ollama import OllamaProvider
from rich import print
import logfire

################################ Environment Setup ################################
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LOGFIRE_API_KEY = os.getenv("LOGFIRE_API_KEY")
# logfire.configure(token=LOGFIRE_API_KEY)
# logfire.instrument_pydantic_ai()

################################ Agent Definition & Configuration ################################
# OpenAI example (uncomment to use)
# model = OpenAIChatModel("gpt-4o-mini", provider=OpenAIProvider(api_key=OPENAI_API_KEY))

# OpenRouter example (uncomment to use)
# model = OpenAIChatModel(
#     "openai/gpt-oss-120b", provider=OpenRouterProvider(api_key=OPENROUTER_API_KEY)
# )

# Ollama example (uncomment to use)
model = OpenAIChatModel(
    model_name="gpt-oss:20b", provider=OllamaProvider(base_url="http://eos.local:11434/v1")
)

# Define a simple agent for weather forecasting
weather_agent = Agent(
    model,  # "openai:gpt-4o-mini",
    system_prompt="Provide weather forecasts using tools.",  # Define the agent's core instruction
    model_settings={
        "temperature": 0,
        # "max_tokens": 500,
    },  # Set a low temperature for more deterministic outputs
)

################################ Tool Implementation ################################


# Register a function as a tool the agent can use
@weather_agent.tool
async def get_weather(ctx: RunContext, location: str, query_date: date) -> str:
    """Fetch weather data for a given location and date."""
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
    print("\n############# Running synchronous demonstration #############")
    print("=== Synchronous Execution ===")
    demo_run_sync()

    # Run all asynchronous demonstrations
    print("\n############# Running asynchronous demonstration #############")
    print("=== Asynchronous Execution ===")
    asyncio.run(demo_run_async())
    print("\n=== Stream Execution ===")
    asyncio.run(demo_run_stream())
    print("\n=== Stream Events Execution ===")
    asyncio.run(demo_run_stream_events())
    print("\n=== Iteration Execution ===")
    asyncio.run(demo_iter())

    print("Tutorial script completed.")


# Ensure the script runs only when executed directly
if __name__ == "__main__":
    main()
