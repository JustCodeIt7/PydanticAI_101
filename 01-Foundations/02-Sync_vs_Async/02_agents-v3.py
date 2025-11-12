import asyncio
import enum
import os
from datetime import date

import logfire
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, UsageLimits, ModelSettings
from pydantic_ai.mcp import ProcessToolCallback
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from rich import print

################### Environment Setup ################
# Load environment variables from a .env file if one exists
load_dotenv()

# Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LOGFIRE_API_KEY = os.getenv("LOGFIRE_API_KEY")

# Optional: Configure Logfire for detailed tracing and debugging
logfire.configure(token=LOGFIRE_API_KEY)
logfire.instrument_pydantic_ai()

############### Agent Definition & Configuration ##################

# Define an OpenAI model (gpt-4o-mini)
# model = OpenAIChatModel("gpt-4o-mini", provider=OpenAIProvider(api_key=OPENAI_API_KEY))

# Define an OpenRouter model
# model = OpenAIChatModel(
#     "openai/gpt-4-turbo", provider=OpenRouterProvider(api_key=OPENROUTER_API_KEY)
# )

# Define an Ollama model to connect to a local LLM instance
model = OpenAIChatModel(
    model_name="gpt-oss:20b",
    provider=OllamaProvider(
        base_url="http://eos.local:11434/v1"
    ),
)

# Create an agent instance
weather_agent = Agent(
    model,
    system_prompt="Provide weather forecasts using tools.",  # Define the agent's core instruction
    model_settings=ModelSettings(temperature=0), # add model settings
)

##################### Tool Implementation ###################


# Define a function and register it as a tool for the agent to use
@weather_agent.tool
async def get_weather(ctx: RunContext, location: str, query_date: date) -> str:
    """Fetch weather data for a given location and date."""
    # Simulate a call to an external weather API
    return f"Sunny in {location} on {query_date}."


################## Agent Execution Demonstrations #################


# Demonstrate the standard async `run()` method
async def demo_run_async():
    """Demonstrate agent.run(), which returns the final result asynchronously."""
    print("[blue]=== agent.run() Example (Async) ===[/blue]")
    # Execute the agent and await the complete `RunResult`
    result = await weather_agent.run("Weather in Paris tomorrow?")
    print(f"Result: {result.output}")


# Demonstrate the synchronous `run_sync()` method
def demo_run_sync():
    """Demonstrate agent.run_sync() for use in synchronous code."""
    print("[cyan]=== agent.run_sync() Example (Sync) ===[/cyan]")
    # Execute the agent and block until the final `RunResult` is available
    result = weather_agent.run_sync("Weather in London today?", usage_limits=UsageLimits(output_tokens_limit=500))
    print(f"Result: {result.output}")


# Demonstrate streaming text output with `run_stream()`
async def demo_run_stream():
    """Demonstrate agent.run_stream() for handling real-time text output."""
    print("[magenta]=== agent.run_stream() Example (Async Stream) ===[/magenta]")
    # Use a context manager to handle the streaming response from the agent
    async with weather_agent.run_stream("Weather in Tokyo next week?", usage_limits=UsageLimits(request_limit=10)) as response:
        # Iterate over incoming text chunks as the LLM generates them
        async for text in response.stream_text():
            print(text, end="")
    print("\n[yellow]Stream complete.[/yellow]")


# Demonstrate processing structured events with `run_stream_events()`
async def demo_run_stream_events():
    """Demonstrate agent.run_stream_events() for detailed, structured updates."""
    print("[purple]=== agent.run_stream_events() Example (Async Events) ===[/purple]")
    events = []
    # Iterate over each event generated during the agent's execution lifecycle
    async for event in weather_agent.run_stream_events("Weather in Berlin?"):
        events.append(event)
        # Check if the event contains the final result and print its output
        if hasattr(event, "result") and hasattr(event.result, "output"):
            print(f"Final Result: {event.result.output}")
    print(f"[yellow]Total Events: {len(events)}[/yellow]")


# Demonstrate iterating over the agent's internal steps with `iter()`
async def demo_iter():
    """Demonstrate agent.iter() to inspect the agent's thought process."""
    print("[purple]=== agent.iter() Example (Async Iteration) ===[/purple]")
    # Use a context manager to access the agent's execution graph
    async with weather_agent.iter("Weather in Rome?") as agent_run:
        nodes = []
        # Iterate through each internal processing node (e.g., tool call, LLM response)
        async for node in agent_run:
            nodes.append(node)
        # Access the final output after the iteration is complete
        print(f"Final Output: {agent_run.result.output}")
        print(f"[yellow]Nodes Processed: {len(nodes)}[/yellow]")


def conversation_example():
    """Demonstrate a multi-turn conversation with the agent."""
    print("[green]=== Multi-turn Conversation Example ===[/green]")
    # Start a conversation with an initial user message
    result1 = weather_agent.run_sync("What's the weather in New York this weekend?")
    print(f"[green] 1. Initial Result: {result1.output}[/green]")
    result2  = weather_agent.run_sync("And in San Francisco?",message_history=result1.new_messages())
    # Add a follow-up user message to the conversation
    print(f"\n[purple] 2. Conversation Result: {result2.output}[/purple]")

################################ Main Execution Block ################################


# Define the main function to orchestrate all demonstrations
def main():
    """Run all agent run method demonstrations."""
    print("[yellow]Starting Pydantic AI Agent Run Methods Tutorial Script[/yellow]\n")

    # Run the synchronous demonstration first
    print("\n[cyan]############# Running synchronous demonstration #############[/cyan]")

    print("[cyan]=== Synchronous Execution ===[/cyan]")
    demo_run_sync()

    # Run all asynchronous demonstrations sequentially
    print("\n[cyan]############# Running asynchronous demonstration #############[/cyan]")

    print("[cyan]=== Asynchronous Execution ===[/cyan]")
    asyncio.run(demo_run_async())

    print("\n[cyan]=== Stream Execution ===[/cyan]")
    asyncio.run(demo_run_stream())

    print("\n[cyan]=== Stream Events Execution ===[/cyan]")
    asyncio.run(demo_run_stream_events())

    print("\n[cyan]=== Iteration Execution ===[/cyan]")
    asyncio.run(demo_iter())

    print("\n[cyan]=== Conversation Execution ===[/cyan]")
    conversation_example()


    print("Tutorial script completed.")


# Execute the main function when the script is run directly
if __name__ == "__main__":
    main()
