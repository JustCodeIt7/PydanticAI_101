import asyncio
import os
from datetime import date

import logfire
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, UsageLimits, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from rich import print

################################ Environment Setup ################################
# Load environment variables from a .env file if one exists
load_dotenv()

# Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LOGFIRE_API_KEY = os.getenv("LOGFIRE_API_KEY")

# Optional: Configure Logfire for detailed tracing and debugging
# logfire.configure(token=LOGFIRE_API_KEY)
# logfire.instrument_pydantic_ai()

################################ Agent Definition & Configuration ################################

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
    ),  # Use the local Ollama server endpoint
)

# Create an agent instance
weather_agent = Agent(
    model,
    system_prompt="Provide weather forecasts using tools.",  # Define the agent's core instruction
    model_settings=ModelSettings(temperature=0),  # Use a low temperature for more predictable and deterministic outputs
)

################################ Tool Implementation ################################


# Define a function and register it as a tool for the agent to use
@weather_agent.tool
async def get_weather(ctx: RunContext, location: str, query_date: date) -> str:
    """Fetch weather data for a given location and date."""


################################ Agent Execution Demonstrations ################################


# Demonstrate the standard async `run()` method
async def demo_run_async():
    """Demonstrate agent.run(), which returns the final result asynchronously."""


# Demonstrate the synchronous `run_sync()` method
def demo_run_sync():
    """Demonstrate agent.run_sync() for use in synchronous code."""


# Demonstrate streaming text output with `run_stream()`
async def demo_run_stream():
    """Demonstrate agent.run_stream() for handling real-time text output."""


# Demonstrate processing structured events with `run_stream_events()`
async def demo_run_stream_events():
    """Demonstrate agent.run_stream_events() for detailed, structured updates."""


# Demonstrate iterating over the agent's internal steps with `iter()`
async def demo_iter():
    """Demonstrate agent.iter() to inspect the agent's thought process."""


def conversation_example():
    """Demonstrate a multi-turn conversation with the agent."""

################################ Main Execution Block ################################


# Define the main function to orchestrate all demonstrations
def main():
    """Run all agent run method demonstrations."""


# Execute the main function when the script is run directly
if __name__ == "__main__":
    main()