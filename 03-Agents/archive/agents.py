# PydanticAI Agents Tutorial
# ==========================
#
# This script demonstrates the key features of PydanticAI agents for a tutorial video.
# It covers agent creation, tools, structured outputs, running methods, and more.
#
# Prerequisites:
# - pydantic-ai installed (`pip install pydantic-ai`)
# - LLM provider libraries installed (e.g., `pip install openai anthropic google-generativeai`)
# - API keys configured as environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)

import asyncio
import os
from datetime import date, datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, AsyncIterator, Tuple
from typing_extensions import TypedDict

from pydantic import BaseModel, Field

from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.exceptions import UsageLimitExceeded, UnexpectedModelBehavior
from pydantic_ai.usage import UsageLimits, Usage
from pydantic_ai.messages import (
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPartDelta,
    ToolCallPartDelta,
)
from pydantic_ai.models.gemini import GeminiModelSettings
from pydantic_ai.agent import capture_run_messages
from pydantic_graph import End

# Import our fake database for examples
from fake_database import DatabaseConn

# Helper function to print section headers
def print_section_header(title):
    print("\n" + "=" * 80)
    print(f"--- {title} ---")
    print("=" * 80 + "\n")

# ============================================================================
# SECTION 1: BASIC AGENT CREATION AND USAGE
# ============================================================================

async def section_1_basic_agent():
    print_section_header("SECTION 1: BASIC AGENT CREATION AND USAGE")

    # Create a simple agent that simulates a roulette wheel
    # This agent takes an integer as dependency (the winning number)
    # and returns a boolean indicating if the player won
    roulette_agent = Agent(
        'openai:gpt-4o',  # The LLM model to use
        deps_type=int,    # The agent expects an integer dependency
        output_type=bool, # The agent will return a boolean
        system_prompt=(
            'Use the `roulette_wheel` function to see if the '
            'customer has won based on the number they provide.'
        ),
    )

    # Define a tool for the agent to use
    @roulette_agent.tool
    async def roulette_wheel(ctx: RunContext[int], square: int) -> str:
        """Check if the square is a winner."""
        print(f"Tool called with square: {square}, winning number: {ctx.deps}")
        return 'winner' if square == ctx.deps else 'loser'

    # Set the winning number
    winning_number = 18
    print(f"Winning number is set to: {winning_number}")

    # Run the agent with a prompt that should result in a win
    print("\nRunning agent with 'Put my money on square eighteen'...")
    result = await roulette_agent.run('Put my money on square eighteen', deps=winning_number)
    print(f"Raw output: {result.output}")
    print(f"Validated output (bool): {result.output}")

    # Run the agent with a prompt that should result in a loss
    print("\nRunning agent with 'I bet five is the winner'...")
    result = await roulette_agent.run('I bet five is the winner', deps=winning_number)
    print(f"Raw output: {result.output}")
    print(f"Validated output (bool): {result.output}")

# ============================================================================
# SECTION 2: DIFFERENT WAYS TO RUN AN AGENT
# ============================================================================

async def section_2_running_agents():
    print_section_header("SECTION 2: DIFFERENT WAYS TO RUN AN AGENT")

    # Create a simple agent
    agent = Agent('openai:gpt-4o')

    # 1. Synchronous run (run_sync)
    print("1. Using agent.run_sync() - Synchronous:")
    result_sync = agent.run_sync('What is the capital of Italy?')
    print(f"Output: {result_sync.output}")

    # 2. Asynchronous run (run)
    print("\n2. Using await agent.run() - Asynchronous:")
    result_async = await agent.run('What is the capital of France?')
    print(f"Output: {result_async.output}")

    # 3. Streaming run (run_stream)
    print("\n3. Using agent.run_stream() - Streaming:")
    print("Streaming full response:")
    async with agent.run_stream('What is the capital of the UK?') as response:
        output = await response.get_output()
        print(f"Full output: {output}")

    print("\nStreaming token by token:")
    async with agent.run_stream('What is the capital of Germany?') as response:
        print("Tokens: ", end="", flush=True)
        async for event in response.events():
            if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                print(event.delta.content_delta, end="", flush=True)
        print()  # New line after streaming completes

# ============================================================================
# SECTION 3: FUNCTION TOOLS AND STRUCTURED OUTPUT
# ============================================================================

async def section_3_tools_and_output():
    print_section_header("SECTION 3: FUNCTION TOOLS AND STRUCTURED OUTPUT")

    # Define a structured output type using Pydantic
    class WeatherReport(BaseModel):
        location: str
        temperature: float
        conditions: str
        forecast_date: date
        humidity: Optional[int] = None

    # Create a weather service dependency
    @dataclass
    class WeatherService:
        api_key: str = "fake_api_key"

        async def get_forecast(self, location: str, forecast_date: date) -> Dict[str, Any]:
            # In a real app, this would call a weather API
            print(f"Getting forecast for {location} on {forecast_date}")
            return {
                "location": location,
                "temperature": 24.5,
                "conditions": "Sunny",
                "forecast_date": forecast_date,
                "humidity": 65
            }

        async def get_historic_weather(self, location: str, historic_date: date) -> Dict[str, Any]:
            # In a real app, this would query a database or API
            print(f"Getting historical weather for {location} on {historic_date}")
            return {
                "location": location,
                "temperature": 18.2,
                "conditions": "Partly cloudy",
                "forecast_date": historic_date,
                "humidity": 72
            }

    # Create an agent with the WeatherService dependency and WeatherReport output
    weather_agent = Agent[WeatherService, WeatherReport](
        'openai:gpt-4o',
        deps_type=WeatherService,
        output_type=WeatherReport,
        system_prompt=(
            "You are a weather assistant. Use the weather_forecast tool to get "
            "weather information for the location and date the user asks about."
        ),
    )

    # Define a tool for the agent to use
    @weather_agent.tool
    async def weather_forecast(
        ctx: RunContext[WeatherService],
        location: str,
        forecast_date: date,
    ) -> Dict[str, Any]:
        """Get weather forecast for a location on a specific date."""
        # Determine if we need a forecast or historical data
        if forecast_date >= date.today():
            return await ctx.deps.get_forecast(location, forecast_date)
        else:
            return await ctx.deps.get_historic_weather(location, forecast_date)

    # Run the agent with a future date
    future_date = date.today() + timedelta(days=3)
    print(f"Asking about weather in Paris on {future_date}")
    result = await weather_agent.run(
        f'What will the weather be like in Paris on {future_date.isoformat()}?',
        deps=WeatherService()
    )

    # The output is a WeatherReport object
    print(f"Output type: {type(result.output)}")
    print(f"Location: {result.output.location}")
    print(f"Temperature: {result.output.temperature}°C")
    print(f"Conditions: {result.output.conditions}")
    print(f"Date: {result.output.forecast_date}")
    print(f"Humidity: {result.output.humidity}%")

    # Run the agent with a past date
    past_date = date.today() - timedelta(days=7)
    print(f"\nAsking about weather in London on {past_date}")
    result = await weather_agent.run(
        f'What was the weather like in London on {past_date.isoformat()}?',
        deps=WeatherService()
    )

    # The output is a WeatherReport object
    print(f"Output type: {type(result.output)}")
    print(f"Location: {result.output.location}")
    print(f"Temperature: {result.output.temperature}°C")
    print(f"Conditions: {result.output.conditions}")
    print(f"Date: {result.output.forecast_date}")
    print(f"Humidity: {result.output.humidity}%")

# ============================================================================
# SECTION 4: SYSTEM PROMPTS AND INSTRUCTIONS
# ============================================================================

async def section_4_prompts_and_instructions():
    print_section_header("SECTION 4: SYSTEM PROMPTS AND INSTRUCTIONS")

    # 1. Static system prompts
    print("1. Static system prompts:")
    agent_static = Agent(
        'openai:gpt-4o',
        system_prompt="You are a helpful assistant that specializes in geography."
    )

    result = await agent_static.run('What is the capital of Brazil?')
    print(f"Output: {result.output}")

    # 2. Dynamic system prompts
    print("\n2. Dynamic system prompts:")
    agent_dynamic = Agent(
        'openai:gpt-4o',
        deps_type=str,  # User's name as dependency
        system_prompt="You are a helpful assistant."
    )

    @agent_dynamic.system_prompt
    def add_user_name(ctx: RunContext[str]) -> str:
        return f"The user's name is {ctx.deps}."

    @agent_dynamic.system_prompt
    def add_current_date() -> str:
        return f"Today's date is {date.today().isoformat()}."

    result = await agent_dynamic.run('Tell me about yourself.', deps='Alice')
    print(f"Output: {result.output}")

    # 3. Instructions vs System Prompts
    print("\n3. Instructions vs System Prompts:")
    print("Instructions are recommended for most use cases.")

    agent_instructions = Agent(
        'openai:gpt-4o',
        deps_type=str,
        instructions="You are a helpful assistant that provides concise answers."
    )

    @agent_instructions.instructions
    def add_user_role(ctx: RunContext[str]) -> str:
        return f"The user is a {ctx.deps}."

    result = await agent_instructions.run('What skills should I develop?', deps='software developer')
    print(f"Output: {result.output}")

# ============================================================================
# SECTION 5: CONVERSATIONS AND MESSAGE HISTORY
# ============================================================================

async def section_5_conversations():
    print_section_header("SECTION 5: CONVERSATIONS AND MESSAGE HISTORY")

    agent = Agent('openai:gpt-4o')

    # First message in conversation
    print("First message: 'Who was Albert Einstein?'")
    result1 = await agent.run('Who was Albert Einstein?')
    print(f"Output: {result1.output}")

    # Second message, with history from first message
    print("\nSecond message with history: 'What was his most famous equation?'")
    result2 = await agent.run(
        'What was his most famous equation?',
        message_history=result1.new_messages()
    )
    print(f"Output: {result2.output}")

    # Third message, without history (for comparison)
    print("\nThird message WITHOUT history: 'What was his most famous equation?'")
    result3 = await agent.run('What was his most famous equation?')
    print(f"Output: {result3.output}")

    # Notice how the second message correctly identifies "his" as Einstein
    # while the third message without history might be confused

# ============================================================================
# SECTION 6: REFLECTION AND SELF-CORRECTION
# ============================================================================

async def section_6_reflection():
    print_section_header("SECTION 6: REFLECTION AND SELF-CORRECTION")

    # Define a structured output type
    class ChatResult(BaseModel):
        user_id: int
        message: str

    # Create an agent that uses our fake database
    agent = Agent(
        'openai:gpt-4o',
        deps_type=DatabaseConn,
        output_type=ChatResult,
        system_prompt=(
            "You are a messaging assistant. Use get_user_by_name to find "
            "the user_id. Then formulate a message to send to that user."
        )
    )

    # Define a tool that can retry if it fails
    @agent.tool(retries=2)  # Allow 2 retries
    def get_user_by_name(ctx: RunContext[DatabaseConn], name: str) -> int:
        """Get a user's ID from their full name."""
        print(f"Tool called with name: {name!r}, retry count: {ctx.retry_count}")

        # Try to get the user ID
        user_id = ctx.deps.users.get(name=name)

        if user_id is None:
            # If not found, raise ModelRetry to tell the model to try again
            raise ModelRetry(
                f'No user found with name {name!r}, remember to provide their full name'
            )

        return user_id

    # Create a database connection
    db = DatabaseConn()

    # Run the agent - it might need to retry if it doesn't use the full name
    print("Running agent with 'Send a message to John asking for coffee next week'")
    result = await agent.run(
        'Send a message to John asking for coffee next week',
        deps=db
    )

    print(f"Output: {result.output}")
    print(f"User ID: {result.output.user_id}")
    print(f"Message: {result.output.message}")

# ============================================================================
# SECTION 7: ERROR HANDLING
# ============================================================================

async def section_7_error_handling():
    print_section_header("SECTION 7: ERROR HANDLING")

    # Create an agent with a tool that will fail unless given the right input
    agent = Agent('openai:gpt-4o', retries=1)

    @agent.tool
    def calculate_answer(x: int) -> int:
        """Calculate the answer to life, the universe, and everything."""
        print(f"Tool called with x={x}")
        if x == 42:
            return x
        else:
            raise ModelRetry("That's not the answer. Try again with the answer to life, the universe, and everything.")

    # Capture messages for diagnostic purposes
    with capture_run_messages() as messages:
        try:
            print("Running agent with a prompt that will likely cause an error...")
            result = await agent.run('Calculate the answer to 2 + 2')
            print(f"Output: {result.output}")  # This might not be reached
        except UnexpectedModelBehavior as e:
            print(f"Caught expected error: {e}")
            print(f"Cause: {repr(e.__cause__)}")

            # Print captured messages for diagnostics
            print("\nCaptured messages:")
            for i, msg in enumerate(messages):
                print(f"Message {i+1}: {type(msg).__name__}")
                # You could print more details about each message here

# ============================================================================
# SECTION 8: USAGE LIMITS AND MODEL SETTINGS
# ============================================================================

async def section_8_limits_and_settings():
    print_section_header("SECTION 8: USAGE LIMITS AND MODEL SETTINGS")

    # 1. Usage Limits
    print("1. Usage Limits:")
    agent = Agent('openai:gpt-4o')

    # Limit response tokens
    print("\nLimiting response tokens to 10:")
    try:
        result = await agent.run(
            'What is the capital of Italy? Answer with just the city name.',
            usage_limits=UsageLimits(response_tokens_limit=10)
        )
        print(f"Output: {result.output}")
        print(f"Usage: {result.usage()}")
    except UsageLimitExceeded as e:
        print(f"Error: {e}")

    # This should exceed the token limit
    print("\nThis should exceed the token limit:")
    try:
        result = await agent.run(
            'What is the capital of Italy? Give me a detailed paragraph about its history.',
            usage_limits=UsageLimits(response_tokens_limit=10)
        )
        print(f"Output: {result.output}")  # This might not be reached
    except UsageLimitExceeded as e:
        print(f"Error: {e}")

    # 2. Model Settings
    print("\n2. Model Settings:")

    # Using temperature to control randomness
    print("\nUsing temperature=0.0 for deterministic output:")
    result = await agent.run(
        'What is the capital of France?',
        model_settings={'temperature': 0.0}
    )
    print(f"Output: {result.output}")

    # Using max_tokens to limit response length
    print("\nUsing max_tokens=20 to limit response length:")
    result = await agent.run(
        'Tell me about Paris.',
        model_settings={'max_tokens': 20}
    )
    print(f"Output: {result.output}")
    print(f"Usage: {result.usage()}")

    # 3. Model-specific settings (Gemini example)
    print("\n3. Model-specific settings (Gemini example):")
    print("Note: This requires a Google API key")

    try:
        gemini_agent = Agent('google:gemini-1.5-flash')

        result = await gemini_agent.run(
            'Write a short poem about AI.',
            model_settings=GeminiModelSettings(
                temperature=0.7,
                safety_settings=[
                    {
                        'category': 'HARM_CATEGORY_HARASSMENT',
                        'threshold': 'BLOCK_MEDIUM_AND_ABOVE',
                    }
                ]
            )
        )
        print(f"Output: {result.output}")
    except Exception as e:
        print(f"Error with Gemini example: {e}")
        print("(This might happen if you don't have a Google API key or the model is not available)")

# ============================================================================
# SECTION 9: ITERATING OVER AN AGENT'S GRAPH
# ============================================================================

async def section_9_iterating_graph():
    print_section_header("SECTION 9: ITERATING OVER AN AGENT'S GRAPH")

    agent = Agent('openai:gpt-4o')

    print("Using 'async for' to iterate over nodes:")
    nodes = []
    async with agent.iter('What is the capital of Spain?') as agent_run:
        async for node in agent_run:
            nodes.append(type(node).__name__)
            print(f"Node executed: {type(node).__name__}")

    print(f"\nTotal nodes: {len(nodes)}")
    print(f"Node sequence: {' -> '.join(nodes)}")

    if agent_run.result:
        print(f"Final output: {agent_run.result.output}")

    print("\nUsing .next() to manually drive iteration:")
    async with agent.iter('What is the capital of Portugal?') as agent_run:
        node = agent_run.next_node
        all_nodes = [type(node).__name__]
        print(f"Initial node: {type(node).__name__}")

        while not isinstance(node, End):
            node = await agent_run.next(node)
            all_nodes.append(type(node).__name__)
            print(f"Next node: {type(node).__name__}")

        print(f"\nTotal nodes: {len(all_nodes)}")
        print(f"Node sequence: {' -> '.join(all_nodes)}")

        if agent_run.result:
            print(f"Final output: {agent_run.result.output}")

# ============================================================================
# SECTION 10: TYPE SAFETY
# ============================================================================

def section_10_type_safety():
    print_section_header("SECTION 10: TYPE SAFETY")

    print("PydanticAI is designed to work well with static type checkers like mypy and pyright.")
    print("This helps catch errors at development time rather than runtime.")

    @dataclass
    class User:
        name: str
        age: int

    # Create an agent with User dependency and string output
    agent = Agent(
        'openai:gpt-4o',
        deps_type=User,
        output_type=str,
    )

    # This system prompt function has the correct type annotation
    @agent.system_prompt
    def add_user_info(ctx: RunContext[User]) -> str:
        return f"The user's name is {ctx.deps.name} and they are {ctx.deps.age} years old."

    # Run the agent
    result = agent.run_sync(
        'Write a short greeting for me.',
        deps=User(name="Alice", age=30)
    )

    print(f"Output: {result.output}")

    print("\nExample of type errors that would be caught by mypy/pyright:")
    print("1. Wrong dependency type in system_prompt function:")
    print("   @agent.system_prompt")
    print("   def wrong_type(ctx: RunContext[str]) -> str:  # Error: should be RunContext[User]")
    print("       return f\"The user's name is {ctx.deps}\"")

    print("\n2. Wrong output usage:")
    print("   def needs_int(x: int) -> None: pass")
    print("   needs_int(result.output)  # Error: result.output is str, not int")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

async def main():
    print("PydanticAI Agents Tutorial")
    print("=========================")
    print("This script demonstrates the key features of PydanticAI agents.")
    print("Each section covers different aspects of working with agents.")
    print()
    print("NOTE: This script makes live LLM API calls which may incur costs.")
    print("Ensure you have the necessary API keys set in your environment.")
    print()

    # Uncomment sections to run them
    await section_1_basic_agent()
    await section_2_running_agents()
    await section_3_tools_and_output()
    await section_4_prompts_and_instructions()
    await section_5_conversations()
    await section_6_reflection()
    await section_7_error_handling()
    await section_8_limits_and_settings()
    await section_9_iterating_graph()
    section_10_type_safety()

    print("\n" + "=" * 80)
    print("Tutorial Complete!")
    print("=" * 80)
    print("\nThis tutorial covered:")
    print("1. Basic agent creation and usage")
    print("2. Different ways to run an agent")
    print("3. Function tools and structured output")
    print("4. System prompts and instructions")
    print("5. Conversations and message history")
    print("6. Reflection and self-correction")
    print("7. Error handling")
    print("8. Usage limits and model settings")
    print("9. Iterating over an agent's graph")
    print("10. Type safety")
    print("\nFor more information, visit the PydanticAI documentation.")

if __name__ == "__main__":
    asyncio.run(main())
