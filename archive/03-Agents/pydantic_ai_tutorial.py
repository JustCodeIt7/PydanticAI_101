import asyncio
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Optional

from pydantic import BaseModel

# Import PydanticAI components
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

#####################################################
# PART 1: Introduction to PydanticAI Agents
#####################################################

# Define a basic agent
basic_agent = Agent('gpt-4.1-nano')  # Use your preferred model


# Convert to async
async def run_basic_agent():
    print("\n=== PART 1: Basic Agent Demo ===")
    result = await basic_agent.run("What is PydanticAI?")
    print(f"Response: {result.output}")
    print(f"Token usage: {result.usage()}")


#####################################################
# PART 2: System Prompts and Instructions
#####################################################

# Agent with static system prompt
agent_with_system_prompt = Agent(
    'gpt-4.1-nano',
    system_prompt="You are a helpful assistant with a focus on Python programming."
)

# Agent with instructions (preferred over system_prompt in most cases)
agent_with_instructions = Agent(
    'gpt-4.1-nano',
    instructions="You are a Python expert who provides concise code examples."
)


# Convert to async
async def run_prompts_example():
    print("\n=== PART 2: System Prompts vs Instructions ===")
    result1 = await agent_with_system_prompt.run("Write a simple function to calculate the factorial of a number")
    print(f"With system prompt: {result1.output[:100]}...")

    result2 = await agent_with_instructions.run("Write a simple function to calculate the factorial of a number")
    print(f"With instructions: {result2.output[:100]}...")


#####################################################
# PART 3: Dynamic System Prompts and Context
#####################################################

# Agent with context dependency and dynamic system prompt
agent_with_dynamic = Agent(
    'gpt-4.1-nano',
    deps_type=str,  # The agent expects a string dependency
)

@agent_with_dynamic.system_prompt
def add_user_context(ctx: RunContext[str]) -> str:
    """Dynamic system prompt that uses the dependency context"""
    return f"The user's name is {ctx.deps}. Always address them by name."

@agent_with_dynamic.system_prompt
def add_date_context() -> str:
    """Dynamic system prompt that adds date information"""
    return f"Today's date is {date.today()}."


# Convert to async
async def run_dynamic_prompts():
    print("\n=== PART 3: Dynamic System Prompts ===")
    result = await agent_with_dynamic.run(
        "What programming language should I learn first?",
        deps="Alice"
    )
    print(f"Response: {result.output}")


#####################################################
# PART 4: Working with Tools
#####################################################

# Agent with tools
tools_agent = Agent(
    'gpt-4.1-nano',
    system_prompt="You are an assistant that can perform various calculations and lookups."
)

@tools_agent.tool_plain
def calculate_area(length: float, width: float) -> float:
    """Calculate the area of a rectangle with the given length and width."""
    return length * width

@tools_agent.tool_plain
def lookup_country_capital(country: str) -> str:
    """Look up the capital city of a country."""
    capitals = {
        "usa": "Washington D.C.",
        "france": "Paris",
        "japan": "Tokyo",
        "australia": "Canberra",
        "brazil": "Brasília",
    }
    country_lower = country.lower()
    if country_lower in capitals:
        return capitals[country_lower]
    return f"Capital for {country} not found in database."


# Convert to async
async def run_tools_example():
    print("\n=== PART 4: Using Tools ===")
    result = await tools_agent.run("What is the area of a rectangle that is 7.5 meters by 12 meters?")
    print(f"Area calculation: {result.output}")

    result = await tools_agent.run("What is the capital of France?")
    print(f"Capital lookup: {result.output}")


#####################################################
# PART 5: Structured Output and Validation
#####################################################

# Define output models
class WeatherForecast(BaseModel):
    location: str
    temperature: float
    conditions: str
    forecast_date: date
    precipitation_chance: Optional[float] = None


# Agent with structured output
structured_agent = Agent(
    'gpt-4.1-nano',
    output_type=WeatherForecast,
    system_prompt="You are a weather forecasting assistant. Provide weather forecasts as structured data."
)


@structured_agent.tool
def get_weather_data(ctx: RunContext, location: str, forecast_date: date) -> Dict[str, Any]:
    """Get weather data for a specific location and date"""
    # In a real application, this would call a weather API
    # Here we're using mock data
    mock_data = {
        "new york": {"temp": 22.5, "conditions": "Partly Cloudy", "precipitation": 0.2},
        "london": {"temp": 18.0, "conditions": "Rainy", "precipitation": 0.7},
        "tokyo": {"temp": 28.0, "conditions": "Sunny", "precipitation": 0.0},
    }

    location_lower = location.lower()
    if location_lower in mock_data:
        data = mock_data[location_lower]
        return {
            "location": location,
            "temperature": data["temp"],
            "conditions": data["conditions"],
            "forecast_date": forecast_date,
            "precipitation_chance": data["precipitation"]
        }

    # Return default data if location not found
    return {
        "location": location,
        "temperature": 20.0,
        "conditions": "Unknown",
        "forecast_date": forecast_date,
        "precipitation_chance": 0.0
    }


# Convert to async
async def run_structured_output():
    print("\n=== PART 5: Structured Output ===")
    result = await structured_agent.run("What's the weather like in Tokyo today?")
    print(f"Weather forecast: {result.output}")
    print(f"Temperature: {result.output.temperature}°C")
    print(f"Conditions: {result.output.conditions}")

    # Type safety in action - this is validated!
    forecast: WeatherForecast = result.output


#####################################################
# PART 6: Error Handling and Retries
#####################################################

retry_agent = Agent(
    'gpt-4.1-nano',
    retries=2,  # Set the number of retries for the agent
    system_prompt="You are an assistant that helps validate user input."
)


@retry_agent.tool(retries=3)  # Override default retries for this specific tool
def validate_email(ctx: RunContext, email: str) -> str:
    """Validate if an email address is correctly formatted."""
    if "@" not in email or "." not in email:
        raise ModelRetry("That doesn't look like a valid email. Please provide a proper email address.")
    return f"The email {email} appears to be valid."


# Convert to async
async def run_retry_example():
    print("\n=== PART 6: Error Handling and Retries ===")
    try:
        result = await retry_agent.run("My email is johndoe at example dot com")
        print(f"Response: {result.output}")
    except UnexpectedModelBehavior as e:
        print(f"Error: {e}")
        print("The agent exceeded its retry limit")


#####################################################
# PART 7: Message History and Conversations
#####################################################

conversation_agent = Agent(
    'gpt-4.1-nano',
    system_prompt="You are a helpful assistant that remembers previous messages."
)


# Convert to async
async def run_conversation_example():
    print("\n=== PART 7: Message History and Conversations ===")

    # First message
    result1 = await conversation_agent.run("My name is Bob and I live in Seattle.")
    print(f"Response 1: {result1.output}")

    # Second message with history
    result2 = await conversation_agent.run(
        "What's my name and where do I live?",
        message_history=result1.new_messages()
    )
    print(f"Response 2: {result2.output}")

    # Third message with history
    result3 = await conversation_agent.run(
        "What's the weather like there this time of year?",
        message_history=result2.new_messages()
    )
    print(f"Response 3: {result3.output}")


#####################################################
# PART 8: Streaming Responses
#####################################################

async def run_streaming_example():
    print("\n=== PART 8: Streaming Responses ===")
    agent = Agent('gpt-4.1-nano')

    print("Streaming response:")
    async with agent.run_stream("What are three benefits of using type hints in Python?") as response:
        async for chunk in response:
            # In a real application, you would print these incrementally
            # Here we'll just preview the first few chunks
            if isinstance(chunk, str) and len(chunk) > 0:
                print(f"Chunk: {chunk[:20]}...")
                break

        # Get the complete output
        full_output = await response.get_output()
        print(f"\nFull output: {full_output[:100]}...")


#####################################################
# PART 9: Advanced Agent with Dependencies
#####################################################

@dataclass
class Database:
    """Simulated database connection"""
    users: Dict[int, str] = None

    def __post_init__(self):
        self.users = {
            1: "Alice Smith",
            2: "Bob Johnson",
            3: "Carol Williams"
        }

    def get_user(self, user_id: int) -> Optional[str]:
        return self.users.get(user_id)

    def add_user(self, user_id: int, name: str) -> bool:
        if user_id in self.users:
            return False
        self.users[user_id] = name
        return True


class UserInfo(BaseModel):
    user_id: int
    name: str
    message: str


advanced_agent = Agent(
    'gpt-4.1-nano',
    deps_type=Database,
    output_type=UserInfo,
    system_prompt="You are a user management assistant that can look up user information."
)


@advanced_agent.tool
def lookup_user(ctx: RunContext[Database], user_id: int) -> Dict[str, Any]:
    """Look up a user by their ID in the database."""
    name = ctx.deps.get_user(user_id)
    if name:
        return {"user_id": user_id, "name": name, "found": True}
    return {"user_id": user_id, "found": False}


@advanced_agent.tool
def create_user(ctx: RunContext[Database], user_id: int, name: str) -> Dict[str, Any]:
    """Create a new user in the database."""
    success = ctx.deps.add_user(user_id, name)
    return {"user_id": user_id, "name": name, "created": success}


# Convert to async
async def run_advanced_agent():
    print("\n=== PART 9: Advanced Agent with Dependencies ===")
    db = Database()

    # Look up existing user
    result = await advanced_agent.run(
        "Get information about user ID 2 and craft a welcome message for them",
        deps=db
    )
    print(f"User ID: {result.output.user_id}")
    print(f"Name: {result.output.name}")
    print(f"Message: {result.output.message}")


#####################################################
# PART 10: Usage Limits and Settings
#####################################################

limit_agent = Agent('gpt-4.1-nano')


# Convert to async and handle UsageLimitExceeded exception
async def run_limits_example():
    print("\n=== PART 10: Usage Limits and Settings ===")

    # Set usage limits with exception handling
    try:
        result = await limit_agent.run(
            "Write a concise paragraph about machine learning",
            usage_limits=UsageLimits(request_limit=3, response_tokens_limit=100)
        )
        print(f"Response: {result.output}")
    except UsageLimitExceeded as e:
        print(f"Expected exception: {e}")
        print("This demonstrates how usage limits can prevent token overuse!")

    # Use a shorter prompt to demonstrate successful usage with limits
    try:
        result = await limit_agent.run(
            "Define AI briefly",
            usage_limits=UsageLimits(response_tokens_limit=100)
        )
        print(f"Short response within limits: {result.output}")
    except UsageLimitExceeded as e:
        print(f"Unexpected exception: {e}")

    # Set model settings
    result = await limit_agent.run(
        "Generate a creative poem about AI",
        model_settings=ModelSettings(temperature=0.9)
    )
    print(f"Creative response: {result.output[:100]}...")

    # Lower temperature for more deterministic responses
    result = await limit_agent.run(
        "Generate a creative poem about AI",
        model_settings=ModelSettings(temperature=0.1)
    )
    print(f"Deterministic response: {result.output[:100]}...")


#####################################################
# PART 11: Advanced Graph Iteration
#####################################################

async def run_graph_iteration():
    print("\n=== PART 11: Advanced Graph Iteration ===")
    agent = Agent('gpt-4.1-nano')

    print("Iterating through agent graph nodes:")
    nodes = []
    async with agent.iter("What is the capital of Italy?") as agent_run:
        async for node in agent_run:
            node_type = type(node).__name__
            nodes.append(node_type)

    print(f"Node types encountered: {nodes}")
    print(f"Final result: {agent_run.result.output}")


#####################################################
# Main function to run all examples
#####################################################

async def main():
    print("==================================================")
    print("     PYDANTIC AI AGENTS TUTORIAL")
    print("==================================================")

    # Run all examples as async functions
    await run_basic_agent()
    await run_prompts_example()
    await run_dynamic_prompts()
    await run_tools_example()
    await run_structured_output()
    await run_retry_example()
    await run_conversation_example()
    await run_advanced_agent()
    await run_limits_example()
    await run_streaming_example()
    await run_graph_iteration()


if __name__ == "__main__":
    asyncio.run(main())