import asyncio
from datetime import datetime, date
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from pydantic import BaseModel, Field

# Import PydanticAI components
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.usage import UsageLimits, Usage
from pydantic_ai.settings import ModelSettings
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import TextPartDelta, PartStartEvent, PartDeltaEvent, FunctionToolCallEvent


#####################################################
# PART 1: Introduction to PydanticAI Agents
#####################################################

# Define a basic agent
basic_agent = Agent('openai:gpt-4o')  # Use your preferred model


# Simple agent run example
def run_basic_agent():
    print("\n=== PART 1: Basic Agent Demo ===")
    result = basic_agent.run_sync("What is PydanticAI?")
    print(f"Response: {result.output}")
    print(f"Token usage: {result.usage()}")


#####################################################
# PART 2: System Prompts and Instructions
#####################################################

# Agent with static system prompt
agent_with_system_prompt = Agent(
    'openai:gpt-4o',
    system_prompt="You are a helpful assistant with a focus on Python programming."
)

# Agent with instructions (preferred over system_prompt in most cases)
agent_with_instructions = Agent(
    'openai:gpt-4o',
    instructions="You are a Python expert who provides concise code examples."
)


def run_prompts_example():
    print("\n=== PART 2: System Prompts vs Instructions ===")
    result1 = agent_with_system_prompt.run_sync("Write a simple function to calculate the factorial of a number")
    print(f"With system prompt: {result1.output[:100]}...")
    
    result2 = agent_with_instructions.run_sync("Write a simple function to calculate the factorial of a number")
    print(f"With instructions: {result2.output[:100]}...")


#####################################################
# PART 3: Dynamic System Prompts and Context
#####################################################

# Agent with context dependency and dynamic system prompt
agent_with_dynamic = Agent(
    'openai:gpt-4o',
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


def run_dynamic_prompts():
    print("\n=== PART 3: Dynamic System Prompts ===")
    result = agent_with_dynamic.run_sync(
        "What programming language should I learn first?", 
        deps="Alice"
    )
    print(f"Response: {result.output}")


#####################################################
# PART 4: Working with Tools
#####################################################

# Agent with tools
tools_agent = Agent(
    'openai:gpt-4o',
    system_prompt="You are an assistant that can perform various calculations and lookups."
)

@tools_agent.tool
def calculate_area(length: float, width: float) -> float:
    """Calculate the area of a rectangle with the given length and width."""
    return length * width

@tools_agent.tool
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


def run_tools_example():
    print("\n=== PART 4: Using Tools ===")
    result = tools_agent.run_sync("What is the area of a rectangle that is 7.5 meters by 12 meters?")
    print(f"Area calculation: {result.output}")
    
    result = tools_agent.run_sync("What is the capital of France?")
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
    'openai:gpt-4o',
    output_type=WeatherForecast,
    system_prompt="You are a weather forecasting assistant. Provide weather forecasts as structured data."
)


@structured_agent.tool
def get_weather_data(location: str, forecast_date: date) -> Dict[str, Any]:
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


def run_structured_output():
    print("\n=== PART 5: Structured Output ===")
    result = structured_agent.run_sync("What's the weather like in Tokyo today?")
    print(f"Weather forecast: {result.output}")
    print(f"Temperature: {result.output.temperature}°C")
    print(f"Conditions: {result.output.conditions}")
    
    # Type safety in action - this is validated!
    forecast: WeatherForecast = result.output


#####################################################
# PART 6: Error Handling and Retries
#####################################################

retry_agent = Agent(
    'openai:gpt-4o',
    retries=2,  # Set the number of retries for the agent
    system_prompt="You are an assistant that helps validate user input."
)


@retry_agent.tool(retries=3)  # Override default retries for this specific tool
def validate_email(email: str) -> str:
    """Validate if an email address is correctly formatted."""
    if "@" not in email or "." not in email:
        raise ModelRetry("That doesn't look like a valid email. Please provide a proper email address.")
    return f"The email {email} appears to be valid."


def run_retry_example():
    print("\n=== PART 6: Error Handling and Retries ===")
    try:
        result = retry_agent.run_sync("My email is johndoe at example dot com")
        print(f"Response: {result.output}")
    except UnexpectedModelBehavior as e:
        print(f"Error: {e}")
        print("The agent exceeded its retry limit")


#####################################################
# PART 7: Message History and Conversations
#####################################################

conversation_agent = Agent(
    'openai:gpt-4o',
    system_prompt="You are a helpful assistant that remembers previous messages."
)


def run_conversation_example():
    print("\n=== PART 7: Message History and Conversations ===")
    
    # First message
    result1 = conversation_agent.run_sync("My name is Bob and I live in Seattle.")
    print(f"Response 1: {result1.output}")
    
    # Second message with history
    result2 = conversation_agent.run_sync(
        "What's my name and where do I live?",
        message_history=result1.new_messages()
    )
    print(f"Response 2: {result2.output}")
    
    # Third message with history
    result3 = conversation_agent.run_sync(
        "What's the weather like there this time of year?",
        message_history=result2.new_messages()
    )
    print(f"Response 3: {result3.output}")


#####################################################
# PART 8: Streaming Responses
#####################################################

async def run_streaming_example():
    print("\n=== PART 8: Streaming Responses ===")
    agent = Agent('openai:gpt-4o')
    
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
    'openai:gpt-4o',
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


def run_advanced_agent():
    print("\n=== PART 9: Advanced Agent with Dependencies ===")
    db = Database()
    
    # Look up existing user
    result = advanced_agent.run_sync(
        "Get information about user ID 2 and craft a welcome message for them",
        deps=db
    )
    print(f"User ID: {result.output.user_id}")
    print(f"Name: {result.output.name}")
    print(f"Message: {result.output.message}")


#####################################################
# PART 10: Usage Limits and Settings
#####################################################

limit_agent = Agent('openai:gpt-4o')


def run_limits_example():
    print("\n=== PART 10: Usage Limits and Settings ===")
    
    # Set usage limits
    result = limit_agent.run_sync(
        "Write a concise paragraph about machine learning",
        usage_limits=UsageLimits(request_limit=3, response_tokens_limit=100)
    )
    print(f"Response: {result.output}")
    
    # Set model settings
    result = limit_agent.run_sync(
        "Generate a creative poem about AI",
        model_settings=ModelSettings(temperature=0.9)
    )
    print(f"Creative response: {result.output[:100]}...")
    
    # Lower temperature for more deterministic responses
    result = limit_agent.run_sync(
        "Generate a creative poem about AI",
        model_settings=ModelSettings(temperature=0.1)
    )
    print(f"Deterministic response: {result.output[:100]}...")


#####################################################
# PART 11: Advanced Graph Iteration
#####################################################

async def run_graph_iteration():
    print("\n=== PART 11: Advanced Graph Iteration ===")
    agent = Agent('openai:gpt-4o')
    
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
    
    # Run sync examples
    run_basic_agent()
    run_prompts_example()
    run_dynamic_prompts()
    run_tools_example()
    run_structured_output()
    run_retry_example()
    run_conversation_example()
    run_advanced_agent()
    run_limits_example()
    
    # Run async examples
    await run_streaming_example()
    await run_graph_iteration()


if __name__ == "__main__":
    asyncio.run(main())