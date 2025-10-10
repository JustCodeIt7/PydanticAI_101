"""
Pydantic AI Agents Tutorial Script
===================================
This script demonstrates core concepts of Pydantic AI agents including:
- Creating agents with typed dependencies and outputs
- Defining custom tools for agents
- Using Pydantic models for structured inputs/outputs
- Synchronous and asynchronous agent execution
- Error handling and validation
"""

import asyncio
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.exceptions import UnexpectedModelBehavior


# ============================================================================
# SECTION 1: Pydantic Models for Structured Data
# ============================================================================


class WeatherRequest(BaseModel):
    """Structured input for weather queries"""

    location: str = Field(description="City name or location")
    units: Literal["celsius", "fahrenheit"] = Field(default="celsius")


class WeatherResponse(BaseModel):
    """Structured weather data output"""

    location: str
    temperature: float
    condition: str
    humidity: int = Field(ge=0, le=100)
    timestamp: datetime


class MathOperation(BaseModel):
    """Mathematical operation with validation"""

    operation: Literal["add", "subtract", "multiply", "divide", "power"]
    operand1: float
    operand2: float


class MathResult(BaseModel):
    """Structured math calculation result"""

    operation: str
    result: float
    expression: str


# ============================================================================
# SECTION 2: Weather Agent - Demonstrating Tools and Dependencies
# ============================================================================

# Mock weather database to simulate real data source
WEATHER_DATA = {
    "london": {"temp": 15.5, "condition": "Cloudy", "humidity": 75},
    "paris": {"temp": 18.0, "condition": "Sunny", "humidity": 60},
    "tokyo": {"temp": 22.0, "condition": "Rainy", "humidity": 85},
    "new york": {"temp": 12.0, "condition": "Partly Cloudy", "humidity": 70},
}


# Create weather agent with typed output
weather_agent = Agent[None, WeatherResponse](
    "openai:gpt-4o",
    output_type=WeatherResponse,
    instructions=(
        "You are a weather information assistant. Use the get_weather tool "
        "to fetch real weather data. Always return properly formatted weather information."
    ),
)


@weather_agent.tool
def get_weather(ctx: RunContext[None], location: str, units: str = "celsius") -> str:
    """
    Fetch weather data for a given location.

    This tool demonstrates:
    - Function parameters that the LLM will call
    - String-based return for flexibility
    - Mock data access (would be API call in production)
    """
    location_key = location.lower().strip()

    if location_key not in WEATHER_DATA:
        # Raise ModelRetry to ask the LLM to try with different input
        raise ModelRetry(f"Location '{location}' not found. Try: {', '.join(WEATHER_DATA.keys())}")

    data = WEATHER_DATA[location_key]
    temp = data["temp"]

    # Convert temperature if needed
    if units == "fahrenheit":
        temp = (temp * 9 / 5) + 32

    return (
        f"Location: {location}, Temperature: {temp}°{units[0].upper()}, "
        f"Condition: {data['condition']}, Humidity: {data['humidity']}%"
    )


# ============================================================================
# SECTION 3: Math Agent - Structured Inputs and Outputs
# ============================================================================

math_agent = Agent[None, MathResult](
    "openai:gpt-4o",
    output_type=MathResult,
    instructions=(
        "You are a mathematical calculation assistant. Use the calculate tool "
        "to perform operations. Return results in the proper MathResult format."
    ),
)


@math_agent.tool
def calculate(ctx: RunContext[None], operation: str, operand1: float, operand2: float) -> str:
    """
    Perform mathematical calculations.

    Demonstrates:
    - Multiple typed parameters
    - Validation and error handling
    - String formatting of results
    """
    try:
        # Validate inputs using Pydantic model
        op = MathOperation(operation=operation, operand1=operand1, operand2=operand2)

        # Perform calculation
        if op.operation == "add":
            result = op.operand1 + op.operand2
            symbol = "+"
        elif op.operation == "subtract":
            result = op.operand1 - op.operand2
            symbol = "-"
        elif op.operation == "multiply":
            result = op.operand1 * op.operand2
            symbol = "*"
        elif op.operation == "divide":
            if op.operand2 == 0:
                raise ModelRetry("Cannot divide by zero. Please use a non-zero divisor.")
            result = op.operand1 / op.operand2
            symbol = "/"
        elif op.operation == "power":
            result = op.operand1**op.operand2
            symbol = "^"

        expression = f"{op.operand1} {symbol} {op.operand2}"
        return f"Operation: {op.operation}, Result: {result}, Expression: {expression}"

    except ValidationError as e:
        raise ModelRetry(f"Invalid operation: {str(e)}")


# ============================================================================
# SECTION 4: Multi-Step Async Agent - Complex Workflows
# ============================================================================

# Data analysis agent that performs multiple steps
analysis_agent = Agent[dict, str](
    "openai:gpt-4o",
    deps_type=dict,
    output_type=str,
    instructions=(
        "You are a data analysis assistant. Use available tools to analyze data, "
        "calculate statistics, and provide insights. Work step-by-step."
    ),
)


@analysis_agent.tool
async def calculate_stats(ctx: RunContext[dict], data_key: str) -> str:
    """
    Calculate statistics from provided data.

    Demonstrates:
    - Async tool functions
    - Accessing context dependencies
    - Multi-value returns
    """
    # Simulate async data fetch
    await asyncio.sleep(0.1)

    data = ctx.deps.get(data_key, [])
    if not data:
        raise ModelRetry(
            f"No data found for key '{data_key}'. Available keys: {list(ctx.deps.keys())}"
        )

    count = len(data)
    total = sum(data)
    average = total / count if count > 0 else 0
    minimum = min(data)
    maximum = max(data)

    return (
        f"Statistics for {data_key}: Count={count}, Sum={total}, "
        f"Average={average:.2f}, Min={minimum}, Max={maximum}"
    )


@analysis_agent.tool
async def find_outliers(ctx: RunContext[dict], data_key: str, threshold: float = 2.0) -> str:
    """
    Identify outliers in numerical data.

    Demonstrates:
    - Optional parameters with defaults
    - More complex async operations
    """
    await asyncio.sleep(0.1)

    data = ctx.deps.get(data_key, [])
    if not data:
        raise ModelRetry(f"No data found for key '{data_key}'")

    # Simple outlier detection using mean ± threshold * range
    mean = sum(data) / len(data)
    data_range = max(data) - min(data)
    threshold_value = data_range * (threshold / 10)

    outliers = [x for x in data if abs(x - mean) > threshold_value]

    return f"Found {len(outliers)} outliers in {data_key}: {outliers[:5]}"


# ============================================================================
# SECTION 5: Execution Examples
# ============================================================================


def demo_weather_agent_sync():
    """Demonstrate synchronous weather agent execution"""
    print("\n" + "=" * 70)
    print("DEMO 1: Weather Agent (Synchronous)")
    print("=" * 70)

    try:
        # Example 1: Valid location
        result = weather_agent.run_sync("What's the weather in Paris?")
        print(f"\n✓ Query: What's the weather in Paris?")
        print(f"  Response: {result.output.model_dump_json(indent=2)}")

        # Example 2: Different location
        result = weather_agent.run_sync("Tell me about the weather in Tokyo")
        print(f"\n✓ Query: Tell me about the weather in Tokyo")
        print(f"  Location: {result.output.location}")
        print(f"  Temperature: {result.output.temperature}°C")
        print(f"  Condition: {result.output.condition}")

    except Exception as e:
        print(f"✗ Error: {e}")


def demo_math_agent_sync():
    """Demonstrate synchronous math agent execution"""
    print("\n" + "=" * 70)
    print("DEMO 2: Math Agent (Synchronous)")
    print("=" * 70)

    try:
        # Example 1: Addition
        result = math_agent.run_sync("Calculate 45 plus 67")
        print(f"\n✓ Query: Calculate 45 plus 67")
        print(f"  Result: {result.output.result}")
        print(f"  Expression: {result.output.expression}")

        # Example 2: Division
        result = math_agent.run_sync("What is 100 divided by 4?")
        print(f"\n✓ Query: What is 100 divided by 4?")
        print(f"  Result: {result.output.result}")

        # Example 3: Power operation
        result = math_agent.run_sync("Calculate 2 to the power of 10")
        print(f"\n✓ Query: Calculate 2 to the power of 10")
        print(f"  Result: {result.output.result}")

    except Exception as e:
        print(f"✗ Error: {e}")


async def demo_analysis_agent_async():
    """Demonstrate asynchronous multi-step agent execution"""
    print("\n" + "=" * 70)
    print("DEMO 3: Analysis Agent (Asynchronous)")
    print("=" * 70)

    # Prepare sample data
    sample_data = {
        "sales": [100, 150, 120, 180, 90, 200, 110],
        "temperatures": [22, 24, 23, 45, 21, 23, 22, 24],  # Note: 45 is an outlier
    }

    try:
        # Example: Multi-step analysis
        result = await analysis_agent.run(
            "Analyze the sales data and find any outliers in temperatures", deps=sample_data
        )
        print(f"\n✓ Query: Analyze the sales data and find any outliers in temperatures")
        print(f"  Analysis Result:\n{result.output}")

        # Show usage statistics
        usage = result.usage()
        print(f"\n📊 Token Usage:")
        print(f"  Requests: {usage.requests}")
        print(f"  Total Tokens: {usage.total_tokens}")

    except UnexpectedModelBehavior as e:
        print(f"✗ Model Error: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")


# ============================================================================
# SECTION 6: Main Execution
# ============================================================================


async def main():
    """
    Main function orchestrating all demos.

    Demonstrates:
    - Combining sync and async execution
    - Proper error handling
    - Clear output formatting for tutorials
    """
    print("\n" + "=" * 70)
    print("PYDANTIC AI AGENTS TUTORIAL")
    print("=" * 70)

    # Run synchronous demos
    demo_weather_agent_sync()
    demo_math_agent_sync()

    # Run async demo
    await demo_analysis_agent_async()

    print("\n" + "=" * 70)
    print("Tutorial Complete! Key Takeaways:")
    print("=" * 70)
    print("1. Agents use typed dependencies and outputs for type safety")
    print("2. Tools allow agents to access external functions and data")
    print("3. Pydantic models provide structured, validated inputs/outputs")
    print("4. Both sync and async execution modes are supported")
    print("5. ModelRetry enables self-correction and retry logic")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Note: Requires OPENAI_API_KEY environment variable to be set
    # Run with: python 02_agents.py
    asyncio.run(main())
