# video_07_agent_tools.py

# --- Video 7: Giving Your Agent Tools (@agent.tool) ---

# Objective: Teach how to define and register functions (tools)
# that the agent's LLM can call.

import os
import random
from dotenv import load_dotenv
from typing import Literal # For specific string return types

# --- Prerequisites ---
# 1. PydanticAI installed with relevant LLM extra (e.g., pip install pydantic-ai[openai])
#    (Tool calling works best with models designed for it, like OpenAI's GPT series)
# 2. API key configured (e.g., OPENAI_API_KEY set as environment variable or in .env)

print("--- PydanticAI Tutorial Series ---")
print("Video 7: Giving Your Agent Tools (@agent.tool)")

# Load environment variables (e.g., API keys) from a .env file if it exists
load_dotenv()
print("\nAttempted to load environment variables (like API keys) from .env file.")

# Check if the required API key is present (using OpenAI as example)
api_key_present = bool(os.getenv("OPENAI_API_KEY"))
print(f"OpenAI API Key detected in environment: {api_key_present}")

if not api_key_present:
    print("\nWarning: OPENAI_API_KEY not found in environment.")
    print("The script might fail as tool calling often relies on capable models like GPT.")
    print("Please ensure your API key is set for a provider that supports function/tool calling.")

# === 1. Why Tools? ===
print("\n=== 1. Why Agents Need Tools ===")
print(" - LLMs have general knowledge but lack access to:")
print("   - Real-time information (weather, stock prices)")
print("   - Private data sources (databases, internal APIs)")
print("   - The ability to perform external actions (sending emails, booking tickets)")
print(" - Tools bridge this gap by allowing LLMs to call Python functions.")

# === 2. Import Agent and Configure ===
print("\n=== 2. Importing and Configuring the Agent ===")
try:
    from pydantic_ai import Agent, AgentRunResult
    print(" - Successfully imported 'Agent' and 'AgentRunResult'.")

    # Use a model known for good tool/function calling (e.g., OpenAI GPT series)
    agent = Agent(
        'openai:gpt-4o', # Or another capable model like gpt-4-turbo
        system_prompt="You are a helpful assistant that can use tools.",
        # Note: No output_type is specified here, as the LLM decides
        # whether to call a tool or respond directly.
    )
    print(f" - Agent configured with model: '{agent.llm.provider}:{agent.llm.model}'")

except ImportError:
    print("\nError: Could not import from 'pydantic_ai'.")
    print("Please ensure PydanticAI is installed correctly (`pip install pydantic-ai[openai]`).")
    exit()
except Exception as e:
    print(f"\nError configuring Agent: {e}")
    print(" - Check API key and installed extras.")
    exit()

# === 3. Defining Tools with @agent.tool ===
print("\n=== 3. Defining Tools with @agent.tool ===")
print(" - Use the '@agent.tool' decorator on standard Python functions.")
print(" - Type hints and docstrings are crucial for the LLM.")

# --- Tool 1: Simple Tool (No Arguments) ---
@agent.tool
def flip_coin() -> Literal['Heads', 'Tails']:
    """
    Flips a virtual coin and returns the result.
    Use this when the user asks to flip a coin or make a random choice between two options.
    """
    print("\n--- Tool Execution: flip_coin ---") # Debug print
    result = random.choice(['Heads', 'Tails'])
    print(f"   - Result: {result}")
    print("--- End Tool Execution ---\n")
    return result

print(" - Defined tool: 'flip_coin'")
print("   - Takes no arguments.")
print("   - Returns 'Heads' or 'Tails'.")
print("   - Docstring explains its purpose to the LLM.")

# --- Tool 2: Tool with Arguments ---
@agent.tool
def get_stock_price(symbol: str) -> float | str:
    """
    Fetches the current stock price for a given stock ticker symbol.

    Use this tool when the user asks for the price of a specific stock.

    Args:
        symbol (str): The stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT').
                      Should be uppercase.
    """
    print("\n--- Tool Execution: get_stock_price ---") # Debug print
    print(f"   - Received symbol: {symbol}")

    # --- Placeholder Implementation ---
    # In a real application, you would call a stock market API here.
    # Example: Use libraries like 'yfinance' or 'requests' to call an API.
    # For demonstration, we'll use placeholder values.
    symbol = symbol.upper() # Normalize symbol
    known_prices = {
        "AAPL": 175.50 + random.uniform(-2, 2),
        "GOOGL": 140.20 + random.uniform(-1.5, 1.5),
        "MSFT": 300.80 + random.uniform(-3, 3),
    }
    price = known_prices.get(symbol)
    # --- End Placeholder ---

    if price is not None:
        result = round(price, 2)
        print(f"   - Returning price: {result}")
        print("--- End Tool Execution ---\n")
        return result
    else:
        error_message = f"Could not find price for symbol '{symbol}'."
        print(f"   - Returning error: {error_message}")
        print("--- End Tool Execution ---\n")
        # Returning a descriptive string is often better than raising an error here,
        # as the LLM can incorporate this information into its response.
        return error_message

print("\n - Defined tool: 'get_stock_price'")
print("   - Takes one argument: 'symbol' (string, type-hinted).")
print("   - Docstring describes the tool and the 'symbol' argument.")
print("   - Returns the price (float) or an error message (str).")

# === 4. How Tools Work ===
print("\n=== 4. How Tools Work ===")
print(" 1. User sends query (e.g., 'Flip a coin', 'What's the price of MSFT?').")
print(" 2. LLM analyzes query and available tools (based on names, docstrings, args).")
print(" 3. If LLM decides a tool is needed, it outputs a request (tool name + args).")
print(" 4. PydanticAI intercepts the request.")
print(" 5. PydanticAI validates LLM-provided arguments against the tool's type hints.")
print(" 6. If valid, PydanticAI executes the Python function (`flip_coin` or `get_stock_price`).")
print(" 7. The function's return value is sent back to the LLM.")
print(" 8. LLM uses the tool's result to formulate the final response to the user.")

# === 5. Running the Agent with Tools ===
print("\n=== 5. Running the Agent with Tools ===")

# --- Example 1: Using flip_coin ---
query1 = "Should I have pizza or pasta for dinner? Flip a coin to decide."
print(f"\n   Sending query 1: \"{query1}\"")
try:
    result1: AgentRunResult = agent.run_sync(query1)
    print("\n   --- Agent Response 1 ---")
    print(result1.output)
    print("   --- End Agent Response 1 ---")
    # Check if the tool was actually called (more advanced introspection)
    # print(f"   Tool calls in run 1: {result1.tool_calls}")
except Exception as e:
    print(f"\n   Error during agent run 1: {e}")

# --- Example 2: Using get_stock_price ---
query2 = "What is the current stock price for Microsoft (MSFT)?"
print(f"\n   Sending query 2: \"{query2}\"")
try:
    result2: AgentRunResult = agent.run_sync(query2)
    print("\n   --- Agent Response 2 ---")
    print(result2.output)
    print("   --- End Agent Response 2 ---")
except Exception as e:
    print(f"\n   Error during agent run 2: {e}")

# --- Example 3: Tool Argument Validation (Implicit) ---
# If the LLM tried to call get_stock_price(symbol=123), Pydantic
# would raise a validation error *before* the tool runs, and PydanticAI
# would inform the LLM of the error, potentially allowing it to retry.

# === 6. Note on @agent.tool_plain ===
print("\n=== 6. Note on @agent.tool_plain ===")
print(" - An alternative decorator `@agent.tool_plain` exists.")
print(" - It's for very simple tools that don't need access to agent context (more on context later).")
print(" - `@agent.tool` is generally preferred as it's more flexible.")

print("\n--- End of Video 7 Agent Tools Example ---")
