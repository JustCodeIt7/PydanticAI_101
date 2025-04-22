# video_04_hello_world.py

# --- Video 4: Your First PydanticAI Agent: Hello World! ---

# Objective: Build and run the simplest possible PydanticAI agent.

import os
from dotenv import load_dotenv

# --- Prerequisites ---
# 1. PydanticAI installed (pip install pydantic-ai)
# 2. Relevant LLM extra installed (e.g., pip install pydantic-ai[openai])
# 3. API key configured (e.g., OPENAI_API_KEY set as environment variable or in .env)

print("--- PydanticAI Tutorial Series ---")
print("Video 4: Your First Agent - Hello World!")

# Load environment variables (e.g., API keys) from a .env file if it exists
load_dotenv()
print("\nAttempted to load environment variables (like API keys) from .env file.")

# Check if the required API key is present (using OpenAI as example)
api_key_present = bool(os.getenv("OPENAI_API_KEY"))
print(f"OpenAI API Key detected in environment: {api_key_present}")

if not api_key_present:
    print("\nWarning: OPENAI_API_KEY not found in environment.")
    print("The script might fail if it tries to contact the OpenAI API.")
    print("Please ensure your API key is set as an environment variable or in a .env file.")

# === 1. Import the Agent Class ===
print("\n=== 1. Importing Agent ===")
try:
    from pydantic_ai import Agent, AgentRunResult # AgentRunResult holds the response details
    print(" - Successfully imported 'Agent' and 'AgentRunResult' from 'pydantic_ai'.")
except ImportError:
    print("\nError: Could not import from 'pydantic_ai'.")
    print("Please ensure PydanticAI is installed correctly (`pip install pydantic-ai`).")
    exit() # Exit if core import fails

# === 2. Configure the Agent ===
print("\n=== 2. Configuring the Agent ===")
# Specify the LLM model using the '<provider>:<model_name>' format.
# Provide an optional system prompt to guide the LLM's behavior.
try:
    # Example using OpenAI's GPT-4o model.
    # Replace 'openai:gpt-4o' with another model if desired (e.g., 'google-gla:gemini-1.5-flash')
    # Ensure the corresponding extra is installed and API key is set.
    agent = Agent(
        'openai:gpt-4o', # Model identifier
        system_prompt='Be concise, reply with one sentence.' # High-level instruction
    )
    print(f" - Agent configured with model: '{agent.llm.provider}:{agent.llm.model}'")
    print(f" - System Prompt: '{agent.system_prompt}'")
except Exception as e:
    print(f"\nError configuring Agent: {e}")
    print(" - Check if the required extra (e.g., pydantic-ai[openai]) is installed.")
    print(" - Verify the API key is correctly set in the environment.")
    exit() # Exit if agent configuration fails

# === 3. Run the Agent Synchronously ===
print("\n=== 3. Running the Agent ===")
# Use the run_sync method for simple, blocking execution.
# Pass the user's query as a string.
user_query = 'Where does "hello world" come from?'
print(f" - Sending query: \"{user_query}\"")

try:
    # This makes the actual call to the LLM provider's API
    result: AgentRunResult = agent.run_sync(user_query)
    print(" - Agent execution complete.")

    # === 4. Access the Output ===
    print("\n=== 4. Accessing the Output ===")
    # The run_sync method returns an AgentRunResult object.
    # The LLM's response text is in the 'output' attribute.
    print(f" - Raw result object type: {type(result)}")
    print(f" - LLM Response (result.output):")
    print(f"   >>> {result.output}") # Accessing the main text output

    # AgentRunResult also contains other useful info (optional to explore)
    # print(f" - Run ID: {result.run_id}")
    # print(f" - Cost: {result.cost}")
    # print(f" - Tokens Used: {result.tokens}")

except Exception as e:
    print(f"\nError during agent execution: {e}")
    print(" - This often happens if the API key is invalid, missing, or quota is exceeded.")
    print(" - Check your provider console (OpenAI, Google AI Studio) for details.")

print("\n--- End of Video 4 Hello World Example ---")
