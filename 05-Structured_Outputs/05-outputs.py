# video_05_structured_output.py

# --- Video 5: Understanding Structured Output with Pydantic Models ---

# Objective: Explain how to enforce structured, validated responses
# from the LLM using Pydantic models.

import os
from dotenv import load_dotenv
from typing import List # For type hinting lists

# --- Prerequisites ---
# 1. PydanticAI installed (pip install pydantic-ai)
# 2. Relevant LLM extra installed (e.g., pip install pydantic-ai[openai])
# 3. API key configured (e.g., OPENAI_API_KEY set as environment variable or in .env)
# 4. Pydantic installed (usually comes with pydantic-ai)

print("--- PydanticAI Tutorial Series ---")
print("Video 5: Structured Output with Pydantic Models")

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

# === 1. The Problem with Unstructured Output ===
print("\n=== 1. The Problem with Unstructured Output ===")
print(" - Relying on raw text strings from LLMs can be unreliable.")
print(" - Strings can be inconsistent, hard to parse, and lead to errors.")

# === 2. Defining a Pydantic Model for Structured Output ===
print("\n=== 2. Defining a Pydantic Model ===")
print(" - Pydantic models define the desired data structure using Python types.")
print(" - Use 'BaseModel' from Pydantic.")
print(" - Use 'Field' to add descriptions, which helps guide the LLM.")

try:
    from pydantic import BaseModel, Field
    from pydantic_ai import Agent, AgentRunResult # Import Agent and result type

    print(" - Successfully imported 'BaseModel', 'Field' from 'pydantic'.")
    print(" - Successfully imported 'Agent', 'AgentRunResult' from 'pydantic_ai'.")

    class CityInfo(BaseModel):
        """Represents structured information about a city."""
        # Field descriptions help the LLM understand what data is expected.
        city_name: str = Field(description="The official name of the city")
        country: str = Field(description="The country where the city is located")
        population: int = Field(description="Estimated population of the city")
        landmarks: List[str] = Field(description="A list of 3-5 famous landmarks or attractions")

    print(f" - Defined Pydantic model: '{CityInfo.__name__}'")
    print(f"   - Fields: {list(CityInfo.model_fields.keys())}")

except ImportError as e:
    print(f"\nError importing Pydantic or PydanticAI components: {e}")
    print("Please ensure both 'pydantic' and 'pydantic-ai' are installed.")
    exit()

# === 3. Configuring the Agent with 'output_type' ===
print("\n=== 3. Configuring Agent for Structured Output ===")
print(" - Pass the Pydantic model class to the Agent constructor via 'output_type'.")

try:
    # Configure the agent, specifying CityInfo as the desired output structure.
    # PydanticAI will use this model to instruct the LLM and validate the response.
    agent = Agent(
        'openai:gpt-4o', # Replace if needed, ensure key/extra are set
        system_prompt='You are an expert geographer. Provide detailed information about the requested city.',
        output_type=CityInfo # Key step: Specify the desired output model
    )
    print(f" - Agent configured with model: '{agent.llm.provider}:{agent.llm.model}'")
    print(f" - Output Type enforced: {agent.output_type.__name__}") # Shows the model being used

except Exception as e:
    print(f"\nError configuring Agent: {e}")
    print(" - Check LLM provider/model string, API key, and installed extras.")
    exit()

# === 4. Running the Agent and Accessing Structured Output ===
print("\n=== 4. Running Agent & Accessing Structured Output ===")
user_query = "Tell me about Paris, France."
print(f" - Sending query: \"{user_query}\"")

try:
    # Run the agent - PydanticAI handles schema generation and validation
    result: AgentRunResult = agent.run_sync(user_query)
    print(" - Agent execution complete.")

    # Access the validated output
    print("\n - Accessing the validated output:")
    # result.output is now an instance of CityInfo, not just a string!
    validated_output: CityInfo = result.output

    print(f"   - Type of result.output: {type(validated_output)}")

    # Access attributes in a type-safe way
    if isinstance(validated_output, CityInfo):
        print(f"   - City Name: {validated_output.city_name}")
        print(f"   - Country:   {validated_output.country}")
        print(f"   - Population: {validated_output.population:,}") # Format population
        print(f"   - Landmarks:")
        for landmark in validated_output.landmarks:
            print(f"     - {landmark}")
    else:
        # This might happen if validation failed and wasn't automatically retried/fixed
        print(f"   - Expected CityInfo object, but got: {type(validated_output)}")
        print(f"   - Raw output: {validated_output}")


    # PydanticAI also handles validation errors internally (can involve re-prompting)
    # We can check if validation was successful or if retries happened (more advanced)
    # print(f" - Validation Success: {result.validation_success}") # Check validation status

except Exception as e:
    print(f"\nError during agent execution: {e}")
    print(" - Check API key validity, network connection, or provider status.")
    print(" - The LLM might have failed to produce output matching the schema.")

# === 5. Benefits of Structured Output ===
print("\n=== 5. Benefits of Structured Output ===")
print(" - Consistency: Predictable response format.")
print(" - Reliability: Validated data reduces runtime errors.")
print(" - Type Safety: Work with Python objects, not raw strings.")
print(" - Simplified Integration: Easier to use LLM output in other systems.")
print(" - This is a core strength and differentiator for PydanticAI.")

print("\n--- End of Video 5 Structured Output Example ---")
