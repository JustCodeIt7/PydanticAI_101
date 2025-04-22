# video_03_configure_llm.py

# --- Video 3: Configuring Your First LLM (OpenAI & Gemini) ---

# Objective: Show how to configure PydanticAI to use common LLM providers,
# focusing on OpenAI and Google Gemini.

import os
from dotenv import load_dotenv # Used to load environment variables from .env file

print("--- PydanticAI Tutorial Series ---")
print("Video 3: Configuring Your First LLM")

# === 1. API Keys ===
print("\n=== 1. API Keys ===")
print(" - To use LLMs (like OpenAI, Gemini), you need API keys from the provider.")
print(" - Obtain keys from their respective consoles (OpenAI Platform, Google AI Studio).")

# === 2. Securely Managing API Keys (Environment Variables) ===
print("\n=== 2. Securely Managing API Keys ===")
print(" - DO NOT hardcode API keys directly in your source code!")
print(" - Recommended method: Use environment variables.")
print("   - Set OS environment variables (e.g., OPENAI_API_KEY, GOOGLE_API_KEY).")
print("   - For local development, use a '.env' file in your project root.")

# Example .env file content:
# OPENAI_API_KEY=sk-your_openai_api_key_here
# GOOGLE_API_KEY=your_google_api_key_here

print("   - Use a library like 'python-dotenv' to load keys from '.env'.")
print("     (Install it: pip install python-dotenv)")

# Attempt to load variables from .env file (if it exists)
load_dotenv()
print("   - Attempted to load environment variables from a '.env' file (if present).")

# Check if keys are loaded (optional demonstration)
openai_key = os.getenv("OPENAI_API_KEY")
google_key = os.getenv("GOOGLE_API_KEY")

print(f"   - OpenAI Key loaded: {'Yes' if openai_key else 'No (or not set)'}")
print(f"   - Google Key loaded: {'Yes' if google_key else 'No (or not set)'}")
print("   - PydanticAI often automatically detects these standard environment variables.")


# === 3. Configuring PydanticAI Agent ===
print("\n=== 3. Configuring the PydanticAI Agent ===")
print(" - Import the Agent class from pydantic_ai.")
try:
    from pydantic_ai import Agent
    print("   - Successfully imported 'Agent' from 'pydantic_ai'.")

    print(" - Specify the LLM using a model identifier string: '<provider>:<model_name>'")

    # --- Example: OpenAI Configuration ---
    print("\n   --- Example: OpenAI (GPT-4o) ---")
    # Assumes OPENAI_API_KEY is set in the environment
    # Assumes 'pip install pydantic-ai[openai]' was run
    try:
        # We only instantiate it here to show the syntax.
        # A real call would require the key to be valid and set.
        agent_openai = Agent(
            'openai:gpt-4o',
            system_prompt='You are a helpful assistant.' # Optional: Set assistant behavior
        )
        print("     - Agent configured for 'openai:gpt-4o'.")
        print(f"     - Provider: {agent_openai.llm.provider}, Model: {agent_openai.llm.model}")
    except Exception as e:
        print(f"     - Could not instantiate OpenAI agent (is key set? is extra installed?): {e}")


    # --- Example: Google Gemini Configuration ---
    print("\n   --- Example: Google Gemini (Gemini 1.5 Flash) ---")
    # Assumes GOOGLE_API_KEY is set in the environment (or other auth like gcloud ADC)
    # Assumes 'pip install pydantic-ai[google-gla]' was run
    try:
        # Instantiate to show syntax. Requires valid key/auth and installation.
        agent_gemini = Agent(
            'google-gla:gemini-1.5-flash',
            system_prompt='You are a creative assistant.' # Optional: Set assistant behavior
        )
        print("     - Agent configured for 'google-gla:gemini-1.5-flash'.")
        print(f"     - Provider: {agent_gemini.llm.provider}, Model: {agent_gemini.llm.model}")
    except Exception as e:
        print(f"     - Could not instantiate Gemini agent (is key/auth set? is extra installed?): {e}")

    # --- Example: Groq Configuration ---
    print("\n   --- Example: Groq (Mixtral) ---")
    # Assumes GROQ_API_KEY is set in the environment
    # Assumes 'pip install pydantic-ai[groq]' was run
    try:
        # Instantiate to show syntax. Requires valid key/auth and installation.
        agent_groq = Agent(
            'groq:mixtral-8x7b-32768',
             system_prompt='You are a fast assistant.' # Optional
        )
        print("     - Agent configured for 'groq:mixtral-8x7b-32768'.")
        print(f"     - Provider: {agent_groq.llm.provider}, Model: {agent_groq.llm.model}")
    except Exception as e:
        print(f"     - Could not instantiate Groq agent (is key set? is extra installed?): {e}")


except ImportError:
    print("\nError: Could not import 'Agent' from 'pydantic_ai'.")
    print("Please ensure PydanticAI is installed correctly (pip install pydantic-ai).")
except Exception as e:
    print(f"\nAn unexpected error occurred during Agent configuration: {e}")


# === 4. Model Agnosticism ===
print("\n=== 4. Model Agnosticism ===")
print(" - The consistent '<provider>:<model_name>' format makes switching easy.")
print(" - Often, only this string needs changing (plus ensuring the correct extra is installed).")
print(" - Reduces vendor lock-in and simplifies experimentation.")

# === 5. Other Supported Providers ===
print("\n=== 5. Other Supported Providers ===")
print(" - PydanticAI supports many providers, including:")
print("   - Anthropic (Claude models)")
print("   - Ollama (Local models)")
print("   - Mistral")
print("   - Cohere")
print("   - AWS Bedrock")
print("   - Deepseek")
print("   - ... and more (check documentation)")
print(" - Refer to the official PydanticAI documentation for the full list and specific model identifiers.")

print("\n--- End of Video 3 Configuration Guide ---")
