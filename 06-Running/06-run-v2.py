# video_06_run_methods.py

# --- Video 6: Diving Deeper: Agent Lifecycle (run, run_sync, run_stream) ---

# Objective: Explain the different ways to execute an agent run and
# introduce asynchronous operations and streaming, using Ollama locally.

import asyncio # Required for async operations (run, run_stream)
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider


print("--- PydanticAI Tutorial Series ---")
print("Video 6: Agent Run Methods (run_sync, run, run_stream)")
print("Using Ollama with model 'llama3.2' (ensure Ollama server is running)")

# === 1. Import Agent and Result Types ===
print("\n=== 1. Importing Agent and Result Types ===")
try:
    from pydantic_ai import Agent, AgentRunResult, StreamedRunResult
    print(" - Successfully imported Agent and result types.")
except ImportError:
    print("\nError: Could not import from 'pydantic_ai'.")
    print("Please ensure PydanticAI is installed correctly (`pip install pydantic-ai[ollama]`).")
    exit()

# === 2. Configure the Agent (Ollama) ===
print("\n=== 2. Configuring Agent for Local Ollama ===")
# No API key needed for local Ollama.
# PydanticAI defaults to http://localhost:11434 for Ollama.
# Ensure the model name 'llama3.2' matches what you have pulled in Ollama.
model = OpenAIModel(
    'llama3.2',
    provider=OpenAIProvider(
        base_url='http://localhost:11434/v1',
        
    ),
)
agent = Agent(model)

# === 3. Synchronous Execution: run_sync ===
print("\n=== 3. Synchronous Execution: run_sync ===")
print(" - Blocks execution until the LLM responds.")
print(" - Suitable for simple scripts or synchronous code.")

user_query_sync = "Explain the difference between synchronous and asynchronous programming."
print(f"\n   Sending query (sync): \"{user_query_sync}\"")
try:
    # This call will wait for the full response
    result_sync: AgentRunResult = agent.run_sync(user_query_sync)
    print("   - Sync execution complete.")
    print("\n   --- Sync Response ---")
    print(result_sync.output)
    print("   --- End Sync Response ---")
except Exception as e:
    print(f"\n   Error during sync execution: {e}")
    print("   - Is the Ollama server running and accessible?")

# === 4. Asynchronous Execution: run ===
print("\n=== 4. Asynchronous Execution: run ===")
print(" - Uses `async` and `await`.")
print(" - Doesn't block the entire program; allows other tasks to run.")
print(" - Ideal for web servers (FastAPI, etc.) or I/O-bound tasks.")

user_query_async = "What are the main benefits of using Python's asyncio?"

async def run_async_example():
    """Demonstrates the asynchronous agent.run() method."""
    print(f"\n   Sending query (async): \"{user_query_async}\"")
    try:
        # 'await' pauses this coroutine, allowing others to run
        result_async: AgentRunResult = await agent.run(user_query_async)
        print("   - Async execution complete.")
        print("\n   --- Async Response ---")
        print(result_async.output)
        print("   --- End Async Response ---")
    except Exception as e:
        print(f"\n   Error during async execution: {e}")
        print("   - Is the Ollama server running and accessible?")

# Run the async function using asyncio
print("\n   Running the async example...")
try:
    asyncio.run(run_async_example())
except RuntimeError as e:
    # Handle cases where asyncio event loop is already running (e.g., in Jupyter)
     if "cannot run nested" in str(e):
         print("   - (Skipping asyncio.run() as loop seems already running)")
     else:
         raise e


# === 5. Asynchronous Streaming: run_stream ===
print("\n=== 5. Asynchronous Streaming: run_stream ===")
print(" - Uses `async` and `await` with an async iterator.")
print(" - Yields response chunks as they become available.")
print(" - Great for real-time feedback (chatbots, etc.).")

user_query_stream = "Tell me a short story about a curious robot exploring a garden."

async def run_stream_example():
    """Demonstrates the asynchronous agent.run_stream() method."""
    print(f"\n   Sending query (stream): \"{user_query_stream}\"")
    print("\n   --- Streamed Response ---")
    try:
        # 'async with' manages the stream context
        async with agent.run_stream(user_query_stream) as response_stream:
            # response_stream is an async iterator (StreamedRunResult)
            async for chunk in response_stream:
                # Print each chunk as it arrives
                print(chunk, end="", flush=True) # end="" prevents newlines, flush ensures immediate display
        print("\n   --- End Streamed Response ---")
        print("\n   - Stream finished.")
    except Exception as e:
        print(f"\n   Error during stream execution: {e}")
        print("   - Is the Ollama server running and accessible?")

# Run the stream example using asyncio
print("\n   Running the stream example...")
try:
    asyncio.run(run_stream_example())
except RuntimeError as e:
    # Handle cases where asyncio event loop is already running (e.g., in Jupyter)
     if "cannot run nested" in str(e):
         print("   - (Skipping asyncio.run() as loop seems already running)")
     else:
         raise e


# === 6. Choosing the Right Method ===
print("\n=== 6. Choosing the Right Method ===")
print(" - `run_sync`: Simple scripts, synchronous code.")
print(" - `run`: Async applications (web servers), concurrent I/O.")
print(" - `run_stream`: Real-time feedback, processing partial responses.")

print("\n--- End of Video 6 Run Methods Example ---")
