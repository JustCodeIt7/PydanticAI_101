import asyncio

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Set up connection to Ollama (using OpenAI-compatible API)
base_url = 'http://100.95.122.242:11434/v1'
model_name = 'qwen3:1.7b'
ollama_model = OpenAIModel(
    model_name=model_name,
    provider=OpenAIProvider(base_url=base_url)
)

# Create the simplest possible agent - no tools, just LLM interaction
basic_agent = Agent(
    ollama_model,
    system_prompt="You are a helpful assistant that provides clear and concise responses."
)

# Synchronous example - simplest way to use the agent
def demo_basic_sync():
    print("\n=== Basic Synchronous Example ===")
    user_input = "What is machine learning in simple terms?"

    result = basic_agent.run_sync(user_input)
    print(f"User: {user_input}")
    print(f"Agent: {result.output}")

# Asynchronous example
async def demo_basic_async():
    print("\n=== Basic Asynchronous Example ===")
    user_input = "Explain how neural networks work in one paragraph."

    result = await basic_agent.run(user_input)
    print(f"User: {user_input}")
    print(f"Agent: {result.output}")

# Streaming example - see response as it's generated
async def demo_basic_streaming():
    print("\n=== Basic Streaming Example ===")
    user_input = "List 3 benefits of artificial intelligence."

    print(f"User: {user_input}")
    print("Agent: ", end="", flush=True)

    # Fixed streaming implementation:
    async with basic_agent.run_stream(user_input) as stream:
        # Get the final output directly - PydanticAI handles the streaming internally
        output = await stream.get_output()
        print(output)

# Alternative streaming approach that shows incremental updates
async def demo_streaming_with_updates():
    print("\n=== Streaming with Updates ===")
    user_input = "What are the key principles of reinforcement learning?"

    print(f"User: {user_input}")
    print("Agent: ", end="", flush=True)

    # This is an alternative approach if the model and provider support incremental streaming
    result = await basic_agent.run(user_input)
    print(result.output)

# Simple interactive chat function
def interactive_chat():
    print("\n=== Interactive Chat ===")
    print("Type 'exit' to quit")

    # To maintain conversation history
    message_history = []

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == '/exit':
            break

        # Run agent with accumulated message history
        result = basic_agent.run_sync(user_input, message_history=message_history)

        # Update message history for next turn
        message_history = result.new_messages()

        print(f"Agent: {result.output}")

if __name__ == "__main__":
    # Run the synchronous example
    demo_basic_sync()

    # Run the asynchronous examples
    asyncio.run(demo_basic_async())
    asyncio.run(demo_basic_streaming())
    asyncio.run(demo_streaming_with_updates())

    # Run the interactive chat example
    interactive_chat()