"""
message_types.py - Video 10: Managing Conversation History

This script explores the different message types in PydanticAI
and demonstrates how they are used in conversation history.
"""

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    UserMessage, 
    AssistantMessage,
    ToolMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart
)
from typing import List
import json
from dataclasses import dataclass

# Initialize the agent with OpenAI's model
agent = Agent(
    'openai:gpt-4o-mini',
    system_prompt="You are a helpful assistant that provides information about cities."
)

# Define a tool to demonstrate ToolMessage creation
@agent.tool
async def get_weather(ctx: RunContext, city: str) -> str:
    """Get the current weather for a city (simulated)."""
    # In a real app, this would call a weather API
    weather_data = {
        "new york": "72°F, Partly Cloudy",
        "london": "65°F, Rainy",
        "tokyo": "80°F, Sunny",
        "paris": "70°F, Clear"
    }
    return weather_data.get(city.lower(), "Weather data not available")

def print_message_details(message: ModelMessage, level: int = 0):
    """Print details about a message in the conversation history."""
    indent = "  " * level
    print(f"{indent}Message Type: {type(message).__name__}")
    print(f"{indent}Timestamp: {message.timestamp}")
    
    # Print role if available
    if hasattr(message, "role"):
        print(f"{indent}Role: {message.role}")
    
    # Print parts
    print(f"{indent}Parts ({len(message.parts)}): ")
    for i, part in enumerate(message.parts):
        part_indent = "  " * (level + 1)
        print(f"{part_indent}Part {i+1} Type: {type(part).__name__}")
        
        # Display part content based on type
        if hasattr(part, "content"):
            # Truncate content if it's too long
            content = str(part.content)
            if len(content) > 50:
                content = content[:47] + "..."
            print(f"{part_indent}Content: {content}")
    
    print()  # Add a blank line for readability

def explore_message_types():
    """Demonstrate and explore different message types in PydanticAI."""
    print("PydanticAI Message Types Explorer")
    print("=================================\n")
    
    # 1. Create a conversation history with various message types
    conversation: List[ModelMessage] = []
    
    # 2. Add a user message
    user_msg = UserMessage(content="Hi, I'm visiting New York next week. What's the weather like?")
    conversation.append(user_msg)
    
    # 3. Process the message with our agent
    result = agent.run_sync(
        "Tell me about the weather in New York.",
        message_history=conversation
    )
    
    # 4. Add the new messages to our history
    conversation.extend(result.new_messages())
    
    # 5. Add another user message that will trigger a tool call
    user_msg2 = UserMessage(content="What about the weather in Tokyo?")
    conversation.append(user_msg2)
    
    # 6. Process this message (will use the tool)
    result2 = agent.run_sync(
        "What's the current weather in Tokyo?",
        message_history=conversation
    )
    
    # 7. Add these messages to our history
    conversation.extend(result2.new_messages())
    
    # 8. Now analyze the conversation history
    print("CONVERSATION HISTORY ANALYSIS")
    print("-----------------------------\n")
    print(f"Total messages in history: {len(conversation)}")
    
    for i, message in enumerate(conversation):
        print(f"Message #{i+1}")
        print("-" * 40)
        print_message_details(message)
    
    # 9. Demonstrate how to extract just the text exchange for display
    print("\nEXTRACTING A CLEAN CONVERSATION VIEW")
    print("-----------------------------------")
    
    clean_conversation = []
    for msg in conversation:
        if isinstance(msg, UserMessage) and len(msg.parts) > 0:
            clean_conversation.append(f"User: {msg.parts[0].content}")
        elif isinstance(msg, ModelResponse) and len(msg.parts) > 0:
            if isinstance(msg.parts[0], TextPart):
                clean_conversation.append(f"Assistant: {msg.parts[0].content}")
    
    print("\nClean Conversation:")
    for line in clean_conversation:
        print(line)
        print("-" * 40)
    
    # 10. Demonstrate creating and examining different message types manually
    print("\nMANUALLY CREATING MESSAGE TYPES")
    print("------------------------------")
    
    # Create a UserMessage
    user_message = UserMessage(content="Hello, this is a user message")
    print("User Message:")
    print_message_details(user_message)
    
    # Create an AssistantMessage
    assistant_message = AssistantMessage(content="Hello, this is an assistant message")
    print("Assistant Message:")
    print_message_details(assistant_message)
    
    # Create a ToolMessage
    tool_message = ToolMessage(
        tool_name="get_weather",
        tool_input={"city": "Paris"},
        content="70°F, Clear"
    )
    print("Tool Message:")
    print_message_details(tool_message)

if __name__ == "__main__":
    explore_message_types()