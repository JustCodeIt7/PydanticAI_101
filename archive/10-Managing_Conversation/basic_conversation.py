"""
basic_conversation.py - Video 10: Managing Conversation History

This script demonstrates the fundamentals of maintaining conversation
history across multiple interactions with a PydanticAI agent.
"""

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

# Initialize the agent with OpenAI's GPT model
# Replace with your preferred model
agent = Agent('openai:gpt-4o-mini')

def main():
    print("PydanticAI Conversation Example")
    print("===============================\n")
    
    # Initialize an empty conversation history
    # This will store all messages exchanged during the conversation
    conversation_history: list[ModelMessage] = []
    
    # First turn of the conversation
    user_query1 = "My name is James. I'm interested in AI frameworks."
    print(f"User: {user_query1}")
    
    # Run the agent with the initial query and empty history
    result1 = agent.run_sync(user_query1, message_history=conversation_history)
    print(f"AI: {result1.output}")
    
    # Update the conversation history with messages from this turn
    # This is the key step for maintaining context!
    conversation_history.extend(result1.all_messages())
    
    # Second turn - asking about frameworks
    user_query2 = "What are some popular Python frameworks for working with LLMs?"
    print(f"\nUser: {user_query2}")
    
    # Run the agent again, but now with the updated history
    result2 = agent.run_sync(user_query2, message_history=conversation_history)
    print(f"AI: {result2.output}")
    
    # Update history again
    conversation_history.extend(result2.new_messages())
    
    # Third turn - ask a question that requires remembering the user's name
    user_query3 = "Which one would you recommend for me specifically?"
    print(f"\nUser: {user_query3}")
    
    # The agent should remember the user's name is James
    result3 = agent.run_sync(user_query3, message_history=conversation_history)
    print(f"AI: {result3.output}")
    
    # Display message history structure
    print("\n----------------")
    print("Message History Structure:")
    print("----------------")
    for i, msg in enumerate(conversation_history):
        print(f"Message {i+1} - Type: {type(msg).__name__}")
    
    print("\nTotal messages in history:", len(conversation_history))

if __name__ == "__main__":
    main()