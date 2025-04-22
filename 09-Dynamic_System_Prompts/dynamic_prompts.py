"""
Example demonstrating dynamic system prompts with PydanticAI.

This script shows how to:
1. Define a dependency container for user profiles
2. Create a dynamic system prompt that adapts based on user preferences
3. Pass dependencies at runtime to personalize agent responses
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

# Ensure you have an OpenAI API key set
# os.environ["OPENAI_API_KEY"] = "your-api-key"

# Define a data model for user preferences (our dependency container)
@dataclass
class UserProfile:
    user_name: str
    language_preference: str
    expertise_level: str  # "beginner", "intermediate", or "expert"
    preferred_tone: str = "friendly"  # default tone

# Define a response model
class TranslationResponse(BaseModel):
    greeting: str = Field(description="Greeting in the user's preferred language")
    explanation: str = Field(description="Explanation of the concept")
    example: Optional[str] = Field(description="A helpful example if appropriate")

# Initialize our agent with the dependency type
agent = Agent(
    "openai:gpt-4o-mini",  # Change to your preferred model
    deps_type=UserProfile,  # Specify that this agent needs UserProfile dependencies
    output_type=TranslationResponse  # Structured output format
)

# Define a dynamic system prompt using the decorator
@agent.system_prompt
async def generate_personalized_prompt(ctx: RunContext[UserProfile]) -> str:
    """Generates a system prompt tailored to the user's profile."""
    user_name = ctx.deps.user_name
    language = ctx.deps.language_preference
    expertise = ctx.deps.expertise_level
    tone = ctx.deps.preferred_tone
    
    # Build a personalized system prompt based on user attributes
    prompt = f"You are a helpful assistant for {user_name}. "
    prompt += f"Please respond primarily in {language}. "
    
    # Adjust explanation depth based on expertise level
    if expertise == "beginner":
        prompt += "Explain concepts in simple terms without technical jargon. "
    elif expertise == "intermediate":
        prompt += "Use some technical terms but provide explanations for complex concepts. "
    else:  # expert
        prompt += "Feel free to use technical terminology and provide in-depth explanations. "
    
    # Set the tone
    prompt += f"Maintain a {tone} tone in your responses. "
    
    # Additional instruction for the structured output
    prompt += "Structure your response with a greeting in the user's language, " \
              "a clear explanation of the concept, and an example if appropriate."
    
    return prompt

# Optional: Add a second dynamic prompt component for additional instructions
@agent.system_prompt
async def add_safety_guidelines(_: RunContext[UserProfile]) -> str:
    """Adds standard safety guidelines to all prompts."""
    return "Always provide accurate information. If you're unsure about something, acknowledge it."

# Function to interact with the agent
async def interact_with_agent(user_id: int, query: str):
    """Simulate fetching a user profile and interacting with the agent."""
    # In a real application, you might fetch the profile from a database
    # Here we're simulating different user profiles
    profiles = {
        1: UserProfile(user_name="Alice", language_preference="French", 
                       expertise_level="beginner", preferred_tone="friendly"),
        2: UserProfile(user_name="Bob", language_preference="English", 
                       expertise_level="expert", preferred_tone="professional"),
        3: UserProfile(user_name="Carlos", language_preference="Spanish", 
                       expertise_level="intermediate", preferred_tone="casual"),
    }
    
    # Get the profile for this user
    profile = profiles.get(user_id, profiles[1])  # Default to user 1 if not found
    
    print(f"\nInteracting as user: {profile.user_name}")
    print(f"Language: {profile.language_preference}")
    print(f"Expertise: {profile.expertise_level}")
    print(f"Preferred tone: {profile.preferred_tone}")
    print(f"Query: {query}")
    print("-" * 50)
    
    # Run the agent with the user's profile as dependencies
    result = await agent.run(query, deps=profile)
    
    # Print the structured result
    print("\nResponse:")
    print(f"Greeting: {result.output.greeting}")
    print(f"Explanation: {result.output.explanation}")
    if result.output.example:
        print(f"Example: {result.output.example}")
    
    return result

async def main():
    """Main function to demonstrate the agent with different users."""
    # Example queries about the same topic for different users
    queries = [
        (1, "Explain how neural networks work"),
        (2, "Explain how neural networks work"),
        (3, "Explain how neural networks work"),
    ]
    
    print("Dynamic System Prompts Demo")
    print("=========================")
    
    for user_id, query in queries:
        await interact_with_agent(user_id, query)
        print("\n" + "=" * 50)

if __name__ == "__main__":
    # Check if API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found in environment variables.")
        print("Uncomment and set the API key in the script or export it as an environment variable.")
    
    # Run the main function
    asyncio.run(main())