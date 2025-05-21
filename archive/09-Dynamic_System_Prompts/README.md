# Dynamic System Prompts in PydanticAI

This example demonstrates how to create dynamic system prompts that adapt based on runtime context using PydanticAI's dependency injection system.

## Key Concepts

- **Dynamic System Prompts**: System prompts that change based on runtime context
- **Dependency Injection**: Providing necessary context to your agent at runtime
- **User Personalization**: Tailoring responses based on user preferences

## Features Demonstrated

1. Defining a dependency container using Python dataclasses
2. Creating dynamic system prompt functions with the `@agent.system_prompt` decorator
3. Accessing injected dependencies via `RunContext`
4. Combining multiple system prompts
5. Passing dependencies at runtime
6. Personalizing responses based on user profiles

## How It Works

The example creates an agent that:
- Takes a `UserProfile` object as a dependency
- Dynamically generates a system prompt based on the user's preferences
- Adapts its responses based on language, expertise level, and tone preferences
- Returns a structured output using Pydantic models

## Running the Example

1. Make sure you have PydanticAI installed:
   ```
   pip install pydantic-ai
   ```

2. Set your OpenAI API key:
   ```
   export OPENAI_API_KEY=your-api-key-here
   ```

3. Run the example:
   ```
   python dynamic_prompts.py
   ```

## Expected Output

The example runs the same query ("Explain how neural networks work") for three different user profiles:
- Alice: A French-speaking beginner who prefers a friendly tone
- Bob: An English-speaking expert who prefers a professional tone
- Carlos: A Spanish-speaking intermediate user who prefers a casual tone

Each response will be personalized according to the user's preferences, demonstrating how dynamic system prompts can create context-aware agent behavior.

## Real-World Applications

- **Multilingual Customer Support**: Respond to customers in their preferred language
- **Educational Assistants**: Adjust explanation complexity based on student level
- **Personalized Services**: Customize agent persona based on user preferences
- **Role-Based Responses**: Adapt information detail based on user role or clearance