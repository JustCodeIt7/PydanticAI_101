# Managing Conversation History in PydanticAI

This tutorial demonstrates how to maintain conversational context across multiple interactions with an LLM using PydanticAI.

## Key Concepts

- **Conversational Memory**: Maintaining context between separate LLM calls
- **Message History**: Passing previous interactions to subsequent agent runs
- **Message Objects**: Understanding the different types of messages (UserMessage, AssistantMessage, ToolMessage)
- **Stateful Conversations**: Creating agents that remember previous interactions

## Features Demonstrated

1. Initializing an empty conversation history
2. Running an agent with the current history
3. Extracting new messages from an agent's response
4. Updating the conversation history with new messages
5. Implementing basic and advanced conversation patterns
6. Persisting conversation history across sessions

## How It Works

PydanticAI agents are stateless by default, meaning they don't automatically remember previous interactions. To create a conversational experience:

1. An empty message history is initialized
2. Each agent interaction passes the current history
3. New messages from each interaction are extracted and added to the history
4. The updated history is passed to the next interaction

## Running the Examples

1. Ensure you have PydanticAI installed:
   ```
   pip install pydantic-ai
   ```

2. Set your OpenAI API key:
   ```
   export OPENAI_API_KEY=your-api-key-here
   ```

3. Run the basic conversation example:
   ```
   python basic_conversation.py
   ```

4. Run the advanced conversation manager example:
   ```
   python conversation_manager.py
   ```

## Real-World Applications

- **Chatbots**: Create engaging chatbots that maintain context
- **Customer Support**: Build agents that remember customer details
- **Virtual Assistants**: Develop assistants that recall personal preferences
- **Education**: Create tutors that adapt to student progress over time
- **Healthcare**: Build agents that remember patient history for appropriate responses