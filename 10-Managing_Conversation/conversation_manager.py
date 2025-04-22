"""
conversation_manager.py - Video 10: Managing Conversation History

This script demonstrates a more advanced approach to conversation management with
a dedicated ConversationManager class that handles persisting history, summarizing
long conversations, and managing multiple conversation sessions.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, UserMessage, AssistantMessage, ModelMessagesTypeAdapter

class ConversationSummary(BaseModel):
    """Model for summarizing a conversation to reduce token usage."""
    main_topics: List[str] = Field(description="The main topics discussed in the conversation")
    key_points: List[str] = Field(description="The key points or facts established in the conversation")
    user_preferences: Dict[str, str] = Field(description="User preferences or information revealed during the conversation")

class ConversationManager:
    """
    Manages conversation history for PydanticAI agents, including:
    - Storing and retrieving conversation history
    - Handling multiple conversation sessions
    - Summarizing long conversations to stay within token limits
    - Persisting conversations to disk
    """
    
    def __init__(self, agent: Agent, session_id: str = "default", history_dir: str = ".conversation_history"):
        """
        Initialize the conversation manager.
        
        Args:
            agent: The PydanticAI Agent to use for conversations
            session_id: Unique identifier for this conversation session
            history_dir: Directory to store conversation histories
        """
        self.agent = agent
        self.session_id = session_id
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(exist_ok=True)
        
        # Initialize or load existing conversation history
        self.history: List[ModelMessage] = self._load_history()
        
        # Create a summarization agent
        self.summary_agent = Agent(
            'openai:gpt-4o-mini',
            output_type=ConversationSummary,
            system_prompt="Analyze the conversation and extract key information."
        )
        
    def _load_history(self) -> List[ModelMessage]:
        """Load conversation history from disk if it exists."""
        history_path = self.history_dir / f"{self.session_id}.json"
        
        if history_path.exists():
            try:
                with open(history_path, 'r') as f:
                    json_data = f.read()
                    return ModelMessagesTypeAdapter.validate_json(json_data)
            except Exception as e:
                print(f"Error loading history: {e}")
                return []
        return []
    
    def _save_history(self) -> None:
        """Save conversation history to disk."""
        history_path = self.history_dir / f"{self.session_id}.json"
        
        try:
            # Convert to JSON using the type adapter
            json_data = ModelMessagesTypeAdapter.dump_json(self.history)
            
            with open(history_path, 'wb') as f:
                f.write(json_data)
        except Exception as e:
            print(f"Error saving history: {e}")
    
    def add_user_message(self, content: str) -> None:
        """
        Add a user message to the history without sending to the agent.
        Useful for adding system context or initial messages.
        """
        message = UserMessage(content=content)
        self.history.append(message)
        self._save_history()
    
    def process_message(self, content: str, save_history: bool = True) -> str:
        """
        Process a user message through the agent and update conversation history.
        
        Args:
            content: The user's message
            save_history: Whether to save history to disk after processing
            
        Returns:
            The agent's response text
        """
        # Check if history might be getting too long and summarize if needed
        if len(self.history) > 20:  # arbitrary threshold
            self._summarize_history()
            
        # Process the message with the agent
        result = self.agent.run_sync(content, message_history=self.history)
        
        # Update history with new messages
        self.history.extend(result.new_messages())
        
        # Optionally save the updated history
        if save_history:
            self._save_history()
            
        return result.output
    
    def _summarize_history(self) -> None:
        """
        Summarize a long conversation history to reduce token usage.
        Replaces older messages with a summary message.
        """
        # Only summarize if we have enough messages
        if len(self.history) < 10:
            return
            
        # Extract the conversation as a formatted string for the summarizer
        formatted_history = "\n".join([
            f"{'User' if isinstance(msg, UserMessage) else 'Assistant'}: {msg.parts[0].content}"
            for msg in self.history
            if isinstance(msg, (UserMessage, AssistantMessage)) and len(msg.parts) > 0
        ])
        
        # Get a summary of the conversation
        summary_result = self.summary_agent.run_sync(formatted_history)
        summary = summary_result.output
        
        # Create a summary message
        summary_text = f"""CONVERSATION SUMMARY:
        Topics: {', '.join(summary.main_topics)}
        Key Points: {', '.join(summary.key_points)}
        User Preferences: {json.dumps(summary.user_preferences, indent=2)}
        """
        
        # Replace older messages with the summary
        # Keep the most recent 5 messages
        recent_messages = self.history[-5:]
        self.history = [UserMessage(content=summary_text)] + recent_messages
    
    def list_available_sessions(self) -> List[str]:
        """List all available conversation sessions from disk."""
        return [p.stem for p in self.history_dir.glob("*.json")]
    
    def switch_session(self, session_id: str) -> None:
        """Switch to a different conversation session."""
        # Save current history first
        self._save_history()
        
        # Switch session ID and load the corresponding history
        self.session_id = session_id
        self.history = self._load_history()
    
    def clear_history(self) -> None:
        """Clear the current conversation history."""
        self.history = []
        self._save_history()


def interactive_demo():
    """Run an interactive conversation demo."""
    agent = Agent('openai:gpt-4o')
    
    print("PydanticAI Conversation Manager Demo")
    print("====================================\n")
    
    # Get or create a session
    print("Available sessions:", end=" ")
    cm = ConversationManager(agent)
    sessions = cm.list_available_sessions()
    
    if sessions:
        print(", ".join(sessions))
        session_id = input("Enter session name or press Enter for new session: ")
        if session_id in sessions:
            cm.switch_session(session_id)
        else:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            cm.switch_session(session_id)
    else:
        print("No existing sessions")
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cm.switch_session(session_id)
    
    print(f"\nUsing session: {cm.session_id}")
    print("Type 'exit' to quit, 'clear' to clear history, 'switch' to change sessions\n")
    
    # Initial system context (optional)
    cm.add_user_message("My name is James and I'm learning about AI frameworks.")
    
    # Main conversation loop
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == "exit":
            break
        elif user_input.lower() == "clear":
            cm.clear_history()
            print("Conversation history cleared.")
            continue
        elif user_input.lower() == "switch":
            new_session = input("Enter session name: ")
            cm.switch_session(new_session)
            print(f"Switched to session: {cm.session_id}")
            continue
        
        # Process the user message
        response = cm.process_message(user_input)
        print(f"AI: {response}")


if __name__ == "__main__":
    interactive_demo()