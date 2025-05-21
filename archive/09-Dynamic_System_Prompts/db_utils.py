"""
Simulated database utility for user profiles.

In a real application, this would connect to an actual database.
This is a simplified mock implementation for demonstration purposes.
"""

import asyncio
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class UserPreferences:
    """User preferences that might be stored in a database."""
    preferred_language: str
    expertise_level: str
    communication_style: str
    special_instructions: Optional[str] = None


class UserDatabase:
    """A mock database for user profiles and preferences."""
    
    def __init__(self):
        """Initialize with some mock data."""
        self._users: Dict[int, UserPreferences] = {
            1: UserPreferences(
                preferred_language="French",
                expertise_level="beginner",
                communication_style="friendly",
                special_instructions="Always include a fun fact in responses"
            ),
            2: UserPreferences(
                preferred_language="English",
                expertise_level="expert",
                communication_style="professional",
                special_instructions="Prioritize scientific accuracy"
            ),
            3: UserPreferences(
                preferred_language="Spanish",
                expertise_level="intermediate",
                communication_style="casual",
                special_instructions="Include practical examples"
            ),
        }
    
    async def get_user_preferences(self, user_id: int) -> Optional[UserPreferences]:
        """
        Simulate an asynchronous database query to get user preferences.
        
        In a real application, this would query a database.
        """
        # Simulate network delay
        await asyncio.sleep(0.2)
        
        return self._users.get(user_id)
    
    async def update_user_preferences(
        self, user_id: int, preferences: UserPreferences
    ) -> bool:
        """Update a user's preferences."""
        # Simulate network delay
        await asyncio.sleep(0.2)
        
        if user_id in self._users:
            self._users[user_id] = preferences
            return True
        return False


# Example usage
async def example():
    db = UserDatabase()
    user_prefs = await db.get_user_preferences(1)
    print(user_prefs)


if __name__ == "__main__":
    asyncio.run(example())