# fake_database.py

class DatabaseConn:
    def __init__(self):
        self._users = {
            "John Doe": 123,
            "Jane Smith": 456,
        }
        self._messages_sent = []

    @property
    def users(self):
        return UserTable(self._users)

    @property
    def messages_sent(self):
        return self._messages_sent

    def record_message(self, user_id: int, message: str):
        self._messages_sent.append({"user_id": user_id, "message": message})
        print(f"FakeDB: Message recorded for user {user_id}: '{message}'")

class UserTable:
    def __init__(self, users_data):
        self._users_data = users_data

    def get(self, name: str):
        return self._users_data.get(name)