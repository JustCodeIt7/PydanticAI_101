"""
hello_world.py  –  PydanticAI Masterclass · Video 4
---------------------------------------------------
Your very first PydanticAI agent in <20 seconds!

Prerequisite:
  • Set an environment variable  OPENAI_API_KEY="sk‑..."   (or use another provider)

Run:
  python hello_world.py
"""

from pydantic_ai import Agent

# 1) Configure the agent (single line ↓)
agent = Agent(
    "openai:gpt-4o",                     # swap provider/model if you like
    system_prompt="Be concise, reply with one sentence.",
)

# 2) Ask a question (synchronous helper)
result = agent.run_sync('Where does "hello world" come from?')

# 3) Print the model’s answer
print(result.output)
