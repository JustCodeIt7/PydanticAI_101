# File: 01_hello_pydantic_ai.py
# %%
import os
from unittest import result
from ollama import Client
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from rich import print
import logfire
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# %%
# --- 0. Optional: Set Up Logging with Logfire ---
# Logfire helps you monitor and debug your Pydantic AI applications.
# It's optional but highly recommended for production use.
# You can sign up for a free account at https://logfire.dev and get your token
# logfire.configure(send_to_logfire="if-token-present")
# logfire.instrument_pydantic_ai()


# %%
# --- 2. Create an Agent with Pydantic AI ---
# An Agent is the core of Pydantic AI. It combines your data model with
# the LLM to extract structured data from unstructured text.
# Here, we create an agent that uses the Ollama LLM via the OllamaTool
model = OpenAIChatModel("gpt-5-nano", provider=OpenAIProvider(api_key=OPENAI_API_KEY))
agent = Agent(model)

result = await agent.run(
    "Hello, my name is Alice. I am 30 years old and I work as a software engineer."
)
print(result.output)


# %%
# --- 1. Define Your Desired Data Structure ---
# Imagine you have unstructured text and you want to pull out specific, clean
# information. With Pydantic, you define a class that represents the "shape"
# of the data you want.
# This gives you type-safety and auto-validation out of the box.
class User(BaseModel):
    """A Pydantic model to represent a user's information."""

    name: str = Field(description="The user's full name")
    age: int = Field(description="The user's age in years")
    role: str = Field(description="The user's role or job title")


# %%
