# File: 01_hello_pydantic_ai.py
# Description: Video 1 - Your first "Hello World" with Pydantic AI.
# This script demonstrates the core functionality of Pydantic AI:
# extracting structured, type-safe data from unstructured text using an LLM.

import os

# We'll use a local Ollama instance for our LLM.
# Pydantic AI is model-agnostic, so you could easily swap this for OpenAI, Anthropic, etc.
from ollama import Client

# Pydantic is used to define the data structure we want to extract.
from pydantic import BaseModel, Field

# PydanticAI is the main class that orchestrates the LLM interaction.
from pydantic_ai import PydanticAI
from rich import print

# --- 1. Define Your Desired Data Structure ---
#
# Imagine you have unstructured text and you want to pull out specific, clean
# information. With Pydantic, you define a class that represents the "shape"
# of the data you want.
# This gives you type-safety and auto-validation out of the box.


class User(BaseModel):
    """A Pydantic model to represent a user's information."""

    name: str = Field(description="The user's full name")
    age: int = Field(description="The user's age in years")
    role: str = Field(description="The user's role or job title")


# --- 2. Set Up The Language Model (LLM) ---
#
# Pydantic AI works with any LLM. Here, we're setting up a client to
# connect to a locally running Ollama instance.
# Make sure you have Ollama installed and a model pulled (e.g., `ollama pull llama3.2`)
try:
    ollama_client = Client()
    # Ping the model to make sure it's running before we proceed.
    ollama_client.chat(model="llama3.2", messages=[{"role": "user", "content": "Hi"}])
    print("[green]✓ Ollama client is connected and model is available.[/green]")
except Exception as e:
    print(f"[red]✗ Error connecting to Ollama:[/red] {e}")
    print(
        "[yellow]Please ensure Ollama is running and you have pulled the 'llama3.2' model.[/yellow]"
    )
    exit()


# --- 3. Instantiate PydanticAI ---
#
# This is the core component. You create an instance of PydanticAI and
# pass it the LLM client you want to use. This object will handle the
# prompt engineering, LLM calls, and parsing for you.

# We are using the `ollama_client.chat.completions.create` method as the llm_engine
# The library is smart enough to adapt to different client interfaces.
ai = PydanticAI(llm_engine=ollama_client.chat.completions.create, llm_params={"model": "llama3.2"})


# --- 4. Provide Unstructured Input and Run ---
#
# This is our "Hello World" example. We have a simple sentence containing
# the information we want to extract.

unstructured_text = "My name is James Brendamour. I'm a 42-year-old Computer Scientist and the founder of Bmours Solutions LLC."

print("\n[b cyan]Unstructured Input Text:[/b cyan]")
print(f"[italic grey50]'{unstructured_text}'[/italic grey50]")


# Now, we call the `run` method. We provide:
# - The Pydantic model we want to populate (`User`)
# - The unstructured text (`unstructured_text`)
# Pydantic AI will automatically generate a prompt, send it to the LLM,
# receive the response, and parse it into an instance of your User class.
try:
    print("\n[b yellow]🤖 Calling PydanticAI to extract structured data...[/b yellow]")
    user_object = ai.run(output_model=User, prompt=unstructured_text)

    # --- 5. Inspect the Result ---
    #
    # The result is not just a dictionary; it's a fully-validated Pydantic object.
    # You can access its attributes with dot notation, and you can be sure the
    # data types are correct (e.g., `age` is an `int`, not a string).

    print("\n[b green]✨ Structured Output Received:[/b green]")
    print(user_object)

    print("\n[b cyan]Type of the output object:[/b cyan]")
    print(type(user_object))

    print("\n[b cyan]Accessing data with type safety:[/b cyan]")
    print(f"Name: {user_object.name} (type: {type(user_object.name).__name__})")
    print(f"Age: {user_object.age} (type: {type(user_object.age).__name__})")
    print(f"Role: {user_object.role} (type: {type(user_object.role).__name__})")

except Exception as e:
    print(f"\n[red]✗ An error occurred during the PydanticAI run:[/red] {e}")
