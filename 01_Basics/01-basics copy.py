# video_01_introduction.py

# --- Video 1: Welcome to PydanticAI! ---

# Objective: Introduce PydanticAI, its core value proposition,
# and set the stage for the series.

print("--- Welcome to the PydanticAI Tutorial Series! ---")

# === What is PydanticAI? ===
# PydanticAI is a Python agent framework from the Pydantic team.
# Goal: Simplify building production-grade Generative AI applications.
# Inspiration: Aims for the ergonomic design and developer experience
#             similar to FastAPI, but for the AI domain.
print("\n1. Introducing PydanticAI: Building Robust AI Apps")
print("   - Developed by the Pydantic team.")
print("   - Focuses on developer experience and production readiness.")

# === Core Principles ===

# 1. Leverages Pydantic: Built upon Pydantic's strengths in data
#    validation and type safety.
#    Example: Defining structured data using Pydantic models.
try:
    from pydantic import BaseModel, Field
    print("\n2. Core Principles:")
    print("   - Principle: Leverages Pydantic for Validation")

    class UserProfile(BaseModel):
        name: str = Field(..., description="User's full name")
        email: str = Field(..., description="User's email address")
        age: int | None = Field(None, description="User's age (optional)")

    print("     - Example: Defined a Pydantic 'UserProfile' model for structured data.")

except ImportError:
    print("\n   - (Note: Pydantic library not installed, skipping model example)")
    print("   - Principle: Leverages Pydantic for Validation")


# 2. Model-Agnostic: Supports various LLMs (OpenAI, Gemini, Anthropic, etc.).
print("   - Principle: Model-Agnostic (Supports various LLMs)")

# 3. Type Safety: Ensures robust and predictable interactions with LLMs.
print("   - Principle: Strong Emphasis on Type Safety")

# 4. Python-Centric: Uses familiar Python control flow and composition.
print("   - Principle: Python-Centric Design")

# 5. Structured Responses: Uses Pydantic models to ensure LLM outputs
#    conform to expected schemas.
print("   - Principle: Excels at Producing Structured Responses (like the UserProfile model)")

# === Powerful Features ===
print("\n3. Powerful Features for Production:")
# - Optional Dependency Injection: Enhances testability and modularity.
print("   - Optional Dependency Injection System")
# - Streaming Support: For real-time LLM responses.
print("   - Streaming LLM Responses")
# - Logfire Integration: Built-in observability.
print("   - Seamless Integration with Pydantic Logfire")
# - Pydantic Graph Support: Manage complex workflows.
print("   - Support for Pydantic Graph")


# === Common Use Cases ===
print("\n4. Potential Applications:")
print("   - Sophisticated Chatbots")
print("   - Data Analysis Assistants")
print("   - Automated Reporting Tools")
print("   - Agents Interacting with APIs")
print("   - Natural Language to SQL Generation")
print("   - Retrieval-Augmented Generation (RAG) Pipelines")

# === PydanticAI vs. Other Frameworks (e.g., LangChain, LlamaIndex) ===
print("\n5. How PydanticAI Compares:")
print("   - Focus: Simplicity, robust type safety via Pydantic, production readiness.")
print("   - Strengths: Structured output validation, maintainability, great for Pydantic/FastAPI users.")
print("   - Positioning: Often seen as a powerful tool integrated into projects,")
print("                  focusing on structured interactions and validation.")
# While others might offer broader features, they can sometimes be more complex.
# PydanticAI targets developers prioritizing reliable, structured AI applications.

# === Tutorial Series Overview ===
print("\n--- What's Next? ---")
print("This series will cover:")
print(" - Setup and Fundamentals")
print(" - Core Capabilities (LLM Interaction, Structured Output)")
print(" - Advanced Features (Streaming, DI, RAG)")
print(" - Practical Project Examples")
print(" - PydanticAI in the AI Ecosystem")

print("\n--- End of Video 1 Introduction ---")

# Note: Actual PydanticAI usage (like importing PydanticAI itself
# and interacting with an LLM) will be shown in subsequent videos.
# This script serves as a conceptual overview reflected in code comments.