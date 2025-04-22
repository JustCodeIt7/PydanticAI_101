\
import asyncio
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from typing import Optional

# --- Mock Database ---
# In a real application, this would interact with an actual database.
class MockDatabaseConn:
    async def customer_balance(self, id: int, include_pending: bool) -> float:
        print(f"\\n--- Mock DB: Fetching balance for customer {id} (include_pending={include_pending}) ---")
        # Simulate fetching data
        await asyncio.sleep(0.1) # Simulate I/O
        if id == 123:
            return 1500.75 if include_pending else 1450.50
        elif id == 456:
            return 50.00 if include_pending else 45.00
        else:
            return 0.0

    async def get_customer_name(self, id: int) -> Optional[str]:
        print(f"\\n--- Mock DB: Fetching name for customer {id} ---")
        await asyncio.sleep(0.1) # Simulate I/O
        if id == 123:
            return "Alice"
        elif id == 456:
            return "Bob"
        else:
            return None

# --- Step 1: Define Dependency Container ---
@dataclass
class SupportDependencies:
    """Container for dependencies needed by support tools."""
    customer_id: int
    db: MockDatabaseConn # Use the mock database connection

# --- Define Output Model (as mentioned in Step 2 description) ---
class SupportOutput(BaseModel):
    """Structured output for the support agent."""
    response: str = Field(description="The final response to the customer")

# --- Step 2: Associate Dependency Type with Agent ---
support_agent = Agent(
    'openai:gpt-4o', # Replace with your preferred model if needed
    system_prompt="You are a helpful bank support assistant. Use the available tools to answer customer queries.",
    deps_type=SupportDependencies, # Associate the dependency type
    output_type=SupportOutput # Specify the desired output structure
)

# --- Step 3: Access Dependencies via RunContext in a Tool ---
@support_agent.tool
async def get_customer_balance(
    ctx: RunContext[SupportDependencies], # Receive RunContext, typed with the dependency class
    include_pending: bool = Field(False, description="Whether to include pending transactions in the balance")
) -> float:
    """
    Returns the current account balance for the customer associated with this support session.
    Use this tool whenever the customer asks about their balance.
    """
    print(f"\\n--- Tool: get_customer_balance called (include_pending={include_pending}) ---")
    # Access dependencies via ctx.deps
    balance = await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id, # Get customer_id from dependencies
        include_pending=include_pending,
    )
    print(f"--- Tool: Returning balance: {balance} ---")
    return balance

@support_agent.tool
async def get_customer_name(
    ctx: RunContext[SupportDependencies] # Receive RunContext
) -> str:
    """Returns the name of the customer associated with this support session."""
    print("\\n--- Tool: get_customer_name called ---")
    name = await ctx.deps.db.get_customer_name(id=ctx.deps.customer_id)
    if name:
        print(f"--- Tool: Returning name: {name} ---")
        return name
    else:
        print("--- Tool: Customer not found ---")
        return "Customer not found"


# --- Step 4: Provide Dependencies at Runtime ---
async def run_agent_with_deps(customer_id: int, query: str):
    print(f"\\n=== Running Agent for Customer {customer_id} ===")
    print(f"Query: {query}")

    # Create an instance of the dependencies for this specific run
    db_connection = MockDatabaseConn() # Initialize database connection (mock)
    dependencies = SupportDependencies(customer_id=customer_id, db=db_connection)

    # Pass the dependencies instance when running the agent
    # Using run_sync for simplicity here, but run (await) works the same way
    try:
        result = await support_agent.run(
            query,
            deps=dependencies # Provide the dependencies instance
        )
        print("\\n--- Agent Result ---")
        print(f"Output Type: {type(result.output)}")
        print(f"Output Value: {result.output}")
        # Access structured output fields
        if isinstance(result.output, SupportOutput):
            print(f"Formatted Response: {result.output.response}")

    except Exception as e:
        print("\n--- Agent Error ---") # Removed unnecessary f-string
        print(f"An error occurred: {e}")

    print("=" * 30)


async def main():
    # Example 1: Alice asks for her balance
    await run_agent_with_deps(
        customer_id=123,
        query="Hi, can you tell me my current account balance, including any pending transactions?"
    )

    # Example 2: Bob asks for his name and balance (without pending)
    await run_agent_with_deps(
        customer_id=456,
        query="What's my name and my balance without pending stuff?"
    )

    # Example 3: Unknown customer
    await run_agent_with_deps(
        customer_id=999,
        query="What is my balance?"
    )

if __name__ == "__main__":
    # Ensure API keys (e.g., OPENAI_API_KEY) are set as environment variables
    # or configure authentication appropriately.
    asyncio.run(main())
