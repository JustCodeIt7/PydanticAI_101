import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field

# 1. Define Mock Database Connection (for demonstration)
class DatabaseConn:
    async def customer_balance(self, id: int, include_pending: bool) -> float:
        # In a real scenario, this would query a database
        print(f"Querying database for customer {id}, include_pending={include_pending}")
        # Simulate different balances based on include_pending
        if include_pending:
            return 1250.75
        else:
            return 1200.50

# 2. Define Dependency Container
@dataclass
class SupportDependencies:
    customer_id: int
    db: DatabaseConn

# 3. Define Output Model (Optional but good practice)
# For this specific tool returning a float, we could use output_type=float
# But let's define a simple model for illustration if needed later
class BalanceResponse(BaseModel):
    balance: float = Field(..., description="The customer's account balance")


# 4. Associate Dependency Type with Agent
support_agent = Agent(
    # Replace with your preferred LLM provider and model
    # e.g., 'openai:gpt-4o', 'ollama/llama3', 'groq/llama3-70b-8192'
    'openai:gpt-4o',
    deps_type=SupportDependencies,
    # Using float directly as the tool returns float
    output_type=float
    # Or use the Pydantic model: output_type=BalanceResponse
)

# 5. Access Dependencies in Tool via RunContext
@support_agent.tool
async def customer_balance(
    ctx: RunContext[SupportDependencies], # Receive RunContext, typed with dependencies
    include_pending: bool = Field(False, description="Whether to include pending transactions")
) -> float:
    """Returns the customer's current account balance."""
    print(f"Tool called: customer_balance(include_pending={include_pending})")
    # Access dependencies via ctx.deps
    balance = await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
        include_pending=include_pending,
    )
    print(f"Balance retrieved from DB: {balance}")
    return balance

# 6. Provide Dependencies at Runtime
async def main():
    # Create an instance of the dependencies
    db_connection = DatabaseConn() # Initialize database connection
    dependencies = SupportDependencies(customer_id=123, db=db_connection)

    print("Running agent without pending transactions...")
    # Pass dependencies when running the agent
    result_no_pending = await support_agent.run(
        'What is my current balance?',
        deps=dependencies
    )
    print(f"Agent Result (no pending): {result_no_pending}")
    # If using BalanceResponse output_type: print(f"Agent Result: {result.output.balance}")

    print("\\nRunning agent with pending transactions...")
    result_with_pending = await support_agent.run(
        'What is my current balance including pending transactions?',
        deps=dependencies
    )
    print(f"Agent Result (with pending): {result_with_pending}")


if __name__ == "__main__":
    asyncio.run(main())
