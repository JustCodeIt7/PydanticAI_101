# pydantic_ai_agents_tutorial.py
import asyncio
from datetime import date, timedelta, datetime
from dataclasses import dataclass
from typing import List, Any # Added Any for some generic contexts
from typing_extensions import TypedDict # For older Python versions if needed

from pydantic import BaseModel, Field

# PydanticAI imports
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.exceptions import UsageLimitExceeded, UnexpectedModelBehavior
from pydantic_ai.messages import (
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPartDelta,
    ToolCallPartDelta,
    UserPromptPart, # Added for clarity in examples
    ToolCallPart,   # Added for clarity in examples
    RetryPromptPart # Added for clarity in examples
)
from pydantic_ai.models import ModelRequest, ModelResponse # Added for clarity
from pydantic_ai.pydantic_graph import End, BaseNode  # For iterating graph nodes
from pydantic_ai.agent import AgentRun, UserPromptNode, ModelRequestNode, CallToolsNode, capture_run_messages # For iterating graph nodes
from pydantic_ai.tools import tool # Can also use @agent.tool_plain
from pydantic_ai.usage import Usage, UsageLimits
from pydantic_ai.settings import ModelSettings

# Model-specific settings (example)
from pydantic_ai.models.gemini import GeminiModelSettings
from pydantic_ai.models.openai import OpenAIModelSettings # Example if needed

# Import for fake database
from fake_database import DatabaseConn

# --- Presenter Note ---
# Before starting, remind viewers that to run these examples, they need:
# 1. pydantic-ai installed (`pip install pydantic-ai`).
# 2. LLM provider libraries installed (e.g., `pip install openai anthropic google-generativeai`).
# 3. API keys configured as environment variables (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY).
# The model strings like 'openai:gpt-4o' will use these settings.
# Some examples might take a few seconds to run due to LLM calls.
# For models other than 'test', ensure you have the necessary API access.

# --- Global Helper ---
def print_section_header(title):
    print("\n" + "=" * 80)
    print(f"--- {title} ---")
    print("=" * 80 + "\n")

# --- Section 1: Introduction to PydanticAI Agents (Conceptual) ---
# Presenter: Explain what PydanticAI Agents are, their components (System Prompts, Tools, Output, Dependencies, LLM, Settings),
# and the analogy to FastAPI apps for reusability. This section is mostly conceptual.

# --- Section 2: Basic Agent Creation and Running ---
async def section_2_basic_agent():
    print_section_header("Section 2: Basic Agent Creation and Running (Roulette Example)")
    # Presenter: Introduce the first example: a roulette wheel agent.
    # Explain Agent instantiation, deps_type, output_type, system_prompt.
    # Explain @agent.tool and RunContext.
    # Demonstrate agent.run_sync().

    # roulette_wheel.py
    roulette_agent = Agent(
        model_id='openai:gpt-4o', # Replace with 'test' if you don't want to make API calls
        deps_type=int,
        output_type=bool,
        system_prompt=(
            'Use the `roulette_wheel` function to see if the '
            'customer has won based on the number they provide.'
        ),
    )

    @roulette_agent.tool
    async def roulette_wheel(ctx: RunContext[int], square: int) -> str:
        """check if the square is a winner"""
        # Presenter: Explain that ctx.deps holds the dependency passed to the agent run.
        print(f"Tool 'roulette_wheel' called with square: {square}, winning number (deps): {ctx.deps}")
        return 'winner' if square == ctx.deps else 'loser'

    success_number = 18
    print(f"Winning number is set to: {success_number}")

    # Presenter: Explain that the LLM will interpret the prompt and decide to call the tool.
    # The output_type=bool means PydanticAI will try to convert the LLM's final text response to a boolean.
    print("\nRunning agent for 'Put my money on square eighteen'...")
    result = await roulette_agent.run('Put my money on square eighteen', deps=success_number)
    print(f"Agent raw output: {result.raw_output}") # Show raw output before Pydantic coercion
    print(f"Agent validated output (bool): {result.output}")
    # Expected: True (or similar, LLM might phrase it, then PydanticAI coerces to bool)

    print("\nRunning agent for 'I bet five is the winner'...")
    result = await roulette_agent.run('I bet five is the winner', deps=success_number)
    print(f"Agent raw output: {result.raw_output}")
    print(f"Agent validated output (bool): {result.output}")
    # Expected: False

# --- Section 3: Different Ways to Run an Agent ---
async def section_3_running_agents():
    print_section_header("Section 3: Different Ways to Run an Agent")
    # Presenter: Explain the four ways to run an agent, focusing on run_sync, run, and run_stream.

    # run_agent.py
    # Using 'test' model to avoid API calls for this simple example,
    # as the focus is on the run methods, not complex LLM interaction.
    # For real LLM behavior, use a model like 'openai:gpt-4o'.
    agent = Agent(model_id='openai:gpt-4o') # Change to 'test' for no API calls

    # 1. agent.run_sync()
    # Presenter: This is a synchronous call, good for scripts or non-async environments.
    print("Running agent.run_sync('What is the capital of Italy?')...")
    try:
        result_sync = agent.run_sync('What is the capital of Italy?')
        print(f"run_sync output: {result_sync.output}")
    except Exception as e:
        print(f"Error with run_sync: {e}. (This might happen if 'test' model is used or API key is missing)")


    # 2. agent.run()
    # Presenter: This is an asynchronous call, returns a coroutine.
    print("\nRunning await agent.run('What is the capital of France?')...")
    try:
        result_async = await agent.run('What is the capital of France?')
        print(f"run output: {result_async.output}")
    except Exception as e:
        print(f"Error with run: {e}. (This might happen if 'test' model is used or API key is missing)")

    # 3. agent.run_stream()
    # Presenter: This is for streaming responses, useful for real-time updates.
    print("\nRunning agent.run_stream('What is the capital of the UK?')...")
    try:
        async with agent.run_stream('What is the capital of the UK?') as response_stream:
            # get_output waits for the full response from the stream
            # For token-by-token streaming, iterate over `response_stream.events()`
            # (More on this in the dedicated streaming section)
            streamed_output = await response_stream.get_output()
            print(f"run_stream output (via get_output()): {streamed_output}")

            # Example of iterating events for more granular streaming:
            print("Streaming events for 'What is the capital of Germany?':")
            async with agent.run_stream('What is the capital of Germany?') as rs:
                async for event in rs.events():
                    if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                        print(f"  Stream delta: {event.delta.content_delta!r}")
                    elif isinstance(event, FinalResultEvent):
                        print(f"  Stream final result: {event.output!r}")
    except Exception as e:
        print(f"Error with run_stream: {e}. (This might happen if 'test' model is used or API key is missing)")

    # Presenter: Briefly mention agent.iter() as the fourth way, which will be covered next.

# --- Section 4: Iterating Over an Agent's Graph ---
async def section_4_iterating_graph():
    print_section_header("Section 4: Iterating Over an Agent's Graph")
    # Presenter: Explain pydantic-graph and how Agent.iter() provides fine-grained control.

    agent = Agent(model_id='openai:gpt-4o') # Change to 'test' for no API calls

    # 4a. `async for` iteration
    # Presenter: Show how to use `async for` with `agent.iter()`.
    print("Iterating with 'async for' for 'What is the capital of Spain?'...")
    try:
        nodes_history_async_for = []
        async with agent.iter('What is the capital of Spain?') as agent_run_async_for:
            async for node in agent_run_async_for:
                nodes_history_async_for.append(node)
                print(f"  Node executed (async for): {type(node).__name__}")
                if isinstance(node, UserPromptNode):
                    print(f"    UserPrompt: {node.user_prompt}")
                # Add more specific node checks if needed for demonstration

        print(f"Total nodes (async for): {len(nodes_history_async_for)}")
        # Presenter: Explain the typical sequence of nodes (UserPromptNode, ModelRequestNode, CallToolsNode/ModelResponseNode, End).
        # The exact nodes can vary based on whether tools are called.
        if agent_run_async_for.result:
            print(f"Final output (async for): {agent_run_async_for.result.output}")
            print(f"Usage (async for): {agent_run_async_for.usage()}")
        else:
            print("Agent run did not complete successfully or 'test' model used.")
    except Exception as e:
        print(f"Error with 'async for' iteration: {e}")


    # 4b. Using `.next(...)` manually
    # Presenter: Show how to manually drive iteration using `agent_run.next()`.
    print("\nIterating manually with '.next()' for 'What is the capital of Portugal?'...")
    try:
        async with agent.iter('What is the capital of Portugal?') as agent_run_manual:
            current_node = agent_run_manual.next_node
            all_nodes_manual = [current_node]
            print(f"  Initial node (manual): {type(current_node).__name__}")

            while not isinstance(current_node, End):
                current_node = await agent_run_manual.next(current_node)
                all_nodes_manual.append(current_node)
                print(f"  Node executed (manual): {type(current_node).__name__}")
                if isinstance(node, UserPromptNode):
                    print(f"    UserPrompt: {node.user_prompt}")


            print(f"Total nodes (manual): {len(all_nodes_manual)}")
            if agent_run_manual.result:
                print(f"Final output (manual): {agent_run_manual.result.output}")
            else:
                print("Agent run did not complete successfully or 'test' model used.")
    except Exception as e:
        print(f"Error with manual '.next()' iteration: {e}")

# --- Section 5: Streaming Agent Runs (Detailed) ---
async def section_5_streaming_detailed():
    print_section_header("Section 5: Streaming Agent Runs (Detailed Weather Example)")
    # Presenter: Deep dive into streaming with a more complex example involving tools.

    @dataclass
    class WeatherService:
        async def get_forecast(self, location: str, forecast_date: date) -> str:
            # In real code: call weather API, DB queries, etc.
            return f'The forecast in {location} on {forecast_date} is 24°C and sunny.'

        async def get_historic_weather(self, location: str, forecast_date: date) -> str:
            # In real code: call a historical weather API or DB
            return (
                f'The weather in {location} on {forecast_date} was 18°C and partly cloudy.'
            )

    weather_agent = Agent[WeatherService, str](
        model_id='openai:gpt-4o', # Needs a capable model for tool use
        deps_type=WeatherService,
        output_type=str,
        system_prompt='Providing a weather forecast at the locations the user provides. Always use the weather_forecast tool.',
    )

    @weather_agent.tool
    async def weather_forecast(
        ctx: RunContext[WeatherService],
        location: str,
        forecast_date: date, # PydanticAI will handle string-to-date conversion
    ) -> str:
        # Presenter: Explain how the LLM needs to provide 'forecast_date' in a format Pydantic can parse (e.g., YYYY-MM-DD).
        print(f"Tool 'weather_forecast' called for {location} on {forecast_date}")
        # Make forecast_date dynamic for the example
        # If the LLM provides a relative date like "Tuesday", PydanticAI/LLM must resolve it.
        # For this tool, we assume an absolute date is passed.
        if forecast_date >= date.today():
            return await ctx.deps.get_forecast(location, forecast_date)
        else:
            return await ctx.deps.get_historic_weather(location, forecast_date)

    output_messages_log: list[str] = []

    # Let's make the date dynamic: 7 days from today
    future_date = date.today() + timedelta(days=7)
    user_prompt = f'What will the weather be like in Paris on {future_date.strftime("%Y-%m-%d")}?'
    # Presenter: Note that the prompt now includes a specific date.
    # The LLM should extract "Paris" and this date for the tool.

    print(f"Running streaming example with prompt: \"{user_prompt}\"")
    try:
        async with weather_agent.iter(user_prompt, deps=WeatherService()) as run:
            async for node in run:
                if Agent.is_user_prompt_node(node):
                    msg = f'=== UserPromptNode: {node.user_prompt} ==='
                    print(msg)
                    output_messages_log.append(msg)
                elif Agent.is_model_request_node(node):
                    msg = '=== ModelRequestNode: streaming partial request tokens ==='
                    print(msg)
                    output_messages_log.append(msg)
                    async with node.stream(run.ctx) as request_stream:
                        async for event in request_stream:
                            if isinstance(event, PartStartEvent):
                                m = f'[Request] Starting part {event.index}: {event.part!r}'
                                print(f"  {m}")
                                output_messages_log.append(m)
                            elif isinstance(event, PartDeltaEvent):
                                if isinstance(event.delta, TextPartDelta):
                                    m = f'[Request] Part {event.index} text delta: {event.delta.content_delta!r}'
                                    print(f"  {m}")
                                    output_messages_log.append(m)
                                elif isinstance(event.delta, ToolCallPartDelta):
                                    m = f'[Request] Part {event.index} args_delta={event.delta.args_delta!r}'
                                    print(f"  {m}")
                                    output_messages_log.append(m)
                            elif isinstance(event, FinalResultEvent): # This event is for when the model directly produces a final output
                                m = f'[Result] The model produced a final output (tool_name={event.tool_name})'
                                print(f"  {m}")
                                output_messages_log.append(m)
                elif Agent.is_call_tools_node(node):
                    msg = '=== CallToolsNode: streaming partial response & tool usage ==='
                    print(msg)
                    output_messages_log.append(msg)
                    async with node.stream(run.ctx) as handle_stream:
                        async for event in handle_stream:
                            if isinstance(event, FunctionToolCallEvent):
                                m = f'[Tools] The LLM calls tool={event.part.tool_name!r} with args={event.part.args} (tool_call_id={event.part.tool_call_id!r})'
                                print(f"  {m}")
                                output_messages_log.append(m)
                            elif isinstance(event, FunctionToolResultEvent):
                                m = f'[Tools] Tool call {event.tool_call_id!r} returned => {event.result.content!r}'
                                print(f"  {m}")
                                output_messages_log.append(m)
                            # Other events like PartStartEvent, PartDeltaEvent for the *response* text can also occur here
                            elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                                m = f'[Response Text Delta] {event.delta.content_delta!r}' # LLM generating text after tool call
                                print(f"  {m}")
                                output_messages_log.append(m)


                elif Agent.is_end_node(node):
                    if run.result: # Check if result is available
                        assert run.result.output == node.data.output
                        msg = f'=== Final Agent Output: {run.result.output} ==='
                        print(msg)
                        output_messages_log.append(msg)
                    else:
                        msg = '=== EndNode reached, but run.result is not yet populated or error occurred ==='
                        print(msg)
                        output_messages_log.append(msg)

        print("\n--- Full Log of Streaming Events ---")
        for log_entry in output_messages_log:
            print(log_entry)
        # Presenter: The output log will show the sequence of user prompt, model requests (possibly for tool calls),
        # tool execution, further model requests (to summarize tool output), and the final answer.
        # The exact sequence and content depend on the LLM's behavior.

    except Exception as e:
        print(f"Error during detailed streaming example: {e}")
        import traceback
        traceback.print_exc()


# --- Section 6: Additional Configuration ---
async def section_6_configuration():
    print_section_header("Section 6: Additional Configuration")

    # 6a. Usage Limits
    # Presenter: Explain UsageLimits for controlling token and request counts.
    print("--- 6a. Usage Limits ---")
    agent_for_limits = Agent(model_id='anthropic:claude-3-5-sonnet-latest') # Using Anthropic as per docs

    print("\nTesting response_tokens_limit (expect success)...")
    try:
        result_sync_tokens_ok = agent_for_limits.run_sync(
            'What is the capital of Italy? Answer with just the city.',
            usage_limits=UsageLimits(response_tokens_limit=10),
        )
        print(f"Output (tokens_ok): {result_sync_tokens_ok.output}")
        print(f"Usage (tokens_ok): {result_sync_tokens_ok.usage()}")
    except UsageLimitExceeded as e:
        print(f"Error (tokens_ok): {e}")
    except Exception as e:
        print(f"General Error (tokens_ok): {e} (Ensure Anthropic API key is set and model is accessible)")


    print("\nTesting response_tokens_limit (expect failure)...")
    try:
        result_sync_tokens_fail = agent_for_limits.run_sync(
            'What is the capital of Italy? Answer with a paragraph.',
            usage_limits=UsageLimits(response_tokens_limit=10),
        )
        print(f"Output (tokens_fail): {result_sync_tokens_fail.output}") # Should not reach here
    except UsageLimitExceeded as e:
        print(f"Successfully caught expected error (tokens_fail): {e}")
    except Exception as e:
        print(f"General Error (tokens_fail): {e}")


    # Presenter: Explain request_limit for preventing loops.
    class NeverOutputType(TypedDict):
        never_use_this: str

    # This agent is designed to potentially loop or retry excessively.
    agent_for_request_limit = Agent(
        model_id='anthropic:claude-3-5-sonnet-latest', # Needs a model that supports tool use and retries
        retries=3, # Agent-level retries
        output_type=NeverOutputType, # An output type the LLM will struggle to produce
        system_prompt='Any time you get a response, call the `infinite_retry_tool` to produce another response. You must conform to NeverOutputType.',
    )

    @agent_for_request_limit.tool(retries=5) # Tool-level retries
    def infinite_retry_tool() -> int:
        # Presenter: This tool always asks the model to retry.
        print("Tool 'infinite_retry_tool' called, raising ModelRetry.")
        raise ModelRetry('Please try again from the tool.')

    print("\nTesting request_limit (expect UsageLimitExceeded)...")
    try:
        # The combination of system prompt, difficult output type, and retrying tool
        # should lead to multiple requests. We limit it to 3.
        result_sync_requests = agent_for_request_limit.run_sync(
            'Begin infinite retry loop!',
            usage_limits=UsageLimits(request_limit=3)
        )
        print(f"Output (request_limit): {result_sync_requests.output}") # Should not reach here
    except UsageLimitExceeded as e:
        print(f"Successfully caught expected error (request_limit): {e}")
    except UnexpectedModelBehavior as e: # Could also be this if retries are exhausted first
         print(f"Caught UnexpectedModelBehavior (request_limit may not have been hit if retries exhausted first): {e}")
    except Exception as e:
        print(f"General Error (request_limit): {e}")


    # 6b. Model Settings
    # Presenter: Explain ModelSettings for fine-tuning requests (e.g., temperature).
    print("\n--- 6b. Model Settings ---")
    agent_for_model_settings = Agent(model_id='openai:gpt-4o')

    print("\nRunning with temperature 0.0 for deterministic output...")
    try:
        result_temp_0 = agent_for_model_settings.run_sync(
            'What is the capital of Italy?', model_settings={'temperature': 0.0}
        )
        print(f"Output (temp 0.0): {result_temp_0.output}")

        # Example of setting default model settings for an agent
        agent_with_default_settings = Agent(
            model_id='openai:gpt-4o',
            model_settings=OpenAIModelSettings(temperature=0.1, max_tokens=50) # Example for OpenAI
        )
        print("\nRunning agent with default low temperature and max_tokens...")
        result_default_settings = await agent_with_default_settings.run(
            'Tell me a very short story about a robot.'
        )
        print(f"Output (default settings): {result_default_settings.output}")
        print(f"Usage (default settings): {result_default_settings.usage()}")

    except Exception as e:
        print(f"Error with model settings example: {e}")


    # 6c. Model Specific Settings
    # Presenter: Show how to use model-specific settings (e.g., Gemini safety settings).
    print("\n--- 6c. Model Specific Settings (Gemini Example) ---")
    # This example requires a Google API key and `google-generativeai`
    agent_gemini = Agent(model_id='google:gemini-1.5-flash-latest') # Updated model name

    print("\nRunning Gemini agent with specific safety settings (expect potential block)...")
    try:
        # This prompt is designed to potentially trigger safety filters.
        result_gemini = agent_gemini.run_sync(
            'Write a list of 5 very rude things that I might say to the universe after stubbing my toe in the dark:',
            model_settings=GeminiModelSettings(
                temperature=0.0,
                safety_settings=[ # Corrected field name from gemini_safety_settings
                    {
                        'category': 'HARM_CATEGORY_HARASSMENT',
                        'threshold': 'BLOCK_LOW_AND_ABOVE',
                    },
                    {
                        'category': 'HARM_CATEGORY_HATE_SPEECH',
                        'threshold': 'BLOCK_LOW_AND_ABOVE',
                    },
                ],
            ),
        )
        print(f"Output (Gemini safety): {result_gemini.output}")
    except UnexpectedModelBehavior as e:
        print(f"Successfully caught expected error (Gemini safety): {e}")
        # Presenter: Explain that this error indicates the safety settings were triggered.
    except Exception as e:
        print(f"General Error (Gemini safety): {e} (Ensure Google API key and correct model name)")


# --- Section 7: Managing Conversations ---
async def section_7_conversations():
    print_section_header("Section 7: Managing Conversations")
    # Presenter: Explain how to maintain conversation context using message_history.

    agent = Agent(model_id='openai:gpt-4o')

    print("Starting a conversation...")
    try:
        # First run
        print("\nRun 1: 'Who was Albert Einstein?'")
        result1 = await agent.run('Who was Albert Einstein?')
        print(f"Output 1: {result1.output}")

        # Second run, passing previous messages
        # Presenter: Emphasize `result1.new_messages()` for context.
        print("\nRun 2: 'What was his most famous equation?' (with history)")
        result2 = await agent.run(
            'What was his most famous equation?',
            message_history=result1.new_messages(),
        )
        print(f"Output 2: {result2.output}")

        # Third run, without history (for comparison)
        print("\nRun 3: 'What was his most famous equation?' (WITHOUT history)")
        result3 = await agent.run('What was his most famous equation?')
        print(f"Output 3 (no history): {result3.output}")
        # Presenter: Note how Output 3 likely won't know who "his" refers to.
    except Exception as e:
        print(f"Error during conversation example: {e}")

# --- Section 8: Type Safety ---
def section_8_type_safety():
    print_section_header("Section 8: Type Safety")
    # Presenter: Explain PydanticAI's design for static type checking.
    # Show the example script and mention running mypy/pyright on it.

    # type_mistakes.py (content)
    type_mistakes_code = """
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext

@dataclass
class User:
    name: str

# Using 'test' model as LLM interaction is not the point here.
agent = Agent(
    model_id='test',
    deps_type=User,  # (1)! Agent expects User dependency
    output_type=bool,
)

@agent.system_prompt
def add_user_name(ctx: RunContext[str]) -> str:  # (2)! Type mismatch: RunContext[str] vs expected RunContext[User]
    return f"The user's name is {ctx.deps}." # This would be a User object, not str

def foobar(x: bytes) -> None:
    pass

# This run call is okay type-wise for deps
result = agent.run_sync('Does their name start with "A"?', deps=User('Anne'))

# (3)! Type mismatch: result.output is bool, foobar expects bytes
# foobar(result.output) # This line would cause a type error, uncomment to see with mypy/pyright

# To make it runnable without type errors at runtime for this demo, let's not call foobar
print(f"Agent output (bool): {result.output}")
print("Presenter: The 'type_mistakes.py' example is designed to show errors with a static type checker like mypy.")
print("Presenter: The @agent.system_prompt decorator for 'add_user_name' has a RunContext[str] type hint,")
print("           but the agent is defined with deps_type=User. This is a type mismatch.")
print("Presenter: The call to foobar(result.output) would also be a type error if uncommented,")
print("           as result.output is bool and foobar expects bytes.")
"""
    print("--- Code for type_mistakes_example (for demonstration) ---")
    print(type_mistakes_code)
    print("--- End of type_mistakes_example code ---")

    # Presenter: To demonstrate, save the above code as `type_mistakes_example.py`
    # Then run `mypy type_mistakes_example.py` or `pyright type_mistakes_example.py`.
    # Expected mypy output (example):
    # type_mistakes_example.py:14: error: Argument 1 to "system_prompt" of "Agent" has incompatible type "Callable[[RunContext[str]], str]"; expected "Callable[[RunContext[User]], str]"  [arg-type]
    # type_mistakes_example.py:23: error: Argument 1 to "foobar" has incompatible type "bool"; expected "bytes"  [arg-type] (if uncommented)

    # For the tutorial script, we'll run a simplified version here to avoid actual type errors during the run
    @dataclass
    class UserForDemo:
        name: str

    agent_type_demo = Agent(
        model_id='test',
        deps_type=UserForDemo,
        output_type=bool,
    )

    # Corrected system prompt for demo purposes
    @agent_type_demo.system_prompt
    def add_user_name_correct(ctx: RunContext[UserForDemo]) -> str:
        return f"The user's name is {ctx.deps.name}."

    result_type_demo = agent_type_demo.run_sync('Does their name start with "A"?', deps=UserForDemo('Anne'))
    print(f"Agent output (type_demo): {result_type_demo.output}")


# --- Section 9: System Prompts vs. Instructions ---
async def section_9_prompts_instructions():
    print_section_header("Section 9: System Prompts vs. Instructions")

    # 9a. System Prompts (Static and Dynamic)
    # Presenter: Explain static and dynamic system prompts.
    print("--- 9a. System Prompts ---")
    agent_sp = Agent(
        model_id='openai:gpt-4o',
        deps_type=str, # Dependency: user's name
        system_prompt="Use the customer's name while replying to them.", # Static
    )

    @agent_sp.system_prompt # Dynamic
    def add_the_users_name_sp(ctx: RunContext[str]) -> str:
        return f"The user's name is {ctx.deps}."

    @agent_sp.system_prompt # Dynamic, no deps needed
    def add_the_date_sp() -> str:
        # Presenter: The output date will be the current date.
        return f'The date is {date.today().isoformat()}.'

    try:
        print("\nRunning agent with system prompts (Frank)...")
        result_sp = await agent_sp.run('What is the date?', deps='Frank')
        print(f"Output (system_prompts, Frank): {result_sp.output}")
        # Expected: Something like "Hello Frank, the date today is <current_date>."
    except Exception as e:
        print(f"Error with system prompts example: {e}")

    # 9b. Instructions
    # Presenter: Explain instructions and their difference from system_prompts (especially with message_history).
    # Recommend using instructions generally.
    print("\n--- 9b. Instructions ---")
    agent_instr = Agent(
        model_id='openai:gpt-4o',
        instructions='You are a helpful assistant that can answer questions and help with tasks.',
    )
    try:
        print("\nRunning agent with static instructions...")
        result_instr_static = await agent_instr.run('What is the capital of France?')
        print(f"Output (static instructions): {result_instr_static.output}")
    except Exception as e:
        print(f"Error with static instructions example: {e}")


    # Dynamic instructions
    agent_dyn_instr = Agent(model_id='openai:gpt-4o', deps_type=str)

    @agent_dyn_instr.instructions
    def add_the_users_name_instr(ctx: RunContext[str]) -> str:
        return f"The user's name is {ctx.deps}."

    @agent_dyn_instr.instructions
    def add_the_date_instr() -> str:
        return f'The date is {date.today().isoformat()}.'

    try:
        print("\nRunning agent with dynamic instructions (Alice)...")
        result_dyn_instr = await agent_dyn_instr.run('What is the date?', deps='Alice')
        print(f"Output (dynamic instructions, Alice): {result_dyn_instr.output}")
    except Exception as e:
        print(f"Error with dynamic instructions example: {e}")

# --- Section 10: Reflection and Self-Correction ---
async def section_10_reflection_self_correction():
    print_section_header("Section 10: Reflection and Self-Correction (Tool Retry)")
    # Presenter: Explain ModelRetry and how validation errors or explicit raises can trigger retries.
    # This example uses the fake_database.py

    class ChatResult(BaseModel):
        user_id: int
        message: str

    # Ensure DatabaseConn is available (defined in fake_database.py)
    agent_retry = Agent(
        model_id='openai:gpt-4o', # Needs a model good at following instructions and using tools
        deps_type=DatabaseConn,
        output_type=ChatResult,
        system_prompt="You are a messaging assistant. Use get_user_by_name to find the user_id. Then formulate the message."
    )

    # Presenter: Explain that `retries=2` on the tool means it can retry twice after the initial call.
    @agent_retry.tool(retries=2)
    def get_user_by_name(ctx: RunContext[DatabaseConn], name: str) -> int:
        """Get a user's ID from their full name."""
        print(f"Tool 'get_user_by_name' called with name: {name!r}. Retry count: {ctx.retry_count}")
        user_id = ctx.deps.users.get(name=name) # Accessing users table from DatabaseConn
        if user_id is None:
            # Presenter: This ModelRetry tells the LLM to try again, providing a hint.
            raise ModelRetry(
                f'No user found with name {name!r}, remember to provide their full name. For example, "John Doe".'
            )
        return user_id

    print("\nRunning agent with tool that might retry (John Doe)...")
    # The LLM might first try "John" then "John Doe" if the system prompt and retry message guide it.
    try:
        # Using a real DatabaseConn instance
        db_conn = DatabaseConn()
        result_retry = await agent_retry.run(
            'Send a message to John Doe asking for coffee next week', deps=db_conn
        )
        print(f"Output (tool_retry): {result_retry.output}")
        # Expected: ChatResult(user_id=123, message='...')
        # Presenter: Check the console for "Tool 'get_user_by_name' called..." messages to see retries.
    except UnexpectedModelBehavior as e:
        print(f"UnexpectedModelBehavior during tool retry example: {e}")
        print(f"Cause: {e.__cause__}")
    except Exception as e:
        print(f"Error during tool retry example: {e}")
        import traceback
        traceback.print_exc()

# --- Section 11: Handling Model Errors ---
async def section_11_model_errors():
    print_section_header("Section 11: Handling Model Errors")
    # Presenter: Explain UnexpectedModelBehavior and capture_run_messages for diagnostics.

    agent_model_error = Agent(
        model_id='openai:gpt-4o', # Or 'test' if the tool logic itself is the focus
        retries=1 # Agent level retries
    )

    # Presenter: This tool is designed to fail unless size is 42.
    # With agent retries=1, it will try once, fail, retry once, fail again.
    @agent_model_error.tool() # Default tool retries is 1, agent retries is 1
    def calc_volume(size: int) -> int:
        print(f"Tool 'calc_volume' called with size: {size}")
        if size == 42:
            return size**3
        else:
            raise ModelRetry('The size is not the answer to life, the universe, and everything. Please try again with the correct size.')

    print("\nRunning agent with a tool designed to cause retries and potentially fail...")
    with capture_run_messages() as messages_context:
        try:
            result_error_handling = await agent_model_error.run(
                'Please get me the volume of a box with size 6.'
            )
            print(f"Output (model_errors): {result_error_handling.output}") # Should not reach if error occurs
        except UnexpectedModelBehavior as e:
            print(f"Caught expected error (model_errors): {e}")
            print(f"Cause: {repr(e.__cause__)}") # Should be ModelRetry
            print("\n--- Captured Messages ---")
            # messages_context is a list of ModelRequest/ModelResponse objects
            for i, msg_obj in enumerate(messages_context):
                print(f"Message {i+1}: Type: {type(msg_obj).__name__}")
                if isinstance(msg_obj, ModelRequest):
                    for part_idx, part in enumerate(msg_obj.parts):
                        print(f"  Request Part {part_idx}: {type(part).__name__}, Content: {getattr(part, 'content', '')[:100]}...")
                elif isinstance(msg_obj, ModelResponse):
                     for part_idx, part in enumerate(msg_obj.parts):
                        print(f"  Response Part {part_idx}: {type(part).__name__}, Content: {getattr(part, 'content', '')[:100]}...")
                # Add more detailed printing if needed
            print("--- End of Captured Messages ---")
        except Exception as e:
            print(f"An unexpected general error occurred: {e}")


# --- Main execution block ---
async def main_tutorial():
    # Presenter: You can uncomment sections to run them one by one.
    # For a full video, you'd go through them sequentially.

    # await section_2_basic_agent()
    # await section_3_running_agents()
    # await section_4_iterating_graph()
    # await section_5_streaming_detailed() # This is a long one
    # await section_6_configuration()
    # await section_7_conversations()
    # section_8_type_safety() # This is synchronous
    # await section_9_prompts_instructions()
    # await section_10_reflection_self_correction()
    await section_11_model_errors()

    print_section_header("Tutorial Complete")
    print("Presenter: Recap key features and encourage viewers to explore PydanticAI documentation.")

if __name__ == "__main__":
    # Presenter: Explain that this script runs the selected sections.
    # Ensure viewers have their environment (API keys, libraries) set up.
    print("Starting PydanticAI Agents Tutorial Script...")
    print("NOTE: Some examples make live LLM calls and may take time or incur costs.")
    print("Ensure your API keys (e.g., OPENAI_API_KEY) are set in your environment.\n")

    # Choose which sections to run by uncommenting them in main_tutorial()
    asyncio.run(main_tutorial())