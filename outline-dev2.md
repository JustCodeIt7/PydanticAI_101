# **PydanticAI Masterclass: A Comprehensive YouTube Series Outline**

## **Part 1: Introduction & Foundations**

### **Video 1: Welcome to PydanticAI\!**

**Objective:** Introduce PydanticAI, its core value proposition, and set the stage for the series.

PydanticAI emerges as a Python agent framework meticulously crafted by the Pydantic team. Its primary design goal is to alleviate the common difficulties encountered when building production-grade applications powered by Generative AI.1 Drawing inspiration from the success of FastAPI in web development, PydanticAI aims to replicate that framework's ergonomic design and intuitive developer experience within the domain of Generative AI application development.1 This focus on developer experience is central to its philosophy.

The framework is built upon several key principles that define its approach. Being developed by the Pydantic team, it inherently leverages the strengths of the Pydantic library, particularly in data validation and type safety.1 It boasts a model-agnostic architecture, supporting a wide range of Large Language Models (LLMs) from various providers.2 A strong emphasis is placed on type safety, ensuring that interactions with LLMs and the data exchanged are robust and predictable.1 The design remains Python-centric, utilizing familiar Python control flow and composition patterns, making it accessible to developers accustomed to standard Python practices.1 PydanticAI excels at producing structured responses, harnessing Pydantic's validation capabilities to ensure LLM outputs are consistent and conform to predefined schemas.1 Additional powerful features include an optional dependency injection system for enhanced testability and modularity, support for streaming LLM responses for real-time applications, seamless integration with Pydantic Logfire for observability, and support for Pydantic Graph to manage complex workflows.1

These features make PydanticAI suitable for a variety of applications. Examples include building sophisticated chatbots, data analysis assistants capable of interacting with data sources, automated reporting tools, agents designed to interact with external APIs, systems for generating SQL queries from natural language, and implementing Retrieval-Augmented Generation (RAG) pipelines.11

When considering PydanticAI in the context of other agent frameworks like LangChain or LlamaIndex, its distinct focus becomes apparent. PydanticAI emphasizes simplicity, robust type safety through Pydantic integration, and features geared towards production readiness, such as structured output validation and dependency injection.11 While other frameworks might offer a broader range of features or a larger ecosystem, they can sometimes be perceived as more complex or abstract.19 PydanticAI appears to target a niche of developers who prioritize maintainable, structured, and validation-centric code, particularly those already comfortable within the Pydantic and FastAPI ecosystems.11 Some might view it initially more as a powerful tool integrated into a project rather than an all-encompassing ecosystem 21, concentrating primarily on structured interactions and validation.11 This strategic positioning, leveraging the reputation and design principles of its sibling projects, makes it an attractive option for developers aiming to build reliable and robust AI applications for production environments.

This tutorial series will guide viewers through the framework, starting with setup and fundamental concepts, progressing to core capabilities and advanced features, demonstrating practical projects, and finally, exploring its place within the broader AI ecosystem.

### **Video 2: Setting Up Your PydanticAI Environment**

**Objective:** Guide viewers through installing PydanticAI and setting up a clean development environment.

Before installing PydanticAI, ensure the development environment meets the prerequisite of having Python version 3.9 or later installed.22 As with most Python projects, establishing an isolated environment is highly recommended to manage dependencies effectively and avoid conflicts between projects. Using Python's built-in venv module is a standard practice. A virtual environment can be created and activated using commands such as:

Bash

python \-m venv.venv  
source.venv/bin/activate \# On Linux/macOS  
\# or  
.\\.venv\\Scripts\\activate \# On Windows

Alternatively, conda environments serve the same purpose.2

Once the virtual environment is active, PydanticAI can be installed using pip, the standard Python package installer 2:

Bash

pip install pydantic-ai

This command installs the core PydanticAI library along with its essential dependencies, including the Pydantic library itself. For developers aiming for minimal installations, particularly in resource-constrained environments, a pydantic-ai-slim package is available, which excludes some optional dependencies.26

PydanticAI's design allows for modularity regarding LLM provider support. While the core library provides the framework, specific integrations might require additional dependencies. These can be installed as extras. For example, to include support for both OpenAI and Google Gemini models, the installation command would be 2:

Bash

pip install pydantic-ai\[openai,google-gla\]

Consult the official PydanticAI documentation for the specific extras required for other supported LLM providers.2 This approach of optional dependencies keeps the core installation lightweight and allows developers to include only the integrations they need, aligning with best practices for dependency management in production systems.26

For an enhanced development experience, Integrated Development Environments (IDEs) like Visual Studio Code or PyCharm are recommended. Their strong support for Python type hints complements PydanticAI's type-safe design, enabling features like autocompletion and static analysis.22

After installation, verify the setup by running a simple Python command to import the library:

Python

import pydantic_ai  
print("PydanticAI installed successfully\!")

Executing this without errors confirms that the library is correctly installed and accessible within the activated virtual environment.

### **Video 3: Configuring Your First LLM (OpenAI & Gemini)**

**Objective:** Show how to configure PydanticAI to use common LLM providers, focusing on OpenAI and Google Gemini.

To utilize LLMs through PydanticAI, API keys or other credentials provided by the LLM vendor are necessary. These keys authenticate requests and link usage to a specific account. Keys can typically be obtained from the provider's console, such as the OpenAI Console or Google AI Studio.2

Handling these keys securely is paramount. Hardcoding sensitive credentials directly into source code is strongly discouraged. The recommended practice is to use environment variables. Set variables like OPENAI_API_KEY or GOOGLE_API_KEY in the operating system environment. For local development, using a .env file in the project root to store these keys is common. Libraries like python-dotenv can automatically load variables from this file into the environment when the application starts.2 PydanticAI typically detects and uses these standard environment variables automatically for authentication.2

Once API keys are configured in the environment, specifying which LLM to use within PydanticAI is straightforward. The Agent class constructor accepts a model identifier string as its first argument. This string follows a standardized format: \<provider_name\>:\<model_name\>.1

For example, to use OpenAI's GPT-4o model:

Python

from pydantic_ai import Agent

\# Assumes OPENAI_API_KEY is set as an environment variable  
agent_openai \= Agent('openai:gpt-4o', system_prompt='Be helpful.')

Similarly, to use Google's Gemini 1.5 Flash model:

Python

from pydantic_ai import Agent

\# Assumes GOOGLE_API_KEY is set as an environment variable (or appropriate auth is configured)  
agent_gemini \= Agent('google-gla:gemini-1.5-flash', system_prompt='Be creative.')

Other examples include models from Groq, like 'groq:mixtral-8x7b-32768'.2

This consistent model specification format is a cornerstone of PydanticAI's model-agnostic design.2 It allows developers to switch between different LLM providers often by simply changing this identifier string, without needing to alter the surrounding agent logic or learn provider-specific SDKs.2 This abstraction significantly enhances flexibility and reduces vendor lock-in at the code level.

PydanticAI supports a growing list of providers beyond OpenAI and Gemini, including Anthropic, Groq, Ollama, Mistral, Cohere, Bedrock, and Deepseek.2 Configuration details and specific model identifiers for these providers can be found in the official PydanticAI documentation, particularly in the section dedicated to models.2

### **Video 4: Your First PydanticAI Agent: Hello World\!**

**Objective:** Build and run the simplest possible PydanticAI agent.

Creating a basic agent with PydanticAI requires minimal code, demonstrating the framework's focus on ease of use and reduced boilerplate. The core component is the Agent class, which needs to be imported first 1:

Python

from pydantic_ai import Agent

Next, instantiate the Agent. This requires specifying the LLM model to use, following the \<provider\>:\<model_name\> format discussed previously.1 Additionally, a static system prompt can be provided directly during instantiation using the system_prompt keyword argument. The system prompt provides high-level instructions or context to the LLM, guiding its behavior and tone throughout the interaction.1

Python

\# Example using OpenAI's GPT-4o model  
agent \= Agent(  
 'openai:gpt-4o',  
 system_prompt='Be concise, reply with one sentence.'  
)

With the agent configured, interaction is initiated using one of the run methods. For simple, synchronous execution, suitable for scripts or basic testing, the run_sync method is used. It takes the user's query as a string argument.2

Python

\# Send a query to the agent  
result \= agent.run_sync('Where does "hello world" come from?')

The run_sync method handles the interaction with the LLM. Internally, PydanticAI sends the system prompt and the user query to the specified LLM provider. The LLM processes this input and generates a response.1

The method returns an AgentRunResult object. This object contains various details about the interaction, but for this basic example, the primary interest is the LLM's response text, which can be accessed via the output attribute.2

Python

\# Print the LLM's response  
print(result.output)

The complete "Hello World" example, as often presented in the documentation 2, is remarkably concise:

Python

\# hello_world.py  
from pydantic_ai import Agent

\# Configure the agent  
agent \= Agent(  
 'openai:gpt-4o', \# Replace with your desired model if needed  
 system_prompt='Be concise, reply with one sentence.'  
)

\# Run the agent synchronously  
result \= agent.run_sync('Where does "hello world" come from?')

\# Print the output  
print(result.output)

\# Expected Output (may vary slightly):  
\# The first known use of "hello, world" was in a 1974 textbook about the C programming language.

This brevity underscores PydanticAI's commitment to an ergonomic developer experience, mirroring the simplicity often associated with FastAPI.1 It provides a very low barrier to entry for performing basic LLM interactions.

### **Video 5: Understanding Structured Output with Pydantic Models**

**Objective:** Explain how to enforce structured, validated responses from the LLM using Pydantic models.

While the basic "Hello World" agent returns a simple text string, relying solely on unstructured text output from LLMs can be problematic in production applications. Raw string responses can be inconsistent, difficult to parse reliably, and may not adhere to expected formats, leading to downstream errors.15 PydanticAI addresses this challenge by deeply integrating Pydantic models for defining and enforcing structured output.

Pydantic itself is a widely used Python library renowned for data validation and settings management using Python type hints.13 Its core component is the BaseModel, which allows developers to define data schemas declaratively. PydanticAI leverages this capability directly.

To obtain structured output, the first step is to define a Pydantic BaseModel that represents the desired structure of the LLM's response. This model specifies the fields, their data types (e.g., str, int, list, other BaseModels), and optionally provides descriptions using Field(description=...) to further guide the LLM.1

Python

from pydantic import BaseModel, Field  
from typing import List

class CityInfo(BaseModel):  
 city_name: str \= Field(description="The name of the city")  
 country: str \= Field(description="The country the city is in")  
 population: int \= Field(description="Estimated population")  
 landmarks: List\[str\] \= Field(description="List of famous landmarks")

Once the output model (CityInfo in this case) is defined, it needs to be associated with the agent. This is done by passing the model class to the Agent constructor using the output_type argument.2

Python

from pydantic_ai import Agent

agent \= Agent(  
 'openai:gpt-4o',  
 system_prompt='Provide information about the requested city.',  
 output_type=CityInfo \# Specify the desired output structure  
)

Behind the scenes, PydanticAI uses the provided output_type model to generate a schema, typically a JSON Schema, which is then included in the prompt sent to the LLM. This schema explicitly instructs the LLM on the required format for its response.2

Crucially, when the LLM returns its response (expected to be in JSON format matching the schema), PydanticAI automatically attempts to parse and validate this response against the CityInfo model.1 If the validation is successful, the result.output attribute will contain an instance of the CityInfo model, not just a raw string. This allows for type-safe attribute access (e.g., result.output.city_name, result.output.population).2 If validation fails (e.g., missing fields, incorrect data types), PydanticAI can even handle this by potentially re-prompting the LLM with the validation error details, asking it to correct its output (covered further in error handling).2

The benefits of this approach are significant:

- **Consistency:** Ensures responses adhere to a predictable structure across different runs or even different LLMs.
- **Reliability:** Guarantees that the data received by the application code is valid according to the defined schema, reducing runtime errors.
- **Type Safety:** Allows developers to work with the LLM output as typed Python objects, improving code clarity and enabling static analysis.
- **Simplified Integration:** Makes it much easier to integrate LLM outputs into downstream processes, databases, or APIs.

This tight integration with Pydantic for structured output validation is a fundamental aspect of PydanticAI's design philosophy. It directly addresses a critical challenge in building dependable AI applications and is a key reason why developers might choose PydanticAI for projects requiring high data integrity and robust interactions.1 This focus on validation is arguably its core differentiator compared to interacting with LLMs via raw SDK calls that merely return strings.15

## **Part 2: Core Agent Capabilities**

### **Video 6: Diving Deeper: Agent Lifecycle (run, run_sync, run_stream)**

**Objective:** Explain the different ways to execute an agent run and introduce asynchronous operations and streaming.

PydanticAI offers flexibility in how agent interactions are executed, catering to different application architectures and requirements through three primary run methods: run_sync, run, and run_stream.

The run_sync method, used in the "Hello World" example, executes the agent interaction synchronously.2 This means the program execution will pause at the agent.run_sync(...) call and wait until the LLM responds and the agent completes its processing before proceeding. This approach is straightforward and suitable for command-line scripts, simple batch processing, or scenarios where asynchronous programming is not necessary or desired.

For applications involving I/O-bound operations (like network requests to LLM APIs or tool APIs) or integration with modern asynchronous web frameworks (such as FastAPI, which PydanticAI aims to emulate in developer experience 1), the asynchronous run method is preferred.8 This method is a coroutine and must be awaited using await agent.run(...). It allows the application to perform other tasks while waiting for the LLM or tools to respond, improving overall responsiveness and throughput. A typical usage pattern involves defining an async def main(): function or using it within an async route handler in a web framework.1

Python

import asyncio  
from pydantic_ai import Agent

agent \= Agent('openai:gpt-4o')

async def main():  
 result \= await agent.run('What is the capital of France?')  
 print(result.output)

if \_\_name\_\_ \== "\_\_main\_\_":  
 asyncio.run(main())

The third method, run_stream, enables streaming responses from the LLM.1 Instead of waiting for the entire response to be generated, run_stream returns an asynchronous iterator that yields chunks of the response as they become available. This is crucial for applications requiring real-time feedback, such as chatbots or interfaces displaying progressively generated text. The basic syntax involves using an async with block 27:

Python

async def stream_example():  
 async with agent.run_stream('Tell me a long story.') as response_stream:  
 async for chunk in response_stream:  
 print(chunk, end="", flush=True)

(Detailed streaming implementation will be covered in Part 3).

The return types differ slightly. Both run_sync and run return an AgentRunResult object once the interaction is complete. This object contains the final output (result.output) and other metadata like message history and usage statistics.8 In contrast, run_stream returns a StreamedRunResult object, which primarily provides the asynchronous iterator to consume the streaming chunks.8

Choosing the appropriate method depends on the context:

- Use run_sync for simple scripts or synchronous codebases.
- Use run for asynchronous applications (like web servers) or when dealing with potentially long-running tool calls alongside the LLM interaction.
- Use run_stream when real-time feedback or processing of partial responses is required.

By providing these three distinct execution modes, PydanticAI offers developers the versatility to integrate AI capabilities effectively into various Python application paradigms.

### **Video 7: Giving Your Agent Tools (@agent.tool)**

**Objective:** Teach how to define and register functions (tools) that the agent's LLM can call.

LLMs possess vast general knowledge but lack access to real-time information, private data sources, or the ability to perform actions in the external world. To overcome these limitations, PydanticAI allows developers to equip agents with "tools" – standard Python functions that the LLM can invoke during its reasoning process to gather information or execute tasks.1

The primary mechanism for registering a function as a tool is the @agent.tool decorator.1 Applying this decorator to a Python function makes it available to the agent's underlying LLM.

Python

from pydantic_ai import Agent  
import random

agent \= Agent('openai:gpt-4o', system_prompt="You can flip a coin.")

@agent.tool  
def flip_coin() \-\> str:  
 """Flips a virtual coin and returns 'Heads' or 'Tails'."""  
 result \= random.choice()  
 print(f"Tool 'flip_coin' executed, returning: {result}") \# For debugging  
 return result

\# Example usage:  
\# result \= agent.run_sync("Flip a coin for me.")  
\# print(result.output) \# LLM should incorporate the tool's result

Defining the tool function follows standard Python practices. It can be synchronous (def) or asynchronous (async def). Crucially, PydanticAI leverages Python's type hints and docstrings to automatically generate the necessary information for the LLM to understand and use the tool.1

- **Tool Parameters:** Any parameters defined in the function signature (excluding the optional RunContext, discussed later) become the input arguments for the tool. PydanticAI uses the type hints for these parameters to create an input schema.1 When the LLM decides to call the tool, it must provide arguments matching this schema. Pydantic automatically validates the arguments provided by the LLM before the tool function is executed. If validation fails, an error is returned to the LLM, prompting it to potentially correct the arguments and retry the tool call.1
- **Docstrings for Descriptions:** The function's docstring plays a vital role. The main docstring is used as the overall description of the tool, explaining its purpose to the LLM. Furthermore, PydanticAI can parse parameter descriptions from the docstring (e.g., using standard formats like Google or NumPy style) and include them in the schema sent to the LLM, providing clearer guidance on what each parameter represents.1

Consider a tool requiring an argument:

Python

@agent.tool  
def get_stock_price(symbol: str) \-\> float:  
 """  
 Fetches the current stock price for a given symbol.

    Args:
        symbol: The stock ticker symbol (e.g., 'AAPL').
    """
    \# (Implementation to call a stock API)
    print(f"Tool 'get\_stock\_price' called for symbol: {symbol}")
    \#... fetch price...
    price \= 175.50 \# Placeholder
    return price

Here, the LLM would be informed that a tool named get_stock_price exists, it requires a string argument named symbol (described as the ticker symbol), and it returns a float.

The overall flow involves the LLM processing the user query and conversation history. If it determines that calling a specific tool would be beneficial for generating a response, it outputs a special request indicating the tool name and arguments. PydanticAI intercepts this, validates the arguments, executes the corresponding Python function, and sends the function's return value back to the LLM. The LLM then uses this result to formulate its final response to the user.

For very simple tools that do not require access to the agent's runtime context (explained in the next section), an alternative decorator @agent.tool_plain can be used.8 However, @agent.tool is more common as it enables access to dependencies via RunContext.

This approach of defining tools using standard Python functions, type hints, and docstrings makes the process highly intuitive and Pythonic, significantly reducing the effort required compared to manually defining tool schemas in JSON or other formats.1

### **Video 8: Enhancing Tools with Dependency Injection**

**Objective:** Introduce the Dependency Injection system for providing context, data, or services to tools.

While simple tools might be self-contained, many practical tools need access to external resources, configuration, or runtime context. For example, a tool to check a user's account balance needs the user's ID and a database connection; a tool interacting with an external API needs an API client or key. Passing such dependencies explicitly and managing their lifecycle is crucial for building modular, testable, and maintainable applications. PydanticAI provides an optional, type-safe Dependency Injection (DI) system specifically for this purpose.1

The core idea is to define a container for the dependencies and then make this container available to tools (and dynamic system prompts) during the agent's run.

1.  **Define Dependency Container:** A standard Python @dataclass is typically used to group related dependencies. This dataclass defines the structure and types of the data or services needed.1  
    Python  
    from dataclasses import dataclass  
    \# Assume DatabaseConn is a class handling database interactions  
    from bank_database import DatabaseConn

    @dataclass  
    class SupportDependencies:  
     customer_id: int  
     db: DatabaseConn

    This SupportDependencies class bundles a customer_id and a db connection object.

2.  **Associate with Agent:** The dependency type is associated with the agent using the deps_type argument in the Agent constructor.11  
    Python  
    support_agent \= Agent(  
     'openai:gpt-4o',  
     deps_type=SupportDependencies, \# Associate the dependency type  
     output_type=SupportOutput \# Assuming SupportOutput is defined  
    )

3.  **Access Dependencies via RunContext:** Tools registered with @agent.tool can access these dependencies through a special object called RunContext. This object is automatically passed as the first argument to the tool function. It should be type-hinted with the dependency dataclass to enable type safety and autocompletion: ctx: RunContext\[YourDependencyClass\].1 Dependencies are accessed via the ctx.deps attribute:  
    Python  
    from pydantic_ai import RunContext

    @support_agent.tool  
    async def customer_balance(  
     ctx: RunContext, \# Receive RunContext  
     include_pending: bool  
    ) \-\> float:  
     """Returns the customer's current account balance."""  
     \# Access dependencies via ctx.deps  
     balance \= await ctx.deps.db.customer_balance(  
     id\=ctx.deps.customer_id,  
     include_pending=include_pending,  
     )  
     return balance

4.  **Provide Dependencies at Runtime:** When executing the agent using run or run_sync, an instance of the dependency dataclass must be provided using the deps keyword argument.4  
    Python  
    async def main():  
     \# Create an instance of the dependencies  
     db_connection \= DatabaseConn() \# Initialize database connection  
     dependencies \= SupportDependencies(customer_id=123, db=db_connection)

        \# Pass dependencies when running the agent
        result \= await support\_agent.run(
            'What is my current balance including pending transactions?',
            deps=dependencies
        )
        print(result.output)

This DI system offers several advantages:

- **Explicit Dependencies:** Makes it clear what external resources a tool requires.
- **Modularity:** Tools are decoupled from how dependencies are created or managed.
- **Testability:** During testing, real dependencies (like database connections) can be easily replaced with mock objects by passing a different instance to the deps argument.1
- **Type Safety:** Using dataclasses and RunContext with type hints ensures that dependencies are accessed correctly and leverages static analysis tools.1

The inclusion of this sophisticated DI system highlights PydanticAI's focus on facilitating the development of robust, production-ready applications by promoting established software engineering best practices.1

### **Video 9: Dynamic System Prompts & Context**

**Objective:** Show how to create system prompts that adapt based on runtime context using dependency injection.

The static system_prompt provided during agent instantiation offers a fixed set of instructions. However, in many scenarios, it's beneficial for the agent's core guidance or persona to adapt based on the specific user, task, or context of the interaction. PydanticAI enables this through dynamic system prompts, leveraging the same Dependency Injection system used for tools.

Instead of just a static string, system prompts can be defined as functions decorated with @agent.system_prompt.2

Python

from pydantic_ai import Agent, RunContext  
from dataclasses import dataclass

@dataclass  
class UserProfile:  
 user_name: str  
 language_preference: str

agent \= Agent(  
 'openai:gpt-4o',  
 deps_type=UserProfile \# Agent needs UserProfile dependencies  
)

@agent.system_prompt  
async def generate_personalized_prompt(ctx: RunContext\[UserProfile\]) \-\> str:  
 """Generates a system prompt tailored to the user."""  
 user_name \= ctx.deps.user_name  
 language \= ctx.deps.language_preference

    \# Fetch user-specific instructions or persona details if needed
    \# e.g., user\_persona \= await ctx.deps.db.get\_persona(user\_name)

    prompt \= f"You are a helpful assistant for {user\_name}. "
    prompt \+= f"Please respond primarily in {language}. "
    \# prompt \+= f"Adopt the following persona: {user\_persona}"
    prompt \+= "Be friendly and concise."
    return prompt

Similar to tools using @agent.tool, functions decorated with @agent.system_prompt receive the RunContext object as an argument.2 This allows the function to access the dependencies provided during the run or run_sync call via ctx.deps.

The decorated function must return a string, which will be used as the system prompt for that specific agent run. This allows the prompt to incorporate dynamic information, such as the user's name, preferences, account status, or any other data available through the injected dependencies.

When running the agent, the necessary dependencies (an instance of UserProfile in this example) must be passed using the deps argument:

Python

async def interact(user_id: int, query: str):  
 \# Fetch user profile based on user_id (e.g., from a database)  
 profile \= UserProfile(user_name="Alice", language_preference="French")  
 result \= await agent.run(query, deps=profile)  
 print(result.output)

It's possible to define multiple dynamic system prompt functions for an agent. PydanticAI will execute all of them and concatenate their returned strings to form the final system prompt sent to the LLM. This allows for modular prompt construction. If a static system_prompt is also provided in the Agent constructor, it is typically prepended to the dynamically generated parts.

Dynamic system prompts provide a powerful mechanism for creating personalized and context-aware agent behavior directly within the framework's structure, going beyond static instructions and enabling more sophisticated and adaptive interactions.

### **Video 10: Managing Conversation History**

**Objective:** Explain how to maintain conversational context across multiple turns.

For an agent to engage in coherent, multi-turn conversations, it needs a mechanism to remember previous interactions – a form of short-term memory.3 LLMs themselves are typically stateless between independent API calls, so the context must be explicitly provided. PydanticAI facilitates this primarily through the message_history argument.

When an agent completes a run (using run or run_sync), the returned AgentRunResult object contains information about the messages exchanged during that specific interaction. This includes the initial user message, any tool calls and their results, and the final assistant response. These messages can be accessed using methods like result.all_messages() (which includes the input messages) or result.new_messages() (which typically includes messages generated during the run, like tool interactions and the final response).3

To maintain context in a subsequent interaction with the same user or within the same conversational thread, this list of messages from the previous run should be passed to the next run or run_sync call via the message_history keyword argument.2

Python

\# conversation_example.py \[3, 27\]  
from pydantic_ai import Agent

agent \= Agent('openai:gpt-4o')  
conversation_history \= \# Initialize empty history

\# First turn  
user_query1 \= "My name is Bob. What is the capital of Italy?"  
result1 \= agent.run_sync(user_query1, message_history=conversation_history)  
print(f"User: {user_query1}")  
print(f"AI: {result1.output}")  
\# Update history with messages from this run  
conversation_history.extend(result1.new_messages())

\# Second turn  
user_query2 \= "What is my name?"  
result2 \= agent.run_sync(user_query2, message_history=conversation_history)  
print(f"User: {user_query2}")  
print(f"AI: {result2.output}") \# Should output "Your name is Bob."  
\# Update history again if conversation continues  
conversation_history.extend(result2.new_messages())

The messages themselves are typically represented by objects like UserMessage, AssistantMessage, or ToolMessage (often subclasses or variations of ModelMessage 2), containing the content and the role ('user', 'assistant', 'tool'). PydanticAI uses this history to construct the appropriate prompt for the LLM, allowing it to understand the context of the ongoing conversation.

This pattern of extracting messages from the result and passing them into the next run forms the basis of conversational memory in PydanticAI. The framework provides the mechanism for passing this turn-by-turn history.

However, for more advanced memory management strategies – such as persisting conversations across different user sessions, summarizing long histories to fit within context window limits, or integrating with external vector stores for long-term knowledge retrieval – developers typically need to implement custom logic. This might involve storing conversation_history in a database keyed by session ID 3 or potentially integrating with specialized memory components from other libraries. PydanticAI itself appears focused on providing the core message passing mechanism rather than offering complex, built-in memory modules like those found in frameworks such as LangChain.11 This aligns with its positioning as potentially more of a focused tool or shim rather than an all-encompassing ecosystem 21, providing the essential building block for conversation while leaving sophisticated memory strategies to the developer or other libraries.

### **Video 11: Handling Errors Gracefully**

**Objective:** Discuss common errors and how PydanticAI helps manage them, especially validation errors.

Building robust AI applications requires anticipating and handling various potential errors. These can range from network issues when calling LLM APIs, errors during the execution of custom tools, or failures related to the LLM's response not conforming to expectations. PydanticAI incorporates mechanisms to handle certain types of errors automatically, particularly those related to data validation.

A key strength of PydanticAI is its handling of Pydantic validation errors.1 These occur in two main scenarios:

1. **Invalid Tool Arguments:** If the LLM attempts to call a tool (defined with @agent.tool) but provides arguments that do not match the function's type hints or validation rules defined in the Pydantic model implicitly generated from the signature, Pydantic raises a validation error. PydanticAI intercepts this error and, instead of crashing, sends the validation error details back to the LLM. This feedback allows the LLM to understand its mistake and potentially retry the tool call with corrected arguments.1
2. **Invalid Structured Output:** If an output_type (a Pydantic model) is specified for the agent, PydanticAI validates the final response from the LLM against this model. If the response structure or data types are incorrect, a validation error occurs. Similar to tool argument errors, PydanticAI can automatically re-prompt the agent, providing the validation error information to the LLM and asking it to generate a compliant response.2

This automatic retry mechanism based on validation feedback significantly enhances the reliability of interactions, increasing the likelihood of obtaining usable, structured data even if the LLM's initial attempt is flawed.1 While this retry logic is powerful, developers should be aware of potential edge cases or limitations, as suggested by issue reports concerning retry loops in specific scenarios in earlier versions.31 Configuration options for retry limits might exist or be planned to prevent infinite loops.

For errors occurring _within_ the execution of a custom tool function (e.g., an external API call fails, a database query errors out), standard Python error handling practices apply. Developers should use try...except blocks within their tool functions to catch expected exceptions, log them appropriately, and return a meaningful error message or status to the agent/LLM if necessary.

PydanticAI also defines specific exceptions for certain conditions. For instance, if usage limits (discussed later) are configured and exceeded during a run, a UsageLimitExceeded exception is raised, which can be caught and handled by the application code.8

General best practices for error handling, such as comprehensive logging (potentially using Pydantic Logfire, covered later) and providing informative feedback to the user or system administrators, remain essential when building applications with PydanticAI. The framework's built-in handling of validation errors, however, significantly reduces the burden on developers for a common class of failures in LLM interactions.

## **Part 3: Advanced Features & Techniques**

### **Video 12: Real-time Responses: Streaming Output**

**Objective:** Implement basic text streaming for immediate feedback.

In many interactive AI applications, such as chatbots or code generation tools, waiting for the LLM to generate its complete response before displaying anything can lead to a poor user experience, especially for longer outputs. PydanticAI addresses this with built-in support for streaming responses, allowing applications to receive and display output incrementally as it's generated by the LLM.1

Streaming is enabled by using the run_stream method instead of run or run_sync.8 This method returns a StreamedRunResult object, which acts as an asynchronous iterator.

The standard pattern for consuming the stream is using an async for loop within an async with block 2:

Python

import asyncio  
from pydantic_ai import Agent

agent \= Agent('openai:gpt-4o')

async def stream_basic_text():  
 print("AI Assistant:")  
 async with agent.run_stream("Tell me a short story about a brave knight.") as response_stream:  
 async for chunk in response_stream:  
 \# Each chunk is typically a string fragment  
 print(chunk, end="", flush=True)  
 print("\\n--- End of Stream \---")

if \_\_name\_\_ \== "\_\_main\_\_":  
 asyncio.run(stream_basic_text())

In this basic text streaming scenario, each chunk yielded by the response_stream iterator is typically a string containing a portion of the LLM's generated text. The code iterates through these chunks, printing each one immediately. Using end="" prevents adding extra newlines between chunks, and flush=True ensures the output buffer is written to the console promptly, creating a smooth, real-time display effect.2

The StreamedRunResult object itself might offer other methods (like await response.get_output() shown in 27, which likely waits for the full output after streaming), but the primary interaction model for streaming is asynchronous iteration over the yielded chunks.

Basic text streaming is particularly useful for:

- **Chatbots:** Displaying the assistant's response word-by-word or sentence-by-sentence.
- **Long Text Generation:** Showing progress as articles, code, or stories are generated.
- **Improving Perceived Performance:** Giving the user immediate feedback that the system is working.

PydanticAI's straightforward implementation of streaming via asynchronous iteration makes it relatively simple to build responsive, real-time AI interfaces without needing complex manual handling of low-level response streams.1

### **Video 13: Advanced Streaming: Structured Data Chunks**

**Objective:** Demonstrate how to stream structured data, validating it chunk by chunk using Pydantic models.

While basic text streaming enhances responsiveness, PydanticAI offers a more powerful capability: streaming structured data validated against a Pydantic model _incrementally_.2 This combines the real-time benefits of streaming with the core PydanticAI strength of data validation and structure enforcement. Handling raw streaming JSON and attempting to parse it piece by piece can be complex and error-prone.

To enable structured streaming, the approach involves two key steps:

1. Define a Pydantic BaseModel representing the desired structured output (as covered in Video 5).
2. Instantiate the Agent with this model specified in the output_type argument.
3. Call the agent.run_stream(...) method.2

Python

import asyncio  
from pydantic import BaseModel, Field  
from pydantic_ai import Agent

class Story(BaseModel):  
 title: str \= Field(description="The title of the story")  
 paragraph: str \= Field(description="A paragraph from the story")

agent \= Agent(  
 'openai:gpt-4o',  
 system_prompt='Generate a very short story with a title and one paragraph, output as JSON.',  
 output_type=Story \# Specify structured output type  
)

async def stream_structured_data():  
 print("Streaming Structured Story:")  
 full_story \= None  
 async with agent.run_stream("Generate a story about a lost robot.") as response_stream:  
 async for chunk in response_stream:  
 if isinstance(chunk, Story):  
 \# Received a validated Story object (or potentially part of one)  
 print(f"\\n--- Validated Story Chunk \---")  
 print(f"Title: {getattr(chunk, 'title', '...')}")  
 print(f"Paragraph: {getattr(chunk, 'paragraph', '...')}")  
 full_story \= chunk \# Keep track of the latest complete object  
 else:  
 \# Received a partial string (part of the JSON)  
 print(chunk, end="", flush=True)

    print("\\n--- End of Stream \---")
    if full\_story:
        print(f"\\nFinal Validated Story Object: {full\_story}")

if \_\_name\_\_ \== "\_\_main\_\_":  
 asyncio.run(stream_structured_data())

When streaming with an output_type defined, PydanticAI attempts to parse the incoming data stream (expected to be JSON) against the specified Pydantic model (Story in this case) incrementally. The asynchronous iterator (response_stream) yields different types of chunks 2:

- **Partial Strings:** Fragments of the raw JSON string as it's being received.
- **Validated Pydantic Objects:** As soon as enough valid JSON has been received to constitute a complete instance (or potentially a valid partial instance) of the output_type model, PydanticAI parses, validates, and yields that object.

The application code needs to check the type of each yielded chunk using isinstance(chunk, Story) to differentiate between these possibilities.2 If the chunk is an instance of the target Pydantic model, its attributes can be accessed directly (e.g., chunk.title).

This "immediate validation" during streaming 1 is a significant advantage. It allows applications to work with guaranteed-valid, structured data as early as possible in the response generation process, without waiting for the entire LLM response. This enables the creation of sophisticated real-time user interfaces or data processing pipelines that react to structured information incrementally.

For even more fine-grained control over the streaming process, PydanticAI provides different event types that can be yielded during a stream, such as PartDeltaEvent, FinalResultEvent, FunctionToolCallEvent, and FunctionToolResultEvent.27 These allow developers to react specifically to text deltas, tool invocations, or the final validated result within the streaming loop, although the basic type-checking approach covers many common use cases.

Structured streaming represents a powerful synergy between PydanticAI's real-time capabilities and its core commitment to data validation, enabling the development of reliable and responsive applications handling complex data structures from LLMs.

### **Video 14: Monitoring Your Agent with Pydantic Logfire**

**Objective:** Introduce Pydantic Logfire and show how to integrate it for debugging and observability.

As AI agent applications become more complex, involving multiple LLM calls, tool interactions, and intricate logic, understanding their behavior and debugging issues becomes challenging. Observability – the ability to monitor and understand a system's internal state based on its outputs – is crucial for development, debugging, and production monitoring. Pydantic Logfire is an observability tool developed by the Pydantic team, designed specifically for monitoring Python applications, with strong support for Pydantic models and AI/LLM workflows.1

PydanticAI is designed for seamless integration with Logfire.1 This tight, first-party integration provides developers with a powerful tool for gaining insights into agent execution without significant configuration overhead. Logfire helps developers understand the flow of an agent run, track performance metrics, monitor token usage, inspect data payloads passed between components (LLM, tools), and ultimately debug unexpected behavior.1

Setting up Logfire integration typically involves a few steps:

1. **Installation:** Install the logfire library (pip install logfire).
2. **Configuration:** Configure Logfire, usually by calling logfire.configure(). This might involve setting up a Logfire account and API key for sending data to the Logfire platform, although local logging might also be possible.
3. **Instrumentation:** Instruct Logfire to automatically instrument relevant libraries. This often includes Pydantic itself (logfire.instrument_pydantic()) and potentially specific LLM client libraries (e.g., logfire.instrument_openai()) or database drivers used in tools.2 PydanticAI might also have specific instrumentation calls or be automatically instrumented when logfire.configure is used.

Python

\# Conceptual Logfire Setup  
import logfire  
from pydantic_ai import Agent  
\# Potentially import instrument functions for specific libraries  
\# from logfire.integrations import instrument_openai

\# Configure Logfire (might require API keys / account setup)  
logfire.configure()

\# Instrument Pydantic and potentially other libraries  
logfire.instrument_pydantic()  
\# instrument_openai() \# If using OpenAI

\# \--- Define and run your PydanticAI agent as usual \---  
agent \= Agent('openai:gpt-4o')  
\#... agent definition...  
\# result \= agent.run_sync("Some query")  
\# \--- Agent execution details will be sent to Logfire \---

Once configured and instrumented, Logfire automatically captures detailed traces of the agent's execution. These traces can typically be viewed in a web-based UI provided by the Logfire service.2 The UI visualizes the entire flow of an agent run, showing:

- The initial user prompt.
- Calls made to the LLM, including the prompts sent and responses received.
- Invocations of tools, including the arguments passed and the results returned.
- Pydantic validation steps (successes and failures).
- Timings for each step.
- Token usage information.

This detailed, structured view is invaluable for debugging complex interactions, identifying performance bottlenecks, and ensuring the agent behaves as expected.1 The built-in, straightforward integration with Logfire reinforces PydanticAI's focus on production readiness by providing developers with essential observability tools directly within the ecosystem.1

### **Video 15: Controlling Costs & Behavior: Usage Limits & Model Settings**

**Objective:** Explain how to set limits on token usage/requests and configure LLM parameters.

Operating LLM agents in production requires careful management of resources and costs. LLM API calls are often priced based on the number of input and output tokens processed. Uncontrolled interactions or verbose responses can lead to unexpectedly high costs. Furthermore, controlling the LLM's behavior, such as its creativity or response length, is often necessary for specific tasks. PydanticAI provides mechanisms to address these needs through configurable usage limits and model settings.

Usage Limits:  
To control costs and prevent excessive resource consumption, PydanticAI offers a UsageLimits structure, typically found within pydantic_ai.settings.8 This structure allows developers to define limits on various aspects of the LLM interaction, such as the maximum number of tokens allowed in the response.

Python

from pydantic_ai import Agent  
from pydantic_ai.settings import UsageLimits  
from pydantic_ai.exceptions import UsageLimitExceeded

agent \= Agent('openai:gpt-4o')

\# Define usage limits \- e.g., limit response to 20 tokens  
limits \= UsageLimits(response_tokens_limit=20)

try:  
 \# Apply limits during the agent run  
 result \= agent.run_sync(  
 'Explain the theory of relativity in detail.',  
 usage_limits=limits  
 )  
 print(result.output)  
except UsageLimitExceeded as e:  
 \# Handle the case where the limit was exceeded  
 print(f"Error: Usage limit exceeded\! {e}")

The UsageLimits instance is passed to the run, run_sync, or run_stream methods via the usage_limits keyword argument.8 If the defined limit is exceeded during the interaction (e.g., the LLM generates more tokens than allowed), PydanticAI raises a UsageLimitExceeded exception, which the application code can catch and handle appropriately.8 Other potential limits might include total tokens (prompt \+ response), number of LLM requests, or number of tool calls per run, though response_tokens_limit is explicitly shown.8

Model Settings:  
Beyond usage limits, developers often need to fine-tune the behavior of the LLM itself. Common parameters control aspects like randomness (temperature), maximum output length (max_tokens), or API call timeouts. PydanticAI allows passing these provider-specific settings during agent execution using the model_settings argument.8 This argument typically accepts a dictionary.

Python

from pydantic_ai import Agent

agent \= Agent('openai:gpt-4o')

\# Define model settings \- e.g., make output more deterministic  
settings \= {'temperature': 0.0}

\# Apply settings during the agent run  
result \= agent.run_sync(  
 'What is the capital of Italy?',  
 model_settings=settings  
)  
print(result.output) \# Expected: Rome (with high probability due to temp=0.0)

Commonly used settings include temperature (lower values make output more focused and deterministic, higher values increase randomness), max_tokens (limits the length of the LLM's generated response), and timeout (sets a time limit for the API call).8 The specific parameters available depend on the LLM provider being used.

These model settings can also be specified as defaults when instantiating the Agent itself, applying them to all runs unless overridden by the model_settings argument in a specific run call.27

Providing these per-run controls for both usage limits and model parameters gives developers fine-grained management capabilities essential for optimizing agent behavior, controlling operational costs, and ensuring predictable performance in production deployments.8

### **Video 16: Orchestrating Complex Workflows with Pydantic Graph (Intro)**

**Objective:** Introduce the concept and basic usage of Pydantic Graph for managing complex agent interactions or state machines.

While many agent tasks can be handled through a sequence of prompts and tool calls managed by the base Agent class, more complex applications might involve intricate workflows, multi-step processes with conditional logic, or collaborations between multiple specialized agents. In such scenarios, relying solely on standard Python control flow can lead to convoluted and hard-to-maintain code, sometimes referred to as "spaghetti code".1

To address this, PydanticAI integrates with Pydantic Graph, a library (also from the Pydantic team) designed for building and running state machines or complex workflows in a structured, type-safe manner using Pydantic models and typing hints.1 Pydantic Graph provides a way to define workflows as a directed graph where nodes represent states or processing steps, and edges represent transitions between them.

The core concepts of Pydantic Graph include 33:

- **BaseNode:** Represents a node (a state or step) in the graph. Each node typically implements a run method that performs some action (e.g., calls an LLM, executes a tool, processes data) and determines the next node in the workflow based on the outcome.
- **State:** A dataclass (or Pydantic model) that holds the data shared and potentially modified across different nodes in the graph during a single run.
- **GraphRunContext:** An object passed to a node's run method, providing access to the current State (ctx.state).
- **Graph:** The main object that defines the structure of the workflow by collecting all the nodes and potentially defining the starting node and initial state.
- **End:** A special node type returned by a BaseNode's run method to signal the successful completion of the workflow, optionally carrying a final result value.

A simple example illustrates these concepts 33:

Python

\# never_42.py \[33\]  
from \_\_future\_\_ import annotations  
from dataclasses import dataclass  
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

@dataclass  
class MyState:  
 number: int

@dataclass  
class Increment(BaseNode): \# Node operating on MyState  
 async def run(self, ctx: GraphRunContext) \-\> Check42: \# Returns next node type  
 ctx.state.number \+= 1  
 return Check42() \# Transition to Check42 node

@dataclass  
class Check42(BaseNode): \# Takes MyState, returns End\[int\] or Increment  
 async def run(self, ctx: GraphRunContext) \-\> Increment | End\[int\]:  
 if ctx.state.number \== 42:  
 \# This condition seems reversed in the snippet, assuming it should end if NOT 42  
 \# return Increment() \# Loop back if 42 (as per snippet logic)  
 return End(ctx.state.number) \# End if 42 (more logical interpretation)  
 else:  
 \# return End(ctx.state.number) \# End if not 42 (as per snippet logic)  
 return Increment() \# Loop back if not 42 (more logical interpretation)

\# Define the graph structure  
never_42_graph \= Graph(nodes=(Increment, Check42))

\# Running the graph would involve calling graph.run() or graph.iter()  
\# with an initial state, e.g., MyState(number=0)

This example defines a graph with two nodes, Increment and Check42, operating on a state containing a single number. The nodes define the logic for incrementing the number and checking its value, determining the transition to the next node or the end of the workflow.

Within PydanticAI, the Agent class itself utilizes Pydantic Graph internally to manage its execution flow (involving prompts, tool calls, validation, etc.).27 Developers can gain access to this underlying graph execution via the agent.iter() method, which returns an AgentRun object allowing iteration over the nodes of the agent's internal graph.27

Beyond observing the agent's internal graph, developers can leverage Pydantic Graph directly to build _custom_ complex workflows. This could involve orchestrating multiple PydanticAI agents, implementing sophisticated decision trees based on tool outputs, or managing long-running, stateful processes that go beyond a single request-response cycle.

While potentially having a steeper learning curve than basic agent usage, the availability of Pydantic Graph provides a powerful, optional tool for tackling complex orchestration problems within the Pydantic ecosystem, offering a structured, type-safe alternative to ad-hoc control flow or potentially more complex graph systems in other frameworks.1 This signals PydanticAI's capability to scale beyond simple tasks to handle sophisticated, multi-step AI applications.

## **Part 4: Practical Projects**

These project videos aim to synthesize the concepts covered in Parts 1-3, demonstrating how the core features of PydanticAI work together to build tangible applications. The progression is designed to introduce complexity gradually.

### **Video 17: Project 1: Building an API Interaction Agent (e.g., Weather Bot)**

**Objective:** Build an agent that calls an external API (like a weather API) using tools.

This project demonstrates a common use case: creating an agent that can fetch real-time data from the external world via an API. The goal is to build an agent capable of answering questions like "What's the weather in London?".

**Steps:**

1. **Define Goal & Scope:** The agent should accept a location name and return current weather conditions (e.g., temperature, description).
2. **Select API:** Choose a public weather API (e.g., OpenWeatherMap, WeatherAPI). Obtain an API key if required and store it securely (e.g., environment variable).
3. **Define API Response Model:** Create a Pydantic BaseModel to represent the relevant parts of the JSON response from the chosen weather API. This helps in parsing the complex API response.
4. **Create Fetch Tool:** Define a Python function (sync or async) decorated with @agent.tool. This function will:
   - Accept the location (e.g., location: str) as an argument.
   - Use an HTTP client library (like httpx or requests) to call the weather API endpoint with the location and API key.
   - Include try...except blocks to handle potential network errors or invalid API responses.
   - Parse the JSON response, potentially using the Pydantic model defined in step 3\.
   - Return the relevant weather information (e.g., a dictionary or a custom object).

Python  
\# Example Tool Snippet  
@agent.tool  
async def get_current_weather(location: str) \-\> dict:  
 """Fetches the current weather for a specified location."""  
 \#... (use httpx/requests to call weather API)...  
 \#... (handle errors)...  
 \#... (parse response)...  
 weather_data \= {"temperature": 15, "description": "Cloudy"} \# Placeholder  
 return weather_data

5. **Define Agent Output Model:** Create a Pydantic BaseModel for the agent's _final_, user-facing structured output. This might be simpler than the full API response model.  
   Python  
   class WeatherReport(BaseModel):  
    city: str  
    temperature_celsius: float  
    conditions: str

6. **Configure Agent:** Instantiate the Agent, providing the get_current_weather tool (via the tools argument or decorator) and setting output_type=WeatherReport. Define a suitable system_prompt instructing the agent on how to use the tool and format the final answer.
7. **Test:** Run the agent with various location queries (e.g., agent.run_sync("How is the weather in Paris?")). Verify that the tool is called and the output conforms to the WeatherReport model.
8. **Refine:** Adjust the system prompt or tool logic for better performance and more natural language interaction.

This project solidifies understanding of the core loop: Agent instantiation, defining and using @agent.tool, interacting with external APIs within tools, handling basic errors, and using Pydantic models for structured agent output (output_type).1

### **Video 18: Project 2: Creating a SQL Generation Assistant**

**Objective:** Build an agent that translates natural language questions into SQL queries for a predefined database schema.

This project tackles the task of natural language to SQL translation, a valuable capability in data analysis and business intelligence.13 The agent will act as an interface between a user asking questions in plain English and a relational database.

**Steps:**

1. **Define Database Schema:** Create or document a sample relational database schema (e.g., tables for customers, products, orders with columns and relationships).
2. **Define Goal & Scope:** The agent should accept a natural language question about the data (e.g., "Show me customers from California who bought laptops") and generate the corresponding SQL query.
3. **Provide Schema to Agent:** The LLM needs to know the database structure. This information (table names, column names, types, relationships) must be provided. Common methods include:
   - Including the schema definition directly within the system_prompt.
   - Creating a tool (@agent.tool) that the LLM can call to retrieve schema information if needed (though less common for basic generation).
4. **Define Output Model:** Create a Pydantic BaseModel to structure the agent's output, primarily containing the generated SQL query string.  
   Python  
   class SqlQueryOutput(BaseModel):  
    sql_query: str \= Field(description="The generated SQL query.")

5. **Configure Agent:** Instantiate the Agent, setting output_type=SqlQueryOutput. Craft a detailed system_prompt that:
   - Provides the database schema.
   - Instructs the LLM to translate user questions into valid SQL for the given schema.
   - Specifies the SQL dialect if necessary (e.g., PostgreSQL, MySQL).
   - Emphasizes generating only the SQL query.
6. **(Optional) Add Validation Tool:** Consider adding a tool that attempts to parse or minimally validate the generated SQL syntax before returning it.
7. **Test:** Run the agent with various natural language questions targeting the schema (e.g., agent.run_sync("Which customers placed orders last week?")). Verify the generated sql_query in the output.
8. **Discuss Limitations:** Address the challenges and risks, especially the potential for generating incorrect or inefficient SQL, and the critical security risk of SQL injection if generated queries are executed directly without sanitization or safeguards.

This project emphasizes careful system_prompt engineering, providing necessary context (the schema) to the LLM, and leveraging output_type for retrieving the specific desired artifact (the SQL query string) in a structured way.13

### **Video 19: Project 3: Implementing a Basic RAG Agent**

**Objective:** Build a simple Retrieval-Augmented Generation agent that answers questions based on provided documents.

Retrieval-Augmented Generation (RAG) enhances LLM responses by grounding them in external knowledge retrieved from a specific document set, making answers more accurate, relevant, and up-to-date.17 This project builds a minimal RAG agent.

**Steps:**

1. **Prepare Knowledge Base:** Gather a small collection of text documents (e.g., text files, markdown files) containing the information the agent should use.
2. **Implement Simple Retrieval:** Create a basic mechanism to find relevant document snippets based on a user query. For simplicity, this could be:
   - Keyword search across documents.
   - A very basic vector search implementation (potentially introducing a library like sentence-transformers for embeddings and numpy/faiss for simple similarity search, keeping it minimal).
3. **Define Goal & Scope:** The agent should answer user questions using _only_ the information found in the provided documents.
4. **Create Retrieval Tool:** Define a function decorated with @agent.tool that:
   - Accepts the user's query (query: str).
   - Uses the retrieval mechanism (step 2\) to find the most relevant text snippets from the knowledge base.
   - Returns these snippets as a single string or a list of strings.

Python  
@agent.tool  
def retrieve_context(query: str) \-\> str:  
 """Searches the knowledge base for relevant information."""  
 \#... (Implement keyword or simple vector search)...  
 retrieved_snippets \= "..." \# Placeholder for found text  
 return retrieved_snippets

5. **Design System Prompt:** Craft a system_prompt that instructs the LLM:
   - Its primary goal is to answer the user's question.
   - It _must_ use the retrieve_context tool to get relevant information.
   - It should base its answer _solely_ on the context provided by the tool.
   - If the context doesn't contain the answer, it should state that.
6. **Configure Agent:** Instantiate the Agent with the retrieve_context tool and the carefully designed system prompt. A simple string output (output_type=str) might suffice initially.
7. **Test:** Ask questions whose answers are present in the documents. Verify that the agent calls the retrieval tool and formulates an answer based on the returned context. Test questions whose answers are _not_ in the documents to ensure the agent responds appropriately.

This project introduces the RAG pattern, demonstrating how tools can be used to inject external knowledge into the LLM's reasoning process, guided by specific instructions in the system prompt.17 It highlights the interplay between retrieval components (implemented as tools) and the generative model.

### **Video 20: Project 4: Building a Custom Chatbot with Memory**

**Objective:** Create a more interactive chatbot that uses tools and maintains conversation history effectively.

This project combines several previously learned concepts to build a more functional chatbot capable of holding a coherent conversation, using multiple tools, and potentially leveraging context passed via dependency injection.3

**Steps:**

1. **Define Chatbot Persona & Capabilities:** Decide on the chatbot's role (e.g., a simple task assistant, a helpful Q\&A bot). Define its personality via the system_prompt.
2. **Implement Multiple Tools:** Create several distinct tools using @agent.tool, for example:
   - A simple calculator tool.
   - A tool to get the current date/time.
   - A basic web search tool (using a library like requests or a dedicated search API).
3. **Manage Conversation History:** Implement the core chat loop using the message_history pattern demonstrated in Video 10\. Store the list of messages between turns.  
   Python  
   \# Conceptual Chat Loop  
   history \=  
   while True:  
    user_input \= input("You: ")  
    if user_input.lower() \== 'quit':  
    break  
    result \= await agent.run(user_input, message_history=history) \# Use async run  
    print(f"AI: {result.output}")  
    history.extend(result.new_messages())

4. **(Optional) Use Dependency Injection:** If the chatbot needs user-specific context (e.g., preferences, session ID) that should persist across turns but isn't part of the LLM's direct memory, use the DI system (deps_type, RunContext, deps argument) to pass this state into tools or dynamic prompts.
5. **Configure Agent:** Instantiate the Agent with the defined system prompt, all the tools, and potentially deps_type if using DI.
6. **Test Multi-Turn Conversations:** Engage in conversations that require:
   - Using different tools (e.g., "What is 5+7? Also, what's the date?").
   - Recalling information from previous turns (e.g., "My favorite color is blue." \-\> later \-\> "What color did I say I liked?").
   - Combining tool use and memory.
7. **Refine:** Adjust prompts and tool logic to improve conversational flow and accuracy.

This project serves as a capstone for the core PydanticAI features, showing how Agent, @agent.tool, message_history, and potentially Dependency Injection come together to create interactive, stateful, and capable AI applications.3

## **Part 5: Ecosystem & Conclusion**

### **Video 21: PydanticAI vs. The World (LangChain, LlamaIndex)**

**Objective:** Provide a more detailed comparison of PydanticAI with major alternative frameworks.

Choosing the right framework is crucial for developing AI agent applications efficiently. This section provides a comparative analysis of PydanticAI against two other popular frameworks: LangChain and LlamaIndex.

PydanticAI Recap:  
PydanticAI's core strengths lie in its simplicity, strong emphasis on type safety inherited from Pydantic, an ergonomic developer experience reminiscent of FastAPI, a primary focus on input/output validation and structured data handling, a clean dependency injection system, and seamless integration with Pydantic Logfire for observability.1  
LangChain Overview:  
LangChain is known for its comprehensive nature and large ecosystem. It offers extensive capabilities for building complex chains of LLM calls and sophisticated agents with numerous integrations for tools and data sources. It features advanced memory modules for managing conversation history and state. The LangChain ecosystem also includes LangSmith for debugging/evaluation and LangServe for deploying chains as APIs.11 However, this breadth can sometimes lead to criticisms regarding its complexity, level of abstraction, and potential maintainability challenges.19  
LlamaIndex Overview:  
LlamaIndex primarily excels in building applications that leverage data, particularly for Retrieval-Augmented Generation (RAG). Its strengths lie in data indexing, ingestion pipelines (connecting to various data sources), and sophisticated retrieval and query engines optimized for querying private or external data alongside LLMs.19 While it supports agentic capabilities, its core focus has traditionally been on the data interaction aspects of LLM applications.  
**Feature Comparison:**

| Feature                          | PydanticAI                                                                  | LangChain                                                                       | LlamaIndex                                                                      |
| :------------------------------- | :-------------------------------------------------------------------------- | :------------------------------------------------------------------------------ | :------------------------------------------------------------------------------ |
| **Core Purpose**                 | Building production-grade, validation-focused AI apps with FastAPI feel     | General-purpose framework for composing LLM applications (chains, agents)       | Framework for connecting LLMs with external data (indexing, retrieval, RAG)     |
| **Primary Strength**             | Structured Output Validation, Type Safety, Simplicity, Developer Experience | Breadth of Integrations, Complex Agent/Chain Orchestration, Ecosystem Size      | Data Indexing & Retrieval, RAG Pipelines, Querying External Data                |
| **Data Validation**              | Strong, built-in via Pydantic models (core feature)                         | Basic validation via output parsers, custom logic; less central than PydanticAI | Primarily focused on data ingestion/querying; output validation less emphasized |
| **Tool Definition**              | Simple, Pythonic (decorators, type hints, docstrings)                       | Flexible via various base classes and decorators; potentially more boilerplate  | Tools often integrated within query engines or agent loops                      |
| **Agent Complexity**             | Designed for simpler, structured agents; Graph for complex flows            | Supports highly complex, multi-step agents and autonomous workflows             | Supports agents, often focused on data interaction and RAG                      |
| **Graph/Workflow Orchestration** | Pydantic Graph integration for complex, stateful workflows                  | LangGraph library for building complex, cyclical agent workflows                | Less emphasis on a dedicated graph framework; uses query pipelines/loops        |
| **Memory Management**            | Basic context passing via message_history; less opinionated                 | Multiple built-in, sophisticated memory modules (buffer, summary, vector)       | Context management primarily within query/retrieval process                     |
| **RAG Focus**                    | Supports RAG via tools, but not the primary focus                           | Supports RAG components, but less specialized than LlamaIndex                   | Strong focus and specialization in RAG pipelines and data components            |
| **Streaming Support**            | Built-in streaming for text and validated structured output                 | Supports streaming output through various components                            | Supports streaming in query responses                                           |
| **Debugging/Observability**      | Seamless integration with Pydantic Logfire (first-party)                    | LangSmith integration for tracing, debugging, evaluation                        | Integrations with observability tools, including LangSmith                      |
| **Ecosystem/Integrations**       | Growing; strong ties to Pydantic/FastAPI ecosystem                          | Very large; extensive integrations with LLMs, tools, vector stores              | Large; focused on data loaders, vector stores, data formats                     |
| **Learning Curve**               | Generally considered simpler, especially for Pydantic users                 | Can be steep due to breadth and abstractions                                    | Moderate, especially if focused on core RAG concepts                            |

**When to Choose Which:**

- **Choose PydanticAI if:** The priority is robust data validation and structured output from LLMs. Type safety and a clean, Pythonic developer experience (similar to FastAPI) are important. The project involves building reliable, production-focused agents where maintainability is key, potentially integrating within an existing Pydantic/FastAPI stack.10
- **Choose LangChain if:** The project requires building highly complex, multi-step agentic workflows with intricate logic. Access to a vast range of pre-built integrations, advanced memory types, or the broader LangChain ecosystem (LangSmith/LangServe) is needed.11
- **Choose LlamaIndex if:** The core task involves building sophisticated RAG applications, requiring advanced data ingestion, indexing, and retrieval strategies over large or diverse datasets.30

It's also worth noting that these frameworks are not always mutually exclusive. It's sometimes feasible to use components from different frameworks together, for instance, using LlamaIndex for its data indexing and retrieval capabilities within an agent orchestrated by PydanticAI or LangChain.34 The choice depends heavily on the specific project requirements and developer preferences regarding abstraction, complexity, and core focus.

### **Video 22: Integrating PydanticAI with FastAPI**

**Objective:** Show how to expose a PydanticAI agent via a FastAPI web API.

FastAPI and PydanticAI share philosophical roots, both originating from the Pydantic ecosystem and emphasizing type safety and developer experience.1 Integrating a PydanticAI agent into a FastAPI application is therefore a natural fit, allowing developers to build web services that leverage LLM capabilities.

**Integration Steps:**

1.  **FastAPI Setup:** Begin with a standard FastAPI application structure. Ensure FastAPI and an ASGI server like Uvicorn are installed (pip install fastapi uvicorn).  
    Python  
    \# main.py  
    from fastapi import FastAPI  
    from pydantic import BaseModel

    app \= FastAPI()

    class QueryRequest(BaseModel):  
     query: str  
     session_id: str | None \= None \# Optional session ID for history

    class AgentResponse(BaseModel):  
     response: str  
     \# Add other fields as needed, e.g., session_id, usage_info

    Here, Pydantic models are used (as is standard in FastAPI) to define the expected structure of incoming requests (QueryRequest) and outgoing responses (AgentResponse).

2.  **Instantiate Agent:** Instantiate the PydanticAI agent within the FastAPI application scope. This might be done globally (for simple cases) or using FastAPI's dependency injection system for better management.  
    Python  
    from pydantic_ai import Agent  
    \# Assume agent is configured appropriately  
    agent \= Agent('openai:gpt-4o', output_type=str) \# Example

3.  **Create API Endpoint:** Define a FastAPI endpoint (e.g., using @app.post("/chat")) that will receive user queries. This endpoint function should be asynchronous (async def).  
    Python  
    @app.post("/chat", response_model=AgentResponse)  
    async def chat_endpoint(request: QueryRequest):  
     \# Endpoint logic here  
     pass

4.  **Call Agent Asynchronously:** Within the endpoint handler, call the agent's asynchronous run method, passing the user's query from the request body. If managing conversation history, retrieve the relevant history (perhaps based on request.session_id) and pass it via the message_history argument.  
    Python  
    @app.post("/chat", response_model=AgentResponse)  
    async def chat_endpoint(request: QueryRequest):  
     user_query \= request.query  
     \# Placeholder for history management based on session_id  
     message_history \= \# Retrieve history for request.session_id if provided

        \# Call the agent asynchronously
        result \= await agent.run(user\_query, message\_history=message\_history)

        \# Prepare and return the response
        return AgentResponse(response=result.output)

5.  **Handle Streaming (Optional):** To provide real-time responses over the API, use FastAPI's StreamingResponse. The endpoint should call agent.run_stream(...) and return an async generator that yields the chunks from the agent's stream.  
    Python  
    from fastapi.responses import StreamingResponse  
    import asyncio

    async def stream_agent_response(query: str, history: list):  
     async with agent.run_stream(query, message_history=history) as stream:  
     async for chunk in stream:  
     \# Process/format chunk if needed (e.g., Server-Sent Events)  
     yield str(chunk) \# Simplest case: yield text chunks  
     await asyncio.sleep(0.01) \# Small sleep to prevent blocking event loop

    @app.post("/chat_stream")  
    async def chat_stream_endpoint(request: QueryRequest):  
     \# Placeholder for history management  
     message_history \=  
     return StreamingResponse(  
     stream_agent_response(request.query, message_history),  
     media_type="text/plain" \# Adjust media type as needed  
     )

This integration leverages FastAPI's native async capabilities and its seamless use of Pydantic for request/response validation, creating a robust and efficient way to expose PydanticAI agents as web services or microservices. A complete example demonstrating a chat application with FastAPI is often available in the PydanticAI examples section.17

### **Video 23: Testing Your PydanticAI Agents**

**Objective:** Introduce strategies for testing PydanticAI applications.

Ensuring the reliability and correctness of AI applications is critical, especially in production. Testing PydanticAI agents involves strategies similar to testing other software, but with specific considerations for mocking LLM interactions and dependencies. PydanticAI's design facilitates testability.1

**Testing Strategies:**

1. **Use Standard Test Harness:** Employ standard Python testing frameworks like pytest for organizing and running tests.8
2. **Mock LLM Interactions:** Making actual LLM API calls during automated tests is slow, expensive, and non-deterministic. PydanticAI provides utilities to replace the real LLM with test doubles 8:
   - **TestModel:** A mock model useful for simulating specific responses or errors.
   - **FunctionModel:** Allows defining a Python function that dynamically generates responses based on the input prompt, useful for more complex mocking scenarios. These test models can be used directly when instantiating an Agent during tests, or applied globally using Agent.override.8
   - **Prevent Accidental Real Calls:** Set the environment variable ALLOW_MODEL_REQUESTS=False globally during tests to prevent any accidental calls to actual LLM APIs if mocks are misconfigured.8
3. **Unit Test Tools:** Tool functions (decorated with @agent.tool or @agent.tool_plain) are standard Python functions and should be unit tested independently. Use standard mocking libraries (like unittest.mock or pytest-mock) to isolate the tool from external dependencies (e.g., mock httpx calls for API tools, mock database connections).15
4. **Leverage Dependency Injection for Testing:** The DI system is highly beneficial for testing. When testing an agent or a tool that uses dependencies (RunContext), simply pass mock instances of the dependencies (e.g., a mock database connection) in the deps argument during the test run.1 This allows testing the agent/tool logic in isolation.
5. **Integration Testing:** Test the end-to-end flow of the agent, but with critical components (LLM, external APIs, databases) mocked. This verifies the interaction between the agent logic, tools, prompts, and validation, without external factors.
6. **Evaluation-Driven Development (Evals):** For assessing the _quality_ of the agent's output (not just functional correctness), evaluation frameworks or techniques might be employed. PydanticAI's documentation sometimes mentions "eval-driven iterative development" in the context of the DI system, suggesting it facilitates structured evaluation.1 This might involve running the agent against a predefined set of test cases (an "eval set") and comparing its output against expected results or using automated metrics.18 Logfire might also play a role in collecting data for evaluation.2

By incorporating these testing strategies, developers can build confidence in their PydanticAI applications. The framework's design, particularly the DI system and provision of test models, actively supports standard testing practices, contributing to its suitability for building production-ready systems.1

### **Video 24: Contributing & The PydanticAI Ecosystem**

**Objective:** Encourage community involvement and awareness of related tools.

PydanticAI is an open-source project, developed and maintained by the Pydantic team.4 Community involvement through usage, feedback, and contributions is vital for its growth and improvement.

**Getting Involved:**

- **GitHub Repository:** The primary hub for the project is the pydantic/pydantic-ai repository on GitHub.1 Here, users can find the source code, official documentation links, examples, report issues, and propose pull requests.
- **Community Channels:** Engage with other users and developers through GitHub Issues/Discussions (if enabled) or potentially relevant online communities like the r/PydanticAI subreddit.19
- **Contribution Process:** For those wishing to contribute code or documentation, a contribution guide is typically available (often CONTRIBUTING.md).26 The process generally involves:
  - Forking the repository.
  - Setting up the development environment, potentially using uv for dependency management and pre-commit for code quality checks.26
  - Running tests and linters using make commands.26
  - Submitting pull requests.
  - Specific rules exist for adding support for new LLM models, often requiring significant usage metrics for the model's underlying library or GitHub organization to manage maintainer workload.26 For other models, creating separate pydantic-ai-xxx packages is recommended.26

The Broader Pydantic Ecosystem:  
PydanticAI benefits from being part of a larger ecosystem of tools developed by the Pydantic organization 32:

- **Pydantic:** The core data validation library underpinning PydanticAI.22
- **Pydantic-Settings:** A library for managing application settings using Pydantic models, often used for loading configuration like API keys.15
- **Logfire:** The integrated observability platform for monitoring and debugging.2
- **Pydantic Graph:** The library used for orchestrating complex workflows.1

Other Integrations:  
PydanticAI is designed to work within the broader Python ecosystem. Examples and discussions show integrations with various other tools and platforms:

- Web Frameworks: FastAPI 1, Gradio.14
- Other AI/Agent Frameworks: Interoperability examples exist for using tools from CrewAI or LangChain within other frameworks (like AutoGen), suggesting potential for cross-compatibility patterns.36
- Deployment/Platform Tools: Apify 37, Riza.24
- Security Frameworks: Permit.io for fine-grained authorization.23

Understanding this ecosystem context helps developers leverage related tools effectively and appreciate PydanticAI's position within the rapidly evolving landscape of AI development in Python.

### **Video 25: Series Recap & Your Next Steps with PydanticAI**

**Objective:** Summarize key learnings and suggest avenues for further exploration.

This series has provided a comprehensive journey through the PydanticAI framework. Key concepts covered include:

- The central Agent class for orchestrating LLM interactions.
- Leveraging Pydantic models for robust output_type definition and validation.
- Equipping agents with capabilities using @agent.tool.
- Managing context and resources via the Dependency Injection system (RunContext, deps).
- Enabling real-time responses with basic and structured run_stream.
- Gaining observability using the integrated Pydantic Logfire.
- Optionally handling complex workflows with Pydantic Graph.
- The core philosophy emphasizing type safety, developer experience, and production readiness.

The primary benefits of using PydanticAI stem from this philosophy: a streamlined development process, particularly for those familiar with Pydantic/FastAPI; enhanced reliability through automatic data validation; improved maintainability via type safety and dependency injection; and features geared towards building robust, production-grade applications.1

For continued learning and deeper exploration, the following resources are highly recommended:

- **Official Documentation:** ai.pydantic.dev is the definitive source for guides, explanations, and API details.1
- **Examples:** Work through the official examples provided in the documentation or repository to see practical implementations.2
- **API Reference:** Consult the API reference for detailed information on classes, methods, and parameters.2
- **GitHub Repository:** Explore the source code, issues, and discussions on pydantic/pydantic-ai.1
- **Contribute:** Consider contributing back to the project if you find bugs or have ideas for improvements.26

To solidify understanding, viewers are encouraged to undertake their own challenge projects, applying the concepts learned to build custom AI agents tailored to their interests or needs.

PydanticAI is a relatively new but rapidly evolving framework. Its focus on structure, validation, and developer experience positions it as a compelling option in the growing landscape of AI agent frameworks. As the field of Generative AI continues to mature, frameworks like PydanticAI that prioritize robustness and integration with established software engineering practices are likely to play an increasingly important role.
