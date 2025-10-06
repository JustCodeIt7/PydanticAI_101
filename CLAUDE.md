# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PydanticAI_101 is a tutorial repository for learning PydanticAI, a Python framework for building AI agents with type safety and structured outputs. The repo contains example scripts organized by tutorial topics and follows a YouTube video series structure.

## Repository Structure

- `01-Configuring_LLMs/` - Examples of configuring different LLM providers (OpenAI, Gemini, Ollama)
- `02-Hello_World/` - Basic "hello world" agent examples
- `03-Agents/` - Advanced agent implementations showing core features
  - `01-basic_pydanticai_agent.py` - Basic agent patterns (sync, async, streaming, interactive chat)
  - `02-pydanticai_advanced_agents.py` - Advanced features (tools, dependencies, retries, conversation history)
- `examples/pydantic_ai_examples/` - Additional reference examples (bank support, RAG, SQL generation, etc.)
- `archive/` - Older tutorial content
- `outline.md` - Comprehensive tutorial series outline

## Development Setup

### Package Management
This project uses **uv** as the package manager. Dependencies are managed via `pyproject.toml` and locked in `uv.lock`.

**Install dependencies:**
```bash
uv sync
```

**Add new dependencies:**
```bash
uv add <package-name>
```

**Run scripts:**
```bash
uv run python <script-path>
```

### Python Environment
- Minimum Python version: 3.9
- Key dependencies: `pydantic-ai`, `yfinance`, `python-dotenv`

### API Keys
Scripts require API keys stored in `.env` files (git-ignored):
- `OPENAI_API_KEY` - For OpenAI models
- `GOOGLE_API_KEY` - For Gemini models
- Local Ollama setup at `http://100.95.122.242:11434/v1` (no key needed)

## Code Architecture

### PydanticAI Agent Patterns

**1. Basic Agent Setup**
```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

model = OpenAIModel(model_name='gpt-4', provider=OpenAIProvider(base_url=...))
agent = Agent(model, system_prompt="You are a helpful assistant.")
```

**2. Execution Methods**
- `agent.run_sync()` - Synchronous execution, returns `AgentRunResult`
- `await agent.run()` - Asynchronous execution
- `async with agent.run_stream()` - Streaming responses

**3. Agent Features**
- **Tools**: Functions decorated with `@agent.tool` that LLM can call
- **Dependencies**: Injected via `deps_type` and `RunContext[T]`, accessed in tools
- **Structured Output**: Specify `output_type` with Pydantic models for validated responses
- **System Prompts**: Static strings or dynamic functions with `@agent.system_prompt`
- **Message History**: Pass `message_history=result.new_messages()` for conversations
- **Retries**: Configure with `retries=N` for error handling and self-correction
- **Usage Limits**: Set token limits with `UsageLimits(response_tokens_limit=N)`

### Common Patterns in Codebase

**Ollama Local Model Setup:**
```python
base_url = 'http://100.95.122.242:11434/v1'
model_name = 'qwen3:1.7b'
ollama_model = OpenAIModel(
    model_name=model_name,
    provider=OpenAIProvider(base_url=base_url)
)
```

**Tool with Dependencies:**
```python
agent = Agent(model, deps_type=int)

@agent.tool
async def roulette_wheel(ctx: RunContext[int], square: int) -> str:
    return "winner" if square == ctx.deps else "loser"
```

**Conversation History:**
```python
first_result = agent.run_sync("Question 1")
second_result = agent.run_sync(
    "Question 2",
    message_history=first_result.new_messages()
)
```

## Code Style

- Uses Ruff for linting and formatting (configured in `pyproject.toml`)
- Line length: 100 characters
- Import organization: stdlib → third-party → first-party (pydanticai)
- Type hints required for function signatures
- Docstrings for tools must describe parameters and return values (LLM uses these)

## Testing

No formal test suite currently exists. Scripts are designed to be run individually for demonstration purposes.

**Run individual scripts:**
```bash
uv run python 01-Configuring_LLMs/03-llms.py
uv run python 03-Agents/01-basic_pydanticai_agent.py
```

## Tutorial Organization

The repository follows a video series structure (see `outline.md`):
- Part 1: Introduction & Foundations (Videos 1-5)
- Part 2: Core Agent Capabilities (Videos 6-11)
- Part 3: Advanced Features (Videos 12-15)

When adding new examples, follow the naming pattern: `<number>-<descriptive-name>.py` within the appropriate topic directory.

## Important Notes

- Scripts use different model providers for demonstration - check imports to understand which provider is active
- Examples in `03-Agents/archive/` are outdated; refer to numbered files in `03-Agents/` for current patterns
- The `.gitignore` excludes `examples/*` except archived content - examples directory contains reference implementations
- Dev outline files (`*-dev.md`) are working notes and not part of the final tutorial content
