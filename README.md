# Agentic Framework

A modular framework for building AI agents with support for Discord integration, crew management, and memory systems.

## Features

- **Agent System**: Create and manage autonomous AI agents
- **Conversation Management**: Handle multi-turn conversations
- **Crew Management**: Orchestrate multiple agents working together
- **Memory System**: Persistent memory for agents
- **Streaming Events**: Real-time event streaming
- **LLM Client**: Unified interface for various LLM providers
- **Tool System**: Extensible tool framework

## Installation

### From GitHub (Recommended for now)

```bash
# Install latest version
pip install git+https://github.com/yourusername/discord-ai-app.git

# Install specific version/tag
pip install git+https://github.com/yourusername/discord-ai-app.git@v0.1.0

# Install in editable mode for development
pip install -e git+https://github.com/yourusername/discord-ai-app.git#egg=agentic-framework
```

### From PyPI (when published)

```bash
pip install agentic-framework
```

### Development Installation

```bash
git clone https://github.com/yourusername/discord-ai-app.git
cd discord-ai-app
pip install -e ".[dev]"
```

## Quick Start

```python
from agentic_framework import Agent, Crew

# Create an agent
agent = Agent(name="assistant")

# Create a crew of agents
crew = Crew(agents=[agent])

# Run the crew
result = crew.run("Hello, how are you?")
```

## Requirements

- Python 3.10+
- See `pyproject.toml` for full dependency list

## License

MIT License
