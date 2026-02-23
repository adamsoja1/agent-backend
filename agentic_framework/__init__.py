"""Agentic Framework - A modular framework for building AI agents.

This package provides tools and components for building AI-powered agents
with support for Discord integration, crew management, and memory systems.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your@email.com"

from agentic_framework.core.agent import Agent
from agentic_framework.core.conversation import Conversation
from agentic_framework.core.crew import Crew
from agentic_framework.core.memory import Memory
from agentic_framework.core.stream_events import StreamEvents
from agentic_framework.llm.client import LLMClient
from agentic_framework.llm.prompt import PromptManager
from agentic_framework.tools.base import BaseTool
from agentic_framework.tools.tool import Tool

__all__ = [
    "Agent",
    "Conversation",
    "Crew",
    "Memory",
    "StreamEvents",
    "LLMClient",
    "PromptManager",
    "BaseTool",
    "Tool",
]
