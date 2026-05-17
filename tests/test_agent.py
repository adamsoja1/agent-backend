import asyncio
from types import SimpleNamespace

import pytest

from agentic_framework.core.agent import Agent
from agentic_framework.core.stream_events import (
    FinalAnswerEvent,
    TextDeltaEvent,
    ToolCallStartEvent,
    ToolResultEvent,
)
from agentic_framework.tools.base import BaseTool, Skill


class FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk


class FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeStream(self._responses.pop(0))


class FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


def chunk(content="", tool_calls=None, finish_reason=None, reasoning_content=None):
    delta = SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
        reasoning_content=reasoning_content,
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


def tool_call_delta(index, call_id=None, name=None, arguments=""):
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=call_id, function=function)


def collect_events(agent, message):
    async def _collect():
        return [event async for event in agent.stream(message)]

    return asyncio.run(_collect())


def test_agent_streams_final_answer_and_records_visible_messages():
    client = FakeClient(
        responses=[
            [
                chunk(content="Hello, ", reasoning_content="hidden "),
                chunk(content="world!", finish_reason="stop"),
            ]
        ]
    )
    agent = Agent(name="assistant", model="test-model", client=client)

    events = collect_events(agent, "Hi")

    assert [event.delta for event in events if isinstance(event, TextDeltaEvent)] == [
        "Hello, ",
        "world!",
    ]
    assert events[-1] == FinalAnswerEvent(agent_name="assistant", answer="Hello, world!")
    assert agent.conversation.get_messages() == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello, world!"},
    ]

    create_call = client.chat.completions.calls[0]
    assert create_call["model"] == "test-model"
    assert create_call["stream"] is True
    assert "tools" not in create_call
    assert "tool_choice" not in create_call


def test_agent_executes_tool_call_then_continues_to_final_answer():
    echo_tool = BaseTool(
        name="echo",
        description="Echo text.",
        func=lambda text: text.upper(),
    )
    client = FakeClient(
        responses=[
            [
                chunk(
                    tool_calls=[
                        tool_call_delta(
                            index=0,
                            call_id="call_1",
                            name="echo",
                            arguments='{"text": "hello"}',
                        )
                    ],
                    finish_reason="tool_calls",
                )
            ],
            [chunk(content="Done.", finish_reason="stop")],
        ]
    )
    agent = Agent(name="assistant", model="test-model", client=client, tools=[echo_tool])

    events = collect_events(agent, "Echo this")

    assert any(
        event == ToolCallStartEvent(
            agent_name="assistant",
            call_id="call_1",
            tool_name="echo",
            arguments_raw='{"text": "hello"}',
        )
        for event in events
    )
    assert any(
        event == ToolResultEvent(
            agent_name="assistant",
            call_id="call_1",
            tool_name="echo",
            result="HELLO",
            is_error=False,
        )
        for event in events
    )
    assert events[-1] == FinalAnswerEvent(agent_name="assistant", answer="Done.")
    assert agent.conversation.get_messages() == [
        {"role": "user", "content": "Echo this"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"text": "hello"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "HELLO"},
        {"role": "assistant", "content": "Done."},
    ]


def test_agent_records_tool_parse_errors_as_tool_results():
    tool_instance = BaseTool(name="echo", description="Echo text.", func=lambda text: text)
    client = FakeClient(
        responses=[
            [
                chunk(
                    tool_calls=[
                        tool_call_delta(
                            index=0,
                            call_id="call_1",
                            name="echo",
                            arguments='{"text": ',
                        )
                    ],
                    finish_reason="tool_calls",
                )
            ],
            [chunk(content="Recovered.", finish_reason="stop")],
        ]
    )
    agent = Agent(name="assistant", model="test-model", client=client, tools=[tool_instance])

    events = collect_events(agent, "Call badly")

    assert events[-1] == FinalAnswerEvent(agent_name="assistant", answer="Recovered.")
    tool_message = agent.conversation.get_messages()[2]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_1"
    assert "Failed to parse arguments for tool 'echo'" in tool_message["content"]


def test_agent_parse_tool_arguments_truncates_extra_json_data():
    parsed, clean = Agent._parse_tool_arguments('{"text": "hi"} trailing')

    assert parsed == {"text": "hi"}
    assert clean == '{"text": "hi"}'


def test_agent_execute_tool_supports_sync_and_async_callables():
    async def async_echo(text):
        return f"async {text}"

    agent = Agent(
        name="assistant",
        model="test-model",
        tools=[
            BaseTool(name="sync_echo", description="Sync.", func=lambda text: f"sync {text}"),
            BaseTool(name="async_echo", description="Async.", func=async_echo),
        ],
    )

    assert asyncio.run(agent._execute_tool("sync_echo", {"text": "ok"})) == "sync ok"
    assert asyncio.run(agent._execute_tool("async_echo", {"text": "ok"})) == "async ok"

    with pytest.raises(ValueError, match="Tool 'missing' not found"):
        asyncio.run(agent._execute_tool("missing", {}))


def test_agent_skill_activation_switches_tool_overlay_and_can_deactivate():
    base_tool = BaseTool(name="base", description="Base.", func=lambda: "base")
    research_tool = BaseTool(name="search", description="Search.", func=lambda query: query)
    write_tool = BaseTool(name="draft", description="Draft.", func=lambda text: text)
    research_skill = Skill(name="research", description="Research.", tools=[research_tool])
    writing_skill = Skill(name="writing", description="Writing.", tools=[write_tool])
    agent = Agent(
        name="assistant",
        model="test-model",
        tools=[base_tool],
        skills=[research_skill, writing_skill],
    )

    assert agent.list_tools() == ["base", "skill_research", "skill_writing"]

    assert agent._activate_skill(research_skill) == "Skill 'skill_research' activated."
    assert set(agent.list_tools()) == {"base", "skill_research", "skill_writing", "search"}

    assert (
        agent._activate_skill(writing_skill)
        == "Skill 'skill_research' deactivated. Skill 'skill_writing' activated."
    )
    assert set(agent.list_tools()) == {"base", "skill_research", "skill_writing", "draft"}

    assert agent._deactivate_skill() == "Skill 'skill_writing' deactivated."
    assert agent.list_tools() == ["base", "skill_research", "skill_writing"]
