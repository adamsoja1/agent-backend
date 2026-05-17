import asyncio

import pytest

from agentic_framework.core.agent import Agent
from agentic_framework.core.conversation import Conversation
from agentic_framework.core.crew import Crew
from agentic_framework.core.stream_events import DelegationEvent, ErrorEvent, FinalAnswerEvent


def collect_crew_events(crew, message):
    async def _collect():
        return [event async for event in crew.invoke(message)]

    return asyncio.run(_collect())


def test_crew_requires_entrypoint_to_be_listed_agent():
    entrypoint = Agent(name="entrypoint", model="test-model")
    outsider = Agent(name="outsider", model="test-model")

    with pytest.raises(ValueError, match="Entrypoint agent must be part"):
        Crew(agents=[outsider], entrypoint_agent=entrypoint)


def test_crew_registers_peer_agents_as_delegation_tools_once():
    router = Agent(name="router", model="test-model")
    specialist = Agent(name="specialist", model="test-model", description="Specialist.")

    crew = Crew(agents=[router, specialist], entrypoint_agent=router)
    crew._register_agents_as_tools()

    assert router.crew is crew
    assert specialist.crew is crew
    assert router.list_tools().count("delegate_to_agent_specialist") == 1
    assert specialist.list_tools().count("delegate_to_agent_router") == 1


def test_crew_can_register_peer_agents_as_ask_tools_with_shared_conversation():
    conversation = Conversation()
    router = Agent(name="router", model="test-model")
    specialist = Agent(name="specialist", model="test-model", description="Specialist.")

    Crew(
        agents=[router, specialist],
        entrypoint_agent=router,
        only_ask_for_info=True,
        shared_knowledge=True,
        conversation=conversation,
    )

    assert router.conversation is conversation
    assert specialist.conversation is conversation
    assert "ask_agent_specialist" in router.list_tools()
    assert "ask_agent_router" in specialist.list_tools()


def test_crew_follows_delegation_to_target_agent(monkeypatch):
    router = Agent(name="router", model="test-model")
    specialist = Agent(name="specialist", model="test-model")
    crew = Crew(agents=[router, specialist], entrypoint_agent=router)

    async def router_stream(message):
        yield DelegationEvent(agent_name="router", target_agent="specialist", task=message)

    async def specialist_stream(message):
        yield FinalAnswerEvent(agent_name="specialist", answer=f"handled {message}")

    monkeypatch.setattr(router, "stream", router_stream)
    monkeypatch.setattr(specialist, "stream", specialist_stream)

    events = collect_crew_events(crew, "task")

    assert events == [FinalAnswerEvent(agent_name="specialist", answer="handled task")]


def test_crew_emits_error_when_transfer_limit_is_exceeded(monkeypatch):
    router = Agent(name="router", model="test-model")
    specialist = Agent(name="specialist", model="test-model")
    crew = Crew(agents=[router, specialist], entrypoint_agent=router, transfer_limit=1)

    async def router_stream(message):
        yield DelegationEvent(agent_name="router", target_agent="specialist", task=message)

    async def specialist_stream(message):
        yield DelegationEvent(agent_name="specialist", target_agent="router", task=message)

    monkeypatch.setattr(router, "stream", router_stream)
    monkeypatch.setattr(specialist, "stream", specialist_stream)

    events = collect_crew_events(crew, "task")

    assert events == [
        ErrorEvent(agent_name="specialist", error="Transfer limit exceeded. Ending delegation.")
    ]
