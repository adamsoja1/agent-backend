from agentic_framework.core.conversation import (
    AssistantToolCallEvent,
    Conversation,
    MessageEvent,
    ReasoningEvent,
    TextDeltaEvent,
    ToolResultEvent,
)


def test_conversation_serializes_llm_visible_messages_only():
    conversation = Conversation(system_prompt="You are concise.")
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"query": "weather"}'},
        }
    ]

    conversation.add_user_message("Hello")
    conversation.add_reasoning("hidden chain of thought")
    conversation.add_delta("streamed text")
    conversation.add_assistant_tool_calls("Let me check.", tool_calls)
    conversation.add_tool_result("call_1", "Sunny")
    conversation.add_assistant_message("It is sunny.")

    assert conversation.get_messages() == [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Let me check.", "tool_calls": tool_calls},
        {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
        {"role": "assistant", "content": "It is sunny."},
    ]


def test_conversation_helpers_preserve_event_history():
    conversation = Conversation()

    conversation.add_user_message("first")
    conversation.add_assistant_message("second")
    conversation.add_tool_result("call_1", "third")

    assert len(conversation) == 3
    assert conversation.get_last_messages(2) == [
        {"role": "assistant", "content": "second"},
        {"role": "tool", "tool_call_id": "call_1", "content": "third"},
    ]

    conversation.clear()

    assert len(conversation) == 0
    assert conversation.get_messages() == []


def test_dump_events_keeps_event_specific_fields():
    conversation = Conversation()

    conversation.events.extend(
        [
            MessageEvent(role="user", content="visible"),
            AssistantToolCallEvent(content="", tool_calls=[]),
            ToolResultEvent(tool_call_id="call_1", content="ok"),
            ReasoningEvent(content="hidden"),
            TextDeltaEvent(delta="partial"),
        ]
    )

    dumped = conversation.dump_events()

    assert dumped[0]["role"] == "user"
    assert dumped[1]["tool_calls"] == []
    assert dumped[2]["tool_call_id"] == "call_1"
    assert dumped[3]["content"] == "hidden"
    assert dumped[4]["delta"] == "partial"
