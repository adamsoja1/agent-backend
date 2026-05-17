import pytest

from agentic_framework.tools.base import BaseTool, Skill, tool


def test_base_tool_executes_wrapped_callable():
    tool_instance = BaseTool(
        name="add",
        description="Add two numbers.",
        func=lambda left, right: left + right,
    )

    assert tool_instance.execute(2, 3) == 5
    assert str(tool_instance) == "add: Add two numbers."


def test_base_tool_without_callable_raises_clear_error():
    tool_instance = BaseTool(name="noop", description="No operation.")

    with pytest.raises(NotImplementedError, match="'noop' has no executable function"):
        tool_instance.execute()


def test_tool_schema_uses_type_hints_and_required_arguments():
    def create_item(name: str, quantity: int, fragile: bool = False, tags: list | None = None):
        return {"name": name, "quantity": quantity, "fragile": fragile, "tags": tags}

    tool_instance = BaseTool(
        name="create_item",
        description="Create inventory item.",
        func=create_item,
    )

    assert tool_instance.to_openai_schema() == {
        "type": "function",
        "function": {
            "name": "create_item",
            "description": "Create inventory item.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "quantity": {"type": "integer"},
                    "fragile": {"type": "boolean"},
                    "tags": {"type": "string"},
                },
                "required": ["name", "quantity"],
            },
        },
    }


def test_tool_decorator_uses_function_name_and_docstring_defaults():
    @tool
    def greet(name: str) -> str:
        """Greet someone."""
        return f"Hello, {name}"

    assert isinstance(greet, BaseTool)
    assert greet.name == "greet"
    assert greet.description == "Greet someone."
    assert greet.execute("Ada") == "Hello, Ada"


def test_tool_decorator_accepts_custom_metadata():
    @tool(name="custom_greet", description="Custom description.")
    def greet(name: str) -> str:
        return f"Hello, {name}"

    assert greet.name == "custom_greet"
    assert greet.description == "Custom description."


def test_skill_requires_tools_and_prefixes_activation_name():
    skill_tool = BaseTool(name="search", description="Search.", func=lambda query: query)

    skill = Skill(name="research", description="Research skill.", tools=[skill_tool])

    assert skill.name == "skill_research"
    assert skill.get_tool("search") is skill_tool
    assert skill.get_tools_schemas() == [skill_tool.to_openai_schema()]

    with pytest.raises(KeyError, match="Tool 'missing' not found"):
        skill.get_tool("missing")

    with pytest.raises(ValueError, match="A skill must have at least one tool"):
        Skill(name="empty", description="Empty skill.")
