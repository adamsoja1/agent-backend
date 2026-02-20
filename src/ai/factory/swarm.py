"""
AITIS Discord AI — multi-agent swarm with handoffs.

Flow:
  1. AITIS receives the message and decides which specialist to call (or answers directly).
  2. AITIS calls a special "handoff" tool (transfer_to_<specialist>).
  3. The loop swaps the active agent to that specialist.
  4. The specialist runs its own loop with its own tools until done.
  5. The final text is returned to Discord.

Agents:
  - aitis             → router / generalist, handles small talk & context
  - lol_specialist    → League of Legends questions
  - gaming_specialist → general gaming questions
  - tech_specialist   → programming / technical questions
  - server_specialist → Discord server info & management
  - researcher        → web research / general knowledge lookups
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from openai import OpenAI
from dotenv import load_dotenv

from ..tools import (
    web_search,
    league_of_legends_search,
    gaming_search,
    technical_search,
    search_scraped_website,
)
from ..discord_tools import (
    get_server_info,
    list_members,
    list_channels,
    list_roles,
    get_server_stats,
    get_member_info,
    get_channel_info,
    create_text_channel,
)

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenAI-compatible client (Ollama / LM Studio / OpenAI)
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL", "https://ollama.com/v1"),
    api_key=os.getenv("OLLAMA_API_KEY", "ollama"),   
)
MODEL = os.getenv("LLM_MODEL", "llama3.1")

# ---------------------------------------------------------------------------
# Tool schema helpers
# ---------------------------------------------------------------------------

def _fn(name: str, description: str, parameters: dict | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters or {"type": "object", "properties": {}},
        },
    }

# ── Handoff tools (used only by AITIS) ──────────────────────────────────────

HANDOFF_LOL = _fn(
    "transfer_to_lol_specialist",
    "Transfer to the League of Legends specialist for LoL champion, item, patch, or meta questions.",
)
HANDOFF_GAMING = _fn(
    "transfer_to_gaming_specialist",
    "Transfer to the gaming specialist for general gaming news, guides, or game recommendations.",
)
HANDOFF_TECH = _fn(
    "transfer_to_tech_specialist",
    "Transfer to the technical specialist for programming, coding, or software questions.",
)
HANDOFF_SERVER = _fn(
    "transfer_to_server_specialist",
    "Transfer to the server specialist for Discord server info, members, channels, stats, or moderation.",
)
HANDOFF_RESEARCHER = _fn(
    "transfer_to_researcher",
    "Transfer to the web researcher for general knowledge, current events, or any topic requiring web search.",
)

# ── Specialist tool schemas ──────────────────────────────────────────────────

TOOL_WEB_SEARCH = _fn(
    "web_search",
    "Search the web for up-to-date information.",
    {
        "type": "object",
        "properties": {
            "query":       {"type": "string"},
            "max_results": {"type": "integer"},
        },
        "required": ["query"],
    },
)
TOOL_LOL_SEARCH = _fn(
    "league_of_legends_search",
    "Search LoL champion data, items, patch notes, and meta.",
    {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
)
TOOL_GAMING_SEARCH = _fn(
    "gaming_search",
    "Search gaming news, reviews, and guides.",
    {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
)
TOOL_TECH_SEARCH = _fn(
    "technical_search",
    "Search Stack Overflow, GitHub, and programming docs.",
    {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
)
TOOL_SCRAPED = _fn(
    "search_scraped_website",
    "Scrape a URL and return sections relevant to given keywords.",
    {
        "type": "object",
        "properties": {
            "url":      {"type": "string"},
            "keywords": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["url", "keywords"],
    },
)
TOOL_SERVER_INFO    = _fn("get_server_info",       "Get general Discord server info.")
TOOL_LIST_MEMBERS   = _fn("list_members",          "List server members.", {"type": "object", "properties": {"limit": {"type": "integer"}}})
TOOL_LIST_CHANNELS  = _fn("list_channels",         "List all server channels.")
TOOL_LIST_ROLES     = _fn("list_roles",            "List all server roles.")
TOOL_SERVER_STATS   = _fn("get_server_stats",      "Get server statistics.")
TOOL_MEMBER_INFO    = _fn("get_member_info",       "Get info about a member.", {"type": "object", "properties": {"member_id": {"type": "string"}}, "required": ["member_id"]})
TOOL_CHANNEL_INFO   = _fn("get_channel_info",      "Get info about a channel.", {"type": "object", "properties": {"channel_id": {"type": "string"}}, "required": ["channel_id"]})
TOOL_CREATE_CHANNEL = _fn("create_text_channel",   "Create a new text channel.", {"type": "object", "properties": {"name": {"type": "string"}, "category_id": {"type": "integer"}}, "required": ["name"]})

# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

@dataclass
class Agent:
    name: str
    system: str
    tools: list[dict] = field(default_factory=list)


AGENTS: dict[str, Agent] = {

    "aitis": Agent(
        name="AITIS",
        system=(
            "You are AITIS, the friendly community assistant for the Unia Ksiazenice Discord server. "
            "Your job is to greet users, answer simple questions directly, and route complex questions "
            "to the right specialist using the transfer tools. "
            "Rules:\n"
            "- Small talk, greetings, simple factual answers → answer directly, no transfer needed.\n"
            "- League of Legends questions → transfer_to_lol_specialist\n"
            "- General gaming questions → transfer_to_gaming_specialist\n"
            "- Programming / technical questions → transfer_to_tech_specialist\n"
            "- Server info, members, channels, moderation → transfer_to_server_specialist\n"
            "- General knowledge, current events, web lookups → transfer_to_researcher\n"
            "Reply in the SAME LANGUAGE as the user. No markdown. Plain Discord text only.\n"
            "Current time: {now}"
        ),
        tools=[
            HANDOFF_LOL,
            HANDOFF_GAMING,
            HANDOFF_TECH,
            HANDOFF_SERVER,
            HANDOFF_RESEARCHER,
        ],
    ),

    "lol_specialist": Agent(
        name="LoL Specialist",
        system=(
            "You are AITIS's League of Legends expert. "
            "You have deep knowledge of champions, items, runes, patch notes, and the current meta. "
            "Use your search tools to find accurate, up-to-date LoL information. "
            "Be concise and helpful. No markdown. Plain Discord text only. "
            "Reply in the SAME LANGUAGE as the user.\n"
            "Current time: {now}"
        ),
        tools=[TOOL_LOL_SEARCH, TOOL_SCRAPED, TOOL_WEB_SEARCH],
    ),

    "gaming_specialist": Agent(
        name="Gaming Specialist",
        system=(
            "You are AITIS's gaming expert. "
            "You know about many games, genres, gaming news, and can give recommendations. "
            "Use your search tools when needed. "
            "Be concise and enthusiastic. No markdown. Plain Discord text only. "
            "Reply in the SAME LANGUAGE as the user.\n"
            "Current time: {now}"
        ),
        tools=[TOOL_GAMING_SEARCH, TOOL_SCRAPED, TOOL_WEB_SEARCH],
    ),

    "tech_specialist": Agent(
        name="Tech Specialist",
        system=(
            "You are AITIS's programming and technical expert. "
            "You help with coding questions, debugging, software tools, and technical concepts. "
            "Use your search tools to find accurate answers. "
            "Be clear and practical. No markdown. Plain Discord text only. "
            "Reply in the SAME LANGUAGE as the user.\n"
            "Current time: {now}"
        ),
        tools=[TOOL_TECH_SEARCH, TOOL_SCRAPED, TOOL_WEB_SEARCH],
    ),

    "server_specialist": Agent(
        name="Server Specialist",
        system=(
            "You are AITIS's Discord server management expert. "
            "You have access to server info, members, channels, roles, stats, and audit logs. "
            "Use your tools to retrieve accurate server data. "
            "Be organized and clear. No markdown. Plain Discord text only. "
            "Reply in the SAME LANGUAGE as the user.\n"
            "Current time: {now}"
        ),
        tools=[
            TOOL_SERVER_INFO, TOOL_LIST_MEMBERS, TOOL_LIST_CHANNELS,
            TOOL_LIST_ROLES, TOOL_SERVER_STATS, TOOL_MEMBER_INFO,
            TOOL_CHANNEL_INFO, TOOL_CREATE_CHANNEL,
        ],
    ),

    "researcher": Agent(
        name="Researcher",
        system=(
            "You are AITIS's web research expert. "
            "You find accurate, current information on any topic using web search. "
            "Summarize results clearly and concisely. No markdown. Plain Discord text only. "
            "Reply in the SAME LANGUAGE as the user.\n"
            "Current time: {now}"
        ),
        tools=[TOOL_WEB_SEARCH, TOOL_SCRAPED],
    ),
}

# Handoff tool name → agent key
HANDOFF_MAP: dict[str, str] = {
    "transfer_to_lol_specialist":    "lol_specialist",
    "transfer_to_gaming_specialist": "gaming_specialist",
    "transfer_to_tech_specialist":   "tech_specialist",
    "transfer_to_server_specialist": "server_specialist",
    "transfer_to_researcher":        "researcher",
}

# ---------------------------------------------------------------------------
# Tool callable registry
# ---------------------------------------------------------------------------

CALLABLE: dict[str, Any] = {
    "web_search":               web_search,
    "league_of_legends_search": league_of_legends_search,
    "gaming_search":            gaming_search,
    "technical_search":         technical_search,
    "search_scraped_website":   search_scraped_website,
    "get_server_info":          get_server_info,
    "list_members":             list_members,
    "list_channels":            list_channels,
    "list_roles":               list_roles,
    "get_server_stats":         get_server_stats,
    "get_member_info":          get_member_info,
    "get_channel_info":         get_channel_info,
    "create_text_channel":      create_text_channel,
}

# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

def _run_tool(name: str, inputs: dict) -> str:
    fn = CALLABLE.get(name)
    if fn is None:
        return f"[error] Unknown tool: {name}"
    try:
        result = fn(**inputs)
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
    except Exception as exc:
        logger.exception("Tool %s raised an exception", name)
        return f"[error] {name} failed: {exc}"

# ---------------------------------------------------------------------------
# Single-agent loop
# ---------------------------------------------------------------------------

def _run_agent_loop(
    agent: Agent,
    messages: list[dict],
    now: str,
    max_iterations: int = 8,
) -> tuple[str | None, str | None]:
    """
    Run one agent until it either:
      - Returns a final text answer  → (answer, None)
      - Requests a handoff           → (None, target_agent_key)
    """
    system = agent.system.format(now=now)
    full_messages = [{"role": "system", "content": system}] + messages

    for iteration in range(max_iterations):
        is_last = iteration == max_iterations - 1
        # On the last iteration force a plain text answer — no more tool calls
        forced_tool_choice = "none" if is_last else ("auto" if agent.tools else "none")

        response = client.chat.completions.create(
            model=MODEL,
            messages=(
                full_messages + [{
                    "role": "user",
                    "content": "Please now give your final answer directly as plain text. Do not call any more tools.",
                }]
                if is_last else full_messages
            ),
            tools=agent.tools if agent.tools and not is_last else [],
            tool_choice=forced_tool_choice,
        )

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        logger.debug("[%s] finish_reason=%s", agent.name, finish_reason)

        # ── No tool calls → final text answer ───────────────────────────────
        if finish_reason == "stop" or not msg.tool_calls:
            return (msg.content or "").strip(), None

        full_messages.append(msg)

        # ── Process tool calls ───────────────────────────────────────────────
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            logger.info("[%s] Tool call: %s", agent.name, tool_name)

            # Handoff → return immediately with target agent key
            if tool_name in HANDOFF_MAP:
                target = HANDOFF_MAP[tool_name]
                logger.info("[%s] Handoff to: %s", agent.name, target)
                return None, target

            # Regular tool call
            try:
                inputs = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                inputs = {}

            result = _run_tool(tool_name, inputs)
            full_messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            })

    return "Przepraszam, nie udało mi się przetworzyć tego zapytania.", None

# ---------------------------------------------------------------------------
# Swarm orchestrator
# ---------------------------------------------------------------------------

def _run_swarm(user_message: str, max_handoffs: int = 5) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    messages = [{"role": "user", "content": user_message}]

    current_agent_key = "aitis"

    for hop in range(max_handoffs + 1):
        agent = AGENTS[current_agent_key]
        logger.info("=== Active agent: %s (hop %d) ===", agent.name, hop)

        answer, next_agent_key = _run_agent_loop(agent, messages, now)

        if next_agent_key:
            # Handoff: tell the next agent who sent it and why
            messages.append({
                "role": "user",
                "content": (
                    f"[Handoff from {agent.name}] "
                    "Please handle the user's question using your expertise and tools."
                ),
            })
            current_agent_key = next_agent_key
            continue

        # We have a final answer
        return answer or "Przepraszam, brak odpowiedzi."

    return "Przepraszam, nie udało mi się przetworzyć tego zapytania."

# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def answer_discord_question(question: str, context: str = "") -> str:
    """Called by the Discord bot. Returns a plain-text answer."""
    question = question.strip()
    if not question:
        return "Nie rozumiem pytania. Napisz coś więcej!"

    user_content = question
    if context:
        user_content = (
            "[Recent conversation context — for reference only]\n"
            f"{context}\n"
            "[End context]\n\n"
            + user_content
        )

    answer = _run_swarm(user_content)

    if len(answer) > 1900:
        answer = answer[:1850] + "\n... (skrócono)"

    return answer