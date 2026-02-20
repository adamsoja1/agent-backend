"""Discord server tools for AITIS swarm — plain functions, no CrewAI dependency."""

import asyncio
from typing import Optional
import discord
from discord.ext import commands
# ---------------------------------------------------------------------------
# Global state (set by bot.py at startup / on each message)
# ---------------------------------------------------------------------------

_bot_instance: Optional[commands.Bot] = None
_current_guild_id: Optional[int] = None


def set_bot_instance(bot: commands.Bot) -> None:
    global _bot_instance
    _bot_instance = bot


def set_current_guild_context(guild_id: int) -> None:
    global _current_guild_id
    _current_guild_id = guild_id


def _get_bot() -> commands.Bot:
    if _bot_instance is None:
        raise RuntimeError("Bot instance not set. Call set_bot_instance() first.")
    return _bot_instance


def _get_guild(guild_id: Optional[int] = None) -> discord.Guild:
    bot = _get_bot()
    gid = guild_id or _current_guild_id
    if gid:
        guild = bot.get_guild(gid)
        if guild:
            return guild
    if not bot.guilds:
        raise RuntimeError("Bot is not in any guilds.")
    return bot.guilds[0]


# ---------------------------------------------------------------------------
# Server information
# ---------------------------------------------------------------------------

def get_server_info(guild_id: Optional[int] = None) -> str:
    """Get general information about the Discord server."""
    try:
        g = _get_guild(guild_id)
        text_ch = sum(1 for c in g.channels if isinstance(c, discord.TextChannel))
        voice_ch = sum(1 for c in g.channels if isinstance(c, discord.VoiceChannel))
        owner = g.owner.name if g.owner else "N/A"
        return (
            f"Server: {g.name} (ID: {g.id})\n"
            f"Owner: {owner}\n"
            f"Members: {g.member_count}\n"
            f"Created: {g.created_at.strftime('%Y-%m-%d')}\n"
            f"Channels: {len(g.channels)} total ({text_ch} text, {voice_ch} voice)\n"
            f"Roles: {len(g.roles)}\n"
            f"Emojis: {len(g.emojis)}"
        )
    except Exception as e:
        return f"Error getting server info: {e}"


def list_members(guild_id: Optional[int] = None, limit: int = 20) -> str:
    """List recent members of the Discord server."""
    try:
        limit = 20
        g = _get_guild(guild_id)
        members = sorted(
            g.members, key=lambda m: m.joined_at or m.created_at, reverse=True
        )[:limit]
        lines = [f"Members of {g.name} (showing {len(members)}):"]
        for m in members:
            kind = "Bot" if m.bot else "User"
            roles = ", ".join(r.name for r in m.roles[1:]) or "None"
            lines.append(f"  {m.name}, {m.id} [{kind}] — roles: {roles}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing members: {e}"


def list_channels(guild_id: Optional[int] = None) -> str:
    """List all channels on the Discord server."""
    try:
        g = _get_guild(guild_id)
        text_ch = [c for c in g.channels if isinstance(c, discord.TextChannel)]
        voice_ch = [c for c in g.channels if isinstance(c, discord.VoiceChannel)]
        lines = [f"Channels in {g.name}:", "  Text:"]
        for c in text_ch:
            lines.append(f"    #{c.name} (ID: {c.id})")
        lines.append("  Voice:")
        for c in voice_ch:
            count = sum(1 for m in g.members if m.voice and m.voice.channel == c)
            lines.append(f"    {c.name} ({count} users) (ID: {c.id})")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing channels: {e}"


def list_roles(guild_id: Optional[int] = None) -> str:
    """List all roles on the Discord server with member counts."""
    try:
        g = _get_guild(guild_id)
        roles = sorted(g.roles, key=lambda r: r.position, reverse=True)
        lines = [f"Roles in {g.name}:"]
        for r in roles:
            lines.append(f"  {r.name} — {len(r.members)} members")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing roles: {e}"


def get_server_stats(guild_id: Optional[int] = None) -> str:
    """Get detailed statistics about the Discord server."""
    try:
        g = _get_guild(guild_id)
        online = sum(1 for m in g.members if m.status == discord.Status.online)
        idle   = sum(1 for m in g.members if m.status == discord.Status.idle)
        dnd    = sum(1 for m in g.members if m.status == discord.Status.dnd)
        offline= sum(1 for m in g.members if m.status == discord.Status.offline)
        bots   = sum(1 for m in g.members if m.bot)
        voice_users = sum(1 for m in g.members if m.voice)
        return (
            f"Stats for {g.name}:\n"
            f"  Total members: {g.member_count} ({g.member_count - bots} humans, {bots} bots)\n"
            f"  Online: {online} | Idle: {idle} | DnD: {dnd} | Offline: {offline}\n"
            f"  In voice channels: {voice_users}\n"
            f"  Channels: {len(g.channels)} | Roles: {len(g.roles)} | Emojis: {len(g.emojis)}"
        )
    except Exception as e:
        return f"Error getting server stats: {e}"


def get_member_info(member_id: str, guild_id: Optional[int] = None) -> str:
    """Get information about a specific Discord member by their user ID."""
    try:
        g = _get_guild(guild_id)
        member = g.get_member(int(member_id))
        if not member:
            return f"Member with ID {member_id} not found on this server."
        roles = ", ".join(r.name for r in member.roles[1:]) or "None"
        joined = member.joined_at.strftime("%Y-%m-%d") if member.joined_at else "N/A"
        return (
            f"Member: {member.name} (ID: {member.id})\n"
            f"  Bot: {member.bot}\n"
            f"  Account created: {member.created_at.strftime('%Y-%m-%d')}\n"
            f"  Joined server: {joined}\n"
            f"  Roles: {roles}"
        )
    except Exception as e:
        return f"Error getting member info: {e}"


def get_channel_info(channel_id: str, guild_id: Optional[int] = None) -> str:
    """Get information about a specific Discord channel by its ID."""
    try:
        g = _get_guild(guild_id)
        channel = g.get_channel(int(channel_id))
        if not channel:
            return f"Channel with ID {channel_id} not found."
        if isinstance(channel, discord.TextChannel):
            return (
                f"Text channel: #{channel.name} (ID: {channel.id})\n"
                f"  Topic: {channel.topic or 'None'}\n"
                f"  Category: {channel.category.name if channel.category else 'None'}\n"
                f"  NSFW: {channel.is_nsfw()}\n"
                f"  Created: {channel.created_at.strftime('%Y-%m-%d')}"
            )
        elif isinstance(channel, discord.VoiceChannel):
            users = sum(1 for m in g.members if m.voice and m.voice.channel == channel)
            return (
                f"Voice channel: {channel.name} (ID: {channel.id})\n"
                f"  Users currently in channel: {users}\n"
                f"  Bitrate: {channel.bitrate // 1000} kbps\n"
                f"  User limit: {channel.user_limit or 'unlimited'}\n"
                f"  Created: {channel.created_at.strftime('%Y-%m-%d')}"
            )
        return f"Channel type not supported: {type(channel).__name__}"
    except Exception as e:
        return f"Error getting channel info: {e}"



def create_text_channel(name: str, category_id: Optional[int] = None) -> str:
    """Create a new text channel on the Discord server."""
    try:
        bot = _get_bot()
        g = _get_guild()

        async def _create():
            category = g.get_channel(category_id) if category_id else None
            channel = await g.create_text_channel(name=name, category=category)
            return f"Created text channel: #{channel.name} (ID: {channel.id})"

        future = asyncio.run_coroutine_threadsafe(_create(), bot.loop)
        return future.result(timeout=10)
    except Exception as e:
        return f"Error creating channel: {e}"