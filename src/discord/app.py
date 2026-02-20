import os
import logging
import asyncio

from dotenv import load_dotenv
import discord
from discord.ext import commands
from ..ai.factory.swarm import answer_discord_question   # <-- changed import
from ..ai import discord_tools

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
if not TOKEN:
    raise SystemExit("DISCORD_TOKEN not set.")

intents = discord.Intents.default()
intents.message_content = True
intents.members = True 
bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    logging.info(f"Logged in as {bot.user} (id: {bot.user.id})")
    print(f"Logged in as {bot.user} (id: {bot.user.id})")
    discord_tools.set_bot_instance(bot)


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    print(f"Message from {message.author}: {message.content}")
    discord_tools.set_current_guild_context(message.guild.id)

    if bot.user not in message.mentions:
        await bot.process_commands(message)
        return

    try:
        await message.add_reaction("✅")
    except Exception:
        logging.exception("Failed to add reaction")

    thinking_msg = await message.reply(
        content="🤖 Myślę...",
        mention_author=False,
    )

    try:
        # Gather last 5 messages as context (exclude current)
        context_lines: list[str] = []
        async for msg in message.channel.history(limit=6):
            if msg.id != message.id:
                context_lines.insert(0, f"{msg.author.name}: {msg.content}")
        context = "\n".join(context_lines)

        # Strip mention from question
        question = message.content.replace(f"<@{bot.user.id}>", "").strip()

        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(
            None,
            answer_discord_question,
            question,
            context,
        )

        await thinking_msg.edit(content=answer or "🤷 Brak odpowiedzi.")

    except Exception as e:
        logging.exception("AI error")
        await thinking_msg.edit(content=f"❌ Błąd: {e}")

    await bot.process_commands(message)


def main():
    logging.basicConfig(level=logging.INFO)
    bot.run(TOKEN)


if __name__ == "__main__":
    main()