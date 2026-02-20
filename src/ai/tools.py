"""Tools for AITIS swarm — plain functions, no CrewAI dependency."""

import re
import requests
from typing import List, Tuple
from bs4 import BeautifulSoup
from ddgs import DDGS


# ---------------------------------------------------------------------------
# Web / Research
# ---------------------------------------------------------------------------

def web_search(query: str, max_results: int = 3) -> str:
    """Search the web using DuckDuckGo."""
    try:
        ddgs = DDGS()
        results = ddgs.text(query, max_results=max_results)
        if not results:
            return f"No results found for: {query}"
        out = f"Web search results for '{query}':\n\n"
        for i, r in enumerate(results, 1):
            out += f"{i}. {r.get('title', 'N/A')}\n"
            out += f"   {r.get('body', '')[:800]}\n"
            out += f"   Link: {r.get('href', 'N/A')}\n\n"
        return out[:5000]
    except Exception as e:
        return f"Error during web search: {e}"


def league_of_legends_search(query: str) -> str:
    """Search League of Legends info — champions, items, patch notes, meta."""
    try:
        ddgs = DDGS()
        lol_query = (
            f"{query}"        )
        results = ddgs.text(lol_query, max_results=5)
        if not results:
            return f"No LoL information found for: {query}"
        out = f"League of Legends — {query}:\n\n"
        for i, r in enumerate(results, 5):
            out += f"{i}. {r.get('title', 'N/A')}\n"
            out += f"   {r.get('body', '')[:1500]}\n\n"
        return out[:5000]
    except Exception as e:
        return f"Error searching LoL info: {e}"


def gaming_search(query: str) -> str:
    """Search general gaming news, guides, and game info."""
    try:
        ddgs = DDGS()
        results = ddgs.text(query, max_results=2)
        if not results:
            return f"No gaming information found for: {query}"
        out = f"Gaming search results for '{query}':\n\n"
        for i, r in enumerate(results, 1):
            out += f"{i}. {r.get('title', 'N/A')}\n"
            out += f"   {r.get('body', '')[:800]}\n\n"
        return out[:3000]
    except Exception as e:
        return f"Error during gaming search: {e}"


def technical_search(query: str) -> str:
    """Search programming docs, Stack Overflow, GitHub."""
    try:
        ddgs = DDGS()
        tech_query = (
            f"{query} "
            "site:stackoverflow.com OR site:github.com OR site:docs.python.org OR site:developer.mozilla.org"
        )
        results = ddgs.text(tech_query, max_results=2)
        if not results:
            results = ddgs.text(query, max_results=2)
        if not results:
            return f"No technical information found for: {query}"
        out = f"Technical search results for '{query}':\n\n"
        for i, r in enumerate(results, 1):
            out += f"{i}. {r.get('title', 'N/A')}\n"
            out += f"   {r.get('body', '')[:800]}\n\n"
        return out[:3000]
    except Exception as e:
        return f"Error during technical search: {e}"


# ---------------------------------------------------------------------------
# Website scraper
# ---------------------------------------------------------------------------

def _scrape_website(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AITISBot/1.0)"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    return chunks


def _keyword_score(chunk: str, keywords: List[str]) -> int:
    lower = chunk.lower()
    return sum(lower.count(kw.lower()) for kw in keywords)


def search_scraped_website(url: str, keywords: List[str], top_k: int = 5) -> str:
    """Scrape a URL and return the most relevant chunks for the given keywords."""
    try:
        text = _scrape_website(url)
        chunks = _chunk_text(text)
        scored = sorted(
            [(chunk, _keyword_score(chunk, keywords)) for chunk in chunks if _keyword_score(chunk, keywords) > 0],
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]
        if not scored:
            return f"No relevant content found on {url} for keywords: {keywords}"
        return "\n\n---\n\n".join(chunk for chunk, _ in scored)
    except Exception as e:
        return f"Error scraping {url}: {e}"