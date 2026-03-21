"""Base agent — shared LLM client, logging, memory helpers."""
from __future__ import annotations
import os
import json
import logging
from agents.memory import store, fetch

log = logging.getLogger(__name__)

def llm_call(prompt: str, max_tokens: int = 400) -> str:
    """LLM call: Groq primary, OpenAI fallback."""
    groq_key   = os.getenv("GROQ_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    if groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            resp = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            log.warning(f"Groq failed: {e}")

    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            log.warning(f"OpenAI failed: {e}")

    return "LLM unavailable"


class BaseAgent:
    name = "BaseAgent"

    def run(self, task: dict) -> dict:
        raise NotImplementedError

    def remember(self, key: str, value: dict) -> None:
        store(f"{self.name}:{key}", value)

    def recall(self, key: str) -> dict | None:
        return fetch(f"{self.name}:{key}")
