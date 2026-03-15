"""
BaseAgent — foundation for all trading agents.

Design principles:
- Tiny system prompts (50-100 words, role-specific)
- Groq (Llama 3.3 70B, FREE) primary — OpenAI gpt-4o-mini fallback
- think_json() uses json.loads(), never eval()
- Timeouts enforced via future.result(timeout=N) in TaskQueue, not signal.alarm
"""

import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-16s] %(message)s",
    datefmt="%H:%M:%S",
)


class BaseAgent:
    """Base class for all trading agents."""

    DEFAULT_TIMEOUT = 120  # seconds; enforced in TaskQueue

    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt
        self.logger = logging.getLogger(name)
        self._groq = None
        self._openai = None
        self._init_llm()

    def _init_llm(self):
        groq_key = os.getenv("GROQ_API_KEY", "")
        if groq_key:
            try:
                from groq import Groq
                self._groq = Groq(api_key=groq_key)
            except ImportError:
                self.log("groq package not installed")

        openai_key = os.getenv("OPENAI_API_KEY", "")
        if openai_key:
            try:
                from openai import OpenAI
                self._openai = OpenAI(api_key=openai_key)
            except ImportError:
                self.log("openai package not installed")

    def think(self, prompt: str, context: str = "", max_tokens: int = 512) -> str:
        """Call LLM. Groq primary (free), OpenAI fallback. Returns plain text."""
        messages = []
        if context:
            messages += [
                {"role": "user", "content": f"Context:\n{context}"},
                {"role": "assistant", "content": "Understood."},
            ]
        messages.append({"role": "user", "content": prompt})

        system = [{"role": "system", "content": self.system_prompt}]

        if self._groq:
            try:
                resp = self._groq.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=system + messages,
                    max_tokens=max_tokens,
                    temperature=0.1,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                self.log(f"Groq error: {e} — trying OpenAI")

        if self._openai:
            try:
                resp = self._openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=system + messages,
                    max_tokens=max_tokens,
                    temperature=0.1,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                self.log(f"OpenAI error: {e}")

        return json.dumps({"error": f"No LLM available for {self.name}"})

    def think_json(self, prompt: str, context: str = "", max_tokens: int = 512):
        """
        Call LLM and parse JSON response.
        Strips markdown fences, comments, trailing commas. Uses json.loads() — never eval().
        Returns dict/list on success, {} on failure.
        """
        raw = self.think(prompt, context=context, max_tokens=max_tokens).strip()
        return self._safe_json_parse(raw)

    @staticmethod
    def _safe_json_parse(text: str):
        """
        Robust JSON parser for LLM output.
        Handles: markdown fences, // comments, trailing commas, surrounding prose.
        """
        import re

        text = text.strip()

        # 1. Strip markdown code fences — try content between first and last fence
        if "```" in text:
            parts = text.split("```")
            # parts[1] is between first pair of fences; strip optional "json" language tag
            for candidate in parts[1::2]:  # every odd part is inside fences
                candidate = candidate.strip()
                if candidate.lower().startswith("json"):
                    candidate = candidate[4:].strip()
                if candidate:
                    text = candidate
                    break

        text = text.strip()

        # 2. Remove // line comments (not valid JSON but LLMs emit them)
        text = re.sub(r"//[^\n]*", "", text)

        # 3. Remove trailing commas before ] or } (common LLM mistake)
        text = re.sub(r",\s*([}\]])", r"\1", text)

        # 4. Direct parse attempt
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 5. Extract first JSON array or object from surrounding prose
        for start, end in [("[", "]"), ("{", "}")]:
            s = text.find(start)
            e = text.rfind(end)
            if s != -1 and e > s:
                try:
                    return json.loads(text[s: e + 1])
                except json.JSONDecodeError:
                    pass

        return {}

    def log(self, msg: str):
        self.logger.info(msg)
