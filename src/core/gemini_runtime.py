"""Small Google GenAI runtime for the Red / Red Advance agents.

The lab agents do not use tools, so a direct ``google-genai`` call is enough
and avoids ADK automatic-function-calling overhead and related quota errors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class GeminiAgent:
    name: str
    instruction: str
    provider: str = "gemini"


@dataclass
class GeminiRunner:
    app_name: str
    model: str
    api_key: str
    provider: str = "gemini"
    temperature: float = 0.4
    input_hooks: list[Callable[[str], str | None]] = field(default_factory=list)
    output_hooks: list[Callable[[str], str]] = field(default_factory=list)

    async def chat(self, agent: GeminiAgent, user_message: str) -> str:
        for hook in self.input_hooks:
            blocked = hook(user_message)
            if blocked:
                return blocked

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        response = await client.aio.models.generate_content(
            model=self.model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=agent.instruction,
                temperature=self.temperature,
            ),
        )
        text = (response.text or "").strip()
        for hook in self.output_hooks:
            text = hook(text)
        return text


def create_gemini_pair(
    *,
    name: str,
    instruction: str,
    app_name: str,
    model: str,
    api_key: str,
    input_hooks: list | None = None,
    output_hooks: list | None = None,
    temperature: float = 0.4,
) -> tuple[GeminiAgent, GeminiRunner]:
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is required for Gemini Red agents")
    return (
        GeminiAgent(name=name, instruction=instruction),
        GeminiRunner(
            app_name=app_name,
            model=model,
            api_key=api_key,
            input_hooks=list(input_hooks or []),
            output_hooks=list(output_hooks or []),
            temperature=temperature,
        ),
    )
