"""Session 1 core: an OpenAI wrapper, conversation memory, and TinyAgent."""

import os
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class Response:
    """Structured response from LLM calls, matching the course example."""

    content: str = ""
    metadata: dict | None = None


class LLM:
    """Keep OpenAI request and response details in one place."""

    def __init__(self, model: str | None = None):
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY in your environment before creating LLM.")

        self.model = (model if model is not None else os.environ.get("TINYAGENT_MODEL", "")).strip()
        if not self.model:
            raise ValueError(
                "Set TINYAGENT_MODEL to a Chat Completions model available to your "
                "OpenAI account, or pass LLM(model=\"model-id\")."
            )

        self.client = OpenAI(api_key=api_key, timeout=60.0, max_retries=0)

    def generate(self, messages: list[dict]) -> Response:
        """Send the conversation and return text plus token usage."""
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        message = completion.choices[0].message
        content = message.content or message.refusal
        if not content:
            raise RuntimeError("The model returned no text. Try another chat model.")

        usage = completion.usage
        return Response(
            content=content,
            metadata={
                "model": completion.model,
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
            },
        )

    def close(self) -> None:
        """Close the SDK's HTTP connection pool when finished."""
        self.client.close()


class Memory:
    """Simple memory module to store conversation history."""

    def __init__(self):
        self.messages = []

    def add(self, role: str, content: str, **kwargs) -> None:
        self.messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict]:
        return self.messages


class TinyAgent:
    """One model call per turn, as in the Session 1 example."""

    def __init__(self, llm: LLM, memory: Memory):
        self.llm = llm
        self.memory = memory

    def run(self, task: str) -> str:
        self.memory.add("user", task)
        return self._step()

    def _step(self) -> str:
        response = self.llm.generate(self.memory.get_messages())
        self.memory.add("assistant", response.content)
        return response.content
