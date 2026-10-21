"""Session 2 core: OpenAI responses with configurable reasoning effort."""

import os
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class Response:
    """Structured response from an LLM call."""

    content: str = ""
    reasoning: str = ""
    metadata: dict | None = None


class LLM:
    """Keep provider configuration and response parsing in one place."""

    def __init__(self, model: str | None = None, reasoning_effort: str = "medium"):
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY in your environment before creating LLM.")

        self.model = (model if model is not None else os.environ.get("TINYAGENT_MODEL", "")).strip()
        if not self.model:
            raise ValueError(
                "Set TINYAGENT_MODEL to a model available to your OpenAI account, "
                "or pass LLM(model=\"model-id\")."
            )
        if not reasoning_effort.strip():
            raise ValueError("reasoning_effort must be a non-empty string.")

        self.reasoning_effort = reasoning_effort.strip()
        self.client = OpenAI(api_key=api_key, timeout=60.0, max_retries=0)

    def generate(self, messages: list[dict]) -> Response:
        """Generate a response and collect a provider-supplied reasoning summary."""
        result = self.client.responses.create(
            model=self.model,
            input=messages,
            reasoning={"effort": self.reasoning_effort, "summary": "auto"},
        )

        content_parts = []
        reasoning_parts = []
        for item in result.output:
            if item.type == "message":
                for part in item.content:
                    if part.type == "output_text":
                        content_parts.append(part.text)
            elif item.type == "reasoning":
                for part in item.summary or []:
                    if part.type == "summary_text":
                        reasoning_parts.append(part.text)

        content = "".join(content_parts)
        if not content:
            raise RuntimeError("The model returned no text. Try another model.")

        usage = result.usage
        details = usage.output_tokens_details if usage else None
        return Response(
            content=content,
            reasoning="\n".join(reasoning_parts),
            metadata={
                "model": result.model,
                "input_tokens": usage.input_tokens if usage else None,
                "output_tokens": usage.output_tokens if usage else None,
                "reasoning_tokens": details.reasoning_tokens if details else None,
            },
        )

    def close(self) -> None:
        """Close the SDK's HTTP client."""
        self.client.close()


class Memory:
    """Store only the user-visible conversation, never reasoning summaries."""

    def __init__(self):
        self.messages: list[dict] = []

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict]:
        return self.messages


class TinyAgent:
    """One user turn and one LLM response, preserving the Session 1 workflow."""

    def __init__(self, llm: LLM, memory: Memory):
        self.llm = llm
        self.memory = memory

    def run(self, task: str) -> Response:
        self.memory.add("user", task)
        response = self.llm.generate(self.memory.get_messages())
        self.memory.add("assistant", response.content)
        return response
