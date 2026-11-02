"""Session 2 core: OpenAI responses with configurable reasoning effort."""

import os
import time
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from openai import OpenAI


@dataclass
class Response:
    """Structured response from an LLM call."""

    content: str = ""
    reasoning: str = ""
    metadata: dict | None = None


@dataclass
class Step:
    """One model invocation recorded for developer inspection."""

    step_id: int
    reasoning_effort: str
    model: str
    input_messages: list[dict]
    latency_seconds: float
    answer: str | None = None
    reasoning: str = ""
    metadata: dict | None = None
    error: dict | None = None


@dataclass
class Run:
    """One agent execution started by a user task."""

    run_id: int
    query: str
    started_at: str
    metadata: dict
    steps: list[Step]
    duration_seconds: float = 0.0


class Trajectory:
    """Keep an inspectable record of agent runs and their model calls."""

    def __init__(self):
        self.runs: list[Run] = []

    def initialize(self, query: str, metadata: dict | None = None) -> Run:
        run = Run(
            run_id=len(self.runs) + 1,
            query=query,
            started_at=datetime.now(ZoneInfo("America/New_York")).isoformat(),
            metadata=dict(metadata or {}),
            steps=[],
        )
        self.runs.append(run)
        return run

    def add(self, run: Run, step: Step) -> None:
        run.steps.append(step)

    def clear(self) -> None:
        """Clear execution history without changing conversation memory."""
        self.runs.clear()

    def to_dict(self) -> dict:
        """Return JSON-serializable trajectory data."""
        return {
            "schema_version": 1,
            "runs": [
                {
                    "run_id": run.run_id,
                    "query": run.query,
                    "started_at": run.started_at,
                    "duration_seconds": run.duration_seconds,
                    "metadata": run.metadata,
                    "steps": [
                        {
                            "step_id": step.step_id,
                            "reasoning_effort": step.reasoning_effort,
                            "model": step.model,
                            "input_messages": step.input_messages,
                            "latency_seconds": step.latency_seconds,
                            "answer": step.answer,
                            "reasoning": step.reasoning,
                            "reasoning_status": (
                                "Provider summary available" if step.reasoning
                                else "No reasoning summary returned by API"
                            ),
                            "metadata": step.metadata,
                            "error": step.error,
                        }
                        for step in run.steps
                    ],
                }
                for run in self.runs
            ],
        }


class LLM:
    """Keep provider configuration and response parsing in one place."""

    def __init__(self, model: str | None = None, reasoning_effort: str = "medium"):
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "Set OPENAI_API_KEY in your environment before creating LLM."
            )

        configured_model = (
            model if model is not None else os.environ.get("TINYAGENT_MODEL", "")
        )
        self.model = configured_model.strip()
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
                "prompt_tokens": usage.input_tokens if usage else None,
                "completion_tokens": usage.output_tokens if usage else None,
                "total_tokens": usage.total_tokens if usage else None,
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
    """Coordinate conversation memory and record each model call."""

    def __init__(self, llm: LLM, memory: Memory, system_instructions=None):
        self.llm = llm
        self.memory = memory
        self.trajectory = Trajectory()
        # Instruction sources are supplied per request and never enter Memory.
        self.system_instructions = list(system_instructions or [])

    def build_messages(self) -> list[dict]:
        messages = [
            {"role": "system", "content": text}
            for text in self.system_instructions
            if text and text.strip()
        ]
        messages.extend(message.copy() for message in self.memory.get_messages())
        return messages

    def run(self, task: str, run_metadata: dict | None = None, remember: bool = True) -> Response:
        run = self.trajectory.initialize(task, run_metadata)
        messages = self.build_messages()
        if remember:
            self.memory.add("user", task)
            messages = self.build_messages()
        else:
            messages.append({"role": "user", "content": task})
        started = time.perf_counter()
        try:
            response = self.llm.generate(messages)
        except Exception as error:
            elapsed = time.perf_counter() - started
            self.trajectory.add(
                run,
                Step(
                    step_id=len(run.steps) + 1,
                    reasoning_effort=self.llm.reasoning_effort,
                    model=self.llm.model,
                    input_messages=messages,
                    latency_seconds=round(elapsed, 4),
                    error={"type": type(error).__name__, "message": str(error)},
                ),
            )
            run.duration_seconds = round(elapsed, 4)
            raise

        elapsed = time.perf_counter() - started
        self.trajectory.add(
            run,
            Step(
                step_id=len(run.steps) + 1,
                reasoning_effort=self.llm.reasoning_effort,
                model=(response.metadata or {}).get("model", self.llm.model),
                input_messages=messages,
                latency_seconds=round(elapsed, 4),
                answer=response.content,
                reasoning=response.reasoning,
                metadata=response.metadata,
            ),
        )
        run.duration_seconds = round(elapsed, 4)
        if remember:
            self.memory.add("assistant", response.content)
        return response
