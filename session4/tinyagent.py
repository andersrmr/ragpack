"""Session 4 TinyAgent: observable manual and autonomous Responses API runs."""

import inspect
import json
import os
import time
import copy
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

from openai import OpenAI


def _safe_error(error):
    message = f"{type(error).__name__}: {error}"
    secret = os.environ.get("OPENAI_API_KEY", "").strip()
    return message.replace(secret, "[REDACTED]") if secret else message


@dataclass
class Response:
    content: str = ""
    reasoning: str = ""
    tool_call: dict | None = None
    metadata: dict | None = None
    protocol_items: list[dict] = field(default_factory=list)


def _serialize_output_item(item):
    """Return the complete Responses output item for protocol replay."""
    if isinstance(item, dict):
        return copy.deepcopy(item)
    serializer = getattr(item, "to_dict", None)
    if callable(serializer):
        return serializer()
    serializer = getattr(item, "model_dump", None)
    if callable(serializer):
        return serializer(mode="json", exclude_none=True)
    if hasattr(item, "__dict__"):
        return {key: _serialize_output_value(value) for key, value in vars(item).items()}
    raise TypeError(f"Cannot serialize Responses output item of type {type(item).__name__}.")


def _serialize_output_value(value):
    if isinstance(value, dict):
        return {key: _serialize_output_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_output_value(item) for item in value]
    if hasattr(value, "__dict__"):
        return {key: _serialize_output_value(item) for key, item in vars(value).items()}
    return value


def message_text(content):
    """Extract displayable text from plain or Responses message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict)
            else getattr(item, "text", "")
            for item in content
            if (item.get("type") if isinstance(item, dict) else getattr(item, "type", None))
            in {"output_text", "text"}
        )
    return str(content) if content is not None else ""


def _observable_protocol_item(item):
    """Hide opaque reasoning payloads while retaining useful protocol metadata."""
    if item.get("type") != "reasoning":
        return copy.deepcopy(item)
    return {
        key: copy.deepcopy(item[key])
        for key in ("type", "id", "summary", "status")
        if key in item
    } | {"protocol_payload": "Opaque provider item retained internally; payload omitted from display."}


def _observable_messages(messages):
    return [_observable_protocol_item(item) if item.get("type") == "reasoning"
            else copy.deepcopy(item) for item in messages]


@dataclass
class Step:
    step_id: int
    kind: str
    reasoning_effort: str
    model: str
    input_messages: list[dict]
    latency_seconds: float
    answer: str | None = None
    action: dict | None = None
    observation: str | None = None
    reasoning: str = ""
    metadata: dict | None = None
    error: dict | None = None


@dataclass
class Run:
    run_id: int
    query: str
    started_at: str
    metadata: dict
    steps: list[Step] = field(default_factory=list)
    duration_seconds: float = 0.0
    mode: str = "manual"
    max_steps: int = 1
    outcome: str = "active"


class Trajectory:
    def __init__(self):
        self.runs: list[Run] = []
        self.events: list[dict] = []

    def initialize(self, query: str, metadata: dict | None = None) -> Run:
        run = Run(len(self.runs) + 1, query,
                  datetime.now(ZoneInfo("America/New_York")).isoformat(),
                  dict(metadata or {}))
        self.runs.append(run)
        return run

    def clear(self):
        self.runs.clear()
        self.events.clear()

    def record_event(self, kind, **details):
        self.events.append({"kind": kind, "time": datetime.now(ZoneInfo("America/New_York")).isoformat(), **details})

    def to_dict(self):
        return {"schema_version": 3, "runs": [
            {**asdict(run), "steps": [asdict(step) for step in run.steps]}
            for run in self.runs], "events": copy.deepcopy(self.events)}


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract the second number (b) from the first (a)."""
    return a - b


def divide(a: float, b: float) -> float:
    """Divide the first number (a) by the second (b)."""
    return a / b


def get_weather(location: str) -> str:
    """Return clearly labeled simulated weather for a location."""
    return f"Simulated weather for {location}: sunny, 72°F (demonstration only; not a real observation)."


_TYPE_MAP = {str: "string", int: "integer", float: "number", bool: "boolean",
             list: "array", dict: "object"}


class NativeTools:
    """Allowlisted Python functions with inspectable JSON schemas and validation."""
    def __init__(self):
        self.registry: dict[str, dict] = {}
        self.add_tool("multiply", multiply, {
            "type": "object", "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"}},
            "required": ["a", "b"], "additionalProperties": False})
        self.add_tool("add", add)
        self.add_tool("subtract", subtract)
        self.add_tool("divide", divide)
        self.add_tool("get_weather", get_weather, {
            "type": "object", "properties": {
                "location": {"type": "string", "description": "Location name"}},
            "required": ["location"], "additionalProperties": False})

    def add_tool(self, name, func, parameters=None):
        if not name.isidentifier() or not callable(func):
            raise ValueError("Tools need an identifier name and a callable function.")
        signature = inspect.signature(func)
        if parameters is None:
            properties, required = {}, []
            for key, param in signature.parameters.items():
                properties[key] = {"type": _TYPE_MAP.get(param.annotation, "string")}
                if param.default is inspect.Parameter.empty:
                    required.append(key)
            parameters = {"type": "object", "properties": properties,
                          "required": required, "additionalProperties": False}
        self.registry[name] = {"function": func, "signature": signature,
                               "parameters": parameters}

    @property
    def schemas(self):
        # Responses API function tool shape (not Chat Completions' nested function shape).
        schemas = []
        for name, item in self.registry.items():
            parameters = item["parameters"]
            properties = parameters.get("properties", {})
            required = set(parameters.get("required", []))
            schema = {"type": "function", "name": name,
                      "description": inspect.getdoc(item["function"]) or name,
                      "parameters": parameters}
            # Responses strict mode requires every property and forbids extras.
            if parameters.get("additionalProperties") is False and required == set(properties):
                schema["strict"] = True
            schemas.append(schema)
        return schemas

    @staticmethod
    def parse_arguments(raw):
        args = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(args, dict):
            raise ValueError("Function arguments must be a JSON object.")
        return args

    def execute(self, call: dict) -> tuple[str, str | None]:
        """Return (visible observation, validation/execution error if any)."""
        name, call_id = call.get("name", ""), call.get("call_id", "")
        try:
            if name not in self.registry:
                raise ValueError(f"Unsupported tool name: {name!r}.")
            args = self.parse_arguments(call.get("arguments", ""))
            item = self.registry[name]
            bound = item["signature"].bind(**args)
            bound.apply_defaults()
            for key, value in bound.arguments.items():
                annotation = item["signature"].parameters[key].annotation
                expected = (int, float) if annotation is float else annotation
                if expected is not inspect.Parameter.empty and (
                    not isinstance(value, expected) or
                    (annotation is float and isinstance(value, bool)) or
                    (annotation is int and isinstance(value, bool))
                ):
                    raise TypeError(f"Argument {key!r} must be {getattr(annotation, '__name__', annotation)}.")
            result = item["function"](*bound.args, **bound.kwargs)
            return str(result), None
        except Exception as error:
            message = _safe_error(error)
            return f"Tool error: {message}", message


class LLM:
    def __init__(self, model=None, reasoning_effort="medium"):
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY in your environment before creating LLM.")
        self.model = (model if model is not None else os.environ.get("TINYAGENT_MODEL", "")).strip()
        if not self.model:
            raise ValueError("Set TINYAGENT_MODEL or pass LLM(model='model-id').")
        self.reasoning_effort = reasoning_effort.strip()
        if not self.reasoning_effort:
            raise ValueError("reasoning_effort must be non-empty.")
        self.client = OpenAI(api_key=api_key, timeout=60.0, max_retries=0)

    def generate(self, messages, tools=None):
        kwargs = {"model": self.model, "input": messages,
                  "reasoning": {"effort": self.reasoning_effort, "summary": "auto"}}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
            kwargs["parallel_tool_calls"] = False
        result = self.client.responses.create(**kwargs)
        texts, reasoning, calls, protocol_items = [], [], [], []
        for item in result.output:
            protocol_items.append(_serialize_output_item(item))
            if item.type == "message":
                for part in item.content:
                    if part.type == "output_text":
                        texts.append(part.text)
            elif item.type == "reasoning":
                for part in getattr(item, "summary", None) or []:
                    if part.type == "summary_text":
                        reasoning.append(part.text)
            elif item.type == "function_call":
                calls.append({"type": "function_call", "id": item.id,
                              "call_id": item.call_id, "name": item.name,
                              "arguments": item.arguments})
        if len(calls) > 1:
            # Preserve all returned calls so the harness can reject the batch visibly.
            call = {"multiple_calls": calls}
        else:
            call = calls[0] if calls else None
        usage = getattr(result, "usage", None)
        details = getattr(usage, "output_tokens_details", None) if usage else None
        return Response("".join(texts), "\n".join(reasoning), call, {
            "model": getattr(result, "model", self.model),
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "prompt_tokens": getattr(usage, "input_tokens", None),
            "completion_tokens": getattr(usage, "output_tokens", None),
            "reasoning_tokens": getattr(details, "reasoning_tokens", None)}, protocol_items)

    def summarize(self, messages):
        """Make a dedicated tool-free Responses API summarization request."""
        result = self.client.responses.create(model=self.model, input=messages)
        texts = []
        for item in result.output:
            if item.type == "message":
                texts.extend(part.text for part in item.content if part.type == "output_text")
        usage = getattr(result, "usage", None)
        metadata = {"model": getattr(result, "model", self.model),
                    "input_tokens": getattr(usage, "input_tokens", None),
                    "output_tokens": getattr(usage, "output_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None)}
        return "".join(texts), metadata

    def close(self):
        self.client.close()


class Memory:
    """Responses API conversation items; persistent instructions stay external."""
    def __init__(self):
        self.messages: list[dict] = []
        self.summary = ""
        self.summary_retained_turns: int | None = None
        self.transcript_archive: list[dict] | None = None
        self.transcript_suffix_start = 0

    def add(self, role, content):
        self.messages.append({"role": role, "content": content})

    def add_function_call(self, call):
        self.messages.append(dict(call))

    def add_function_output(self, call_id, output):
        self.messages.append({"type": "function_call_output", "call_id": call_id,
                              "output": str(output)})

    def get_messages(self):
        return self.messages

    def display_messages(self):
        return _observable_messages(self.messages)

    def clear(self):
        self.messages.clear()
        self.summary = ""
        self.summary_retained_turns = None
        self.transcript_archive = None
        self.transcript_suffix_start = 0

    def transcript_messages(self):
        if self.transcript_archive is None:
            return self.messages
        return self.transcript_archive + self.messages[self.transcript_suffix_start:]

    def complete_turns(self):
        """Group from each user item through the following complete response sequence."""
        turns, current = [], []
        for item in self.messages:
            if item.get("role") == "user" and current:
                turns.append(current)
                current = []
            current.append(item)
        if current:
            turns.append(current)
        # An unmatched tool call cannot be summarized or retained as a complete turn.
        for turn in turns:
            calls = {item.get("call_id") for item in turn if item.get("type") == "function_call"}
            outputs = {item.get("call_id") for item in turn if item.get("type") == "function_call_output"}
            if calls != outputs:
                raise RuntimeError("Memory contains an unmatched tool call/output sequence.")
        if any(not turn or turn[-1].get("role") != "assistant" for turn in turns):
            raise RuntimeError("Memory contains an incomplete conversation turn.")
        return turns

    def compact(self, llm, retain_turns=2):
        if not isinstance(retain_turns, int) or retain_turns < 0:
            raise ValueError("retain_turns must be a non-negative integer.")
        turns = self.complete_turns()
        if len(turns) <= retain_turns:
            raise ValueError("There are no older complete turns to compact.")
        older, recent = turns[:-retain_turns] if retain_turns else turns, turns[-retain_turns:] if retain_turns else []
        source = []
        for turn in older:
            for item in turn:
                if item.get("role") in {"user", "assistant"}:
                    source.append({"role": item["role"], "content": message_text(item.get("content", ""))})
                elif item.get("type") == "function_call":
                    source.append({"role": "assistant", "content": f"Requested tool {item.get('name')}: {item.get('arguments')}"})
                elif item.get("type") == "function_call_output":
                    source.append({"role": "assistant", "content": f"Tool observation: {item.get('output', '')}"})
        prompt = [{"role": "developer", "content": (
            "Summarize only the supplied completed conversation turns for continuity. "
            "Keep user goals, preferences expressed in these turns, important facts, and unresolved questions. "
            "Merge any supplied previous compacted summary so continuity is preserved. Do not add facts. "
            "Do not include system instructions, personal context files, source code, quiz instructions, or hidden reasoning.")},
            {"role": "user", "content": json.dumps({"previous_compacted_summary": self.summary,
                                                       "older_complete_turns": source}, ensure_ascii=False)}]
        # Dedicated request: no native tools and no persistent/system instructions.
        summary, metadata = llm.summarize(prompt)
        if not summary.strip():
            raise ValueError("Summarization returned an empty summary.")
        prior_transcript = copy.deepcopy(self.transcript_messages())
        self.transcript_archive = prior_transcript
        self.summary = summary.strip()
        self.summary_retained_turns = retain_turns
        self.messages = [item for turn in recent for item in turn]
        self.transcript_suffix_start = len(self.messages)
        return {"summary": self.summary, "retained_turns": retain_turns, "usage": metadata}


class CompactMemory:
    """Named operation for explicit, user-triggered conversation compaction."""

    def __init__(self, memory, llm):
        self.memory, self.llm = memory, llm

    def run(self, retain_turns=2):
        return self.memory.compact(self.llm, retain_turns)


class TinyAgent:
    def __init__(self, llm, memory, system_instructions=None, tools=None, rag_memory=None):
        self.llm, self.memory = llm, memory
        self.system_instructions = list(system_instructions or [])
        self.tools = tools or NativeTools()
        self.trajectory = Trajectory()
        self.pending_continue = False
        self.active_run: Run | None = None
        self.rag_memory = rag_memory
        self.retrieval_mode = "Off"
        self.turn_retrieval = None
        self._include_retrieval_once = False
        self._run_lock = threading.Lock()
        self._stop_requested = threading.Event()
        if self.rag_memory:
            def search_documents(query: str, top_k: int = 3):
                """Search the local learning corpus using a context-aware query."""
                results = self.rag_memory.search(query, top_k)
                if self.rag_memory.last_error:
                    raise RuntimeError(self.rag_memory.last_error)
                return json.dumps({"query": query, "matches": results}, ensure_ascii=False)
            self.tools.add_tool("search_documents", search_documents, {
                "type": "object", "properties": {
                    "query": {"type": "string", "description": "Search query using relevant conversation context"},
                    "top_k": {"type": "integer", "description": "Maximum matching passages"}},
                "required": ["query"], "additionalProperties": False})

    def set_retrieval_mode(self, mode):
        if mode not in {"Off", "Automatic RAG", "Model-directed retrieval"}:
            raise ValueError("Unknown retrieval mode.")
        if mode != "Off" and self.rag_memory is None:
            raise RuntimeError("Document retrieval is not configured.")
        self.retrieval_mode = mode

    def build_messages(self):
        instructions = [{"role": "system", "content": text}
                        for text in self.system_instructions if text and text.strip()]
        if self.memory.summary:
            instructions.append({"role": "developer", "content": "## Compacted conversation summary\n\n" + self.memory.summary})
        if self._include_retrieval_once and self.turn_retrieval is not None:
            matches = self.turn_retrieval["matches"]
            text = ("## Temporary local retrieval context\n" + json.dumps(matches, ensure_ascii=False)
                    if matches else "## Temporary local retrieval context\nNo relevant local passage was retrieved.")
            instructions.append({"role": "developer", "content": text})
        return instructions + [dict(item) for item in self.memory.get_messages()]

    def request_stop(self):
        if not self._run_lock.locked():
            return False
        self._stop_requested.set()
        return True

    @property
    def run_active(self):
        return self._run_lock.locked()

    def run(self, task, run_metadata=None, remember=True, mode="manual", max_steps=10):
        if mode not in {"manual", "autonomous"}:
            raise ValueError("mode must be manual or autonomous")
        if not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("A run is already active.")
        self._stop_requested.clear()
        try:
            return self._run(task, run_metadata, remember, mode, max_steps)
        except Exception as error:
            if self.active_run is not None:
                self.active_run.outcome = "error"
                self.trajectory.record_event("run_finished", run_id=self.active_run.run_id,
                                             mode=mode, outcome="error",
                                             error={"type": type(error).__name__,
                                                    "message": _safe_error(error)})
            raise
        finally:
            self._run_lock.release()

    def _run(self, task, run_metadata=None, remember=True, mode="manual", max_steps=10):
        if self.pending_continue:
            raise RuntimeError("A tool action is waiting. Use Continue before a new request.")
        if remember:
            self.memory.add("user", task)
        else:
            self.memory.messages.append({"role": "user", "content": task})
        metadata = dict(run_metadata or {})
        self.turn_retrieval = None
        if self.retrieval_mode == "Automatic RAG":
            # Deliberately use this user turn alone: no query rewriting or history merge.
            matches = self.rag_memory.search(task)
            self.turn_retrieval = {"query": task, "matches": matches,
                                   "error": self.rag_memory.last_error,
                                   "threshold": self.rag_memory.threshold,
                                   "top_k": self.rag_memory.top_k,
                                   "threshold_decision": "matches accepted" if matches else "no passage cleared threshold",
                                   "text_added": ("## Temporary local retrieval context\n" + json.dumps(matches, ensure_ascii=False)
                                                  if matches else "## Temporary local retrieval context\nNo relevant local passage was retrieved.")}
            metadata["automatic_retrieval"] = copy.deepcopy(self.turn_retrieval)
            self.trajectory.record_event("automatic_retrieval", query=task,
                                         result_ids=[m["document_id"] for m in matches],
                                         scores=[m["score"] for m in matches],
                                         error=self.rag_memory.last_error or None)
            self._include_retrieval_once = True
        else:
            self._include_retrieval_once = False
        self.active_run = self.trajectory.initialize(task, metadata)
        self.active_run.mode, self.active_run.max_steps = mode, max_steps
        if mode == "manual":
            result = self._step()
            if self.active_run.outcome == "stopped":
                pass
            elif not result.tool_call:
                self.active_run.outcome = "completed"
            else:
                self.active_run.outcome = "waiting_continue"
            return result
        result = self._autonomous_loop(max_steps)
        return result

    def _autonomous_loop(self, max_steps):
        run = self.active_run
        result = Response()
        while len(run.steps) < max_steps:
            if self._stop_requested.is_set():
                run.outcome = "stopped"
                self.pending_continue = False
                break
            result = self._step()
            if run.outcome == "stopped":
                break
            if not result.tool_call:
                run.outcome = "completed"
                break
            if len(run.steps) >= max_steps:
                run.outcome = "max_steps"
                self.pending_continue = False
                break
            if self._stop_requested.is_set():
                run.outcome = "stopped"
                self.pending_continue = False
                break
        else:
            run.outcome = "max_steps"
        if run.outcome == "active":
            run.outcome = "max_steps"
        self.trajectory.record_event("run_finished", run_id=run.run_id, mode=run.mode,
                                     outcome=run.outcome, steps=len(run.steps), max_steps=max_steps)
        result.metadata = dict(result.metadata or {}, run_outcome=run.outcome,
                               steps_used=len(run.steps), max_steps=max_steps)
        return result

    def continue_step(self):
        if not self.pending_continue or self.active_run is None:
            raise RuntimeError("No tool action is waiting for Continue.")
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("A run is already active.")
        try:
            self.pending_continue = False
            try:
                result = self._step()
            except Exception as error:
                self.active_run.outcome = "error"
                self.trajectory.record_event("run_finished", run_id=self.active_run.run_id,
                                             mode=self.active_run.mode, outcome="error",
                                             error={"type": type(error).__name__,
                                                    "message": _safe_error(error)})
                raise
            if self.active_run.outcome == "stopped":
                pass
            elif not result.tool_call:
                self.active_run.outcome = "completed"
            else:
                self.active_run.outcome = "waiting_continue"
            return result
        finally:
            self._run_lock.release()

    def compact_memory(self, retain_turns=2):
        if self.pending_continue:
            raise RuntimeError("Cannot compact while a tool result is waiting for Continue.")
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("Cannot compact while a run is active.")
        started = datetime.now(ZoneInfo("America/New_York")).isoformat()
        try:
            try:
                result = CompactMemory(self.memory, self.llm).run(retain_turns)
                self.trajectory.record_event("compaction", started_at=started, success=True,
                                             retained_turns=retain_turns, summary=result["summary"],
                                             usage=result["usage"])
                return result
            except Exception as error:
                self.trajectory.record_event("compaction", started_at=started, success=False,
                                             retained_turns=retain_turns,
                                             error={"type": type(error).__name__,
                                                    "message": _safe_error(error).partition(": ")[2]})
                raise
        finally:
            self._run_lock.release()

    def _step(self):
        run = self.active_run
        messages = self.build_messages()
        include_tool = self.retrieval_mode == "Model-directed retrieval"
        schemas = self.tools.schemas if include_tool else [s for s in self.tools.schemas if s["name"] != "search_documents"]
        self._include_retrieval_once = False
        started = time.perf_counter()
        if self._stop_requested.is_set():
            run.outcome = "stopped"
            self.pending_continue = False
            self.trajectory.record_event("stop_observed", run_id=run.run_id,
                                         reason="requested before model request")
            return Response(content="Stopped before the next model request.",
                            metadata={"run_outcome": "stopped"})
        try:
            response = self.llm.generate(messages, tools=schemas)
        except Exception as error:
            elapsed = round(time.perf_counter() - started, 4)
            run.steps.append(Step(len(run.steps)+1, "model_request", self.llm.reasoning_effort,
                                  self.llm.model, _observable_messages(messages), elapsed,
                                  error={"type": type(error).__name__, "message": _safe_error(error)}))
            run.duration_seconds += elapsed
            raise
        elapsed = round(time.perf_counter() - started, 4)
        if self._stop_requested.is_set():
            run.outcome = "stopped"
            run.steps.append(Step(len(run.steps)+1, "discarded_response", self.llm.reasoning_effort,
                                  self.llm.model, _observable_messages(messages), elapsed,
                                  action={"tool": (response.tool_call or {}).get("name")
                                          if response.tool_call and "multiple_calls" not in response.tool_call else "pending"},
                                  observation="Stop requested during model request; the uncommitted response and any pending action were discarded."))
            run.duration_seconds += elapsed
            self.pending_continue = False
            return Response(content="Stopped. The returned pending action was discarded and the conversation is consistent.",
                            metadata={"run_outcome": "stopped"})
        model = (response.metadata or {}).get("model", self.llm.model)
        base = dict(step_id=len(run.steps)+1, reasoning_effort=self.llm.reasoning_effort,
                    model=model, input_messages=_observable_messages(messages), latency_seconds=elapsed,
                    reasoning=response.reasoning or "Unavailable: the provider supplied no reasoning summary.",
                    metadata=response.metadata)
        call = response.tool_call
        if response.protocol_items:
            # Responses reasoning items are opaque protocol state. Retain the
            # provider's complete output sequence for the next API request.
            self.memory.messages.extend(copy.deepcopy(response.protocol_items))
        if call and "multiple_calls" in call:
            # Keep the exact provider requests visible, reject as a single explicit error.
            call_id = call["multiple_calls"][0].get("call_id", "")
            text = f"Rejected {len(call['multiple_calls'])} function calls: Stage A executes one action per step."
            if response.content and not response.protocol_items:
                self.memory.add("assistant", response.content)
            if not response.protocol_items:
                self.memory.messages.extend(call["multiple_calls"])
            for item in call["multiple_calls"]:
                self.memory.add_function_output(item.get("call_id", ""), text)
            action = {"tool": "multiple_calls", "calls": call["multiple_calls"]}
            run.steps.append(Step(kind="tool_action", **base, action=action,
                                  observation=text, error={"type": "MultipleToolCalls", "message": text}))
            run.duration_seconds += elapsed
            return Response(content=text, tool_call=action, metadata=response.metadata)
        if call:
            if response.content and not response.protocol_items:
                self.memory.add("assistant", response.content)
            if not response.protocol_items:
                self.memory.add_function_call(call)
            if call.get("name") == "search_documents" and self.retrieval_mode != "Model-directed retrieval":
                error = "search_documents is available only in Model-directed retrieval mode."
                observation = f"Tool error: RuntimeError: {error}"
            else:
                observation, error = self.tools.execute(call)
            self.memory.add_function_output(call.get("call_id", ""), observation)
            action = {"call_id": call.get("call_id"), "tool": call.get("name"),
                      "arguments": call.get("arguments")}
            run.steps.append(Step(kind="tool_action", **base, action=action,
                                  observation=observation,
                                  error={"type": "ToolExecutionError", "message": error} if error else None))
            run.duration_seconds += elapsed
            self.pending_continue = True
            return Response(content=observation, reasoning=response.reasoning,
                            tool_call=action, metadata=response.metadata)
        if response.protocol_items:
            pass
        elif response.content:
            self.memory.add("assistant", response.content)
        run.steps.append(Step(kind="final_answer", **base, answer=response.content))
        run.duration_seconds += elapsed
        self.pending_continue = False
        return response
