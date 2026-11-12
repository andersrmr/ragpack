# TinyAgent Session 4 Stage A: Observable ReAct Runs

Session 4 is an independent checkpoint copied from the completed Session 3
Stage B implementation. It adds selectable manual and autonomous execution on
the same OpenAI Responses API, allowlisted tools, retrieval, Memory, and
Trajectory. Session 1, Session 2, and Session 3 source files are unchanged.
This checkpoint does not add Plan-and-Solve, planning tools, `update_plan`,
`mark_done`, persistent plan state, or planning UI.

## Setup and launch

From the repository root:

```bash
python -m pip install -r session4/requirements.txt
python -m pip install -r session4/requirements-gradio.txt  # optional browser UI
```

Set `OPENAI_API_KEY` and `TINYAGENT_MODEL`. Launch the terminal app with
`cd session4 && python main.py`; launch Gradio with `cd session4 && python
gradio_app.py` (localhost only). Retrieval uses the configured OpenAI embedding
model and the local `documents/` corpus; embedding requests can occur on a
cache miss.

## Modes and run control

Both interfaces default to **Manual Continue** to preserve Session 3 behavior.
In the terminal select `/mode manual` or `/mode autonomous`; `/steps N` sets a
positive maximum number of model requests for autonomous runs. In Gradio use
the Execution mode selector and Maximum model steps slider.

Submitting a task starts a run. Each Responses API request is one model step.
Manual mode sends one request for the submitted task; if a tool is requested,
the tool executes, its observation is recorded, and the run pauses. Each
Continue control or `/continue` makes exactly one further request.

Autonomous mode continues after each single valid tool action and its
observation. A response without a tool call ends normally; that is protocol
completion, not proof that the task was correct. At the step limit, the last
tool action is executed and recorded, but no additional request is made. That
run ends as **max_steps** (incomplete), never as a completed answer. A final
allowed request with no tool call ends normally. Tool failures become visible
observations and autonomous execution may continue. Model-directed retrieval
is a tool action and follows the selected mode. Automatic RAG runs once at the
start of the user turn, not on each autonomous iteration.

Gradio's Stop run control requests a cooperative stop. It cannot cancel an
in-flight network request or synchronous tool. A response arriving after a
stop request is discarded before execution, so no unmatched function-call
item enters conversation Memory. A tool already executing is allowed to
finish; its observation is recorded before the loop stops. Conflicting
conversation mutations are serialized with the active Gradio action. The
terminal interface exposes mode and step limit and runs autonomous work on a
background worker so `/stop` remains available at the prompt. While a run is
active, other commands are deferred until it stops or finishes.

## Tools, memory, and observability

Responses API function calls preserve provider `call_id` correlation; schemas,
arguments, tool names, and execution are validated against the allowlist.
Parallel calls are disabled in requests and a returned batch is visibly
rejected without executing any call. Memory contains conversation protocol
items; Trajectory separately stores run mode, configured limit, each model
request, step number, action, observation, errors, outcome and available
provider metadata. Only provider-supplied reasoning summaries are shown; when
none is supplied the UI labels it unavailable. Hidden reasoning and secrets
are not requested or stored as reasoning.

Off, Automatic RAG, and Model-directed retrieval remain available. `/compact`
and the Gradio Compact Memory control retain the existing explicit safe-boundary
summarization behavior. Reloadable `about_me.md`, optional source context,
manual quiz activation and artifacts, transcript export, independent clearing,
and session-local Gradio state are carried forward.

## Tests

Run the Session 4 offline suite:

```bash
python -m unittest discover -s session4 -p 'test_*.py' -v
```

Run Session 3 regression tests:

```bash
python -m unittest discover -s session3 -p 'test_*.py' -v
```

If desired, run Session 1 and Session 2 regressions with the same command and
`-s session1` or `-s session2`. Offline tests use fake Responses API results,
fixed embedding vectors, and mocked tools. They do not verify live account or
model behavior, answer quality, or a live browser session.
