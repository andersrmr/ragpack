# TinyAgent Session 3: Native Tool Calling

Session 3 is a separate checkpoint based on Session 2. It preserves the OpenAI
Python SDK and Responses API integration while adding native function calling,
an allowlisted `NativeTools` registry, and one-action-per-step control flow.
Session 1 and Session 2 remain independent and were not modified. The current
repository checkout has no root `README.md`; setup conventions are consistent
with `session1/README.md` and `session2/README.md`.

## Setup and launch

Install the terminal dependencies from the repository root:

```bash
python -m pip install -r session3/requirements.txt
```

For the optional browser interface:

```bash
python -m pip install -r session3/requirements-gradio.txt
```

Set `OPENAI_API_KEY` and `TINYAGENT_MODEL` in the environment. From the
repository root, launch the terminal app with:

```bash
cd session3
python main.py
```

Launch Gradio with `python gradio_app.py` from `session3/`. It binds to
localhost. Terminal and browser interfaces use the same `TinyAgent` core.

Terminal commands include `/continue`, `/clear`, `/save filename.md`,
`/context reload`, `/source on|off|status|reload`, `/quiz [topic]`, and
`/quiz end`. A tool step pauses after one execution. `/continue` makes exactly
one subsequent model request. A final answer ends that pending interaction.
Gradio's Continue button is enabled only while a tool call is waiting for the
next step.

## Responses API tool protocol

The LLM sends function tool schemas using the Responses API format: each tool
has `type: "function"`, a top-level `name`, `description`, and JSON Schema
`parameters`. A model request can return a `function_call` output item containing
its item `id`, `call_id`, function `name`, and JSON `arguments`. TinyAgent stores
that item in Memory, checks the allowlisted registry and validates arguments,
then executes one Python function. It appends a
`function_call_output` item with the same `call_id` and the visible result or
error. Continue sends this accumulated input back to the model. No automatic
follow-up model request occurs after a tool execution. Parallel tool calls are
disabled in the request. If a provider response still contains multiple calls,
the harness retains each item, pairs each with a visible rejection output, and
does not execute a batch.

Two demonstration tools are registered: `multiply(a, b)` and
`get_weather(location)`. The weather result is explicitly simulated and does
not represent a live weather observation. Malformed JSON, unsupported names,
argument validation errors, and Python execution errors are returned as visible
tool observations, with diagnostic details in Trajectory.

## Architecture and state

- **Tools** are registered Python functions selected by model-generated calls.
  Only registered functions can execute.
- **Skills** are instruction text under `skills/`. The quiz skill remains
  manually activated and orchestrated by the quiz UI/commands; it is not a
  native model-directed tool.
- **Memory** stores conversational messages and Responses API function-call and
  function-output items. Persistent system instructions remain separate.
- **Trajectory** records model requests, answers, tool actions, arguments,
  observations, errors, latency, and available usage. Provider reasoning
  summaries are recorded only when returned; hidden chain-of-thought is never
  requested or fabricated.

`about_me.md` remains persistent and reloadable. Optional TinyAgent source
context loads `session3/tinyagent.py`. Quiz progress and Markdown artifacts,
conversation clearing, transcript export, reasoning-effort configuration, and
the Chat, Context, and Trajectory tabs are retained from Session 2.

To reference the implementation from Gradio, open the **Context** tab and turn
on **Include session3/tinyagent.py in model requests**. The source is off by
default. When enabled, it is added to future model requests as context; it is
not displayed as a code viewer in the Chat tab. The Context tab's JSON view
shows the loaded text under `source_code_context.content` and the latest
effective request under `effective_messages_for_latest_run`. Use **Reload
tinyagent.py** after editing the file. You can also use `/source on`,
`/source off`, `/source status`, and `/source reload` in the Gradio chat box.

Clearing conversation also cancels any pending Continue state. Clearing the
diagnostic trajectory does not change Memory. The Gradio app keeps agents and
their state isolated by browser session.

## Offline tests

Run Session 3 tests from the repository root:

```bash
python -m unittest discover -s session3 -p 'test_*.py' -v
```

Run regression suites for the earlier checkpoints:

```bash
python -m unittest discover -s session1 -p 'test_*.py' -v
python -m unittest discover -s session2 -p 'test_*.py' -v
```

Tests use fake Responses API results and do not call OpenAI. Gradio construction
checks run when the optional dependency is installed.

## Scope and limitations

This is Stage A. There is no autonomous agent loop, embeddings, RAG memory,
compaction, vector store, model-directed document retrieval, progressive skill
discovery, video search, or MCP integration. Tool execution is synchronous and
local. The provider/model may refuse a tool request or return output that the
harness visibly rejects. Live model behavior and actual account/model schema
compatibility require a manual experiment; no live API calls are part of the
offline validation.
