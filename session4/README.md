# TinyAgent Session 4 Stage B: ReAct and Explicit Planning

ReAct provides iterative decision-making and execution: the model can request
an action, see its observation, and decide what to do next. Plan-and-Solve
adds an explicit checklist that the model can revise through native tool calls.
The same TinyAgent harness governs tool execution, continuation, step budgets,
and stopping in either strategy. It uses the OpenAI Responses API throughout.

## Setup and launch

From the repository root:

```bash
source .venv/bin/activate  # if using the repository's virtual environment
python -m pip install -r session4/requirements.txt
python -m pip install -r session4/requirements-gradio.txt  # optional browser UI
```

Set `OPENAI_API_KEY` and `TINYAGENT_MODEL`. Launch the terminal app with
`cd session4 && python main.py`; launch Gradio with `cd session4 && python
gradio_app.py` (localhost only). Retrieval uses the configured OpenAI embedding
model and the local `documents/` corpus; embedding requests can occur on a
cache miss.

## Modes and run control

Both interfaces default to **ReAct + Manual Continue**. Select a planning
strategy independently of the execution mode.

| Planning strategy | Manual Continue | Autonomous |
| --- | --- | --- |
| **ReAct** | Inspect each tool action before continuing. | The harness advances through successive tool actions. |
| **Plan-and-Solve** | Inspect each tool action and checklist change before continuing. | The harness advances through ordinary and planning tool actions on the same budget. |

- **Terminal** — `/planner react`, `/planner plan-and-solve`, `/planner status`,
  and `/plan`. Use `/mode manual` or `/mode autonomous`, `/steps N` to set the
  autonomous model-request limit, `/continue`, and `/stop`.
- **Gradio** — Use the **Planning strategy**, **Execution mode**, and
  **Maximum model steps** controls. The **Plan** tab shows the selected
  strategy, the current or most recent run's strategy and outcome, and its
  revision and checklist if one exists. Chat shows tool arguments and
  observations. Trajectory shows plan changes and per-run metrics.

Continue, stop, or clear a paused run before changing its planning strategy.
A running action must reach a safe boundary first. Strategy changes after a
run affect the next task. The completed or stopped run remains inspectable.

Submitting a task starts a run. Each Responses API model request is one step.
Manual mode sends one request for the submitted task; if a tool is requested,
the tool executes, its observation is recorded, and the run pauses. Each
Continue control or `/continue` makes exactly one further request. A planning
tool consumes a step on the same terms as any other function call.

Autonomous mode continues after each single valid tool action and its
observation. A response without a tool call ends normally; that is protocol
completion, not proof that the task was correct. At the step limit, the last
tool action is executed and recorded, but no additional request is made. That
run ends as **max_steps** (incomplete), never as a completed answer. A final
allowed request with no tool call ends normally. Tool failures become visible
observations and autonomous execution may continue. Model-directed retrieval
is a tool action and follows the selected mode. Automatic RAG runs once at the
start of the user turn, not on each autonomous iteration.

Stop run and `/stop` can end a paused manual run without another request.
During execution, stop is cooperative. An in-flight network request or
synchronous tool finishes before stop takes effect. A response arriving after a
stop request is discarded before execution, so no unmatched function-call
item enters conversation Memory. A tool already executing is allowed to
finish; its observation is recorded before the loop stops. Conflicting
conversation mutations are serialized with the active Gradio action. The
terminal runs autonomous work on a background worker so `/stop` and read-only
plan inspection remain available. The terminal prints the recorded actions
and observations when the worker returns.

## Planning tools and state

Plan-and-Solve supplies instructions to create a plan for non-trivial tasks,
execute the work with permitted tools, revise the checklist as evidence
changes, and declare progress using the current task numbers. For a trivial
request, the model can answer directly. ReAct requests omit the planning
instructions and tool schemas.

| Function | Responses API arguments | Result |
| --- | --- | --- |
| `update_plan(tasks)` | Required `tasks` is an array of strings. | Create or revise the ordered checklist and return it. |
| `mark_done(index)` | Required `index` is a one-based integer. | Declare the task at the current position done and return the checklist. |

Both are strict `function` tools with `additionalProperties: false`. The
harness rejects an empty list, blank or non-string descriptions, and invalid
indices as recoverable tool observations. Plan-and-Solve uses the same
registry, schema generation, and tool-call dispatcher as the other tools.

Each task has a run-local integer ID assigned in creation order, a description,
and a `pending` or `done` status. The revision starts at zero, increments on
checklist changes and new completion declarations, and stays unchanged on an
identical ordered checklist after whitespace normalization or repeated
`mark_done`. Replanning collapses
whitespace and compares descriptions case-insensitively. A description that
matches exactly once in both the old and new plans keeps its ID and status,
even if reordered. A rewritten description or an ambiguous duplicate on either
side of a changed plan gets a new pending ID. There is no semantic matching.
Each `plan_changed` event lists the new revision, additions, removals, and
any declared completion, along with the step and provider call ID.

The active checklist belongs to the current run and starts empty for each new
task, even when Memory contains earlier plan observations. It stays visible
after the run completes, stops, fails, or exhausts its budget. Clearing the
conversation clears that checklist and its run association while retaining
Trajectory and the selected strategy. Clearing Trajectory leaves the current
checklist alone. Each Gradio session has its own state.

A checked item is a **model declaration**. A request that ends normally is
**protocol completion**. A fully checked plan can still end at `max_steps`
or fail, and a final answer can leave items pending. Inspect the run outcome,
the Plan tab, and the `run_finished` event's `pending_plan_tasks` field to see
these cases. The harness does not verify achievement of the user's goal.

## Tools, memory, and observability

Responses API function calls preserve provider `call_id` correlation; schemas,
arguments, tool names, and execution are validated against the allowlist.
For reasoning models, the complete provider output items, including opaque
reasoning protocol items, are replayed with tool outputs so a follow-up request
can continue the same function-call turn. Those opaque payloads are not shown
in the Context view, Trajectory request display, or transcript; only supported
provider-supplied reasoning summaries are displayed.
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
and session-local Gradio state are carried forward. Quiz generation, grading,
and feedback summaries are tool-free operations controlled by the quiz
harness. The chosen strategy still applies to ordinary tutoring.

## Compare ReAct and Plan-and-Solve

In Gradio, choose **Autonomous**, turn retrieval **Off**, and set a budget of
15 model requests. Keep the model and other settings fixed. Start with **ReAct**
and try one of these prompts. Clear the conversation, switch to
**Plan-and-Solve**, and submit the same prompt again.

```text
What is 17% of 2,350, minus 3 × 41?
```

The expected answer is **276.5**. For a task with more intermediate results, try:

```text
I'm buying 3 notebooks at $4.25 each and 2 pens at $1.80 each,
plus 8% sales tax. What's the total?
```

The expected total rounded to cents is **$17.66**. Use Chat or the final-answer
step in **Trajectory** to check the answer. For each run, record `outcome`,
`metrics.model_requests`, `metrics.tool_calls`, `metrics.planning_tool_calls`,
`metrics.total_tokens`, `metrics.elapsed_seconds`, and `plan.pending_count`
if a plan exists. `tool_calls` counts parsed requests recorded as actions,
including invalid calls and rejected batches. `elapsed_seconds` measures active
harness time, including synchronous tools and retrieval, excluding time waiting
for manual Continue. The token total covers Responses requests and is unavailable
if any step lacks usage data. Terminal autonomous runs print these metrics, and
`/plan` shows checklist state.

Compare which operations were used, whether the answer was correct, and what
the checklist helped you see. A plan can improve coordination and visibility
while using steps, tokens, and time that could go to the task itself. These
prompts are optional live experiments. The scripted purchase comparison can
also be run offline from the repository root:

```bash
python -m unittest discover -s session4 -p test_planning.py -k test_same_purchase_task -v
```

In that fixed sequence, both strategies return $17.66. ReAct makes five model
requests, and Plan-and-Solve makes eight, including three requests for planning
actions. These counts illustrate the orchestration cost in the script.

## Tests

With the Gradio requirements installed, run the Session 4 offline suite:

```bash
python -m unittest discover -s session4 -p 'test_*.py' -v
```

Run the earlier regression suites:

```bash
python -m unittest discover -s session3 -p 'test_*.py' -v
python -m unittest discover -s session2 -p 'test_*.py' -v
python -m unittest discover -s session1 -p 'test_*.py' -v
```

Offline tests use fake Responses API results, fixed embedding vectors, and
mock tools. They exercise the Gradio callbacks without a live browser. Live
model behavior, account compatibility, and answer quality remain unmeasured.
The plan lives in memory for the session. Stop is cooperative and Gradio
normally refreshes the plan after each Submit or Continue completes. The
existing compaction rule still rejects Memory containing an incomplete turn,
including one that ended with a tool observation at a stop or step limit.
