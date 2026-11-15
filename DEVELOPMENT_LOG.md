# TinyAgent Development Log

## Session 4 — Observable autonomous ReAct loop, Stage A (2026-11-11)

- Created an independent `session4/` checkpoint by copying the completed
  Session 3 Stage B implementation. Session 1, Session 2, and Session 3 source
  files were left unchanged. The existing Responses API request shape,
  provider `call_id` pairing, allowlisted tools, retrieval corpus and modes,
  Memory/Trajectory split, compaction, context, quiz, transcript, and Gradio
  session patterns were retained.
- Added selectable Manual Continue and Autonomous modes with Manual Continue as
  the safe default. Each model request counts as one step. Autonomous execution
  makes sequential requests after one tool action and observation, terminates
  normally on a no-tool-call response, and records a distinct incomplete
  `max_steps` outcome after executing an action on the final allowed request.
  Tool errors remain observations and can be followed by another request.
- Added explicit run outcomes, mode, configured step limit, step details, and
  run-finished trajectory events. Stop requests are cooperative. Gradio Stop
  uses a separate concurrency lane; terminal autonomous execution runs on a
  worker so `/stop` can be entered. A response returned after stop is
  discarded before committing function calls to Memory. Active Gradio actions
  serialize conflicting conversation operations.
- Automatic RAG remains once per user submission; model-directed retrieval is
  a tool action in the same mode loop. Responses parallel tool calls remain
  disabled and unexpected batches are rejected visibly without execution.
  Provider-supplied reasoning summaries are preserved; an unavailable summary
  is labeled rather than synthesized. No Plan-and-Solve functionality,
  `update_plan`, `mark_done`, planner tools, plan state, or planning UI was
  added.
- Added `session4/test_autonomy.py`, copied the Session 3 offline regression
  tests, and wrote `session4/README.md` with setup, control semantics,
  observability, limits, retrieval, and test instructions. The terminal and
  Gradio interfaces both expose mode and step limit; Gradio also exposes Stop.
  Added Session 4 generated-output ignore rules to `.gitignore`.

### Validation

- Session 4: **35 offline tests passed** using fake Responses results,
  mocked tools, fixture embeddings, and fake Gradio sessions.
- Session 3: **26 tests passed**; Session 2: **9 tests passed**; Session 1:
  **28 tests passed**.
- `compileall` completed for Session 4. `git diff --check` is recorded after
  the final documentation and code review below.
- No live API, embedding, or browser session was performed. Account/model
  compatibility and answer quality remain unverified. Stop does not cancel an
  in-flight provider request or synchronous tool call; it takes effect at the
  next safe boundary.

## Session 4 — Preserve Responses reasoning protocol items (2026-11-15)

- Diagnosed a live Continue failure from the provider error: the follow-up
  request replayed a `function_call` without its required preceding reasoning
  output item. `LLM.generate` had extracted only the provider-supplied summary
  and discarded the full protocol item.
- Updated Session 4 to retain and replay the complete Responses output-item
  sequence in original order before appending each function output. This keeps
  the opaque provider item available for API protocol continuity without
  exposing its encrypted payload in the Gradio Context view, Trajectory input
  display, chat history, or transcript. Supported provider reasoning summaries
  remain visible as before.
- Added offline regression coverage for the exact reasoning/function-call /
  function-output sequence, successful Continue behavior, and display
  redaction. No live API retry was made.
- Validation: **36 Session 4 offline tests passed** and `compileall` completed.
  The successful continuation is verified with a fake Responses result; the
  original live failure was not retried against an API account.

## Session 3 — Native tool calling, Stage A (2026-11-07)

- Created an independent `session3/` implementation by adapting Session 2's
  context, manually controlled quiz, transcripts, and Gradio/terminal patterns.
  Session 1 and Session 2 source files were left unchanged. The inspected
  checkout did not contain a repository-root `README.md`; Session 3 setup and
  architecture are documented in `session3/README.md`.
- Added Responses API function tool schemas, parsing of `function_call` output
  items, and correlated `function_call_output` items using provider `call_id`s.
  Provider reasoning summaries and token usage remain available, while hidden
  reasoning is neither requested as chain-of-thought nor synthesized.
- Added allowlisted `NativeTools` with schema/argument validation and the
  `multiply(a, b)` and explicitly simulated `get_weather(location)` examples.
  Invalid JSON, unsupported names, bad arguments, and execution exceptions
  become visible observations and trajectory errors. Parallel function calls
  are disabled at the request; a returned multi-call batch is retained and
  visibly rejected without executing any function.
- Extended Memory with distinct Responses API call/output items and Trajectory
  with model-request, answer, action, observation, error, latency, and usage
  data. Tool execution pauses. Terminal `/continue` and Gradio Continue each
  make one model request; clearing Memory cancels pending continuation.
- Adapted both interfaces and carried forward reloadable `about_me.md`, opt-in
  Session 3 source context, manual quiz skills/artifacts, transcript export,
  reasoning effort, session-local Gradio state, and Chat/Context/Trajectory
  tabs. The browser transcript shows each action and its resulting observation.
- Added offline protocol, one-action, Continue, validation/error, multi-call,
  clear-state, and UI tests. No live API calls were made.

### Validation

- Session 3: **15 tests passed**; `compileall` completed; `git diff --check`
  passed. Gradio construction and fake session flow were exercised.
- Session 2 regression: **9 tests passed**.
- Session 1 regression: **28 tests passed**. Its existing suite emitted one
  `ResourceWarning` about an unclosed asyncio event loop while tests passed.
- No live browser session or OpenAI account/model compatibility was tested.

### Scope and follow-up

- Stage A only: no autonomous loop, embeddings, RAG, compaction, retrieval,
  progressive skill discovery, video search, or MCP support.
- Useful manual experiment: ask for `5.1 × 7.3`, inspect the single multiply
  action and observation in Trajectory, press Continue once, and confirm the
  model's final response uses `37.23`. Repeat with a weather request to see the
  simulated label and try malformed/unsupported requests through a mocked test.

## Session 3 — Retrieval and compaction, Stage B (2026-11-10)

- Added `EmbeddingModel` and `RAGMemory` in `session3/retrieval.py`. Both
  interfaces pass the existing OpenAI SDK chat client to the embeddings
  endpoint. The installed SDK was 2.54.0; `requirements.txt` permits
  `openai>=1.66,<3`. The configurable default embedding model is
  `text-embedding-3-small`, separate from `TINYAGENT_MODEL`.
- Added a local Markdown collection with stable IDs, provenance, and one fact
  per passage. Unlike the initial Stage B assumption, the repository's
  `course_materials/session_3_memory_and_tools.py` contains the Session 3
  source facts. It says Ilse is a software engineer at a renewable energy
  startup and does not state her languages; it says Sarah speaks fluent
  Spanish and Portuguese. The corpus uses those source facts and includes a
  clearly labeled synthetic curator note describing the missing Ilse language
  fact. This makes the expected distinction explicit: context-free follow-up
  retrieval may return Sarah's language fact, while an Ilse-directed query has
  no source fact establishing Ilse's languages.
- Added cosine similarity, configurable top-k/threshold, graceful empty/error
  handling, and a local cache keyed by document content and embedding-model
  identity. The ignored cache is rebuildable with
  `python retrieval.py --rebuild-cache` from `session3/`.
- Added Off, Automatic RAG, and Model-directed retrieval modes. Automatic RAG
  uses only the latest user turn, injects labeled temporary request context
  once, and records query, scores, result IDs, threshold decision, and errors.
  Model-directed mode conditionally exposes `search_documents`; its call and
  correlated output remain in Memory and Trajectory and wait for explicit
  Continue. No query-rewriting call or autonomous tool loop was added.
- Added explicit `CompactMemory` operation with a dedicated tool-free
  Responses API request, safe complete-turn boundaries, separate summary,
  retained recent turns, failure recovery, retained transcript history, and
  available usage metadata. Added matching terminal and Gradio controls and
  retrieval/compaction observability.
- Updated `session3/tinyagent.py`, `retrieval.py`, both interfaces,
  `transcripts.py`, `session3/README.md`, and `.gitignore`. Added course-derived
  corpus files and `session3/test_session3_stage_b.py`. Session 1 and Session 2
  sources and package dependencies were unchanged.

### Validation

- Session 3: **26 offline tests passed**, including SDK request shapes with
  fakes, fixture evidence expectations, cache/ranking behavior, retrieval mode
  and Continue semantics, UI/session observability, and compaction boundaries.
- Session 1: **28 tests passed**; Session 2: **9 tests passed**. The existing
  Session 1 asyncio event-loop `ResourceWarning` still appears while its suite
  passes.
- `compileall` and `git diff --check` passed. No live API requests or model
  answer quality evaluation was performed.

### Limitations and deferred work

The course does not provide an Ilse language fact; the synthetic note makes
that absence searchable without inventing an answer. Automatic RAG intentionally
uses the latest message verbatim and may retrieve Sarah's language fact for an
ambiguous follow-up. Model-directed search depends on model query quality.
Query rewriting, hosted vector databases, autonomous tool loops, progressive
skill discovery, video search, MCP, and Session 4 work remain deferred.

## Session 2 — Reasoning comparison (started 2026-10-21)

Session 1 remains the reproducible checkpoint in `session1/`. Session 2 starts
in its own directory with Experiment 1 (reasoning comparison), followed by
Experiment 2 (trajectory inspection), before file-based skills or attachment
support.

### Initial implementation

- Added `session2/tinyagent.py` with configurable reasoning effort, a structured
  `Response.reasoning` field, provider-supplied reasoning summaries, and token
  metadata. `Memory` stores only user and assistant content.
- Added `session2/reasoning_comparison.py`, a repeatable benchmark over five
  fixed questions. It reports exact-match accuracy, latency, input/output and
  reasoning token counts, plus optional cost estimates from user-supplied
  current per-million-token rates.
- Results are written as timestamped JSON under `session2/results/`, which is
  ignored by Git. Existing result files are not overwritten. Each trial records
  the answer and available reasoning summary for local inspection.
- Added standalone setup and experiment instructions in `session2/README.md`.
  The tool uses the OpenAI Responses API; model availability and supported
  reasoning settings depend on the user's account and selected model.

## Session 2 — Trajectory inspection

- Added `Trajectory`, `Run`, and `Step` records to `session2/tinyagent.py`.
  TinyAgent records model inputs, reasoning effort, model name, latency,
  response summary, answer, token usage, and call errors. Reasoning stays out of
  conversation memory.
- Routed each fixed-question benchmark call through `TinyAgent`; the JSON
  experiment file now contains both the existing trial summary and structured
  trajectory runs.
- Added `session2/trajectory_viewer.py`, which creates a self-contained HTML
  report from new trajectory data or earlier reasoning comparison JSON files.
  The report shows aggregate metrics and expandable call details.
- Documented report generation in `session2/README.md`. No new package
  dependency is required.

## Session 2 — Persistent context, manual quiz skill, and tutor UI (2026-11-01)

- Kept the Session 2 `LLM`, `Response`, `Memory`, `Trajectory`, `Step`, and
  `TinyAgent` as the core. Session 1's prompt, transcript, and Gradio patterns
  informed the new interfaces without replacing the Session 2 architecture.
- Added `about_me.md` with editable placeholder sections and a reloadable
  persistent-context loader. Tutor instructions and personal context are
  combined into one system message and remain outside conversation Memory.
- Added `skills/quiz_me.md` and `QuizManager`. Explicit `/quiz` activation loads
  the skill, validates structured JSON questions, tracks responses, and writes
  unique Markdown artifacts under `quizzes/`. `/quiz end` removes skill
  instructions. Model requests do not write files or invoke Python tools.
- Added Session 2 terminal and Gradio applications using the same TinyAgent.
  Gradio has Chat, Trajectory, and Context tabs, session-local chat state,
  transcript downloads, quiz controls, context reload, and independent
  conversation/trajectory clearing.
- Expanded trajectory usage metadata to include prompt, completion, and total
  token aliases. The view labels absent reasoning summaries instead of
  manufacturing reasoning text.
- Added offline tests for context loading/reload/clearing, quiz activation and
  completion, artifact contents, skill deactivation, trajectory records, and
  Gradio session observability. Tests use fake model responses only.
- Updated `session2/README.md` with launch commands and the distinctions among
  persistent instructions, temporary skills, Memory, Trajectory, and manual
  orchestration.

### Validation

- Session 1 regression suite: 28 tests passed.
- Session 2 suite: 8 offline tests passed, including Responses API parsing with
  a fake response, Gradio construction, session observability, safe quiz
  filenames, and overwrite protection.
- Python compilation check completed without syntax errors.
- No live API requests were made. Browser UI verification is based on
  construction and offline session checks, not a live Gradio browser session.

### Limitations and follow-up

- Quiz quality and answer evaluation depend on the selected model; tests verify
  the harness, not educational correctness of live generated quizzes.
- Reasoning summaries and token metadata depend on model/API availability.
- Quiz artifacts are plain Markdown in the local workspace and remain available
  for local inspection after the chat session ends.

## Session 2 — Optional agent source context (2026-11-05)

- Added an opt-in source context that loads only `session2/tinyagent.py` and
  supplies it as a labeled system message while enabled. It stays out of
  conversation Memory.
- Added a Context tab checkbox and reload button, plus terminal `/source on`,
  `/source off`, `/source status`, and `/source reload` commands. The Context
  view shows whether the source is active and the content supplied.
- Added offline checks for opt-in behavior, reload after edits, removal when
  disabled, and inclusion in effective model messages.
- Validation: 9 Session 2 offline tests passed; Python compilation and
  `git diff --check` passed. No live API calls were made.

### Next

- Review the first experiment results and decide whether to expand the fixed
  question set or proceed to file-based skills.

## Session 1 — OpenAI API Integration

**Course:** Introduction to Agents  
**Project:** TinyAgent Tutor  
**Coding assistant:** OpenAI Codex

### Objective
Adapt the course's TinyAgent implementation to use the OpenAI API instead of Ollama.

### Changes implemented
- Created `tinyagent.py` using the course architecture.
- Integrated the official OpenAI Python SDK.
- Configured API access using `OPENAI_API_KEY`.
- Made the model configurable using `TINYAGENT_MODEL`.
- Added tests and setup documentation.

### Validation
- Successfully connected to an OpenAI model.
- Confirmed response text and token metadata.
- Ran offline tests successfully.

### Key architectural lesson
The LLM is responsible for generating responses, while the TinyAgent harness manages conversation history and orchestrates the interaction.

### Next steps
- Implement a minimal terminal chat interface.
- Add `/clear` and `/quit` commands.
- Introduce a course-specific tutoring system prompt.

## 2026-10-18 — Terminal tutor, transcripts, and Gradio

This entry supersedes the earlier next-step list. Work remains within Session 1:
chat and tutoring, without RAG, tool calling, or autonomous execution.

### Accomplishments and files

- Built terminal chat with conversation history, model identification, `/clear`,
  `/quit`, graceful API errors, and client cleanup. Added Rich Markdown rendering.
- Added a persistent tutor prompt: plain-language explanations, small Python
  examples, hints for exercises, direct answers when requested, and no invented
  course material or claims of access to unseen notes.
- Added `/save filename.md` and browser transcript downloads. Both preserve
  Markdown, omit system instructions, and use America/New_York save timestamps.
  Same-second saves append `-2`, `-3`, etc.; exclusive creation prevents overwrites.
- Built browser chat with separate sessions and Send, Clear conversation, Save
  transcript, and End session buttons. The terminal interface remains available.
- Created `main.py`, `prompts.py`, `transcripts.py`, `gradio_app.py`,
  `requirements-gradio.txt`, `test_main.py`, `test_gradio_app.py`, and
  `test_transcripts.py` under `session1/`, plus the root `.gitignore`.
- Modified `tinyagent.py`, `test_tinyagent.py`, `requirements.txt`, and `README.md`;
  also applied Ruff changes to `main.py`, `smoke_test.py`, `test_main.py`, and
  `transcripts.py`. The OpenAI provider and smoke-test baseline remain in use.
- Saved study transcripts and chat archives under `session1/notes/`. The older
  study transcript was renamed using its last-modified time and committed.

### Architectural decisions and lessons

- Preserved `Response`, `LLM`, `Memory`, and `TinyAgent`. `LLM` owns SDK calls and
  response metadata; `Memory` stores chat turns; the agent coordinates one model
  request per turn. Both UIs use these classes rather than separate agent logic.
- Added an optional `system_prompt` to `TinyAgent`. It is prepended to each
  request rather than stored in memory, so `/clear` retains instructions without
  duplicating them. Omitting it preserves plain chat; the smoke test is unaffected.
- Kept presentation and file operations outside the model. Rich/Gradio render
  Markdown; shared transcript helpers format and save it. UI error handling
  removes an unanswered failed turn while retaining completed conversation.
- Browser clients stay server-side, keyed by session; Gradio state holds only
  the key. Actions share a serial queue to prevent history races. End session
  closes that client's connection but retains history for saving; disconnect,
  state deletion, and server shutdown provide further cleanup.
- The LLM generates text from the messages it receives. It does not remember
  previous independent API calls, clear history, or write transcripts itself.
  The harness supplies history and instructions and implements those operations.
  A tutor prompt guides model behavior; offline tests verify delivery and state
  handling, not the quality or consistency of live teaching.

### Problems and resolutions

- Raw Markdown was awkward in the terminal: added Rich while keeping original
  response text in memory and exports.
- Pasted tutor instructions disappeared after clearing: moved them into the
  persistent system prompt.
- Reusing transcript filenames risked collisions: centralized timestamp naming
  and exclusive writes with collision counters for both interfaces.
- Yellow editor underlines were confirmed as Ruff import/formatting diagnostics;
  applied the suggested changes and reran tests.
- Sandbox restrictions blocked dependency downloads and Git writes: approved
  elevated commands installed dependencies and created the requested commits.
  In-process Gradio checks also needed execution outside the sandbox; the page
  fixture required the theme setup normally supplied by `launch()`.
- **Still open:** the suite emits `ResourceWarning: unclosed event loop` while
  passing. It predates the latest Ruff change; its exact source is not yet traced.
  Gradio test setup is a candidate, not a confirmed cause.

### Working state and next steps

- Session 1 used `gpt-6-luna` for `TINYAGENT_MODEL`.
- Latest verification: **28 offline tests passed**, including terminal/browser
  history, tutor persistence, error recovery, exports, timestamp conversion,
  overwrite protection, and client cleanup. Separate Gradio chat/download and
  page/configuration checks passed with fake model responses; no live API calls
  were made by these checks. The terminal has also been exercised interactively.
- From `session1`, activate `.venv`, set `OPENAI_API_KEY` and `TINYAGENT_MODEL`,
  then run `python main.py` or `python gradio_app.py`. Browser dependencies are in
  `requirements-gradio.txt`. Run tests with `python -m unittest discover -v`.
- Latest commit: `2a0fe2c` (Ruff formatting and UTC alias in transcript utilities).
  Before this documentation edit, tracked files were clean; these new notes were
  untracked: `notes/codex_session_2026-10-18.md` and
  `notes/study-notes-2026-10-18_22-04-53.md`. This log update is not yet committed.
- Next: review and commit the new notes/log; investigate the event-loop warning;
  reconcile README's Python 3.10 minimum with `datetime.UTC`, which requires
  Python 3.11+. Use the installed Python 3.14 environment for now.
- Continue trying tutor questions and comparing prompt/model behavior. Keep
  conversation archives as snapshots and this log as the concise resume point.
  Save before clearing/exiting; do not add later-session agent features yet.
