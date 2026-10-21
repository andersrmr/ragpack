# TinyAgent Development Log

## Session 2 — Reasoning comparison (started 2026-10-21)

Session 1 remains the reproducible checkpoint in `session1/`. Session 2 starts
in its own directory and begins with Experiment 1: compare reasoning effort on
the same questions before adding trajectory inspection, file-based skills, or
attachment support.

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

### Next

- Configure `TINYAGENT_MODEL` to a reasoning-capable model, choose supported
  effort values, and run a small comparison to validate the live API path.
- Review the first results and decide whether to add more questions or proceed
  to trajectory inspection.

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
