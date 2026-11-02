# Session 2: transparent TinyAgent tutor

Session 2 remains a separate workspace so `session1/` stays the reproducible
checkpoint. It preserves the course Session 2 architecture in `tinyagent.py`:
`LLM`, `Response`, `Memory`, `Trajectory`, `Step`, and `TinyAgent`. Session 1
serves as a reference for UI behavior and utilities; its older core classes are
not the foundation for this implementation.

## Setup

From the repository root, install core and terminal dependencies:

```bash
python -m pip install -r session2/requirements.txt
```

Install the optional browser dependency for Gradio:

```bash
python -m pip install -r session2/requirements-gradio.txt
```

Set `OPENAI_API_KEY` and `TINYAGENT_MODEL` as described in
[Session 1](../session1/README.md). Keep the API key in the environment; it is
never written to quiz artifacts.

## Launch

From the repository root:

```bash
cd session2
python main.py
```

For the browser app, run `python gradio_app.py` from `session2/`. It binds to
localhost. Each browser session gets its own agent, memory, quiz state, and API
client. Both interfaces use the same Session 2 TinyAgent implementation.

Terminal commands: `/clear`, `/save filename.md`, `/quiz [topic]`, `/quiz end`,
`/context reload`, and `/quit`. Gradio supports the quiz topic and Start/End
quiz buttons, plus the same `/quiz` commands in its message box.

## Persistent context and quiz skill

Edit `about_me.md` with information that should be available on every model
request. Its starter sections cover technical background, experience, learning
objectives, and instructional preferences. It uses placeholders and invents no
personal details. The app combines this file with persistent tutor instructions
into one system message, separate from conversation `Memory`. Missing or blank
files are handled as empty personal context. Use Reload about_me.md in Gradio or
`/context reload` in the terminal to apply edits without restarting. Clearing
conversation memory leaves persistent instructions in effect.

The quiz is a manually activated skill defined in `skills/quiz_me.md`. Use
`/quiz` to use available course material or `/quiz <topic>` to select a topic.
The user controls activation and deactivation; `/quiz end` returns to ordinary
tutoring. The harness validates the generated question JSON, tracks progress,
and writes Markdown artifacts under `quizzes/` using exclusive, non-overwriting
filenames. Artifacts contain topic, questions, answer information, progress,
learner responses, feedback, and final summary.

The skill is an instruction file, not an executable function. Python handles
activation, quiz state, validation, and file operations. The model generates
questions and evaluates answers. This is manual orchestration, not autonomous
tool selection or execution.

## Trajectory and Context views

The Trajectory tab shows recorded Runs and their Steps, including the request,
model, reasoning effort, answer, available usage metadata, latency, errors, and
provider reasoning summary when returned. The current agent makes one model
call per run; the viewer reads the step list so it can show multiple steps as
the course grows. Clear conversation affects `Memory`; Clear trajectory affects
`Trajectory`.

The Context tab distinguishes persistent tutor instructions, `about_me.md`, an
active quiz skill, and conversation `Memory`. It also shows the effective
message sequence for the latest model request. Memory preserves chat continuity;
Trajectory records execution inputs and outcomes. Neither is a replacement for
the other.

The OpenAI Responses API may return a provider-generated reasoning summary and
token usage depending on the model and API response. TinyAgent records a summary
only when it is actually exposed; it does not reconstruct hidden reasoning.
The interface labels when no summary was returned and displays available token
metadata. Reasoning summaries are not private chain-of-thought transcripts.

## Offline tests

Run the Session 2 checks from the repository root:

```bash
python -m unittest discover -s session2 -p 'test_*.py' -v
```

They use mocked model responses and make no API requests. Gradio construction
and session checks run when the optional dependency is installed.

## Reasoning comparison experiment

The original Session 2 benchmark remains available. It compares reasoning
effort on five short questions, reporting exact-match accuracy, latency, token
counts, and optional cost estimates. Run it from `session2/`:

```bash
python reasoning_comparison.py --efforts low,medium,high --repetitions 3
```

The default run makes 45 API calls and may incur charges. Results are written
under `results/`; existing files are not overwritten. Use
`trajectory_viewer.py results/reasoning-YYYYMMDD-HHMMSS.json` to render a local
HTML report. To estimate cost, supply current per-million-token rates with
`--input-price` and `--output-price`. The benchmark uses OpenAI's Responses API,
requests a reasoning summary when available, and keeps reasoning out of Memory.
