# TinyAgent Session 3: Native Tools, Retrieval, and Memory

Session 3 is an independent checkpoint based on Session 2. Stage A added native
Responses API function calling with one action per step. Stage B adds a small
local document retriever, two retrieval policies, and explicit conversation
compaction. Session 1 and Session 2 source files remain unchanged.

## Setup and launch

Install terminal dependencies from the repository root:

```bash
python -m pip install -r session3/requirements.txt
```

For Gradio, also install:

```bash
python -m pip install -r session3/requirements-gradio.txt
```

Set `OPENAI_API_KEY` and `TINYAGENT_MODEL`. From the repository root, launch
the terminal app with:

```bash
cd session3
python main.py
```

Launch the browser interface from `session3/` with `python gradio_app.py`. It
binds to localhost. Both interfaces use the same `TinyAgent` core. Retrieval
uses the same OpenAI SDK client as chat; the SDK request is
`client.embeddings.create(model=..., input=[...])`. The checked environment used
OpenAI Python SDK 2.54.0; `requirements.txt` allows versions `>=1.66,<3`.

## Retrieval configuration and corpus

`retrieval.py` provides `EmbeddingModel` and `RAGMemory`. The local Markdown
corpus is under `documents/`. Each file has YAML-like front matter containing
stable `id`, `title`, and `source` fields, followed by passage text. Only those
files are loaded; the repository, `about_me.md`, transcripts, and arbitrary
user files are not scanned.

The fact passages are adapted from the repository course source,
`course_materials/session_3_memory_and_tools.py`, section 2.3. The source says
Ilse is a software engineer at a renewable energy startup and lists her
residence and interests. It does not state which languages Ilse speaks. The
same source says Sarah speaks fluent Spanish and Portuguese. A clearly labeled
synthetic curator note records that absence so retrieval does not turn it into
a claim about Ilse.

Settings:

- `TINYAGENT_EMBEDDING_MODEL` defaults to `text-embedding-3-small` and is
  separate from `TINYAGENT_MODEL`.
- `TINYAGENT_RETRIEVAL_TOP_K` defaults to `3`.
- `TINYAGENT_RETRIEVAL_THRESHOLD` defaults to cosine similarity `0.30`.
- The local embedding cache is `session3/.cache/embeddings.json`, ignored by
  Git. Cache keys include the embedding model identity and a hash of document
  text. Changed content or model identity is embedded again; stale vectors are
  removed when the cache is refreshed.
- To inspect settings or rebuild all current document embeddings, run from
  `session3/`: `python retrieval.py --rebuild-cache`. This requires the API key
  and may make embedding requests. Empty corpora and no matches return no
  passages; API and cache errors are visible in Context or tool observations.

## Retrieval modes and the Ilse comparison

Select a mode in the Gradio Chat tab. In the terminal, use
`/retrieval off`, `/retrieval automatic`, or `/retrieval directed`. The selected
mode is shown in the Gradio status and Context view and reported after each
terminal mode command.

- **Off** performs no retrieval or embedding request. The existing Stage A
  demonstration tools remain available.
- **Automatic RAG** embeds and searches using only the latest user turn, once at
  the start of that turn. Accepted matches are added to that one effective
  model request as labeled temporary developer context. They are not inserted
  into conversation Memory and are not repeated on Continue. If no passage
  clears the threshold, the request says no relevant local passage was
  retrieved.
- **Model-directed retrieval** exposes `search_documents(query, top_k)` as an
  allowlisted Responses API function tool. The model decides whether to call it
  and supplies its query. The harness searches the same corpus, returns
  source-labeled matches with scores, records the function call/output items,
  and pauses. Continue makes exactly one next model request. It may answer or
  request another single tool action, which again waits for Continue.

Run the sequence once in Automatic RAG:

1. `What does Ilse do for work?`
2. `Does she speak any languages?`

Then clear the conversation, choose Model-directed retrieval, and repeat both
questions. Automatic RAG searches the follow-up wording alone; the course
example shows that it can retrieve Sarah's Spanish/Portuguese fact for “she.”
In the directed run, the model can use the conversation to query for “Ilse's
languages” (the actual query is visible in the function arguments). The source
has no language fact for Ilse, so the expected relevant evidence is the labeled
note that the source is silent; neither retrieval mode should infer a language
for her. Model query formulation and scores can vary, and offline tests use
fixed vectors to verify these evidence expectations. Query rewriting is not
silently added to either mode; it remains a separate possible experiment.

The Context tab shows the selected mode, top-k and threshold, latest automatic
query/matches/scores/threshold decision, exact temporary context in the latest
effective request, model-directed tool schema/action/observation, and compacted
summary boundary. Trajectory stores structured automatic retrieval events,
model requests, tool actions and observations, answers, and compaction events.
The Chat view renders model-directed actions and observations.

## Compacting conversation Memory

Compaction is explicit and only works at a safe turn boundary. In the terminal,
run `/compact` (retains the two most recent complete turns) or
`/compact 3`. In Gradio, set “Recent turns to retain” and choose “Compact
Memory”. Compaction makes a separate, tool-free Responses API summarization
request using only older complete conversation turns. On success, Memory holds
the summary separately plus the retained complete turns. A tool call/output
pair is kept together; unresolved tool state blocks compaction. On failure,
working Memory is unchanged and the error is shown.

Original transcript export history remains available after compaction, while
the effective request uses the compacted summary and retained turns. Clear
conversation removes the summary and compacted working state but preserves
`about_me.md`, source context, and Trajectory. Clearing Trajectory does not
change Memory. Summarization usage is recorded in the compaction Trajectory
event when the Responses API returns it (`input_tokens`, `output_tokens`,
`total_tokens`, and model); the request does not store credentials or hidden
reasoning.

## Preserved Stage A behavior

Responses API function calls and matching `function_call_output` items preserve
the provider `call_id`. `NativeTools` validates allowlisted calls and records
errors as observations. Only one tool action executes per step; terminal
`/continue` and the Gradio Continue button each make exactly one model request.
Memory and Trajectory remain separate. Persistent `about_me.md`, optional
reloadable source context, manually activated `skills/quiz_me.md`, quiz
commands/artifacts, transcript export, independent clearing, and session-local
Gradio agents remain in place. Persistent instructions, personal context,
skills, retrieval results, conversation Memory, and Trajectory remain distinct.

## Offline tests

Run Session 3 tests from the repository root:

```bash
python -m unittest discover -s session3 -p 'test_*.py' -v
```

Run the earlier regression suites:

```bash
python -m unittest discover -s session1 -p 'test_*.py' -v
python -m unittest discover -s session2 -p 'test_*.py' -v
```

Tests use fake embedding vectors, fake Responses API results, and mocks. They do
not require credentials or make live embedding/chat calls. Gradio session
construction checks run when its optional dependency is installed.

## Stage B limitations and deferred work

This is a small in-process cosine-similarity demonstration over ten short
course-source fact passages and one synthetic curator note. Retrieval quality
depends on the selected embedding model, small corpus, and fixed threshold.
Automatic RAG intentionally uses the latest turn verbatim and can miss
context-dependent follow-ups. Model-directed retrieval depends on the model
choosing a useful query. No hosted vector
database, automatic query-rewriting call, autonomous tool loop, progressive
skill discovery, video search, MCP, or coding-agent challenge is implemented.
Those remain outside Stage B; live account/model compatibility and answer
quality require a manual experiment.
