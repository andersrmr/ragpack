# Session 2: reasoning comparison

Session 2 is a separate workspace so `session1/` remains the reproducible
checkpoint. This first experiment compares reasoning effort on the same five
short questions. Each condition reports exact-match accuracy, latency, input and
output tokens, reasoning-token usage, and an optional estimated cost.

The comparison uses the OpenAI Responses API. It requests a provider-generated
reasoning summary when available; the summary and answer are saved in the local
results file. Conversation `Memory` stores only user and assistant messages,
never reasoning summaries. The summary is not a verbatim private chain of thought.

## Setup

Choose either option below. Session 2 does not require a dedicated virtual
environment.

### Use an already active environment

From the repository root, with the environment you want to use activated:

```bash
python -m pip install -r session2/requirements.txt
```

### Create a separate Session 2 environment

From the repository root:

```bash
python3 -m venv session2/.venv
source session2/.venv/bin/activate
python -m pip install -r session2/requirements.txt
```

Set `OPENAI_API_KEY` and `TINYAGENT_MODEL` as described in [Session 1](../session1/README.md).
The API key stays in the environment and must not be committed.

## Run the comparison

```bash
cd session2
python reasoning_comparison.py --efforts low,medium,high --repetitions 3
```

The default run makes 45 API calls (5 questions × 3 effort settings × 3
repetitions), uses API quota, and may incur charges. Change the effort list or
repetition count to control the run size. Model support varies: the API rejects
unsupported effort values for a selected model, so use settings documented for
your model. The current [Chat Completions reference](https://platform.openai.com/docs/api-reference/chat/object)
lists supported effort values; the Responses API uses the same values under
`reasoning.effort`.

Results are written to a timestamped JSON file under `session2/results/`.
Existing files are never overwritten. Every trial contains the model answer,
answer key, exact-match result, latency, usage counts, and any reasoning summary.
The summary includes accuracy, mean and median latency, and aggregate token
counts. Exact-match grading normalizes case and punctuation; it does not grade
semantically equivalent explanations.

To estimate cost, supply the current per-million-token prices for the model you
are using:

```bash
python reasoning_comparison.py --input-price INPUT_USD_PER_MILLION \
  --output-price OUTPUT_USD_PER_MILLION
```

Prices are intentionally supplied by the user because they vary by model and
can change. The estimate uses input and output token totals and excludes any
other account-level charges or discounts.

## Session 2 components

- `LLM` accepts a reasoning effort, requests a reasoning summary, and captures
  response and token metadata.
- `Response` includes a `reasoning` field.
- `Memory` stores conversation turns without reasoning summaries.
- `TinyAgent` still performs one model call per user turn and returns a
  structured `Response`.

Trajectory viewing, file attachments, file-loaded instructions, and additional
skills are later experiments; they are not part of this first comparison.
