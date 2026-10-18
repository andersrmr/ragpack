# Session 1: connect TinyAgent to OpenAI

Requires Python 3.10 or newer. This keeps the course's `Response`, `LLM`,
`Memory`, and `TinyAgent` classes. Only `LLM` knows about the provider.

## Set up

Run these commands from the `session1` folder:

```bash
python3 -m venv .venv  # Skip this if the virtual environment already exists.
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Set the key without putting it into your shell history (Bash):

```bash
read -r -s -p "OpenAI API key: " OPENAI_API_KEY
echo
export OPENAI_API_KEY
```

Choose a model ID from your OpenAI account that supports Chat Completions:

```bash
export TINYAGENT_MODEL="replace-with-your-model-id"
```

There is deliberately no default model. You can change models by changing this
variable, or explicitly use `LLM(model="your-model-id")` in Python. An explicit
model takes precedence over the environment variable.

The program reads the process environment; it does not automatically load `.env`
files. Never put your API key into the Python files or commit it.

## Launch terminal chat

After completing setup and setting both environment variables, run from `session1`:

```bash
source .venv/bin/activate
python main.py
```

The startup message shows the configured model. Type a question and press Enter;
TinyAgent prints the reply and remembers previous turns for follow-up questions.
Input is one line per message. Empty lines are ignored.

Replies are rendered as Markdown using [Rich](https://rich.readthedocs.io/en/stable/markdown.html),
so emphasis, lists, and fenced code blocks are easier to read. Code blocks have
syntax highlighting in terminals that support color. Rendering changes only the
display; conversation memory keeps the model's original text.

If you installed dependencies before this feature was added, update them first:

```bash
python -m pip install -r requirements.txt
```

- `/clear` resets conversation history without making an API request.
- `/quit` exits. Ctrl+C or end-of-input (Ctrl+D on Linux/macOS) also exits.

API errors are reported and the chat stays open. A failed request's unanswered
user turn is removed so you can re-enter it; completed turns remain available.
The client is closed when the chat exits, including after errors or Ctrl+C.
History is held in memory only and is lost when the program exits. Each question
sent to the model makes a live API request; commands do not.

## Run the one-message smoke test

```bash
python smoke_test.py
```

This makes one live API request, which uses your account's API quota and may
incur a charge. It prints a short greeting followed by metadata:

```json
{
  "model": "the-model-returned-by-the-api",
  "prompt_tokens": 15,
  "completion_tokens": 8
}
```

The numbers above are illustrative. Actual counts and model names vary. Token
counts are `null` if the provider omits usage. Missing configuration produces a
clear error before any request. Authentication, connection, quota, and API errors
produce a message and a nonzero exit code. The SDK uses a 60-second timeout and
automatic retries are disabled so a smoke test does not silently retry.

## Use the course classes

```python
from tinyagent import LLM, Memory, TinyAgent

llm = LLM()
try:
    agent = TinyAgent(llm=llm, memory=Memory())
    print(agent.run("What does conversation memory do?"))
    print(agent.run("Explain that with a small example."))
finally:
    llm.close()
```

`TinyAgent.run()` still returns a string and stores the user and assistant turns
in memory. `LLM.generate()` returns a `Response` containing text and metadata.
As in the course example, a failed request leaves the latest user turn in memory.

The wrapper uses the SDK's Chat Completions method to keep the course's message
format. Requests contain only `model` and `messages`; temperature and reasoning
settings are omitted. `main.py` supplies the terminal interface without changing
the core classes. A tutor prompt and graphical interface remain later changes.

Official references: [OpenAI Python SDK](https://developers.openai.com/api/reference/python)
and [API quickstart](https://developers.openai.com/api/docs/quickstart).

## Verify without API access

```bash
python -m unittest discover -v
```

These checks use the real SDK against simulated HTTP responses. They verify
configuration errors, model selection, request contents, metadata, and multi-turn
memory without sending any requests to OpenAI or requiring a real API key.
Terminal tests simulate input and model responses to verify follow-up history,
clearing, quitting, error recovery, missing configuration, and client cleanup
after normal exit, end-of-input, Ctrl+C, and unexpected errors.
They also verify Markdown rendering while preserving the original response in memory.
