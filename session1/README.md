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

- `/clear` resets conversation history, keeping the tutor instructions, without making an API request.
- `/quit` exits. Ctrl+C or end-of-input (Ctrl+D on Linux/macOS) also exits.
- `/save filename.md` exports the current conversation as Markdown.

For example, type `/save study-notes.md` at the `You:` prompt. The file contains
your questions and TinyAgent's answers in order, preserving Markdown and code
blocks. Tutor instructions and terminal status messages are omitted. Open the
file in VS Code and use Markdown Preview to read it.

Relative paths are resolved from the directory where you launched the program.
You can use an absolute path or `~/...`; spaces in filenames work without quotes,
for example `/save study notes.md`. The parent directory must already exist.
Existing files are never overwritten; choose a new filename for each export.
If history is empty, nothing is written. Save errors leave the chat running.
Saving makes no model request and does not clear or change the conversation.
Use `/save` before `/clear` or exiting if you want to keep those turns; saved
files remain on disk after either operation.

API errors are reported and the chat stays open. A failed request's unanswered
user turn is removed so you can re-enter it; completed turns remain available.
The client is closed when the chat exits, including after errors or Ctrl+C.
Unsaved history is held in memory only and is lost when the program exits. Each question
sent to the model makes a live API request; commands do not.

## Session 1 tutor instructions

Terminal chat automatically uses `TUTOR_SYSTEM_PROMPT` from `prompts.py`. You no
longer need to paste the tutor instructions at the start of a conversation.
The prompt asks for plain-language explanations tied to the course's three
components, small Python examples, hints for exercises, direct answers when
requested, and one question at a time when checking understanding. It also asks
the tutor to avoid inventing course material or claiming access to unseen notes.

Edit `prompts.py` and restart the chat to experiment with these instructions.
`TinyAgent` prepends them as one `system` message on every request, separately
from `Memory`, which stores only user and assistant turns. Consequently `/clear`
and failed-request recovery do not remove or duplicate the instructions.
The prompt uses input tokens on each request. It guides behavior; offline tests
verify how it is sent, rather than whether a live model always follows it.

Try asking “What does Memory do?”, requesting a hint for an exercise, and then
using `/clear` before another course question to compare the tutoring behavior.

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
from prompts import TUTOR_SYSTEM_PROMPT

llm = LLM()
try:
    agent = TinyAgent(llm=llm, memory=Memory(), system_prompt=TUTOR_SYSTEM_PROMPT)
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
the overall class structure. `TinyAgent` accepts an optional `system_prompt`;
omitting it keeps the original plain-chat behavior. Direct `LLM.generate()` calls
(including the smoke test) are unaffected. A graphical interface remains a later change.

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
Tutor tests check instruction ordering on every turn, memory separation, and
persistence after `/clear` and failed requests. No live API calls are made.
Transcript tests use temporary files to check Markdown and Unicode preservation,
empty history, missing filenames, overwrite protection, and recovery from file
errors while retaining the conversation.
