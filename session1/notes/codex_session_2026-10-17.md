# Explore TinyAgent tutor improvements

> [
>
> I'm taking an introductory course on Agents, where we're gradually building up our own "TinyAgent". We're currently in Session 1, just getting started, so all it has is basic chat functionality.
>
> We're supposed to play around with the code, and explore adding new features that would make the TinyAgent into a better tutor for the course.
>
> Techniques like RAG and tool calling come in later sessions, so for now we're exploring ideas like:
>
> - Trying other models
> - Writing a system prompt to make the TinyAgent a better tutor
> - Refining the UI
>
> Note that I'll want to:
>
> - Use an inference provider with an API key from an environment variable (e.g.,, I have an openai api key).
> - Build our own chat interface, either on the command line or using gradio.
>
> So, where do you think we should start?
>
> ]
>
> User context
>
> Attachments:
> - # 3 - Source Code <br/> Raw examples &nbsp; &nbsp; → &nbsp; &nbsp; Object-orien…: `~/.codex/attachments/c2461d85-a738-4d94-b84d-6114d4b2ce98/Pasted text.txt`

<details><summary>3 previous messages</summary>

> I’ll read the course example and check the workspace, then suggest a starting point that fits Session 1.
>
> The example already separates the model, conversation memory, agent, and UI. I’d keep that structure and start with a tutor prompt and a simple command-line chat. I’m using the openai-docs skill to check the provider setup guidance.
>
> <details><summary>Searched the web</summary>
>
> Searched the web for `OpenAI API environment variable API key`
>
> Searched the web for `https://developers.openai.com/api/docs/quickstart`
>
> </details>
</details>

I’d start with **a command-line TinyAgent, connected to your inference provider, with a carefully written tutor prompt**. That gives you a small working baseline for experiments.

Your course code already has useful boundaries: `LLM` makes requests, `Memory` holds the conversation, and `TinyAgent` connects them. Keep those.

I’d make changes in this order:

1. **Connect the provider.** Read the key from `OPENAI_API_KEY` and make the model configurable. OpenAI’s Python SDK reads that environment variable automatically. Keep provider details inside `LLM`. ([Official quickstart](https://developers.openai.com/api/docs/quickstart))
2. **Build a minimal terminal chat.** Add a conversation loop plus `/clear` and `/quit`. This makes it easy to inspect what happens on each turn.
3. **Add the tutor prompt.** Send it with every request, including after clearing the conversation.
4. **Compare behavior.** Try the same questions with and without the prompt, then compare models while keeping the prompt fixed.
5. **Add Gradio** once the tutoring behavior is useful.

Here’s the tutor prompt I’d begin with:

```text
You are a patient tutor for an introductory course on AI agents.

The learner is in Session 1. Their TinyAgent consists of an LLM
wrapper, conversation memory, and a chat interface.

Explain concepts in plain language and connect explanations to
these components. Use small Python examples when helpful.

For exercises, offer a hint before a full solution. If the learner
asks for a direct explanation or solution, provide it.

Ask one focused question at a time when checking understanding.
Correct misconceptions kindly and explain why.

Keep suggested implementations within basic chat functionality.
Briefly explain later topics when asked, but avoid requiring RAG,
tool calling, or agent frameworks.

Do not invent course material or claim access to unseen notes.
Ask the learner to paste relevant material when necessary.
```

Two details in the pasted code deserve attention: `reasoning_effort="none"` should be optional rather than sent unconditionally, and the UI’s `8192` context limit belongs to the original local setup. Neither should be assumed for another model.

For your first experiment, use three prompts:

- “Explain what `Memory` does in our TinyAgent.”
- “I think the model remembers our previous API calls automatically. Is that right?”
- “Help me add a clear-conversation command. Give me a hint first.”

Judge the answers on **correctness, clarity, and whether they help you learn**. That will give you a concrete basis for deciding whether a prompt change or a model change improves your tutor.

> ok, what would you do to help me start by implementing change 1.  start by telling me.  do nothing yet.

I’d adapt your existing `LLM` class to use OpenAI while keeping the course’s `Response`, `Memory`, and `TinyAgent` structure.

Specifically, I would:

1. **Create a runnable Python module** from the pasted example, initially containing the core classes.
2. **Use the official OpenAI Python SDK** inside `LLM.generate()`, replacing the manual HTTP request.
3. **Read `OPENAI_API_KEY` from the environment**, with a clear error if it’s missing. The key would stay out of the source code.
4. **Make the model configurable** through an environment variable such as `TINYAGENT_MODEL`.
5. **Omit optional model settings initially**, including the hardcoded `reasoning_effort="none"`.
6. **Preserve the existing return format**: response text plus model name and token counts.
7. **Provide setup instructions and a one-message smoke test** so you can verify the connection before building the chat interface.

The result would still be your course’s TinyAgent, with its model requests going to OpenAI instead of local Ollama.

I haven’t changed any files or made any API calls.

> implement these 7 steps for change 1

<details><summary>4 previous messages</summary>

> I’ll implement the provider connection, keep the course’s core classes, and add setup instructions plus a one-message smoke test. I’ll check the workspace and current SDK guidance first.
>
> <details><summary>Searched the web</summary>
>
> Searched the web for `OpenAI Python SDK chat completions environment key`
>
> Searched the web for `https://developers.openai.com/api/reference/python`
>
> </details>
>
> The `session1` folder has a virtual environment but no source files yet. I’ll add the core module and a separate smoke-test script there. The API key and model will both come from environment variables; neither is currently configured, so I can verify the code locally but can’t test a live response yet.
>
> <details><summary>Created 6 files, edited a file</summary>
>
> - Wrote `./session1/.gitignore` (+4 -0)
> - Wrote `./session1/README.md` (+83 -0)
> - Wrote `./session1/requirements.txt` (+1 -0)
> - Wrote `./session1/smoke_test.py` (+44 -0)
> - Wrote `./session1/tinyagent.py` (+88 -0)
> - Wrote `./session1/README.md` (+10 -0)
> - Wrote `./session1/test_tinyagent.py` (+79 -0)
>
> </details>
</details>

Implemented all seven steps in `session1`.

- [Core classes]( /home/rianderson/sandbox/ragpack/session1/tinyagent.py) use the OpenAI SDK, environment configuration, and the course’s response format.
- [Smoke test](./session1/smoke_test.py) sends one message and prints token metadata.
- [Setup instructions](./session1/README.md) explain installation, key configuration, and model selection.

The SDK is installed, and all three offline tests pass. Live verification remains pending because `OPENAI_API_KEY` and `TINYAGENT_MODEL` aren’t configured. The smoke test correctly reports the missing key.

> Let's implement Change 2: a minimal terminal chat interface.
>
> Please create a `main.py` file that imports and uses the existing `LLM`, `Memory`, and `TinyAgent` classes from `tinyagent.py`.
>
> Requirements:
>
> 1. Preserve the original Session 1 architecture. Don't refactor the core classes unless absolutely necessary.
> 2. Implement a terminal-based interactive chat loop where I can enter questions and see the model's responses.
> 3. Maintain conversation history using the existing `Memory` class.
> 4. Support `/clear` to reset conversation history and `/quit` to exit.
> 5. Ensure the LLM client is properly closed when exiting, including after errors or Ctrl+C.
> 6. Display a brief startup message identifying the model in use.
> 7. Handle API errors gracefully without crashing or unnecessarily losing conversation history.
> 8. Keep the implementation simple and educational, using standard Python without additional frameworks.
> 9. Add clear instructions to README.md explaining how to launch the interface.
> 10. Add appropriate offline tests without making live API calls.
>
> For now, do NOT implement the tutoring system prompt, Gradio interface, RAG, tool calling, or autonomous agent features. Those belong to later changes.
>
> Before implementing, briefly explain your proposed design and identify the files you intend to modify.

<details><summary>2 previous messages</summary>

> I’ll add `session1/main.py` with an input loop that passes messages to `TinyAgent.run()`. `/clear` will empty the existing memory, and `/quit`, Ctrl+C, or end-of-input will exit through a `finally` block that closes the client.
>
> If an API request fails, the loop will report the error and remove the unanswered user turn so you can retry while keeping the completed conversation. I’ll add `session1/test_main.py` and update `session1/README.md`; the core classes will stay unchanged.
>
> <details><summary>Created 2 files, edited a file, explored a list, ran 2 commands</summary>
>
> <details><summary>Explored a list</summary>
>
> - Listed files
>
> </details>
>
> - Ran `cat session1/tinyagent.py session1/smoke_test.py session1/test_tinyagent.py session1/README.md`
>
> - Wrote `./session1/README.md` (+27 -1)
> - Wrote `./session1/main.py` (+68 -0)
> - Wrote `./session1/test_main.py` (+105 -0)
>
> - Ran `.venv/bin/python -m unittest discover -v`
>
> </details>
</details>

Implemented [main.py](./session1/main.py) with conversation history, `/clear`, `/quit`, model identification, API error recovery, and client cleanup on exit or Ctrl+C. Core classes remain unchanged.

After configuring your environment variables, launch it with:

```bash
cd session1
source .venv/bin/activate
python main.py
```

Updated the [README](./session1/README.md) and added offline tests. **All 10 tests pass**, with no live API calls.
