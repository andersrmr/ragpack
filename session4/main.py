"""Terminal chat for Session 4 TinyAgent."""

import sys
import threading

from openai import APIConnectionError, APIError, APIStatusError, AuthenticationError, RateLimitError
from context import PersistentContext, SourceCodeContext, compose_instructions
from quiz import QuizManager
from rich.console import Console
from rich.markdown import Markdown
from tinyagent import LLM, Memory, NativeTools, TinyAgent
from retrieval import EmbeddingModel, RAGMemory
from transcripts import format_transcript, write_transcript


def chat(agent, context, quiz, source_context):
    console = Console()
    active_worker = None
    print(f"TinyAgent Session 4 Stage A | Model: {agent.llm.model}")
    print(f"Retrieval mode: {agent.retrieval_mode}")
    execution_mode, max_steps = "manual", 10
    print("Execution mode: Manual Continue (default), max steps: 10")
    print("Commands: /continue, /mode manual|autonomous, /steps N, /stop, /compact [retain_turns], /retrieval off|automatic|directed, /clear, /save filename.md,")
    print("/quiz [topic], /quiz end, /context reload, /source on|off|status|reload, /quit\n")
    while True:
        text = input("You: ").strip()
        if not text:
            continue
        if text == "/quit":
            if active_worker and active_worker.is_alive():
                agent.request_stop()
                active_worker.join()
            break
        if active_worker and active_worker.is_alive():
            if text == "/stop":
                print("Stop requested. Any in-flight request or tool will finish safely before stopping.\n")
                agent.request_stop()
            else:
                print("A run is active. Use /stop or wait for it to finish.\n")
            continue
        if text.startswith("/mode "):
            requested = text.partition(" ")[2].strip().lower()
            if requested not in {"manual", "autonomous"}:
                print("Usage: /mode manual|autonomous\n")
            else:
                execution_mode = requested
                print(f"Execution mode: {execution_mode}\n")
            continue
        if text.startswith("/steps "):
            try:
                max_steps = int(text.partition(" ")[2])
                if max_steps < 1:
                    raise ValueError
                print(f"Maximum model steps: {max_steps}\n")
            except ValueError:
                print("Usage: /steps positive_integer\n")
            continue
        if text == "/stop":
            print("No run is active.\n")
            continue
        if text == "/clear":
            agent.memory.clear()
            agent.pending_continue = False
            agent.active_run = None
            print("Conversation cleared; persistent context and trajectory retained.\n")
            continue
        if text.startswith("/retrieval "):
            mode = {"off": "Off", "automatic": "Automatic RAG",
                    "directed": "Model-directed retrieval"}.get(text.partition(" ")[2].strip().lower())
            if mode is None:
                print("Usage: /retrieval off|automatic|directed\n")
            else:
                try:
                    agent.set_retrieval_mode(mode)
                    print(f"Retrieval mode: {mode}\n")
                except (RuntimeError, ValueError) as error:
                    print(f"Could not select retrieval mode: {error}\n", file=sys.stderr)
            continue
        if text == "/compact" or text.startswith("/compact "):
            raw = text.partition(" ")[2].strip()
            try:
                retained = int(raw) if raw else 2
                result = agent.compact_memory(retained)
                print(f"Memory compacted; retained {retained} complete recent turns.\n{result['summary']}\n")
            except (ValueError, RuntimeError) as error:
                print(f"Could not compact Memory: {error}\n", file=sys.stderr)
            continue
        if text == "/continue":
            try:
                compose_instructions(agent, context, source_context, quiz)
                result = agent.continue_step()
                if result.tool_call:
                    console.print("TinyAgent requested another tool action; it has been executed.")
                    console.print("[bold]Action[/bold]", result.tool_call)
                    console.print("[bold]Observation[/bold]", result.content)
                else:
                    console.print("TinyAgent:")
                    console.print(Markdown(result.content))
                print()
            except Exception as error:
                print(f"Could not continue: {error}\n", file=sys.stderr)
            continue
        if text.startswith("/save"):
            filename = text.partition(" ")[2].strip()
            transcript = format_transcript(agent.memory)
            if not filename or not transcript:
                print("Usage: /save filename.md (conversation must not be empty)\n")
            else:
                try:
                    print(f"Saved to {write_transcript(filename, transcript)}\n")
                except OSError as error:
                    print(f"Could not save transcript: {error}\n", file=sys.stderr)
            continue
        if text == "/context reload":
            context.reload()
            compose_instructions(agent, context, source_context, quiz)
            print("Persistent context reloaded.\n")
            continue
        if text in {"/source on", "/source off", "/source status", "/source reload"}:
            if text == "/source on":
                source_context.set_enabled(True)
                message = ("TinyAgent source code will be included in model requests." if source_context.source
                           else source_context.error)
            elif text == "/source off":
                source_context.set_enabled(False)
                message = "TinyAgent source code is no longer included in model requests."
            elif text == "/source reload":
                source_context.reload()
                message = "TinyAgent source code reloaded."
            else:
                message = "TinyAgent source context is " + ("on." if source_context.enabled else "off.")
            compose_instructions(agent, context, source_context, quiz)
            print(message, "\n")
            continue
        if text == "/quiz end":
            print(quiz.end(agent), "\n")
            continue
        if text == "/quiz" or text.startswith("/quiz "):
            try:
                first_question = quiz.start(agent, text.partition(" ")[2])
                agent.memory.add("user", text)
                agent.memory.add("assistant", first_question)
                print(first_question, "\n")
            except (ValueError, OSError, RuntimeError) as error:
                print(f"Could not start quiz: {error}\n", file=sys.stderr)
            continue

        if execution_mode == "autonomous":
            def run_autonomously(task):
                compose_instructions(agent, context, source_context, quiz)
                history_length = len(agent.memory.messages)
                try:
                    result = agent.run(task, mode="autonomous", max_steps=max_steps)
                    if result.tool_call:
                        console.print("[bold]Final action observation[/bold]")
                        console.print(result.content)
                    elif result.content:
                        console.print("TinyAgent:")
                        console.print(Markdown(result.content))
                    console.print(f"Run outcome: {agent.active_run.outcome}; {len(agent.active_run.steps)}/{max_steps} model steps used.\n")
                except Exception as error:
                    del agent.memory.messages[history_length:]
                    print(f"Run outcome: error ({type(error).__name__}: {error})\n", file=sys.stderr)
            active_worker = threading.Thread(target=run_autonomously, args=(text,), daemon=True)
            active_worker.start()
            print("Autonomous run started. Enter /stop to request a cooperative stop.\n")
            continue

        compose_instructions(agent, context, source_context, quiz)
        history_length = len(agent.memory.messages)
        try:
            if quiz.active:
                answer = quiz.answer(agent, text)
            else:
                result = agent.run(text, mode=execution_mode, max_steps=max_steps)
                answer = result.content
        except Exception as error:
            del agent.memory.messages[history_length:]
            if isinstance(error, AuthenticationError):
                message = "Authentication failed. Check OPENAI_API_KEY."
            elif isinstance(error, RateLimitError):
                message = "Request limited. Check your API quota, billing, and rate limits."
            elif isinstance(error, APIConnectionError):
                message = "Could not connect to OpenAI. Check your network connection."
            elif isinstance(error, APIStatusError):
                message = f"OpenAI returned HTTP {error.status_code}. Check model and account access."
            elif isinstance(error, (APIError, RuntimeError, ValueError)):
                message = f"Request failed: {error}"
            else:
                message = "Unexpected request error; previous conversation was kept."
            print(f"{message}\n", file=sys.stderr)
            continue
        if result.tool_call:
            console.print("[bold]Action[/bold]")
            console.print(f"Tool: {result.tool_call.get('tool')}\nArguments: {result.tool_call.get('arguments')}")
            console.print("[bold]Observation[/bold]")
            console.print(answer)
            if execution_mode == "manual":
                console.print("Use /continue to request exactly one next model step.")
            else:
                console.print(f"Run outcome: {agent.active_run.outcome}; {len(agent.active_run.steps)}/{max_steps} model steps used.")
        else:
            console.print("TinyAgent:")
            console.print(Markdown(answer))
            if execution_mode == "autonomous":
                console.print(f"Run outcome: {agent.active_run.outcome}; {len(agent.active_run.steps)}/{max_steps} model steps used.")
        console.print()


def main() -> int:
    llm = None
    try:
        context = PersistentContext()
        source_context = SourceCodeContext()
        llm = LLM()
        rag = RAGMemory(embedding_model=EmbeddingModel(llm.client))
        agent = TinyAgent(llm, Memory(), [context.system_message], NativeTools(), rag)
        chat(agent, context, QuizManager(), source_context)
    except (ValueError, KeyboardInterrupt, EOFError) as error:
        if isinstance(error, ValueError):
            print(f"Configuration error: {error}", file=sys.stderr)
    finally:
        if llm is not None:
            llm.close()
    print("Goodbye!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
