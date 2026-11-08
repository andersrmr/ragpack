"""Terminal chat for Session 3 TinyAgent."""

import sys

from openai import APIConnectionError, APIError, APIStatusError, AuthenticationError, RateLimitError
from context import PersistentContext, SourceCodeContext, compose_instructions
from quiz import QuizManager
from rich.console import Console
from rich.markdown import Markdown
from tinyagent import LLM, Memory, NativeTools, TinyAgent
from transcripts import format_transcript, write_transcript


def chat(agent, context, quiz, source_context):
    console = Console()
    print(f"TinyAgent Session 3 | Model: {agent.llm.model}")
    print("Commands: /continue, /clear, /save filename.md, /quiz [topic], /quiz end, /context reload, ")
    print("/source on|off|status|reload, /quit\n")
    while True:
        text = input("You: ").strip()
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/clear":
            agent.memory.messages.clear()
            agent.pending_continue = False
            agent.active_run = None
            print("Conversation cleared; persistent context and trajectory retained.\n")
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

        compose_instructions(agent, context, source_context, quiz)
        history_length = len(agent.memory.messages)
        try:
            if quiz.active:
                answer = quiz.answer(agent, text)
            else:
                result = agent.run(text)
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
            console.print("Use /continue to request exactly one next model step.")
        else:
            console.print("TinyAgent:")
            console.print(Markdown(answer))
        console.print()


def main() -> int:
    llm = None
    try:
        context = PersistentContext()
        source_context = SourceCodeContext()
        llm = LLM()
        agent = TinyAgent(llm, Memory(), [context.system_message], NativeTools())
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
