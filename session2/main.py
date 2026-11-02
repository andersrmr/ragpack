"""Terminal chat for Session 2 TinyAgent."""

import sys

from openai import APIConnectionError, APIError, APIStatusError, AuthenticationError, RateLimitError
from context import PersistentContext
from quiz import QuizManager
from rich.console import Console
from rich.markdown import Markdown
from tinyagent import LLM, Memory, TinyAgent
from transcripts import format_transcript, write_transcript


def chat(agent, context, quiz):
    console = Console()
    print(f"TinyAgent Session 2 | Model: {agent.llm.model}")
    print("Commands: /clear, /save filename.md, /quiz [topic], /quiz end, /context reload, /quit\n")
    while True:
        text = input("You: ").strip()
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/clear":
            agent.memory.messages.clear()
            print("Conversation cleared; persistent context and trajectory retained.\n")
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
            agent.system_instructions[0] = context.system_message
            print("Persistent context reloaded.\n")
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

        instructions = [context.system_message]
        if quiz.active and quiz.skill_text not in agent.system_instructions:
            instructions.append(quiz.skill_text)
        agent.system_instructions = instructions
        history_length = len(agent.memory.messages)
        try:
            if quiz.active:
                answer = quiz.answer(agent, text)
            else:
                answer = agent.run(text).content
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
        console.print("TinyAgent:")
        console.print(Markdown(answer))
        console.print()


def main() -> int:
    llm = None
    try:
        context = PersistentContext()
        llm = LLM()
        agent = TinyAgent(llm, Memory(), [context.system_message])
        chat(agent, context, QuizManager())
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
