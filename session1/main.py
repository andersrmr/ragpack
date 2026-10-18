"""Minimal terminal chat using the Session 1 TinyAgent classes."""

import sys

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    AuthenticationError,
    RateLimitError,
)
from rich.console import Console
from rich.markdown import Markdown
from tinyagent import LLM, Memory, TinyAgent


def chat(agent: TinyAgent) -> None:
    """Read one line at a time and keep completed turns in the agent's memory."""
    console = Console()
    print(f"TinyAgent | Model: {agent.llm.model}")
    print("Ask a question. /clear resets history; /quit exits.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question == "/quit":
            return
        if question == "/clear":
            agent.memory.messages.clear()
            print("Conversation cleared.\n")
            continue

        # TinyAgent adds the user turn before making the request. On failure,
        # remove that unfinished turn while retaining the completed conversation.
        history_length = len(agent.memory.get_messages())
        try:
            answer = agent.run(question)
        except (APIError, RuntimeError) as error:
            del agent.memory.messages[history_length:]
            if isinstance(error, AuthenticationError):
                message = "Authentication failed. Check OPENAI_API_KEY."
            elif isinstance(error, RateLimitError):
                message = (
                    "Request limited. Check your API quota, billing, and rate limits."
                )
            elif isinstance(error, APIConnectionError):
                message = "Could not connect to OpenAI. Check your network connection."
            elif isinstance(error, APIStatusError):
                message = f"OpenAI returned HTTP {error.status_code}. Check your model and account access."
            else:
                message = "The request failed or returned no usable text."
            print(
                f"{message} Previous conversation kept; you can try again.\n",
                file=sys.stderr,
            )
            continue

        print("TinyAgent:")
        console.print(Markdown(answer))
        console.print()


def main() -> int:
    llm = None
    try:
        llm = LLM()
        chat(TinyAgent(llm=llm, memory=Memory()))
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        # Ctrl+C (including during an API request) and Ctrl+D exit cleanly.
        print()
    finally:
        if llm is not None:
            llm.close()
    print("Goodbye!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
