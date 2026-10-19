"""Markdown transcript formatting shared by the terminal and browser."""

from tinyagent import Memory


def format_transcript(memory: Memory) -> str:
    messages = [
        message for message in memory.get_messages()
        if message["role"] in ("user", "assistant")
    ]
    if not messages:
        return ""
    sections = ["# TinyAgent conversation"]
    for message in messages:
        speaker = "You" if message["role"] == "user" else "TinyAgent"
        sections.append(f"## {speaker}\n\n{message['content']}")
    return "\n\n".join(sections) + "\n"
