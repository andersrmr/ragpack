"""Markdown transcript formatting and collision-safe export."""

from datetime import datetime
from itertools import count
from pathlib import Path


def format_transcript(memory) -> str:
    source = memory.transcript_messages() if hasattr(memory, "transcript_messages") else memory.get_messages()
    messages = [m for m in source if m.get("role") in {"user", "assistant"}]
    if not messages:
        return ""
    blocks = ["# TinyAgent conversation"]
    for message in messages:
        speaker = "You" if message["role"] == "user" else "TinyAgent"
        blocks.append(f"## {speaker}\n\n{message['content']}")
    return "\n\n".join(blocks) + "\n"


def write_transcript(filename: str | Path, transcript: str) -> Path:
    requested = Path(filename).expanduser()
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    stem = f"{requested.stem}-{stamp}"
    for number in count(1):
        suffix = "" if number == 1 else f"-{number}"
        path = requested.with_name(f"{stem}{suffix}{requested.suffix}")
        try:
            with path.open("x", encoding="utf-8") as output:
                output.write(transcript)
        except FileExistsError:
            continue
        return path
