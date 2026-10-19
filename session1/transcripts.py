"""Markdown transcript formatting and saving shared by both interfaces."""

from datetime import UTC, datetime
from itertools import count
from pathlib import Path
from zoneinfo import ZoneInfo

from tinyagent import Memory


def write_transcript(filename: str | Path, transcript: str) -> Path:
    """Save using New York time, adding a counter for same-second collisions."""
    path = Path(filename).expanduser()
    saved_at = datetime.now(UTC).astimezone(ZoneInfo("America/New_York"))
    stem = f"{path.stem}-{saved_at:%Y-%m-%d_%H-%M-%S}"
    for number in count(1):
        extra = "" if number == 1 else f"-{number}"
        destination = path.with_name(f"{stem}{extra}{path.suffix}")
        try:
            # Exclusive creation also protects against concurrent saves.
            with destination.open("x", encoding="utf-8") as file:
                file.write(transcript)
        except FileExistsError:
            continue
        return destination


def format_transcript(memory: Memory) -> str:
    messages = [
        message
        for message in memory.get_messages()
        if message["role"] in ("user", "assistant")
    ]
    if not messages:
        return ""
    sections = ["# TinyAgent conversation"]
    for message in messages:
        speaker = "You" if message["role"] == "user" else "TinyAgent"
        sections.append(f"## {speaker}\n\n{message['content']}")
    return "\n\n".join(sections) + "\n"
