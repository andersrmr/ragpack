"""Load persistent instruction sources independently of conversation memory."""

from pathlib import Path

from prompts import TUTOR_SYSTEM_PROMPT


class PersistentContext:
    def __init__(self, about_me_path: str | Path | None = None):
        self.about_me_path = Path(about_me_path or Path(__file__).with_name("about_me.md"))
        self.about_me = ""
        self.reload()

    def reload(self) -> dict:
        try:
            self.about_me = self.about_me_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            self.about_me = ""
        return self.sources

    @property
    def sources(self) -> dict[str, str]:
        return {"Persistent tutor instructions": TUTOR_SYSTEM_PROMPT,
                "Personal context (about_me.md)": self.about_me}

    @property
    def system_message(self) -> str:
        sections = [f"## {name}\n\n{text}" for name, text in self.sources.items() if text]
        return "\n\n".join(sections)
