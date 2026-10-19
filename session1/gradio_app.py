"""Browser chat for the Session 1 tutor. Run with python gradio_app.py."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from uuid import uuid4

import gradio as gr
from openai import APIConnectionError, APIError, AuthenticationError, RateLimitError
from prompts import TUTOR_SYSTEM_PROMPT
from tinyagent import LLM, Memory, TinyAgent
from transcripts import format_transcript


class ChatSession:
    """One tab's conversation, client, and temporary transcript downloads."""

    def __init__(self):
        self.memory = Memory()
        self.llm = None
        self.agent = None
        self.ended = False
        self.downloads = None
        self.lock = RLock()

    def history(self) -> list[dict]:
        return [message.copy() for message in self.memory.get_messages()]

    def send(self, question: str) -> tuple[str, str]:
        """Return remaining input and a status message; preserve input on failure."""
        with self.lock:
            if self.ended:
                return question, "Session ended. Refresh the page to start a new chat."
            if not question.strip():
                return question, "Enter a question."
            length = len(self.memory.messages)
            try:
                if self.llm is None:
                    self.llm = LLM()
                    self.agent = TinyAgent(self.llm, self.memory, TUTOR_SYSTEM_PROMPT)
                self.agent.run(question.strip())
            except (APIError, RuntimeError, ValueError) as error:
                del self.memory.messages[length:]
                if isinstance(error, AuthenticationError):
                    detail = "Authentication failed. Check OPENAI_API_KEY."
                elif isinstance(error, RateLimitError):
                    detail = "Request limited. Check quota, billing, and rate limits."
                elif isinstance(error, APIConnectionError):
                    detail = "Could not connect to OpenAI. Check your network."
                elif isinstance(error, ValueError):
                    detail = str(error)
                else:
                    detail = "Request failed. Check your model and account access."
                return question, f"{detail} Previous conversation kept; you can retry."
            return "", f"Model: {self.llm.model} | Session 1 tutor"

    def clear(self) -> str:
        with self.lock:
            self.memory.messages.clear()
            return "Conversation cleared. Tutor instructions retained."

    def save(self) -> tuple[str | None, str]:
        with self.lock:
            transcript = format_transcript(self.memory)
            if not transcript:
                return None, "No conversation to save."
            try:
                if self.downloads is None:
                    self.downloads = TemporaryDirectory(prefix="tinyagent-")
                path = Path(self.downloads.name) / f"conversation-{uuid4().hex}.md"
                path.write_text(transcript, encoding="utf-8")
            except (OSError, UnicodeError):
                return None, "Could not prepare the transcript. Conversation kept; try again."
            return str(path), "Transcript ready. Click the file below to download it."

    def end(self) -> str:
        with self.lock:
            self.ended = True
            if self.llm is not None:
                self.llm.close()
                self.llm = None
                self.agent = None
            return "Session ended. You can still save the transcript. Refresh to start again."

    def dispose(self) -> None:
        """Release resources on tab disconnect, state deletion, or server exit."""
        with self.lock:
            try:
                self.end()
            finally:
                if self.downloads is not None:
                    self.downloads.cleanup()
                    self.downloads = None


class Sessions:
    """Keep live clients server-side, separate from Gradio's copied state values."""

    def __init__(self):
        self.items = {}
        self.lock = RLock()

    def get(self, key: str) -> ChatSession:
        with self.lock:
            if key not in self.items:
                self.items[key] = ChatSession()
            return self.items[key]

    def remove(self, key: str | None) -> None:
        with self.lock:
            session = self.items.pop(key, None)
        if session is not None:
            session.dispose()

    def close(self) -> None:
        for key in list(self.items):
            self.remove(key)


def build_app() -> tuple[gr.Blocks, Sessions]:
    sessions = Sessions()
    model = os.environ.get("TINYAGENT_MODEL", "").strip() or "not configured"

    def load(request: gr.Request):
        sessions.get(request.session_hash)
        return f"Model: {model} | Session 1 tutor", request.session_hash

    def respond(question, request: gr.Request):
        session = sessions.get(request.session_hash)
        remaining, status = session.send(question)
        return remaining, session.history(), status, None, request.session_hash

    def clear(request: gr.Request):
        session = sessions.get(request.session_hash)
        status = session.clear()
        return "", [], status, None, request.session_hash

    def save(request: gr.Request):
        session = sessions.get(request.session_hash)
        path, status = session.save()
        return path, status, request.session_hash

    def end(request: gr.Request):
        status = sessions.get(request.session_hash).end()
        return status, gr.Textbox(interactive=False), gr.Button(interactive=False)

    def unload(request: gr.Request):
        sessions.remove(request.session_hash)

    with gr.Blocks(
        title="TinyAgent course tutor", analytics_enabled=False,
        delete_cache=(3600, 3600),
    ) as app:
        # State holds only the session key, never an HTTP client or a lock.
        state = gr.State(value=None, delete_callback=sessions.remove)
        gr.Markdown("# TinyAgent course tutor\nAsk questions about Session 1.")
        status = gr.Textbox(value=f"Model: {model}", label="Status", interactive=False)
        conversation = gr.Chatbot(label="Conversation", height=450)
        question = gr.Textbox(label="Your question", lines=3, placeholder="Ask about TinyAgent…")
        with gr.Row():
            send_button = gr.Button("Send", variant="primary")
            clear_button = gr.Button("Clear conversation")
            save_button = gr.Button("Save transcript")
            end_button = gr.Button("End session")
        download = gr.File(label="Markdown transcript", interactive=False)
        gr.Markdown("Save before clearing or leaving. End session affects only this tab. "
                    "Refresh to start a new session.")

        # A shared queue group prevents Send/Clear/Save/End from racing.
        event_options = {"concurrency_id": "chat-actions", "concurrency_limit": 1}
        outputs = [question, conversation, status, download, state]
        app.load(load, outputs=[status, state], **event_options)
        send_button.click(respond, inputs=[question], outputs=outputs, **event_options)
        question.submit(respond, inputs=[question], outputs=outputs, **event_options)
        clear_button.click(clear, outputs=outputs, **event_options)
        save_button.click(save, outputs=[download, status, state], **event_options)
        end_button.click(end, outputs=[status, question, send_button], **event_options)
        app.unload(unload)
    return app, sessions


def main() -> None:
    app, sessions = build_app()
    try:
        app.launch(server_name="127.0.0.1", share=False)
    finally:
        try:
            sessions.close()
        finally:
            app.close()


if __name__ == "__main__":
    main()
