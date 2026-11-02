"""Gradio chat and observability views for Session 2 TinyAgent."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock

import gradio as gr
from openai import APIConnectionError, APIError, APIStatusError, AuthenticationError, RateLimitError
from context import PersistentContext
from quiz import QuizManager
from tinyagent import LLM, Memory, TinyAgent
from transcripts import format_transcript, write_transcript


class ChatSession:
    def __init__(self):
        self.context = PersistentContext()
        self.memory = Memory()
        self.quiz = QuizManager()
        self.llm = None
        self.agent = None
        self.ended = False
        self.downloads = None
        self.lock = RLock()

    def ensure_agent(self):
        if self.agent is None:
            self.llm = LLM()
            self.agent = TinyAgent(self.llm, self.memory, [self.context.system_message])

    def history(self):
        return [{"role": m["role"], "content": m["content"]} for m in self.memory.get_messages()]

    def snapshot(self):
        trajectory = self.agent.trajectory.to_dict() if self.agent else {"schema_version": 1, "runs": []}
        latest = trajectory["runs"][-1]["steps"][-1] if trajectory["runs"] and trajectory["runs"][-1]["steps"] else None
        return trajectory, {
            "persistent_sources": self.context.sources,
            "active_skill": self.quiz.skill_text if self.quiz.active else "",
            "conversation_memory": self.memory.get_messages(),
            "effective_messages_for_latest_run": latest["input_messages"] if latest else [],
            "latest_run_metadata": (trajectory["runs"][-1] if trajectory["runs"] else None),
        }

    def status(self):
        model = self.llm.model if self.llm else (os.environ.get("TINYAGENT_MODEL", "not configured"))
        mode = f"Quiz active: {self.quiz.quiz['topic']}" if self.quiz.active else "Ordinary tutoring"
        return f"Model: {model} | {mode}"

    def send(self, text):
        with self.lock:
            if self.ended:
                return text, "Session ended. Refresh to start a new session."
            text = (text or "").strip()
            if not text:
                return text, "Enter a message."
            try:
                self.ensure_agent()
                history_length = len(self.memory.messages)
                if text == "/quiz end":
                    answer = self.quiz.end(self.agent)
                elif text == "/quiz" or text.startswith("/quiz "):
                    answer = self.quiz.start(self.agent, text.partition(" ")[2])
                elif text == "/context reload":
                    self.context.reload()
                    self.agent.system_instructions[0] = self.context.system_message
                    answer = "Persistent context reloaded from `about_me.md`."
                elif text == "/clear":
                    self.memory.messages.clear()
                    answer = "Conversation cleared. Persistent context and trajectory retained."
                elif self.quiz.active:
                    answer = self.quiz.answer(self.agent, text)
                else:
                    answer = self.agent.run(text).content
            except Exception as error:
                if self.agent is not None:
                    del self.memory.messages[history_length:]
                if isinstance(error, AuthenticationError):
                    detail = "Authentication failed. Check OPENAI_API_KEY."
                elif isinstance(error, RateLimitError):
                    detail = "Request limited. Check quota, billing, and rate limits."
                elif isinstance(error, APIConnectionError):
                    detail = "Could not connect to OpenAI. Check your network."
                elif isinstance(error, APIStatusError):
                    detail = f"OpenAI returned HTTP {error.status_code}. Check model and account access."
                elif isinstance(error, (APIError, RuntimeError, ValueError)):
                    detail = str(error)
                else:
                    detail = "Unexpected request error."
                return text, f"{detail} Previous conversation kept; you can retry."
            if text == "/quiz" or text.startswith("/quiz "):
                self.memory.add("user", text)
                self.memory.add("assistant", answer)
            return "", answer

    def save(self):
        with self.lock:
            transcript = format_transcript(self.memory)
            if not transcript:
                return None, "No conversation to save."
            try:
                if self.downloads is None:
                    self.downloads = TemporaryDirectory(prefix="tinyagent-session2-")
                path = write_transcript(Path(self.downloads.name) / "conversation.md", transcript)
                return str(path), "Transcript ready to download."
            except OSError as error:
                return None, f"Could not save transcript: {error}"

    def dispose(self):
        with self.lock:
            if self.llm:
                self.llm.close()
                self.llm = None
            if self.downloads:
                self.downloads.cleanup()
                self.downloads = None
            self.ended = True


class Sessions:
    def __init__(self):
        self.items = {}
        self.lock = RLock()

    def get(self, key):
        with self.lock:
            return self.items.setdefault(key, ChatSession())

    def remove(self, key):
        with self.lock:
            session = self.items.pop(key, None)
        if session:
            session.dispose()

    def close(self):
        for key in list(self.items):
            self.remove(key)


def build_app():
    sessions = Sessions()

    def current(request: gr.Request):
        return sessions.get(request.session_hash)

    def update(text, request: gr.Request):
        session = current(request)
        remaining, answer = session.send(text)
        trajectory, context = session.snapshot()
        return remaining, session.history(), session.status(), trajectory, context

    def activate_quiz(topic, request: gr.Request):
        return update("/quiz " + (topic or ""), request)

    def deactivate_quiz(request: gr.Request):
        return update("/quiz end", request)

    def clear(request: gr.Request):
        session = current(request)
        session.memory.messages.clear()
        trajectory, context = session.snapshot()
        return [], "Conversation cleared. Persistent context and trajectory retained.", trajectory, context

    def reload_context(request: gr.Request):
        session = current(request)
        session.context.reload()
        if session.agent:
            session.agent.system_instructions[0] = session.context.system_message
        trajectory, context = session.snapshot()
        return "Persistent context reloaded from `about_me.md`.", context, session.status()

    def clear_trajectory(request: gr.Request):
        session = current(request)
        if session.agent:
            session.agent.trajectory.clear()
        trajectory, _ = session.snapshot()
        return trajectory

    def save(request: gr.Request):
        session = current(request)
        path, status = session.save()
        return path, status

    def end(request: gr.Request):
        session = current(request)
        session.dispose()
        return "Session ended. Refresh to start a new session."

    def unload(request: gr.Request):
        sessions.remove(request.session_hash)

    with gr.Blocks(title="TinyAgent Session 2", analytics_enabled=False) as app:
        state = gr.State(None, delete_callback=sessions.remove)
        gr.Markdown("# TinyAgent course tutor · Session 2\nPersistent context, manual quiz skills, and execution observability.")
        status = gr.Textbox(label="Status", value=f"Model: {os.environ.get('TINYAGENT_MODEL', 'not configured')}", interactive=False)
        with gr.Tabs():
            with gr.Tab("Chat"):
                conversation = gr.Chatbot(label="Conversation", height=440)
                question = gr.Textbox(label="Message or command", lines=3, placeholder="Ask a question, /quiz [topic], or /quiz end")
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear_button = gr.Button("Clear conversation")
                    save_button = gr.Button("Save transcript")
                    end_button = gr.Button("End session")
                    reload_button = gr.Button("Reload about_me.md")
                with gr.Row():
                    quiz_topic = gr.Textbox(label="Quiz topic", placeholder="Optional; defaults to available course material")
                    quiz_start = gr.Button("Start quiz")
                    quiz_end = gr.Button("End quiz")
                download = gr.File(label="Transcript download", interactive=False)
                feedback = gr.Markdown()
            with gr.Tab("Trajectory"):
                gr.Markdown("Runs contain one or more Steps. Each step shows request messages, model, reasoning summary when returned, answer, usage metadata, latency, and errors.")
                trajectory_view = gr.JSON(label="Run → Step → Thought / Action / Observation / Answer / Metadata")
                trajectory_clear = gr.Button("Clear trajectory")
            with gr.Tab("Context"):
                gr.Markdown("Persistent instructions and active skill are separate from Memory. Effective messages show the last recorded model request.")
                context_view = gr.JSON(label="Context sources and latest effective request")

        outputs = [question, conversation, status, trajectory_view, context_view]
        options = {"concurrency_id": "session2-actions", "concurrency_limit": 1}
        def load(request: gr.Request):
            session = current(request)
            trajectory, context = session.snapshot()
            return session.status(), trajectory, context

        app.load(load, outputs=[status, trajectory_view, context_view], **options)
        trajectory_clear.click(clear_trajectory, outputs=[trajectory_view], **options)
        send.click(update, inputs=[question], outputs=outputs, **options)
        question.submit(update, inputs=[question], outputs=outputs, **options)
        quiz_start.click(activate_quiz, inputs=[quiz_topic], outputs=outputs, **options)
        quiz_end.click(deactivate_quiz, outputs=outputs, **options)
        clear_button.click(clear, outputs=[conversation, feedback, trajectory_view, context_view], **options)
        reload_button.click(reload_context, outputs=[feedback, context_view, status], **options)
        save_button.click(save, outputs=[download, feedback], **options)
        end_button.click(end, outputs=[feedback], **options)
        app.unload(unload)
    return app, sessions


def main():
    app, sessions = build_app()
    try:
        app.launch(server_name="127.0.0.1", share=False)
    finally:
        sessions.close()
        app.close()


if __name__ == "__main__":
    main()
