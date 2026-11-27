"""Gradio chat and observability views for Session 4 TinyAgent."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock

import gradio as gr
from openai import APIConnectionError, APIError, APIStatusError, AuthenticationError, RateLimitError
from context import PersistentContext, SourceCodeContext, compose_instructions
from planning import PLANNING_STRATEGIES, REACT
from quiz import QuizManager
from tinyagent import LLM, Memory, NativeTools, TinyAgent, message_text
from retrieval import EmbeddingModel, RAGMemory
from transcripts import format_transcript, write_transcript


class ChatSession:
    def __init__(self):
        self.context = PersistentContext()
        self.source_context = SourceCodeContext()
        self.memory = Memory()
        self.quiz = QuizManager()
        self.llm = None
        self.agent = None
        self.ended = False
        self.downloads = None
        self.lock = RLock()
        self.retrieval_mode = "Off"
        self.rag_memory = None
        self.execution_mode = "manual"
        self.planning_strategy = REACT
        self.max_steps = 10

    def ensure_agent(self):
        if self.agent is None:
            self.llm = LLM()
            self.rag_memory = RAGMemory(embedding_model=EmbeddingModel(self.llm.client))
            self.agent = TinyAgent(self.llm, self.memory, [self.context.system_message], NativeTools(), self.rag_memory)
            self.agent.set_retrieval_mode(self.retrieval_mode)
            self.agent.set_planning_strategy(self.planning_strategy)

    def history(self):
        displayed, actions = [], {}
        for item in self.memory.get_messages():
            if item.get("role") in {"user", "assistant"}:
                displayed.append({"role": item["role"], "content": message_text(item.get("content", ""))})
            elif item.get("type") == "function_call":
                try:
                    args = json.dumps(json.loads(item.get("arguments", "{}")), ensure_ascii=False)
                except (TypeError, json.JSONDecodeError):
                    args = item.get("arguments", "")
                actions[item.get("call_id")] = len(displayed)
                displayed.append({"role": "assistant", "content":
                                  f"**Action**\n\nTool: `{item.get('name')}`\n\n"
                                  f"Arguments requested: `{args}`\n\n**Observation**\n\nWaiting for harness result."})
            elif item.get("type") == "function_call_output":
                index = actions.get(item.get("call_id"))
                if index is not None:
                    displayed[index]["content"] = displayed[index]["content"].replace(
                        "Waiting for harness result.", item.get("output", ""))
        return displayed

    def snapshot(self):
        trajectory = self.agent.trajectory.to_dict() if self.agent else {"schema_version": 1, "runs": []}
        latest = trajectory["runs"][-1]["steps"][-1] if trajectory["runs"] and trajectory["runs"][-1]["steps"] else None
        latest_run = trajectory["runs"][-1] if trajectory["runs"] else None
        latest_turn_items = []
        for index in range(len(self.memory.messages) - 1, -1, -1):
            if self.memory.messages[index].get("role") == "user":
                latest_turn_items = self.memory.messages[index:]
                break
        retrieval_calls = {item.get("call_id") for item in latest_turn_items
                           if item.get("type") == "function_call" and item.get("name") == "search_documents"}
        return trajectory, {
            "persistent_sources": self.context.sources,
            "source_code_context": self.source_context.view,
            "active_skill": self.quiz.skill_text if self.quiz.active else "",
            "conversation_memory": self.memory.display_messages(),
            "effective_messages_for_latest_run": latest["input_messages"] if latest else [],
            "latest_run_metadata": (trajectory["runs"][-1] if trajectory["runs"] else None),
            "retrieval": {"selected_mode": self.retrieval_mode,
                          "top_k": self.rag_memory.top_k if self.rag_memory else 3,
                          "similarity_threshold": self.rag_memory.threshold if self.rag_memory else 0.30,
                          "corpus_load_errors": self.rag_memory.corpus_errors if self.rag_memory else [],
                          "latest_automatic_result": (latest_run or {}).get("metadata", {}).get("automatic_retrieval"),
                          "temporary_context_added": bool(latest_run and latest_run.get("metadata", {}).get("automatic_retrieval")),
                          "latest_summary": self.memory.summary,
                          "retained_turn_boundary": self.memory.summary_retained_turns,
                          "model_directed_tool_schema": next((s for s in self.agent.tools.schemas if s["name"] == "search_documents"), None) if self.agent and self.retrieval_mode == "Model-directed retrieval" else None,
                          "latest_model_directed_actions": [step for step in ((latest_run or {}).get("steps", [])) if (step.get("action") or {}).get("tool") == "search_documents"],
                          "latest_model_directed_items": [item for item in latest_turn_items if item.get("call_id") in retrieval_calls]},
        }

    def status(self):
        model = self.llm.model if self.llm else (os.environ.get("TINYAGENT_MODEL", "not configured"))
        mode = f"Quiz active: {self.quiz.quiz['topic']}" if self.quiz.active else "Ordinary tutoring"
        mode += f" | Retrieval: {self.retrieval_mode}"
        mode += f" | Planning: {self.planning_strategy}"
        mode += f" | Execution: {self.execution_mode} | Run: {self.agent.active_run.outcome if self.agent and self.agent.active_run else 'idle'}"
        if self.source_context.enabled:
            mode += " | TinyAgent source included" if self.source_context.source else " | TinyAgent source unavailable"
        return f"Model: {model} | {mode}"

    def plan_view(self):
        if self.agent:
            return self.agent.plan_status()
        return f"Planning strategy: {self.planning_strategy}\n\nNo plan exists for this run."

    def set_planning_strategy(self, strategy):
        with self.lock:
            if strategy not in PLANNING_STRATEGIES:
                raise ValueError("Unknown planning strategy.")
            if self.agent:
                self.agent.set_planning_strategy(strategy)
            self.planning_strategy = strategy

    def clear_conversation(self):
        with self.lock:
            if self.agent:
                self.agent.clear_conversation()
            else:
                self.memory.clear()

    def set_source_enabled(self, enabled):
        with self.lock:
            self.source_context.set_enabled(enabled)
            if self.agent:
                compose_instructions(self.agent, self.context, self.source_context, self.quiz)

    def set_retrieval_mode(self, mode):
        with self.lock:
            if mode not in {"Off", "Automatic RAG", "Model-directed retrieval"}:
                raise ValueError("Unknown retrieval mode.")
            self.retrieval_mode = mode
            if self.agent:
                self.agent.set_retrieval_mode(mode)

    def reload_source(self):
        with self.lock:
            self.source_context.reload()
            if self.agent:
                compose_instructions(self.agent, self.context, self.source_context, self.quiz)

    def send(self, text, execution_mode=None, max_steps=None, planning_strategy=None):
        with self.lock:
            if self.ended:
                return text, "Session ended. Refresh to start a new session."
            text = (text or "").strip()
            if not text:
                return text, "Enter a message."
            if text == "/clear":
                self.clear_conversation()
                return "", "Conversation and active plan cleared. Persistent context and trajectory retained."
            history_length = len(self.memory.messages)
            try:
                self.ensure_agent()
                compose_instructions(self.agent, self.context, self.source_context, self.quiz)
                if text == "/continue":
                    result = self.agent.continue_step()
                    answer = result.content
                elif text == "/quiz end":
                    answer = self.quiz.end(self.agent)
                elif text == "/quiz" or text.startswith("/quiz "):
                    answer = self.quiz.start(self.agent, text.partition(" ")[2])
                elif text == "/context reload":
                    self.context.reload()
                    compose_instructions(self.agent, self.context, self.source_context, self.quiz)
                    answer = "Persistent context reloaded from `about_me.md`."
                elif text == "/source on":
                    self.set_source_enabled(True)
                    answer = ("TinyAgent source code will be included in model requests." if self.source_context.source
                              else self.source_context.error)
                elif text == "/source off":
                    self.set_source_enabled(False)
                    answer = "TinyAgent source code is no longer included in model requests."
                elif text == "/source reload":
                    self.reload_source()
                    answer = "TinyAgent source code reloaded."
                elif text == "/source status":
                    answer = "TinyAgent source context is " + ("on." if self.source_context.enabled else "off.")
                elif text == "/compact":
                    result = self.agent.compact_memory(2)
                    answer = f"Memory compacted; retained 2 complete recent turns.\n\n{result['summary']}"
                elif self.quiz.active:
                    answer = self.quiz.answer(self.agent, text)
                else:
                    if self.agent.pending_continue:
                        raise RuntimeError("Continue, stop, or clear the paused run before submitting a new task.")
                    if planning_strategy is not None:
                        self.set_planning_strategy(planning_strategy)
                    if execution_mode in {"Manual Continue", "Autonomous"}:
                        self.execution_mode = "autonomous" if execution_mode == "Autonomous" else "manual"
                    if max_steps is not None:
                        self.max_steps = max(1, int(max_steps))
                    result = self.agent.run(text, mode=self.execution_mode, max_steps=self.max_steps)
                    if self.execution_mode == "autonomous":
                        run = self.agent.active_run
                        answer = f"**Run outcome: {run.outcome}** · {len(run.steps)} / {run.max_steps} model steps."
                    elif self.agent.pending_continue:
                        answer = "Use Continue for one model step."
                    else:
                        answer = result.content or f"Run outcome: {self.agent.active_run.outcome}."
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
            if text == "/quiz" or text.startswith("/quiz ") or text.startswith("/source "):
                self.memory.add("user", text)
                self.memory.add("assistant", answer)
            return "", answer

    def stop(self):
        if self.agent and self.agent.request_stop():
            if not self.agent.run_active and self.agent.active_run and self.agent.active_run.outcome == "stopped":
                return "Run stopped."
            return "Stop requested. Any in-flight request or tool will finish before the stop is honored."
        return "No run is active or paused."

    def save(self):
        with self.lock:
            transcript = format_transcript(self.memory)
            if not transcript:
                return None, "No conversation to save."
            try:
                if self.downloads is None:
                    self.downloads = TemporaryDirectory(prefix="tinyagent-session4-")
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

    def control_state(session):
        paused = bool(session.agent and session.agent.pending_continue)
        running = bool(session.agent and session.agent.run_active)
        return (gr.update(interactive=paused and not running),
                gr.update(value=session.planning_strategy, interactive=not paused and not running),
                session.plan_view())

    def update_state(session, answer):
        trajectory, context = session.snapshot()
        history = session.history()
        feedback = "" if history and history[-1]["content"] == answer else answer
        return (history, feedback, session.status(), trajectory, context,
                session.source_context.enabled, *control_state(session))

    def update(text, mode, limit, strategy, request: gr.Request):
        session = current(request)
        prior_length = len(session.memory.messages)
        remaining, answer = session.send(text, mode, limit, strategy)
        if session.quiz.active and text and not text.lstrip().startswith("/") and len(session.memory.messages) > prior_length:
            answer = ""  # Quiz feedback and the next question are already in Chat.
        return (remaining, *update_state(session, answer))

    def set_planning(strategy, request: gr.Request):
        session = current(request)
        try:
            session.set_planning_strategy(strategy)
            message = ""
        except (RuntimeError, ValueError) as error:
            message = str(error)
        return message, session.status(), *control_state(session)

    def stop_action(request: gr.Request):
        session = current(request)
        message = session.stop()
        trajectory = (session.agent.trajectory.to_dict()
                      if session.agent and not session.agent.run_active else gr.skip())
        return message, session.status(), trajectory, *control_state(session)

    def continue_action(request: gr.Request):
        session = current(request)
        _, answer = session.send("/continue")
        return update_state(session, answer)

    def activate_quiz(topic, request: gr.Request):
        session = current(request)
        _, answer = session.send("/quiz " + (topic or ""))
        return "", *update_state(session, answer)

    def deactivate_quiz(request: gr.Request):
        session = current(request)
        _, answer = session.send("/quiz end")
        return "", *update_state(session, answer)

    def clear(request: gr.Request):
        session = current(request)
        session.clear_conversation()
        return update_state(session, "Conversation and active plan cleared. Persistent context and trajectory retained.")

    def reload_context(request: gr.Request):
        session = current(request)
        session.context.reload()
        if session.agent:
            compose_instructions(session.agent, session.context, session.source_context, session.quiz)
        trajectory, context = session.snapshot()
        return "Persistent context reloaded from `about_me.md`.", context, session.status()

    def set_retrieval(mode, request: gr.Request):
        session = current(request)
        try:
            session.set_retrieval_mode(mode)
            message = f"Retrieval mode set to **{mode}**."
        except Exception as error:
            message = f"Could not select retrieval mode: {error}"
        trajectory, context = session.snapshot()
        return message, session.status(), trajectory, context

    def compact(retain_turns, request: gr.Request):
        session = current(request)
        try:
            session.ensure_agent()
            result = session.agent.compact_memory(int(retain_turns))
            message = f"Memory compacted; retained {int(retain_turns)} complete recent turns.\n\n{result['summary']}"
        except Exception as error:
            message = f"Memory compaction failed; working Memory was preserved. {error}"
        trajectory, context = session.snapshot()
        return message, trajectory, context

    def set_source(enabled, request: gr.Request):
        session = current(request)
        session.set_source_enabled(enabled)
        _, context = session.snapshot()
        return context, session.status()

    def reload_source(request: gr.Request):
        session = current(request)
        session.reload_source()
        _, context = session.snapshot()
        return context, session.status()

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

    with gr.Blocks(title="TinyAgent Session 4", analytics_enabled=False) as app:
        state = gr.State(None, delete_callback=sessions.remove)
        gr.Markdown("# TinyAgent course tutor · Session 4 Stage B")
        status = gr.Textbox(label="Status", value=f"Model: {os.environ.get('TINYAGENT_MODEL', 'not configured')}", interactive=False)
        with gr.Tabs():
            with gr.Tab("Chat"):
                conversation = gr.Chatbot(label="Conversation", height=440)
                question = gr.Textbox(label="Message or command", lines=3, placeholder="Ask a question, /quiz [topic], or /quiz end")
                with gr.Row():
                    planning_strategy = gr.Dropdown(list(PLANNING_STRATEGIES), value=REACT, label="Planning strategy")
                    execution_mode = gr.Dropdown(["Manual Continue", "Autonomous"], value="Manual Continue", label="Execution mode")
                    max_steps = gr.Slider(1, 30, value=10, step=1, label="Maximum model steps")
                    stop_button = gr.Button("Stop run")
                with gr.Row():
                    retrieval_mode = gr.Dropdown(["Off", "Automatic RAG", "Model-directed retrieval"],
                                                 value="Off", label="Retrieval mode")
                    retained_turns = gr.Number(value=2, precision=0, label="Recent turns to retain")
                    compact_button = gr.Button("Compact Memory")
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    continue_button = gr.Button("Continue", interactive=False)
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
            with gr.Tab("Plan"):
                plan_display = gr.Markdown(f"Planning strategy: {REACT}\n\nNo plan exists for this run.")
            with gr.Tab("Trajectory"):
                gr.Markdown("Each run includes metrics, model-request steps, and any plan. Events show plan revisions, retrieval, and compaction.")
                trajectory_view = gr.JSON(label="Runs and events")
                trajectory_clear = gr.Button("Clear trajectory")
            with gr.Tab("Context"):
                gr.Markdown("Persistent instructions and active skill are separate from Memory. Effective messages show the last recorded model request; opaque reasoning protocol payloads are retained for API continuation but redacted from these views and transcripts.")
                source_toggle = gr.Checkbox(label="Include session4/tinyagent.py in model requests (adds source tokens)", value=False)
                source_reload = gr.Button("Reload tinyagent.py")
                context_view = gr.JSON(label="Context sources and latest effective request")

        controls = [continue_button, planning_strategy, plan_display]
        state_outputs = [conversation, feedback, status, trajectory_view, context_view, source_toggle, *controls]
        outputs = [question, *state_outputs]
        options = {"concurrency_id": "session4-actions", "concurrency_limit": 1}
        def load(request: gr.Request):
            session = current(request)
            trajectory, context = session.snapshot()
            return (session.status(), trajectory, context, session.source_context.enabled,
                    session.retrieval_mode, *control_state(session))

        app.load(load, outputs=[status, trajectory_view, context_view, source_toggle, retrieval_mode, *controls], **options)
        planning_strategy.input(set_planning, inputs=[planning_strategy], outputs=[feedback, status, *controls], **options)
        source_toggle.change(set_source, inputs=[source_toggle], outputs=[context_view, status], **options)
        retrieval_mode.change(set_retrieval, inputs=[retrieval_mode], outputs=[feedback, status, trajectory_view, context_view], **options)
        compact_button.click(compact, inputs=[retained_turns], outputs=[feedback, trajectory_view, context_view], **options)
        source_reload.click(reload_source, outputs=[context_view, status], **options)
        trajectory_clear.click(clear_trajectory, outputs=[trajectory_view], **options)
        send_inputs = [question, execution_mode, max_steps, planning_strategy]
        send.click(update, inputs=send_inputs, outputs=outputs, **options)
        question.submit(update, inputs=send_inputs, outputs=outputs, **options)
        stop_button.click(stop_action, outputs=[feedback, status, trajectory_view, *controls], concurrency_id="session4-stop", concurrency_limit=None)
        continue_button.click(continue_action, outputs=state_outputs, **options)
        quiz_start.click(activate_quiz, inputs=[quiz_topic], outputs=outputs, **options)
        quiz_end.click(deactivate_quiz, outputs=outputs, **options)
        clear_button.click(clear, outputs=state_outputs, **options)
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
