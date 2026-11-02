"""Offline tests for Session 2 context, quiz orchestration, and trajectory."""

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from context import PersistentContext
from quiz import QuizManager
from tinyagent import LLM, Memory, Response, TinyAgent


class FakeLLM:
    model = "fake-model"
    reasoning_effort = "low"

    def __init__(self, replies=None):
        self.replies = list(replies or ["answer"])
        self.requests = []

    def generate(self, messages):
        self.requests.append([m.copy() for m in messages])
        value = self.replies.pop(0)
        return Response(content=value, reasoning="", metadata={"model": self.model,
                                                                 "input_tokens": 4,
                                                                 "output_tokens": 3,
                                                                 "total_tokens": 7})

    def close(self):
        pass


class ContextTests(unittest.TestCase):
    def test_persistent_context_is_sent_and_separate_from_memory_after_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "about_me.md"
            path.write_text("I prefer diagrams.", encoding="utf-8")
            context = PersistentContext(path)
            llm = FakeLLM(["first", "second"])
            memory = Memory()
            agent = TinyAgent(llm, memory, [context.system_message])
            agent.run("Explain memory")
            self.assertEqual(llm.requests[0][0]["role"], "system")
            self.assertIn("I prefer diagrams", llm.requests[0][0]["content"])
            memory.messages.clear()
            agent.run("Try again")
            self.assertIn("I prefer diagrams", llm.requests[1][0]["content"])
            self.assertEqual(sum(m["role"] == "system" for m in llm.requests[1]), 1)

    def test_reload_missing_and_empty_context(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "about_me.md"
            context = PersistentContext(path)
            self.assertEqual(context.about_me, "")
            path.write_text("Updated", encoding="utf-8")
            context.reload()
            self.assertEqual(context.about_me, "Updated")
            path.write_text("  ", encoding="utf-8")
            context.reload()
            self.assertEqual(context.about_me, "")


class ProviderTests(unittest.TestCase):
    def test_openai_response_parsing_preserves_reasoning_and_usage(self):
        result = SimpleNamespace(
            output=[
                SimpleNamespace(type="message", content=[
                    SimpleNamespace(type="output_text", text="answer")]),
                SimpleNamespace(type="reasoning", summary=[
                    SimpleNamespace(type="summary_text", text="brief summary")]),
            ],
            model="returned-model",
            usage=SimpleNamespace(input_tokens=5, output_tokens=7, total_tokens=12,
                                  output_tokens_details=SimpleNamespace(reasoning_tokens=2)),
        )
        calls = []

        class FakeResponses:
            def create(self, **kwargs):
                calls.append(kwargs)
                return result

        llm = LLM.__new__(LLM)
        llm.model = "configured-model"
        llm.reasoning_effort = "high"
        llm.client = SimpleNamespace(responses=FakeResponses())
        response = llm.generate([{"role": "user", "content": "question"}])
        self.assertEqual(response.content, "answer")
        self.assertEqual(response.reasoning, "brief summary")
        self.assertEqual(response.metadata["prompt_tokens"], 5)
        self.assertEqual(response.metadata["completion_tokens"], 7)
        self.assertEqual(response.metadata["total_tokens"], 12)
        self.assertEqual(calls[0]["reasoning"], {"effort": "high", "summary": "auto"})


class QuizTests(unittest.TestCase):
    def test_quiz_lifecycle_artifact_progress_and_no_skill_leak(self):
        generated = json.dumps({"topic": "Memory", "questions": [
            {"question": "What stores turns?", "answer": "Memory", "explanation": "It stores turns."},
            {"question": "What records runs?", "answer": "Trajectory", "explanation": "It records runs."},
        ]})
        llm = FakeLLM([generated, "Correct: Memory stores the turns.", "Correct: Trajectory records runs.", "Review both boundaries.", "Ordinary answer."])
        agent = TinyAgent(llm, Memory(), ["persistent tutor"])
        with tempfile.TemporaryDirectory() as directory:
            quiz = QuizManager(Path(directory))
            self.assertIn("Question 1 of 2", quiz.start(agent, "Memory"))
            self.assertTrue(quiz.active)
            self.assertIn("What records runs?", quiz.answer(agent, "Memory"))
            self.assertIn("Quiz complete", quiz.answer(agent, "Trajectory"))
            artifact = Path(quiz.quiz["path"])
            saved = artifact.read_text(encoding="utf-8")
            self.assertIn("Trajectory", saved)
            self.assertIn("Progress: 2/2", saved)
            quiz.end(agent)
            self.assertFalse(quiz.active)
            agent.run("ordinary")
            self.assertNotIn(quiz.skill_text, [m["content"] for m in llm.requests[-1] if m["role"] == "system"])
            self.assertEqual(llm.requests[-1][0]["content"], "persistent tutor")

    def test_bad_quiz_json_is_rejected_without_leaving_skill_active(self):
        agent = TinyAgent(FakeLLM(["not json"]), Memory(), ["persistent"])
        with tempfile.TemporaryDirectory() as directory:
            quiz = QuizManager(directory)
            with self.assertRaises(ValueError):
                quiz.start(agent, "test")
            self.assertFalse(quiz.active)
            self.assertEqual(agent.system_instructions, ["persistent"])
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_artifact_names_are_safe_and_never_overwrite(self):
        document = json.dumps({"topic": "../../ secret topic", "questions": [
            {"question": "Q?", "answer": "A", "explanation": "Why."},
        ]})
        agent = TinyAgent(FakeLLM([document, document]), Memory(), ["persistent"])
        with tempfile.TemporaryDirectory() as directory:
            quiz = QuizManager(directory)
            quiz.start(agent, "topic")
            first = Path(quiz.quiz["path"])
            original = first.read_text(encoding="utf-8")
            quiz.end(agent)
            quiz.start(agent, "topic")
            second = Path(quiz.quiz["path"])
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_text(encoding="utf-8"), original)
            self.assertNotIn("/", first.name)

    def test_trajectory_records_messages_and_usage_separately_from_memory(self):
        llm = FakeLLM()
        memory = Memory()
        agent = TinyAgent(llm, memory, ["persistent"])
        agent.run("hello")
        run = agent.trajectory.to_dict()["runs"][0]
        step = run["steps"][0]
        self.assertEqual(run["query"], "hello")
        self.assertEqual(step["metadata"]["total_tokens"], 7)
        self.assertEqual(memory.messages, [{"role": "user", "content": "hello"},
                                           {"role": "assistant", "content": "answer"}])
        agent.trajectory.clear()
        self.assertEqual(memory.get_messages()[0]["content"], "hello")
        self.assertEqual(agent.trajectory.runs, [])


@unittest.skipUnless(importlib.util.find_spec("gradio"), "install requirements-gradio.txt")
class GradioTests(unittest.TestCase):
    def test_session_observability_and_app_construction(self):
        from gradio_app import ChatSession, Sessions, build_app

        session = ChatSession()
        session.llm = FakeLLM(["browser answer"])
        session.agent = TinyAgent(session.llm, session.memory, [session.context.system_message])
        remaining, answer = session.send("Explain trajectory")
        self.assertEqual(remaining, "")
        self.assertIn("browser answer", answer)
        trajectory, context = session.snapshot()
        self.assertEqual(trajectory["runs"][0]["steps"][0]["answer"], "browser answer")
        self.assertIn("effective_messages_for_latest_run", context)
        session.memory.messages.clear()
        self.assertEqual(len(session.snapshot()[0]["runs"]), 1)
        session.send("/clear")
        self.assertEqual(session.memory.messages, [])
        self.assertEqual(len(session.snapshot()[0]["runs"]), 1)
        session.dispose()

        isolated = Sessions()
        self.assertIsNot(isolated.get("one").memory, isolated.get("two").memory)
        isolated.close()

        app, sessions = build_app()
        self.assertIsNotNone(app)
        sessions.close()
        app.close()


if __name__ == "__main__":
    unittest.main()
