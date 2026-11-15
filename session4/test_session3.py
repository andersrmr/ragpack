"""Offline tests for Session 2 context, quiz orchestration, and trajectory."""

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from context import PersistentContext, SourceCodeContext, compose_instructions
from quiz import QuizManager
from tinyagent import LLM, Memory, NativeTools, Response, TinyAgent


class FakeLLM:
    model = "fake-model"
    reasoning_effort = "low"

    def __init__(self, replies=None):
        self.replies = list(replies or ["answer"])
        self.requests = []

    def generate(self, messages, tools=None):
        self.requests.append([m.copy() for m in messages])
        self.tool_schemas = tools
        value = self.replies.pop(0)
        if isinstance(value, Response):
            return value
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

    def test_source_context_is_opt_in_and_reloadable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tinyagent.py"
            path.write_text("class TinyAgent: pass", encoding="utf-8")
            source = SourceCodeContext(path)
            self.assertFalse(source.instruction)
            source.set_enabled(True)
            self.assertIn("class TinyAgent", source.instruction)
            path.write_text("class TinyAgent:\n    pass", encoding="utf-8")
            source.reload()
            self.assertIn("    pass", source.instruction)
            source.set_enabled(False)
            self.assertFalse(source.instruction)


class ProviderTests(unittest.TestCase):
    def test_openai_response_parsing_preserves_reasoning_and_usage(self):
        result = SimpleNamespace(
            output=[
                SimpleNamespace(type="message", content=[
                    SimpleNamespace(type="output_text", text="answer")]),
                SimpleNamespace(type="reasoning", summary=[
                    SimpleNamespace(type="summary_text", text="brief summary")],
                               id="rs_test", encrypted_content="sealed-protocol-payload"),
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
        self.assertEqual(response.protocol_items[1]["type"], "reasoning")
        self.assertEqual(response.protocol_items[1]["encrypted_content"], "sealed-protocol-payload")
        self.assertEqual(response.metadata["prompt_tokens"], 5)
        self.assertEqual(response.metadata["completion_tokens"], 7)
        self.assertEqual(response.metadata["total_tokens"], 12)
        self.assertEqual(calls[0]["reasoning"], {"effort": "high", "summary": "auto"})

    def test_responses_function_call_item_and_schema_are_preserved(self):
        item = SimpleNamespace(type="function_call", id="fc_item", call_id="call_42",
                               name="multiply", arguments='{"a":5.1,"b":7.3}')
        result = SimpleNamespace(output=[item], model="returned-model", usage=None)
        calls = []

        class FakeResponses:
            def create(self, **kwargs):
                calls.append(kwargs)
                return result

        llm = LLM.__new__(LLM)
        llm.model, llm.reasoning_effort = "configured", "low"
        llm.client = SimpleNamespace(responses=FakeResponses())
        schemas = NativeTools().schemas
        response = llm.generate([{"role": "user", "content": "multiply"}], tools=schemas)
        self.assertEqual(response.tool_call, {"type": "function_call", "id": "fc_item",
                                              "call_id": "call_42", "name": "multiply",
                                              "arguments": '{"a":5.1,"b":7.3}'})
        self.assertEqual(calls[0]["tools"], schemas)
        self.assertFalse(calls[0]["parallel_tool_calls"])
        self.assertEqual(calls[0]["tool_choice"], "auto")


def function_call(call_id="call_1", name="multiply", arguments='{"a":5.1,"b":7.3}'):
    return Response(tool_call={"type": "function_call", "id": "fc_1", "call_id": call_id,
                               "name": name, "arguments": arguments},
                    metadata={"model": "fake-model", "input_tokens": 5, "output_tokens": 3})


class NativeToolTests(unittest.TestCase):
    def test_response_api_tool_shapes_and_call_output_id_continue_once(self):
        llm = FakeLLM([function_call(), "The product is 37.23."])
        agent = TinyAgent(llm, Memory(), ["persistent"])
        first = agent.run("Multiply 5.1 by 7.3")
        self.assertAlmostEqual(float(first.content), 37.23)
        self.assertEqual(len(llm.requests), 1, "tool execution must not make another model request")
        self.assertTrue(agent.pending_continue)
        self.assertEqual(llm.tool_schemas[0]["type"], "function")
        self.assertEqual(llm.tool_schemas[0]["name"], "multiply")
        self.assertNotIn("function", llm.tool_schemas[0])
        call, output = agent.memory.messages[-2:]
        self.assertEqual(call["type"], "function_call")
        self.assertEqual(output, {"type": "function_call_output", "call_id": "call_1", "output": "37.23"})
        second = agent.continue_step()
        self.assertEqual(second.content, "The product is 37.23.")
        self.assertEqual(len(llm.requests), 2)
        self.assertFalse(agent.pending_continue)
        self.assertEqual(llm.requests[1][-1]["call_id"], "call_1")
        self.assertEqual(agent.trajectory.runs[0].steps[0].action["arguments"], '{"a":5.1,"b":7.3}')
        self.assertEqual(agent.trajectory.runs[0].steps[0].observation, "37.23")

    def test_invalid_calls_are_visible_and_paired_for_continuation(self):
        for name, args in [("nope", "{}"), ("multiply", "not json"),
                           ("multiply", '{"a":"five","b":2}')]:
            llm = FakeLLM([function_call(name=name, arguments=args)])
            agent = TinyAgent(llm, Memory())
            result = agent.run("try a tool")
            self.assertIn("Tool error:", result.content)
            self.assertTrue(agent.pending_continue)
            self.assertEqual(agent.memory.messages[-1]["call_id"], "call_1")
            self.assertIn("Tool error:", agent.memory.messages[-1]["output"])
            self.assertIsNotNone(agent.trajectory.runs[0].steps[0].error)

    def test_multiple_calls_are_preserved_and_rejected_without_execution(self):
        a = function_call("one").tool_call
        b = function_call("two", "get_weather", '{"location":"Rome"}').tool_call
        agent = TinyAgent(FakeLLM([Response(tool_call={"multiple_calls": [a, b]})]), Memory())
        result = agent.run("do two things")
        self.assertIn("Rejected 2 function calls", result.content)
        self.assertEqual(len([m for m in agent.memory.messages if m.get("type") == "function_call"]), 2)
        outputs = [m for m in agent.memory.messages if m.get("type") == "function_call_output"]
        self.assertEqual([m["call_id"] for m in outputs], ["one", "two"])
        self.assertFalse(agent.pending_continue)

    def test_clear_cancels_pending_continuation_and_weather_is_labeled_simulated(self):
        agent = TinyAgent(FakeLLM([function_call(name="get_weather", arguments='{"location":"Paris"}')]), Memory())
        self.assertIn("Simulated", agent.run("weather").content)
        agent.memory.clear()
        agent.pending_continue = False
        agent.active_run = None
        with self.assertRaises(RuntimeError):
            agent.continue_step()
        self.assertEqual(agent.memory.messages, [])

    @unittest.skipUnless(importlib.util.find_spec("gradio"), "install requirements-gradio.txt")
    def test_gradio_session_shows_action_observation_and_explicit_continue(self):
        from gradio_app import ChatSession
        session = ChatSession()
        session.llm = FakeLLM([function_call(), "Computed from the observation."])
        session.agent = TinyAgent(session.llm, session.memory, [session.context.system_message])
        session.send("multiply 5.1 by 7.3")
        shown = session.history()[-1]["content"]
        self.assertIn("**Action**", shown)
        self.assertIn("**Observation**", shown)
        self.assertIn("37.23", shown)
        self.assertTrue(session.agent.pending_continue)
        session.send("/continue")
        self.assertEqual(session.llm.requests.__len__(), 2)
        self.assertFalse(session.agent.pending_continue)
        self.assertIn("Computed from the observation", session.history()[-1]["content"])
        session.dispose()


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
            next_question = quiz.answer(agent, "Memory")
            self.assertIn("What records runs?", next_question)
            self.assertEqual(next_question.count("What records runs?"), 1)
            grading_prompt = llm.requests[1][-1]["content"]
            self.assertIn("Do not ask a question", grading_prompt)
            self.assertIn("harness will present the next question", grading_prompt)
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
        session.set_source_enabled(True)
        session.llm.replies.append("source aware answer")
        session.send("What does the source say?")
        request_system = [message["content"] for message in session.llm.requests[-1] if message["role"] == "system"]
        self.assertTrue(any("Optional TinyAgent source code reference" in item for item in request_system))
        session.set_source_enabled(False)
        session.llm.replies.append("source free answer")
        session.send("Ordinary question")
        request_system = [message["content"] for message in session.llm.requests[-1] if message["role"] == "system"]
        self.assertFalse(any("Optional TinyAgent source code reference" in item for item in request_system))
        self.assertFalse(any("TinyAgent source code" in str(message) for message in session.memory.get_messages()))
        trajectory, context = session.snapshot()
        self.assertEqual(trajectory["runs"][0]["steps"][0]["answer"], "browser answer")
        self.assertIn("effective_messages_for_latest_run", context)
        session.memory.messages.clear()
        self.assertEqual(len(session.snapshot()[0]["runs"]), 3)
        session.send("/clear")
        self.assertEqual(session.memory.messages, [])
        self.assertEqual(len(session.snapshot()[0]["runs"]), 3)
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
