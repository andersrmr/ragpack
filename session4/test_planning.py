"""Offline acceptance coverage for Session 4 Stage B."""

import io
import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from context import PersistentContext, SourceCodeContext
from planning import PLAN_AND_SOLVE, REACT
from quiz import QuizManager
from test_autonomy import FakeLLM, call
from tinyagent import LLM, Memory, NativeTools, Response, TinyAgent
from transcripts import format_transcript


class RecordingLLM(FakeLLM):
    def __init__(self, responses):
        super().__init__(responses)
        self.schemas = []

    def generate(self, messages, tools=None):
        self.schemas.append(tools)
        return super().generate(messages, tools)


def tool(name, call_id, **arguments):
    return call(name, json.dumps(arguments), call_id)


def planning_agent(responses, **kwargs):
    agent = TinyAgent(RecordingLLM(responses), Memory(), **kwargs)
    agent.set_planning_strategy(PLAN_AND_SOLVE)
    return agent


def changes(agent):
    return [event for event in agent.trajectory.events if event["kind"] == "plan_changed"]


class PlanningTests(unittest.TestCase):
    def test_reconcile_by_unique_normalized_description_and_record_changes(self):
        agent = planning_agent([
            tool("update_plan", "create", tasks=["Find evidence", "Check facts", "Write report"]),
            tool("mark_done", "first", index=1),
            tool("mark_done", "second", index=2),
            tool("update_plan", "revise", tasks=["Write report", "  FIND   evidence ", "Verify totals", "Review"]),
            tool("mark_done", "repeat-done", index=2),
            tool("update_plan", "repeat-plan", tasks=["Write report", "FIND evidence", "Verify totals", "Review"]),
            Response(content="Report drafted."),
        ])
        first = agent.run("Prepare a report")
        self.assertIn("- [ ] 1. Find evidence", first.content)
        self.assertIn("- [x] 1. Find evidence", agent.continue_step().content)
        while agent.pending_continue:
            agent.continue_step()

        plan = agent.active_run.plan.snapshot()
        self.assertEqual(plan["revision"], 4)
        self.assertEqual([(task["task_id"], task["status"]) for task in plan["tasks"]],
                         [(3, "pending"), (1, "done"), (4, "pending"), (5, "pending")])
        create, complete_first, complete_second, revise = changes(agent)
        self.assertEqual(create["plan"]["pending_count"], 3)
        self.assertEqual(complete_first["declared_done_task_ids"], [1])
        self.assertEqual(complete_second["declared_done_task_ids"], [2])
        self.assertEqual(revise["call_id"], "revise")
        self.assertEqual(revise["step_id"], 4)
        self.assertEqual(revise["removed"], [{"task_id": 2, "description": "Check facts", "status": "done"}])
        self.assertEqual([task["task_id"] for task in revise["added"]], [4, 5])
        self.assertEqual(agent.active_run.outcome, "completed")
        self.assertEqual(agent.trajectory.events[-1]["pending_plan_tasks"], 3)
        self.assertIn("unfinished plan items", agent.plan_status())

    def test_ambiguous_duplicates_never_inherit_done(self):
        agent = planning_agent([
            tool("update_plan", "p1", tasks=["Gather data"]), tool("mark_done", "d", index=1),
            tool("update_plan", "p2", tasks=["Gather data", "gather data"]),
            tool("update_plan", "p3", tasks=["Gather data"]), Response(content="done"),
        ])
        agent.run("Investigate", mode="autonomous")
        self.assertEqual(agent.active_run.plan.snapshot()["tasks"], [
            {"task_id": 4, "description": "Gather data", "status": "pending"},
        ])
        self.assertEqual(changes(agent)[2]["removed"][0]["status"], "done")

    def test_bad_plan_arguments_are_recoverable_and_do_not_change_state(self):
        attempts = [
            ("mark_done", '{"index":1}', "No plan exists"),
            ("update_plan", '{"tasks":[]}', "non-empty"),
            ("update_plan", '{"tasks":["  "]}', "non-blank"),
            ("update_plan", '{"tasks":[false]}', "list of strings"),
            ("update_plan", '{"tasks":"one"}', "list of strings"),
            ("update_plan", '{"tasks":["one"],"extra":0}', "unexpected keyword"),
            ("update_plan", 'invalid json', "JSONDecodeError"),
            ("update_plan", '{}', "missing"),
            ("update_plan", '{"tasks":["one"]}', "Plan revision 1"),
            ("mark_done", '{"index":0}', "one-based"),
            ("mark_done", '{"index":2}', "one-based"),
            ("mark_done", '{"index":true}', "must be int"),
            ("mark_done", '{"index":1}', "- [x] 1. one"),
        ]
        agent = planning_agent([call(name, args, str(i)) for i, (name, args, _) in enumerate(attempts)]
                               + [Response(content="finished")])
        for i, (_, _, expected) in enumerate(attempts):
            result = agent.run("task") if i == 0 else agent.continue_step()
            self.assertIn(expected, result.content)
            self.assertTrue(agent.pending_continue)
            self.assertEqual(agent.memory.messages[-1]["call_id"], str(i))
        agent.continue_step()
        self.assertEqual(agent.active_run.plan.revision, 2)
        self.assertEqual(len(changes(agent)), 2)

    def test_all_four_strategy_and_execution_combinations(self):
        for strategy in (REACT, PLAN_AND_SOLVE):
            for mode in ("manual", "autonomous"):
                with self.subTest(strategy=strategy, mode=mode):
                    action = (tool("update_plan", "first", tasks=["Compute the total"])
                              if strategy == PLAN_AND_SOLVE else call())
                    llm = RecordingLLM([action, Response(content="final", metadata={"total_tokens": 9})])
                    action.metadata = {"total_tokens": 5}
                    agent = TinyAgent(llm, Memory())
                    self.assertEqual(agent.planning_strategy, REACT)
                    agent.set_planning_strategy(strategy)
                    result = agent.run("task", mode=mode, max_steps=2)
                    if mode == "manual":
                        self.assertEqual(llm.calls, 1)
                        self.assertEqual(agent.active_run.outcome, "waiting_continue")
                        result = agent.continue_step()
                    self.assertEqual(result.content, "final")
                    self.assertEqual(llm.calls, 2)
                    self.assertEqual(agent.active_run.outcome, "completed")
                    for request, schemas in zip(llm.requests, llm.schemas, strict=True):
                        names = {schema["name"] for schema in schemas}
                        planning = strategy == PLAN_AND_SOLVE
                        self.assertIn("multiply", names)
                        self.assertEqual("update_plan" in names, planning)
                        self.assertEqual("mark_done" in names, planning)
                        self.assertEqual(any("## Plan-and-Solve" in str(item.get("content")) for item in request), planning)
                    metrics = agent.trajectory.to_dict()["runs"][0]["metrics"]
                    self.assertEqual(metrics["model_requests"], 2)
                    self.assertEqual(metrics["tool_calls"], 1)
                    self.assertEqual(metrics["planning_tool_calls"], int(strategy == PLAN_AND_SOLVE))
                    self.assertEqual(metrics["total_tokens"], 14)
                    self.assertGreaterEqual(metrics["elapsed_seconds"], 0)

    def test_same_purchase_task_with_and_without_a_plan(self):
        task = ("I'm buying 3 notebooks at $4.25 each and 2 pens at $1.80 each, "
                "plus 8% sales tax. What's the total?")

        def arithmetic():
            return [tool("multiply", "notebooks", a=3, b=4.25), tool("multiply", "pens", a=2, b=1.8),
                    tool("add", "subtotal", a=12.75, b=3.6), tool("multiply", "taxed", a=16.35, b=1.08)]

        final = "The total is $17.66."
        react = TinyAgent(RecordingLLM(arithmetic() + [Response(content=final)]), Memory())
        notebooks, pens, subtotal, taxed = arithmetic()
        planned = planning_agent([tool("update_plan", "p", tasks=["Compute subtotal", "Apply tax"]),
                                  notebooks, pens, subtotal, tool("mark_done", "subtotal-done", index=1),
                                  taxed, tool("mark_done", "tax-done", index=2), Response(content=final)])
        for agent, expected_calls in ((react, 5), (planned, 8)):
            self.assertEqual(agent.run(task, mode="autonomous", max_steps=15).content, final)
            run = agent.active_run
            self.assertEqual(run.outcome, "completed")
            self.assertEqual(run.metrics()["model_requests"], expected_calls)
            self.assertEqual(run.metrics()["tool_calls"], expected_calls - 1)
            actual_total = next(step.observation for step in run.steps if (step.action or {}).get("call_id") == "taxed")
            self.assertEqual(f"{float(actual_total):.2f}", "17.66")
        self.assertIsNone(react.active_run.plan)
        self.assertEqual(planned.active_run.plan.snapshot()["pending_count"], 0)
        self.assertEqual(planned.active_run.metrics()["planning_tool_calls"], 3)

    def test_real_adapter_replays_plan_function_call_and_opaque_reasoning_in_order(self):
        inputs = []
        reasoning = SimpleNamespace(type="reasoning", id="rs_1", summary=[], encrypted_content="opaque-test-data")
        action = SimpleNamespace(type="function_call", id="fc_1", call_id="plan_1", name="update_plan",
                                 arguments='{"tasks":["Check evidence"]}')
        answer = SimpleNamespace(type="message", role="assistant", status="completed", id="msg_1",
                                 content=[SimpleNamespace(type="output_text", text="Checked.")])
        outputs = iter([[reasoning, action], [answer]])

        def create(**kwargs):
            inputs.append(kwargs)
            return SimpleNamespace(output=next(outputs), usage=None)

        llm = LLM.__new__(LLM)
        llm.model, llm.reasoning_effort = "fake-model", "medium"
        llm.client = SimpleNamespace(responses=SimpleNamespace(create=create))
        agent = TinyAgent(llm, Memory())
        agent.set_planning_strategy(PLAN_AND_SOLVE)
        agent.run("Check the evidence")
        self.assertEqual(len(inputs), 1)
        self.assertFalse(inputs[0]["parallel_tool_calls"])
        schemas = {schema["name"]: schema for schema in inputs[0]["tools"]}
        for name, argument, field in (("update_plan", "tasks", {"type": "array", "items": {"type": "string"}}),
                                      ("mark_done", "index", {"type": "integer"})):
            self.assertEqual(schemas[name]["type"], "function")
            self.assertTrue(schemas[name]["strict"])
            self.assertEqual(schemas[name]["parameters"], {
                "type": "object", "properties": {argument: field}, "required": [argument], "additionalProperties": False,
            })
        self.assertEqual(agent.continue_step().content, "Checked.")
        replay = inputs[1]["input"][-3:]
        self.assertEqual([item["type"] for item in replay], ["reasoning", "function_call", "function_call_output"])
        self.assertEqual(replay[0]["encrypted_content"], "opaque-test-data")
        self.assertEqual(replay[1]["call_id"], replay[2]["call_id"])
        self.assertEqual(replay[2]["call_id"], "plan_1")
        self.assertIn("Check evidence", replay[2]["output"])
        self.assertNotIn("opaque-test-data", json.dumps(agent.trajectory.to_dict()))
        self.assertNotIn("opaque-test-data", json.dumps(agent.memory.display_messages()))
        self.assertNotIn("opaque-test-data", format_transcript(agent.memory))

    def test_unoffered_planning_calls_and_parallel_batches_do_not_mutate_plan(self):
        create = tool("update_plan", "p", tasks=["Check"])
        react = TinyAgent(RecordingLLM([create, Response(content="recovered")]), Memory())
        self.assertIn("available only in Plan-and-Solve", react.run("task").content)
        self.assertIsNone(react.active_run.plan)
        self.assertEqual(react.continue_step().content, "recovered")

        both = [create.tool_call, tool("mark_done", "d", index=1).tool_call]
        planned = planning_agent([Response(tool_call={"multiple_calls": both})])
        self.assertIn("Rejected 2", planned.run("task").content)
        self.assertEqual(planned.active_run.plan.revision, 0)
        self.assertFalse(planned.pending_continue)
        self.assertEqual(changes(planned), [])
        self.assertEqual([item["call_id"] for item in planned.memory.messages if item.get("type") == "function_call_output"],
                         ["p", "d"])

    def test_run_boundary_preserves_history_and_trivial_tasks_can_skip_planning(self):
        agent = planning_agent([
            tool("update_plan", "p", tasks=["Check"]), tool("mark_done", "d", index=1),
            Response(content="Done"), Response(content="Hello!"),
            tool("update_plan", "new", tasks=["Check"]), Response(content="Done again"),
        ])
        agent.run("work", mode="autonomous")
        original = agent.active_run
        self.assertEqual(original.plan.snapshot()["pending_count"], 0)
        self.assertIn("declared done", agent.plan_status())
        agent.run("hello")
        self.assertIn("No plan exists", agent.plan_status())
        self.assertEqual(agent.active_run.outcome, "completed")
        agent.run("new work", mode="autonomous")
        self.assertEqual(agent.active_run.plan.snapshot()["pending_count"], 1)
        self.assertEqual(original.plan.snapshot()["pending_count"], 0)
        self.assertEqual(len(changes(agent)), 3)
        agent.trajectory.clear()
        self.assertEqual(agent.active_run.plan.revision, 1)
        agent.clear_conversation()
        self.assertEqual(agent.memory.messages, [])
        self.assertEqual(agent.planning_strategy, PLAN_AND_SOLVE)
        self.assertIsNone(agent.active_run)

    def test_a_checked_plan_does_not_override_the_step_limit_or_provider_failure(self):
        replies = [tool("update_plan", "p", tasks=["Check"]), tool("mark_done", "d", index=1)]
        bounded = planning_agent(replies)
        result = bounded.run("task", mode="autonomous", max_steps=2)
        self.assertEqual(result.metadata["run_outcome"], "max_steps")
        self.assertEqual(bounded.active_run.plan.snapshot()["pending_count"], 0)
        self.assertFalse(bounded.pending_continue)
        self.assertIsNone(bounded.active_run.steps[-1].answer)
        self.assertIsNone(bounded.active_run.metrics()["total_tokens"])

        failing = planning_agent(replies + [RuntimeError("provider failed")])
        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            failing.run("task", mode="autonomous")
        self.assertEqual(failing.active_run.outcome, "error")
        self.assertEqual(failing.active_run.plan.snapshot()["pending_count"], 0)
        self.assertFalse(failing.pending_continue)

    def test_paused_plan_rejects_strategy_change_and_can_be_stopped_or_cleared(self):
        agent = planning_agent([tool("update_plan", "p", tasks=["Check"]),
                                tool("update_plan", "p2", tasks=["Restart"])])
        agent.run("work")
        with self.assertRaisesRegex(RuntimeError, "paused"):
            agent.set_planning_strategy(REACT)
        with self.assertRaisesRegex(RuntimeError, "waiting"):
            agent.run("another task")
        self.assertEqual(agent.active_run.outcome, "waiting_continue")
        self.assertTrue(agent.request_stop())
        self.assertEqual(agent.active_run.outcome, "stopped")
        self.assertEqual(agent.active_run.plan.revision, 1)
        self.assertFalse(agent.pending_continue)
        agent.set_planning_strategy(REACT)
        agent.set_planning_strategy(PLAN_AND_SOLVE)
        agent.run("another task")
        agent.clear_conversation()
        self.assertIn("No plan exists", agent.plan_status())
        self.assertEqual(len(changes(agent)), 2)
        self.assertEqual([run.outcome for run in agent.trajectory.runs], ["stopped", "stopped"])
        with self.assertRaisesRegex(RuntimeError, "No tool action"):
            agent.continue_step()

    def test_stop_during_model_request_keeps_previous_plan_and_discards_revision(self):
        agent = planning_agent([tool("update_plan", "p", tasks=["Keep"]),
                                tool("update_plan", "discard", tasks=["Discard"])])
        entered, release = threading.Event(), threading.Event()
        generate = agent.llm.generate

        def block_after_plan(messages, tools=None):
            if agent.llm.calls == 1:
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("Test did not release the request.")
            return generate(messages, tools)

        agent.llm.generate = block_after_plan
        with ThreadPoolExecutor(1) as executor:
            future = executor.submit(agent.run, "task", mode="autonomous", max_steps=2)
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaisesRegex(RuntimeError, "active"):
                    agent.set_planning_strategy(REACT)
                self.assertTrue(agent.request_stop())
            finally:
                release.set()
            future.result(timeout=5)
        self.assertEqual(agent.active_run.outcome, "stopped")
        self.assertEqual(agent.active_run.plan.tasks[0].description, "Keep")
        self.assertEqual(len(changes(agent)), 1)
        self.assertEqual(agent.active_run.steps[-1].kind, "discarded_response")
        self.assertNotIn("discard", [item.get("call_id") for item in agent.memory.messages])

    def test_dispatched_plan_tool_finishes_before_a_cooperative_stop(self):
        agent = planning_agent([tool("update_plan", "p", tasks=["Keep"]), Response(content="unused")])
        update = agent.planner.update_plan

        def stop_during_update(tasks: list[str]) -> str:
            self.assertTrue(agent.request_stop())
            return update(tasks)

        agent.tools.add_tool("update_plan", stop_during_update)
        agent.run("task", mode="autonomous", max_steps=1)
        self.assertEqual(agent.active_run.outcome, "stopped")
        self.assertEqual(agent.memory.messages[-1]["call_id"], "p")
        self.assertEqual(agent.active_run.plan.tasks[0].description, "Keep")
        self.assertEqual(len(changes(agent)), 1)

    def test_shared_registry_template_does_not_share_bound_plan_tools(self):
        tools = NativeTools()
        first = planning_agent([tool("update_plan", "a", tasks=["First"])], tools=tools)
        second = planning_agent([tool("update_plan", "b", tasks=["Second"])], tools=tools)
        first.run("First task")
        second.run("Second task")
        self.assertEqual(first.active_run.plan.tasks[0].description, "First")
        self.assertEqual(second.active_run.plan.tasks[0].description, "Second")


class IntegrationTests(unittest.TestCase):
    def test_retrieval_tools_and_plan_observations_compose_and_compact(self):
        class Rag:
            last_error, threshold, top_k = "", 0.3, 3

            def __init__(self):
                self.queries = []

            def search(self, query, top_k=None):
                self.queries.append(query)
                return [{"document_id": "doc", "score": 0.9, "text": "Ilse plays cello."}]

        rag = Rag()
        agent = planning_agent([
            tool("update_plan", "p", tasks=["Search for evidence"]),
            tool("search_documents", "s", query="Ilse hobby"),
            tool("mark_done", "d", index=1), Response(content="Ilse plays cello."),
            tool("update_plan", "p2", tasks=["Read the passage"]), Response(content="The passage mentions cello."),
        ], rag_memory=rag)
        agent.set_retrieval_mode("Model-directed retrieval")
        agent.run("What does Ilse play?", mode="autonomous")
        self.assertEqual(rag.queries, ["Ilse hobby"])
        self.assertIn("Ilse plays cello", agent.llm.requests[2][-1]["output"])
        self.assertEqual([step.action["tool"] for step in agent.active_run.steps[:-1]],
                         ["update_plan", "search_documents", "mark_done"])
        agent.set_retrieval_mode("Automatic RAG")
        agent.run("Follow-up", mode="autonomous")
        self.assertEqual(rag.queries, ["Ilse hobby", "Follow-up"])
        self.assertTrue(any("Temporary local retrieval" in str(item.get("content")) for item in agent.llm.requests[4]))
        self.assertFalse(any("Temporary local retrieval" in str(item.get("content")) for item in agent.llm.requests[5]))
        transcript = format_transcript(agent.memory)
        agent.compact_memory(1)
        self.assertEqual(format_transcript(agent.memory), transcript)
        self.assertEqual(agent.active_run.plan.tasks[0].description, "Read the passage")

    def test_quiz_with_planning_selected_keeps_its_tool_free_flow_and_artifacts(self):
        document = {"questions": [{"question": "Q?", "answer": "A", "explanation": "Why."}]}
        agent = planning_agent([Response(content=json.dumps(document)), Response(content="Correct"),
                                Response(content="Summary"), Response(content="Ordinary reply")])
        with TemporaryDirectory() as directory:
            quiz = QuizManager(directory)
            self.assertIn("Q?", quiz.start(agent, "Example"))
            self.assertIn("Quiz complete", quiz.answer(agent, "A"))
            self.assertIn("Progress: 1/1", quiz.quiz["path"].read_text())
            self.assertEqual(agent.llm.schemas, [[], [], []])
            self.assertFalse(any("## Plan-and-Solve" in str(message.get("content"))
                                 for request in agent.llm.requests for message in request))
            quiz.end(agent)
        self.assertEqual(agent.planning_strategy, PLAN_AND_SOLVE)
        agent.run("ordinary tutoring")
        self.assertIn("update_plan", {schema["name"] for schema in agent.llm.schemas[-1]})

    def test_gradio_control_callbacks_refresh_real_plan_and_isolate_sessions(self):
        from gradio_app import build_app

        app, sessions = build_app()
        self.addCleanup(app.close)
        self.addCleanup(sessions.close)
        session = sessions.get("first")
        session.llm = RecordingLLM([tool("update_plan", "p", tasks=["Draft article"]),
                                    Response(content="Finished")])
        session.agent = TinyAgent(session.llm, session.memory)
        request = SimpleNamespace(session_hash="first")
        callbacks = {block.fn.__name__: block.fn for block in app.fns.values() if block.fn}
        selected = callbacks["set_planning"](PLAN_AND_SOLVE, request)
        self.assertIn(PLAN_AND_SOLVE, selected[-1])
        result = callbacks["update"]("Draft", "Manual Continue", 10, PLAN_AND_SOLVE, request)
        self.assertIn("Draft article", result[-1])
        self.assertFalse(result[-2]["interactive"])
        self.assertIn("update_plan", session.history()[-1]["content"])
        self.assertIn("paused", callbacks["set_planning"](REACT, request)[0])
        self.assertIn("No plan exists", sessions.get("second").plan_view())
        self.assertEqual(sessions.get("second").planning_strategy, REACT)
        continued = callbacks["continue_action"](request)
        self.assertIn("unfinished plan items", continued[-1])
        self.assertTrue(continued[-2]["interactive"])
        callbacks["clear_trajectory"](request)
        self.assertIn("Draft article", session.plan_view())
        self.assertIn("No plan exists", callbacks["clear"](request)[-1])

    def test_gradio_stop_callback_keeps_paused_plan_visible(self):
        from gradio_app import build_app

        app, sessions = build_app()
        self.addCleanup(app.close)
        self.addCleanup(sessions.close)
        session = sessions.get("paused")
        session.llm = RecordingLLM([tool("update_plan", "p", tasks=["Check results"])])
        session.agent = TinyAgent(session.llm, session.memory)
        session.send("Audit results", "Manual Continue", 10, PLAN_AND_SOLVE)
        stop_action = next(block.fn for block in app.fns.values() if block.fn and block.fn.__name__ == "stop_action")
        message, _, trajectory, continue_control, strategy_control, plan = stop_action(SimpleNamespace(session_hash="paused"))
        self.assertEqual(message, "Run stopped.")
        self.assertEqual(trajectory["runs"][0]["outcome"], "stopped")
        self.assertFalse(continue_control["interactive"])
        self.assertTrue(strategy_control["interactive"])
        self.assertIn("Check results", plan)
        self.assertEqual(session.llm.calls, 1)

    def test_terminal_planner_commands_and_manual_continue(self):
        from main import chat

        llm = RecordingLLM([tool("update_plan", "p", tasks=["Check figures"]), Response(content="Finished")])
        agent = TinyAgent(llm, Memory())
        commands = ["/planner plan-and-solve", "Do a task", "/plan", "/planner react", "/continue",
                    "/planner react", "/planner status", "/clear", "/plan", "/quit"]
        output, errors = io.StringIO(), io.StringIO()
        with TemporaryDirectory() as directory, patch("builtins.input", side_effect=commands):
            with redirect_stdout(output), redirect_stderr(errors):
                chat(agent, PersistentContext(), QuizManager(directory), SourceCodeContext())
        self.assertEqual(llm.calls, 2)
        self.assertIn("paused run", errors.getvalue())
        self.assertIn("Check figures", output.getvalue())
        self.assertIn("Finished", output.getvalue())
        self.assertIn("No plan exists", output.getvalue())
        self.assertEqual(agent.planning_strategy, REACT)
        self.assertEqual(agent.trajectory.runs[0].planning_strategy, PLAN_AND_SOLVE)


if __name__ == "__main__":
    unittest.main()
