"""Offline checks for Session 4 Stage A run control."""
import threading
import unittest

from tinyagent import Memory, Response, TinyAgent


class FakeLLM:
    model = "fake-model"
    reasoning_effort = "low"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, messages, tools=None):
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, threading.Event):
            item.set()
            release.wait(2)
            return self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def summarize(self, messages):
        return "summary", {}

    def close(self):
        pass


release = threading.Event()


def call(name="multiply", args='{"a":2,"b":3}', call_id="c1"):
    return Response(tool_call={"type": "function_call", "id": "i" + call_id,
                               "call_id": call_id, "name": name, "arguments": args}, metadata={})


class AutonomyTests(unittest.TestCase):
    def test_manual_pauses_and_continue_is_one_request(self):
        llm = FakeLLM([call(), Response(content="done")])
        agent = TinyAgent(llm, Memory())
        self.assertTrue(agent.run("task").tool_call)
        self.assertEqual(llm.calls, 1)
        self.assertEqual(agent.continue_step().content, "done")
        self.assertEqual(llm.calls, 2)

    def test_autonomous_sequential_actions_and_normal_completion(self):
        llm = FakeLLM([call(args='{"a":2,"b":4}', call_id="1"),
                       call(args='{"a":3,"b":4}', call_id="2"), Response(content="done")])
        agent = TinyAgent(llm, Memory())
        result = agent.run("task", mode="autonomous", max_steps=3)
        self.assertEqual(llm.calls, 3)
        self.assertEqual(agent.active_run.outcome, "completed")
        self.assertEqual(len(agent.active_run.steps), 3)
        self.assertEqual(result.content, "done")

    def test_max_steps_executes_last_action_then_is_incomplete(self):
        llm = FakeLLM([call(), call(call_id="2")])
        agent = TinyAgent(llm, Memory())
        result = agent.run("task", mode="autonomous", max_steps=2)
        self.assertEqual(llm.calls, 2)
        self.assertEqual(agent.active_run.outcome, "max_steps")
        self.assertEqual(result.metadata["run_outcome"], "max_steps")
        self.assertFalse(agent.pending_continue)

    def test_stop_during_request_discards_pending_call_and_keeps_memory_consistent(self):
        entered = threading.Event()
        llm = FakeLLM([entered, call()])
        agent = TinyAgent(llm, Memory())
        result_holder = []
        worker = threading.Thread(target=lambda: result_holder.append(
            agent.run("task", mode="autonomous", max_steps=4)))
        worker.start()
        self.assertTrue(entered.wait(1))
        self.assertTrue(agent.request_stop())
        release.set()
        worker.join(2)
        self.assertEqual(agent.active_run.outcome, "stopped")
        self.assertFalse(any(x.get("type") == "function_call" for x in agent.memory.messages))
        self.assertEqual(agent.memory.messages[0]["role"], "user")
        release.clear()

    def test_tool_failure_is_an_observation_and_loop_continues(self):
        llm = FakeLLM([call("multiply", "not-json"), Response(content="finished")])
        agent = TinyAgent(llm, Memory())
        agent.run("task", mode="autonomous", max_steps=2)
        self.assertEqual(llm.calls, 2)
        self.assertIn("Tool error", agent.active_run.steps[0].observation)
        self.assertEqual(agent.active_run.outcome, "completed")

    def test_multiple_calls_are_rejected_without_running_any(self):
        calls = [{"type": "function_call", "id": "1", "call_id": "1", "name": "multiply", "arguments": '{"a":2,"b":3}'},
                 {"type": "function_call", "id": "2", "call_id": "2", "name": "multiply", "arguments": '{"a":4,"b":5}'}]
        llm = FakeLLM([Response(tool_call={"multiple_calls": calls}), Response(content="done")])
        agent = TinyAgent(llm, Memory())
        first = agent.run("task")
        self.assertIn("Rejected 2", first.content)
        self.assertFalse(agent.pending_continue)
        self.assertEqual(len([x for x in agent.memory.messages if x.get("type") == "function_call_output"]), 2)

    def test_stop_after_observation_prevents_following_request(self):
        llm = FakeLLM([call(), Response(content="must not be requested")])
        agent = TinyAgent(llm, Memory())
        agent.tools.registry["multiply"]["function"] = lambda a, b: (agent.request_stop() and "observed")
        agent.run("task", mode="autonomous", max_steps=5)
        self.assertEqual(llm.calls, 1)
        self.assertEqual(agent.active_run.outcome, "stopped")
        self.assertFalse(agent.pending_continue)

    def test_directed_retrieval_continues_and_automatic_rag_is_once(self):
        class Rag:
            last_error = None
            threshold = 0.3
            top_k = 3

            def __init__(self):
                self.searches = []

            def search(self, query, top_k=None):
                self.searches.append(query)
                return [{"document_id": "d", "text": "fact", "score": 0.9}]

        rag = Rag()
        llm = FakeLLM([Response(tool_call={"type": "function_call", "id": "1", "call_id": "r1",
                                           "name": "search_documents", "arguments": '{"query":"contextual query"}'}),
                       Response(content="answer")])
        agent = TinyAgent(llm, Memory(), rag_memory=rag)
        agent.set_retrieval_mode("Model-directed retrieval")
        self.assertEqual(agent.run("question", mode="autonomous", max_steps=2).content, "answer")
        self.assertEqual(rag.searches, ["contextual query"])
        self.assertEqual(llm.calls, 2)

        automatic_rag = Rag()
        auto = TinyAgent(FakeLLM([call(), Response(content="answer")]), Memory(), rag_memory=automatic_rag)
        auto.set_retrieval_mode("Automatic RAG")
        auto.run("one user turn", mode="autonomous", max_steps=2)
        self.assertEqual(automatic_rag.searches, ["one user turn"])

    def test_gradio_session_selects_autonomous_and_reports_outcome(self):
        from gradio_app import ChatSession
        session = ChatSession()
        session.llm = FakeLLM([Response(content="finished")])
        session.agent = TinyAgent(session.llm, session.memory)
        _, answer = session.send("task", "Autonomous", 4)
        self.assertIn("Run outcome: completed", answer)
        self.assertIn("Execution: autonomous", session.status())
        self.assertEqual(session.agent.active_run.max_steps, 4)
        session.dispose()


if __name__ == "__main__":
    unittest.main()
