"""Offline coverage for Session 3 retrieval and explicit compaction."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from retrieval import EmbeddingModel, RAGMemory
from tinyagent import LLM, Memory, NativeTools, Response, TinyAgent
from transcripts import format_transcript


class FakeEmbeddings:
    def __init__(self, mapping=None, error=None):
        self.mapping = mapping or {}
        self.error = error
        self.calls = []

    def create(self, model, input):
        self.calls.append((model, list(input)))
        if self.error:
            raise self.error
        return SimpleNamespace(data=[SimpleNamespace(embedding=self.mapping.get(text, [1.0, 0.0]))
                                     for text in input])


class FakeLLM:
    model = "offline-model"
    reasoning_effort = "low"

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def generate(self, messages, tools=None):
        self.requests.append((messages, tools))
        result = self.replies.pop(0)
        return result if isinstance(result, Response) else Response(content=result, metadata={"model": self.model})

    def summarize(self, messages):
        self.requests.append((messages, None))
        result = self.replies.pop(0)
        if isinstance(result, Exception):
            raise result
        return result, {"model": self.model, "input_tokens": 9, "output_tokens": 4, "total_tokens": 13}

    def close(self):
        pass


def call(name="search_documents", call_id="search-1", arguments='{"query":"Ilse languages","top_k":2}'):
    return Response(tool_call={"type": "function_call", "id": "fc-1", "call_id": call_id,
                               "name": name, "arguments": arguments},
                    metadata={"model": "offline-model"})


def document(directory, doc_id, body):
    (Path(directory) / f"{doc_id}.md").write_text(
        f"---\nid: {doc_id}\ntitle: {doc_id.title()}\nsource: synthetic\n---\n{body}\n", encoding="utf-8")


class RetrievalTests(unittest.TestCase):
    def test_course_ilse_fixture_has_expected_automatic_and_directed_evidence(self):
        corpus = Path(__file__).with_name("documents")
        paths = sorted(corpus.glob("*.md"))
        mapping = {path.read_text(encoding="utf-8").split("---\n", 2)[-1].strip(): [0, 0, 1]
                   for path in paths}
        mapping["Sarah speaks fluent Spanish and Portuguese."] = [0, 1, 0]
        mapping["The course source lists Ilse's work, residence, cello playing, favorite author, and favorite animals, but does not state which languages Ilse speaks. This note describes the source's omission; it is not a course fact about Ilse."] = [1, 0, 0]
        mapping["Does she speak any languages?"] = [0, 1, 0]
        mapping["Ilse's languages"] = [1, 0, 0]
        endpoint = FakeEmbeddings(mapping)
        with tempfile.TemporaryDirectory() as directory:
            rag = RAGMemory(corpus, EmbeddingModel(SimpleNamespace(embeddings=endpoint), "fixture-embedding"),
                            Path(directory) / "cache.json", top_k=3, threshold=.3)
            automatic = rag.search("Does she speak any languages?")
            directed = rag.search("Ilse's languages")
        self.assertEqual([match["document_id"] for match in automatic], ["sarah-languages"])
        self.assertIn("ilse-language-note", [match["document_id"] for match in directed])
        self.assertIn("software engineer at a renewable energy startup", next(
            doc["text"] for doc in rag.documents if doc["id"] == "ilse-work"))

    def test_embedding_endpoint_maps_inputs_and_propagates_errors(self):
        endpoint = FakeEmbeddings({"one": [0.1, 0.2], "two": [0.3, 0.4]})
        model = EmbeddingModel(SimpleNamespace(embeddings=endpoint), model="fake-embedding")
        self.assertEqual(model.embed(["one", "two"]), [[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(endpoint.calls[0][0], "fake-embedding")
        broken = EmbeddingModel(SimpleNamespace(embeddings=FakeEmbeddings(error=RuntimeError("offline"))))
        with self.assertRaisesRegex(RuntimeError, "offline"):
            broken.embed("question")

    def test_corpus_cache_identity_ranking_threshold_and_empty_cases(self):
        with tempfile.TemporaryDirectory() as root:
            corpus, cache = Path(root) / "docs", Path(root) / "cache.json"
            corpus.mkdir()
            document(corpus, "alpha", "Ilse speaks Dutch and English.")
            document(corpus, "beta", "Mara speaks Spanish.")
            (corpus / "malformed.md").write_text("not front matter", encoding="utf-8")
            endpoint = FakeEmbeddings({"Ilse speaks Dutch and English.": [1, 0],
                                       "Mara speaks Spanish.": [0, 1], "languages": [1, 0]})
            rag = RAGMemory(corpus, EmbeddingModel(SimpleNamespace(embeddings=endpoint), "e1"), cache, top_k=1, threshold=.5)
            self.assertTrue(rag.corpus_errors)
            matches = rag.search("languages")
            self.assertEqual([item["document_id"] for item in matches], ["alpha"])
            self.assertEqual(matches[0]["title"], "Alpha")
            self.assertEqual(matches[0]["source"], "synthetic")
            first_call_count = len(endpoint.calls)
            rag.search("languages")
            self.assertEqual(len(endpoint.calls), first_call_count + 1)  # query only; docs came from cache
            document(corpus, "alpha", "Ilse speaks German.")
            rag = RAGMemory(corpus, EmbeddingModel(SimpleNamespace(embeddings=endpoint), "e1"), cache)
            rag.index()
            self.assertIn("Ilse speaks German.", endpoint.calls[-1][1])
            rag2 = RAGMemory(corpus, EmbeddingModel(SimpleNamespace(embeddings=endpoint), "e2"), cache)
            before = len(endpoint.calls)
            rag2.index()
            self.assertEqual(len(endpoint.calls), before + 1)
            empty = RAGMemory(Path(root) / "absent", EmbeddingModel(SimpleNamespace(embeddings=endpoint)), cache)
            self.assertEqual(empty.search("anything"), [])
            self.assertEqual(empty.search(""), [])
            low = RAGMemory(corpus, EmbeddingModel(SimpleNamespace(embeddings=endpoint)), cache, threshold=1.1)
            self.assertEqual(low.search("unknown"), [])

    def test_automatic_rag_uses_only_latest_turn_once_and_off_does_not_search(self):
        class StubRAG:
            top_k, threshold, last_error = 2, .4, ""
            def __init__(self): self.queries = []
            def search(self, query, top_k=None):
                self.queries.append(query)
                return [{"document_id": "ilse", "title": "Ilse", "source": "synthetic",
                         "score": .8, "text": "Ilse speaks Dutch and English."}]
        rag = StubRAG()
        math_call = Response(tool_call={"type": "function_call", "id": "fc-m", "call_id": "math-1",
                                        "name": "multiply", "arguments": '{"a":2,"b":3}'})
        llm = FakeLLM(["off answer", math_call, "auto answer"])
        agent = TinyAgent(llm, Memory(), tools=NativeTools(), rag_memory=rag)
        agent.run("What does Ilse do for work?")
        self.assertEqual(rag.queries, [])
        self.assertNotIn("search_documents", [schema["name"] for schema in llm.requests[0][1]])
        agent.set_retrieval_mode("Automatic RAG")
        agent.run("Does she speak any languages?")
        self.assertEqual(rag.queries, ["Does she speak any languages?"])
        request = llm.requests[-1][0]
        retrieval_item = next(item for item in request if "Temporary local retrieval context" in item.get("content", ""))
        self.assertIn("Temporary local retrieval context", retrieval_item["content"])
        self.assertNotIn("What does Ilse do for work?", json.dumps(retrieval_item))
        self.assertNotIn("Temporary local retrieval context", json.dumps(agent.memory.messages))
        self.assertEqual(agent.trajectory.runs[-1].metadata["automatic_retrieval"]["query"], "Does she speak any languages?")
        self.assertEqual(agent.trajectory.runs[-1].metadata["automatic_retrieval"]["threshold_decision"], "matches accepted")
        agent.continue_step()
        self.assertEqual(rag.queries, ["Does she speak any languages?"])
        self.assertNotIn("Temporary local retrieval context", json.dumps(llm.requests[-1][0]))

    def test_directed_tool_is_mode_limited_and_pauses_for_one_continue(self):
        class StubRAG:
            top_k, threshold, last_error = 3, .3, ""
            def search(self, query, top_k=None):
                self.query = query
                return [{"document_id": "ilse", "title": "Ilse", "source": "synthetic", "score": .9,
                         "text": "Ilse speaks Dutch and English."}]
        rag = StubRAG()
        llm = FakeLLM([call(), "final"])
        agent = TinyAgent(llm, Memory(), rag_memory=rag)
        agent.set_retrieval_mode("Model-directed retrieval")
        result = agent.run("Does she speak any languages?")
        self.assertEqual(len(llm.requests), 1)
        self.assertIn("search_documents", [schema["name"] for schema in llm.requests[0][1]])
        self.assertTrue(agent.pending_continue)
        self.assertEqual(agent.memory.messages[-1]["call_id"], "search-1")
        self.assertIn("Ilse speaks Dutch", agent.memory.messages[-1]["output"])
        step = agent.trajectory.runs[0].steps[0]
        self.assertEqual(step.action["arguments"], call().tool_call["arguments"])
        self.assertIn("Dutch", step.observation)
        agent.continue_step()
        self.assertEqual(len(llm.requests), 2)
        self.assertFalse(agent.pending_continue)
        self.assertIn("search_documents", [schema["name"] for schema in llm.requests[1][1]])

    def test_compaction_is_explicit_atomic_and_transcript_survives(self):
        memory = Memory()
        for number in range(3):
            memory.add("user", f"question {number}")
            memory.add("assistant", f"answer {number}")
        before = format_transcript(memory)
        llm = FakeLLM(["Summary of the first conversation."])
        agent = TinyAgent(llm, memory)
        result = agent.compact_memory(1)
        self.assertEqual(memory.summary, "Summary of the first conversation.")
        self.assertEqual(result["usage"]["total_tokens"], 13)
        self.assertEqual([item["content"] for item in memory.messages if item.get("role") == "user"], ["question 2"])
        self.assertEqual(format_transcript(memory), before)
        llm.replies.append("new response")
        agent.run("new question")
        self.assertIn("new question", format_transcript(memory))
        self.assertIn("question 0", format_transcript(memory))
        pending_rag = SimpleNamespace(search=lambda *_: [], last_error="", top_k=3, threshold=.3)
        pending = TinyAgent(FakeLLM([call()]), Memory(), rag_memory=pending_rag)
        pending.set_retrieval_mode("Model-directed retrieval")
        # Replace the registered callable with a local deterministic result for this pending-state test.
        pending.tools.registry["search_documents"]["function"] = lambda **_: "observation"
        pending.run("search")
        with self.assertRaisesRegex(RuntimeError, "tool result"):
            pending.compact_memory()
        broken = Memory()
        broken.add("user", "old")
        broken.add_function_call({"type": "function_call", "call_id": "x", "name": "f", "arguments": "{}"})
        with self.assertRaisesRegex(RuntimeError, "unmatched"):
            broken.compact(FakeLLM([]), 0)

    def test_failed_compaction_keeps_working_memory_and_clear_keeps_persistent_context_external(self):
        memory = Memory()
        memory.add("user", "one")
        memory.add("assistant", "two")
        memory.add("user", "three")
        memory.add("assistant", "four")
        original = json.dumps(memory.messages)
        agent = TinyAgent(FakeLLM([RuntimeError("summary failure")]), memory, ["persistent context"])
        with self.assertRaisesRegex(RuntimeError, "summary failure"):
            agent.compact_memory(1)
        self.assertEqual(json.dumps(memory.messages), original)
        self.assertFalse(agent.trajectory.events[-1]["success"])
        memory.summary = "old summary"
        memory.clear()
        self.assertEqual(memory.summary, "")
        self.assertEqual(agent.system_instructions, ["persistent context"])

    def test_compaction_rejects_incomplete_turn(self):
        memory = Memory()
        memory.add("user", "not answered")
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            memory.compact(FakeLLM([]), 0)


class ProviderEmbeddingTests(unittest.TestCase):
    def test_openai_sdk_v2_embedding_request_shape(self):
        endpoint = FakeEmbeddings({"hello": [1, 2]})
        model = EmbeddingModel(SimpleNamespace(embeddings=endpoint), "text-embedding-3-small")
        self.assertEqual(model.embed("hello"), [[1, 2]])
        self.assertEqual(endpoint.calls, [("text-embedding-3-small", ["hello"])])

    def test_compaction_uses_tool_free_responses_request_and_records_usage(self):
        captured = []

        class Responses:
            def create(self, **kwargs):
                captured.append(kwargs)
                return SimpleNamespace(output=[SimpleNamespace(type="message", content=[
                    SimpleNamespace(type="output_text", text="summary")])], model="returned",
                    usage=SimpleNamespace(input_tokens=3, output_tokens=2, total_tokens=5))

        llm = LLM.__new__(LLM)
        llm.model = "configured"
        llm.client = SimpleNamespace(responses=Responses())
        summary, usage = llm.summarize([{"role": "developer", "content": "summarize"},
                                        {"role": "user", "content": "turns"}])
        self.assertEqual(summary, "summary")
        self.assertEqual(usage["total_tokens"], 5)
        self.assertEqual(captured[0]["model"], "configured")
        self.assertNotIn("tools", captured[0])
        self.assertNotIn("reasoning", captured[0])


class GradioRetrievalTests(unittest.TestCase):
    def test_context_trajectory_chat_and_session_mode_are_isolated(self):
        from gradio_app import ChatSession, Sessions

        class StubRAG:
            top_k, threshold, last_error = 3, .3, ""
            corpus_errors = []
            def search(self, query, top_k=None):
                self.query = query
                return [{"document_id": "ilse", "title": "Ilse", "source": "synthetic",
                         "score": .95, "text": "Ilse speaks Dutch and English."}]

        session = ChatSession()
        session.rag_memory = StubRAG()
        session.llm = FakeLLM([call(), "answered from observation"])
        session.agent = TinyAgent(session.llm, session.memory, rag_memory=session.rag_memory)
        session.set_retrieval_mode("Model-directed retrieval")
        session.send("Does she speak any languages?")
        trajectory, context = session.snapshot()
        self.assertEqual(context["retrieval"]["selected_mode"], "Model-directed retrieval")
        self.assertEqual(context["retrieval"]["latest_model_directed_items"][0]["name"], "search_documents")
        self.assertEqual(trajectory["runs"][0]["steps"][0]["action"]["tool"], "search_documents")
        self.assertIn("Dutch", trajectory["runs"][0]["steps"][0]["observation"])
        self.assertIn("Dutch", session.history()[1]["content"])
        sessions = Sessions()
        first, second = sessions.get("a"), sessions.get("b")
        first.retrieval_mode = "Automatic RAG"
        self.assertEqual(second.retrieval_mode, "Off")
        self.assertIsNot(first.memory, second.memory)
        session.dispose()
        sessions.close()


if __name__ == "__main__":
    unittest.main()
