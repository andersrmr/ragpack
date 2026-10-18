"""Offline checks using the real SDK with a simulated HTTP server."""

import json
import os
import unittest
from unittest.mock import Mock, patch

import httpx
from openai import OpenAI

from tinyagent import LLM, Memory, Response, TinyAgent


class TutorPromptTests(unittest.TestCase):
    def test_instructions_precede_each_turn_without_entering_memory(self):
        llm = Mock()
        llm.generate.return_value = Response(content="A hint")
        memory = Memory()
        agent = TinyAgent(llm, memory, system_prompt="Tutor instructions")
        agent.run("First question")
        agent.run("Follow-up")

        first, second = [call.args[0] for call in llm.generate.call_args_list]
        instruction = {"role": "system", "content": "Tutor instructions"}
        self.assertEqual(first, [instruction, {"role": "user", "content": "First question"}])
        self.assertEqual(second, [instruction, *memory.get_messages()[:-1]])
        self.assertEqual([m["role"] for m in memory.get_messages()],
                         ["user", "assistant", "user", "assistant"])
        memory.messages.clear()
        agent.run("New conversation")
        self.assertEqual(llm.generate.call_args.args[0], [
            instruction, {"role": "user", "content": "New conversation"},
        ])


class ProviderTests(unittest.TestCase):
    def test_missing_configuration_fails_before_creating_client(self):
        with patch.dict(os.environ, {}, clear=True), patch("tinyagent.OpenAI") as client:
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                LLM()
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                with self.assertRaisesRegex(ValueError, "TINYAGENT_MODEL"):
                    LLM()
            client.assert_not_called()

    def test_sdk_request_metadata_and_conversation_history(self):
        requests = []

        def respond(request):
            self.assertEqual(request.url.path, "/v1/chat/completions")
            self.assertEqual(request.headers["authorization"], "Bearer test-key")
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
                "model": "returned-model",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "Hello!"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            })

        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key", "TINYAGENT_MODEL": "environment-model",
        }, clear=True):
            llm = LLM()
            self.addCleanup(llm.close)
            self.assertEqual(llm.model, "environment-model")
            llm.client.close()
            llm.client = OpenAI(
                api_key="test-key", base_url="https://example.test/v1",
                http_client=httpx.Client(transport=httpx.MockTransport(respond)),
            )
            response = llm.generate([{"role": "user", "content": "Hello"}])
            self.assertEqual(response.content, "Hello!")
            self.assertEqual(response.metadata, {
                "model": "returned-model", "prompt_tokens": 10, "completion_tokens": 2,
            })

            agent = TinyAgent(llm, Memory(), system_prompt="Be a patient tutor.")
            self.assertEqual(agent.run("First question"), "Hello!")
            self.assertEqual(agent.run("Follow-up"), "Hello!")
            self.assertEqual(requests[-1], {
                "model": "environment-model",
                "messages": [
                    {"role": "system", "content": "Be a patient tutor."},
                    {"role": "user", "content": "First question"},
                    {"role": "assistant", "content": "Hello!"},
                    {"role": "user", "content": "Follow-up"},
                ],
            })
            self.assertEqual(len(agent.memory.get_messages()), 4)

            # Omitting the optional prompt retains the original course behavior.
            TinyAgent(llm, Memory()).run("Plain chat")
            self.assertEqual(requests[-1]["messages"], [
                {"role": "user", "content": "Plain chat"},
            ])

    def test_explicit_model_overrides_environment(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key", "TINYAGENT_MODEL": "environment-model",
        }, clear=True):
            llm = LLM(model="explicit-model")
            self.addCleanup(llm.close)
            self.assertEqual(llm.model, "explicit-model")


if __name__ == "__main__":
    unittest.main()
