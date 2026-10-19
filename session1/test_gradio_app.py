"""Offline browser-session tests. No OpenAI requests or listening server."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
from openai import APIConnectionError
from prompts import TUTOR_SYSTEM_PROMPT
from tinyagent import Response

if importlib.util.find_spec("gradio") is not None:
    import gradio_app
else:
    gradio_app = None


@unittest.skipIf(gradio_app is None, "Install requirements-gradio.txt for browser tests")
class BrowserTests(unittest.TestCase):
    def setUp(self):
        self.llm = Mock(model="test-model")
        self.requests = []

        def generate(messages):
            self.requests.append([message.copy() for message in messages])
            return Response(content="**Mémoire**\n\n```python\nx = []\n```")

        self.llm.generate.side_effect = generate
        self.factory = patch("gradio_app.LLM", return_value=self.llm).start()
        self.addCleanup(patch.stopall)
        self.session = gradio_app.ChatSession()
        self.addCleanup(self.session.dispose)

    def test_multiturn_tutor_and_clear(self):
        self.session.send("First")
        self.session.send("Follow-up")
        self.assertEqual(len(self.requests[1]), 4)
        self.assertEqual(self.requests[1][0], {
            "role": "system", "content": TUTOR_SYSTEM_PROMPT,
        })
        self.assertEqual(len(self.session.history()), 4)
        self.assertEqual(self.session.history()[0]["role"], "user")
        self.session.clear()
        self.session.send("Fresh start")
        self.assertEqual(len(self.requests[-1]), 2)
        self.assertEqual(self.requests[-1][0]["role"], "system")
        self.assertEqual(self.requests[-1][1]["content"], "Fresh start")

    def test_failed_request_keeps_input_and_completed_turns(self):
        self.session.send("First")
        previous = self.session.history()
        self.llm.generate.side_effect = APIConnectionError(
            request=httpx.Request("POST", "https://example.test"),
        )
        remaining, status = self.session.send("Retry me")
        self.assertEqual(remaining, "Retry me")
        self.assertIn("Previous conversation kept", status)
        self.assertEqual(self.session.history(), previous)
        self.llm.generate.side_effect = None
        self.llm.generate.return_value = Response(content="Recovered")
        self.assertEqual(self.session.send("Retry me")[0], "")

    def test_configuration_error_and_blank_input(self):
        self.session.send("  ")
        self.factory.assert_not_called()
        self.factory.side_effect = ValueError("Set OPENAI_API_KEY")
        remaining, status = self.session.send("Hello")
        self.assertEqual(remaining, "Hello")
        self.assertIn("Set OPENAI_API_KEY", status)
        self.assertEqual(self.session.history(), [])

    def test_save_after_end_preserves_markdown_and_downloads_are_unique(self):
        self.assertEqual(self.session.save()[0], None)
        self.session.send("Explain")
        history = self.session.history()
        self.session.end()
        first, _ = self.session.save()
        second, _ = self.session.save()
        self.assertNotEqual(first, second)
        transcript = Path(first).read_text(encoding="utf-8")
        self.assertIn("## You\n\nExplain", transcript)
        self.assertIn("**Mémoire**\n\n```python\nx = []\n```", transcript)
        self.assertNotIn(TUTOR_SYSTEM_PROMPT, transcript)
        self.assertEqual(self.session.history(), history)
        self.session.dispose()
        self.assertFalse(Path(first).exists())
        self.llm.close.assert_called_once_with()

    def test_end_blocks_new_messages_and_is_idempotent(self):
        self.session.send("Hello")
        self.session.end()
        self.session.end()
        _, status = self.session.send("Another question")
        self.assertIn("Session ended", status)
        self.llm.generate.assert_called_once()
        self.llm.close.assert_called_once()

    def test_export_error_keeps_history(self):
        self.session.send("Hello")
        previous = self.session.history()
        with patch("gradio_app.Path.write_text", side_effect=OSError("disk full")):
            path, status = self.session.save()
        self.assertIsNone(path)
        self.assertIn("Could not prepare", status)
        self.assertEqual(self.session.history(), previous)

    def test_sessions_are_separate_and_cleanup_closes_each_client(self):
        sessions = gradio_app.Sessions()
        self.addCleanup(sessions.close)
        other_llm = Mock(model="other-model")
        other_llm.generate.return_value = Response(content="Other answer")
        self.factory.side_effect = [self.llm, other_llm]
        first = sessions.get("tab-one")
        second = sessions.get("tab-two")
        first.send("Private question")
        self.assertEqual(second.history(), [])
        second.send("Different question")
        sessions.remove("tab-one")
        self.llm.close.assert_called_once()
        other_llm.close.assert_not_called()
        self.assertEqual(len(second.history()), 2)
        sessions.close()
        other_llm.close.assert_called_once()

    def test_gradio_builds_without_opening_client_and_registers_controls(self):
        app, sessions = gradio_app.build_app()
        self.addCleanup(sessions.close)
        config = app.get_config_file()
        labels = [component["props"].get("value") for component in config["components"]]
        for label in ["Send", "Clear conversation", "Save transcript", "End session"]:
            self.assertIn(label, labels)
        self.assertTrue(config["dependencies"])
        self.factory.assert_not_called()

    def test_server_shutdown_closes_sessions_even_after_launch_error(self):
        app, sessions = Mock(), Mock()
        app.launch.side_effect = RuntimeError("Server failed")
        with patch("gradio_app.build_app", return_value=(app, sessions)):
            with self.assertRaises(RuntimeError):
                gradio_app.main()
        sessions.close.assert_called_once()
        app.close.assert_called_once()

    def test_browser_event_callbacks_and_chatbot_message_format(self):
        app, sessions = gradio_app.build_app()
        self.addCleanup(sessions.close)
        callbacks = {event.fn.__name__: event.fn for event in app.fns.values()}
        request = gradio_app.gr.Request(session_hash="browser-test")
        _, key = callbacks["load"](request)
        self.assertEqual(key, "browser-test")
        remaining, history, status, _, key = callbacks["respond"]("Hello", request)
        self.assertEqual(remaining, "")
        self.assertIn("test-model", status)
        chatbot = next(block for block in app.blocks.values()
                       if isinstance(block, gradio_app.gr.Chatbot))
        self.assertIsNotNone(chatbot.postprocess(history))
        path, _, _ = callbacks["save"](request)
        self.assertTrue(Path(path).exists())
        callbacks["end"](request)
        self.llm.close.assert_called_once()
        callbacks["unload"](request)
        self.assertFalse(Path(path).exists())
        self.assertNotIn(key, sessions.items)


if __name__ == "__main__":
    unittest.main()
