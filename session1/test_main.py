"""Offline terminal tests: scripted input and a fake LLM, with real agent memory."""

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import httpx
import main
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    RateLimitError,
)
from prompts import TUTOR_SYSTEM_PROMPT
from tinyagent import Response


class TerminalTests(unittest.TestCase):
    def run_chat(self, inputs, replies=()):
        llm = Mock(model="test-model")
        histories = []
        replies = iter(replies)

        def generate(messages):
            histories.append([message.copy() for message in messages])
            reply = next(replies)
            if isinstance(reply, BaseException):
                raise reply
            return Response(content=reply)

        llm.generate.side_effect = generate
        output, errors = io.StringIO(), io.StringIO()
        with (
            patch("main.LLM", return_value=llm),
            patch("builtins.input", side_effect=inputs),
            redirect_stdout(output),
            redirect_stderr(errors),
        ):
            result = main.main()
        llm.close.assert_called_once_with()
        return result, histories, output.getvalue(), errors.getvalue()

    def test_save_exports_markdown_and_keeps_chat_history(self):
        answer = "**Mémoire** stores turns.\n\n```python\nx = []\n```"
        with TemporaryDirectory() as directory:
            path = Path(directory) / "study notes.md"
            result, histories, output, errors = self.run_chat(
                ["Explain memory", f"/save {path}", "Continue", "/quit"],
                [answer, "More detail"],
            )
            self.assertEqual(path.read_text(encoding="utf-8"),
                             "# TinyAgent conversation\n\n## You\n\nExplain memory"
                             f"\n\n## TinyAgent\n\n{answer}\n")
        self.assertEqual(result, 0)
        self.assertEqual(len(histories), 2)
        self.assertEqual(histories[1][2], {"role": "assistant", "content": answer})
        self.assertIn("Transcript saved", output)
        self.assertEqual(errors, "")

    def test_save_missing_filename_and_empty_history_make_no_requests(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "empty.md"
            _, histories, output, errors = self.run_chat(
                ["/save", f"/save {path}", "Hello", "/clear", f"/save {path}", "/quit"],
                ["Hi"],
            )
            self.assertFalse(path.exists())
        self.assertEqual(len(histories), 1)
        self.assertIn("Usage: /save filename.md", output)
        self.assertEqual(output.count("No conversation to save"), 2)
        self.assertEqual(errors, "")

    def test_save_preserves_existing_file_and_can_retry(self):
        with TemporaryDirectory() as directory:
            existing = Path(directory) / "existing.md"
            existing.write_text("My existing notes", encoding="utf-8")
            new = Path(directory) / "new.md"
            _, histories, output, errors = self.run_chat(
                ["Hello", f"/save {existing}", f"/save {new}", "Continue", "/quit"],
                ["Hi", "More detail"],
            )
            self.assertEqual(existing.read_text(encoding="utf-8"), "My existing notes")
            self.assertIn("## TinyAgent\n\nHi", new.read_text(encoding="utf-8"))
        self.assertEqual(len(histories[1]), 4)
        self.assertIn("File already exists", errors)
        self.assertIn("Transcript saved", output)

    def test_save_file_error_does_not_end_chat_or_lose_history(self):
        with patch("main.Path.open", side_effect=PermissionError("Permission denied")):
            result, histories, output, errors = self.run_chat(
                ["Hello", "/save notes.md", "Continue", "/quit"],
                ["Hi", "More detail"],
            )
        self.assertEqual(result, 0)
        self.assertEqual(len(histories[1]), 4)
        self.assertIn("Could not save transcript", errors)
        self.assertIn("More detail", output)

    def test_history_blank_input_and_quit(self):
        result, histories, output, errors = self.run_chat(
            ["  ", "Hello", "Follow-up", "/quit"],
            ["Hi", "More detail"],
        )
        self.assertEqual(result, 0)
        self.assertEqual(
            histories,
            [
                [
                    {"role": "system", "content": TUTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": "Hello"},
                ],
                [
                    {"role": "system", "content": TUTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                    {"role": "user", "content": "Follow-up"},
                ],
            ],
        )
        self.assertIn("Model: test-model", output)
        self.assertIn("TinyAgent:\nMore detail", output)
        self.assertEqual(errors, "")

    def test_clear_removes_history_without_calling_model(self):
        _, histories, output, _ = self.run_chat(
            ["Hello", "/clear", "Fresh start", "/quit"],
            ["Hi", "Welcome"],
        )
        self.assertEqual(histories[-1], [
            {"role": "system", "content": TUTOR_SYSTEM_PROMPT},
            {"role": "user", "content": "Fresh start"},
        ])
        self.assertIn("Conversation cleared", output)

    def test_errors_keep_completed_history_and_allow_retry(self):
        request = httpx.Request("POST", "https://example.test/v1/chat/completions")
        failures = [APIConnectionError(request=request), RuntimeError("No text")]
        for error_type, status in [
            (AuthenticationError, 401),
            (RateLimitError, 429),
            (APIStatusError, 500),
        ]:
            failures.append(
                error_type(
                    "Simulated failure",
                    response=httpx.Response(status, request=request),
                    body=None,
                )
            )
        for failure in failures:
            with self.subTest(error=type(failure).__name__):
                result, histories, output, errors = self.run_chat(
                    ["First", "Try this", "Try this", "/quit"],
                    ["Answer", failure, "Recovered"],
                )
                self.assertEqual(result, 0)
                self.assertEqual(histories[1], histories[2])
                self.assertEqual(len(histories[2]), 4)
                self.assertEqual(histories[2][0], {
                    "role": "system", "content": TUTOR_SYSTEM_PROMPT,
                })
                self.assertIn("Previous conversation kept", errors)
                self.assertIn("TinyAgent:\nRecovered", output)

    def test_markdown_is_rendered_but_original_text_stays_in_memory(self):
        answer = "**Memory** stores turns.\n\n```python\nprint('hello')\n```"
        _, histories, output, errors = self.run_chat(
            ["Explain memory", "Continue", "/quit"],
            [answer, "Done"],
        )
        self.assertIn("Memory stores turns.", output)
        self.assertIn("print('hello')", output)
        self.assertNotIn("**Memory**", output)
        self.assertNotIn("```", output)
        self.assertEqual(histories[1][2], {"role": "assistant", "content": answer})
        self.assertEqual(errors, "")

    def test_eof_and_ctrl_c_close_client(self):
        for interruption in [EOFError(), KeyboardInterrupt()]:
            with self.subTest(interruption=type(interruption).__name__):
                result, _, output, _ = self.run_chat([interruption])
                self.assertEqual(result, 0)
                self.assertIn("Goodbye!", output)

    def test_ctrl_c_during_request_closes_client(self):
        result, _, _, _ = self.run_chat(["Hello"], [KeyboardInterrupt()])
        self.assertEqual(result, 0)

    def test_unexpected_error_still_closes_client(self):
        llm = Mock(model="test-model")
        llm.generate.side_effect = LookupError("Unexpected problem")
        with (
            patch("main.LLM", return_value=llm),
            patch("builtins.input", return_value="Hello"),
            redirect_stdout(io.StringIO()),
            self.assertRaises(LookupError),
        ):
            main.main()
        llm.close.assert_called_once_with()

    def test_missing_configuration_exits_with_message(self):
        with (
            patch("main.LLM", side_effect=ValueError("Set OPENAI_API_KEY")),
            redirect_stderr(io.StringIO()) as errors,
        ):
            self.assertEqual(main.main(), 1)
        self.assertIn("Configuration error: Set OPENAI_API_KEY", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
