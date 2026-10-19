"""Offline timestamp and overwrite protection checks."""

import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from transcripts import write_transcript


class TranscriptTests(unittest.TestCase):
    def test_new_york_time_in_summer_and_winter(self):
        cases = [
            (datetime(2026, 10, 18, 21, 38, 50, tzinfo=timezone.utc),
             "study-notes-2026-10-18_17-38-50.md"),
            (datetime(2026, 1, 2, 2, 0, 0, tzinfo=timezone.utc),
             "study-notes-2026-01-01_21-00-00.md"),
        ]
        for now, expected in cases:
            with self.subTest(now=now), TemporaryDirectory() as directory:
                with patch("transcripts.datetime") as clock:
                    clock.now.return_value = now
                    path = write_transcript(Path(directory) / "study-notes.md", "Notes")
                self.assertEqual(path.name, expected)
                self.assertEqual(path.read_text(encoding="utf-8"), "Notes")

    def test_repeated_saves_preserve_all_existing_files(self):
        with TemporaryDirectory() as directory, patch("transcripts.datetime") as clock:
            clock.now.return_value = datetime(2026, 10, 18, 21, 38, 50, tzinfo=timezone.utc)
            filename = Path(directory) / "study-notes.md"
            filename.write_text("Original", encoding="utf-8")
            paths = [write_transcript(filename, content) for content in ["First", "Second", "Third"]]
            self.assertEqual([path.name for path in paths], [
                "study-notes-2026-10-18_17-38-50.md",
                "study-notes-2026-10-18_17-38-50-2.md",
                "study-notes-2026-10-18_17-38-50-3.md",
            ])
            self.assertEqual([path.read_text(encoding="utf-8") for path in paths],
                             ["First", "Second", "Third"])
            self.assertEqual(filename.read_text(encoding="utf-8"), "Original")


if __name__ == "__main__":
    unittest.main()
