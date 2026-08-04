import argparse
import asyncio
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audiobooker import (
    AudioSegment,
    _run_ffmpeg_concat,
    _run_ffmpeg_silence,
    is_valid_audio_file,
    parse_audio_segments,
    prepare_audio_segments,
    process_single_file,
    synthesize_silence_mp3,
)


class PauseTagParserTests(unittest.TestCase):
    def test_pause_tag_preserves_speech_pause_speech_order(self) -> None:
        segments = parse_audio_segments(
            "Текст до. [[PAUSE: 5000]] Текст после.",
            chunk_size=100,
        )

        self.assertEqual(
            segments,
            [
                AudioSegment(text="Текст до."),
                AudioSegment(pause_ms=5000),
                AudioSegment(text="Текст после."),
            ],
        )

    def test_pause_tag_is_case_insensitive_and_adjacent_pauses_are_summed(self) -> None:
        segments = parse_audio_segments(
            "[[ pause : 2000 ]] \n [[PAUSE:3000]]",
            chunk_size=4,
        )

        self.assertEqual(segments, [AudioSegment(pause_ms=5000)])

    def test_leading_and_trailing_pauses_are_preserved(self) -> None:
        segments = parse_audio_segments(
            "[[PAUSE: 1000]]Слово[[PAUSE: 2000]]",
            chunk_size=100,
        )

        self.assertEqual(
            segments,
            [
                AudioSegment(pause_ms=1000),
                AudioSegment(text="Слово"),
                AudioSegment(pause_ms=2000),
            ],
        )

    def test_tag_is_parsed_before_small_text_chunks_are_created(self) -> None:
        segments = parse_audio_segments(
            "До[[PAUSE: 5000]]После",
            chunk_size=3,
        )

        self.assertEqual(
            [segment.pause_ms for segment in segments], [None, 5000, None, None]
        )
        self.assertEqual(
            "".join(segment.text or "" for segment in segments),
            "ДоПосле",
        )

    def test_unknown_double_bracket_tag_remains_speech(self) -> None:
        segments = parse_audio_segments("[[NOTE: 5]] текст", chunk_size=100)

        self.assertEqual(segments, [AudioSegment(text="[[NOTE: 5]] текст")])

    def test_invalid_pause_tags_raise_before_synthesis(self) -> None:
        cases = [
            "[[PAUSE: 0]]",
            "[[PAUSE: -1]]",
            "[[PAUSE: 1.5]]",
            "[[PAUSE: abc]]",
            "[[PAUSE: 5000]",
        ]

        for text in cases:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "PAUSE"):
                parse_audio_segments(text, chunk_size=100)

    def test_malformed_tag_is_rejected_before_text_normalization(self) -> None:
        with self.assertRaisesRegex(ValueError, "PAUSE"):
            prepare_audio_segments(
                "До паузы. [[PAUSE::5000]] После паузы.",
                chunk_size=100,
            )

    def test_per_tag_and_total_pause_limits_are_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "один тег"):
            parse_audio_segments(
                "[[PAUSE: 101]]",
                chunk_size=100,
                max_pause_ms=100,
                max_total_pause_ms=1000,
            )

        with self.assertRaisesRegex(ValueError, "Суммарная"):
            parse_audio_segments(
                "[[PAUSE: 60]][[PAUSE: 60]]",
                chunk_size=100,
                max_pause_ms=100,
                max_total_pause_ms=100,
            )


class SilenceChunkTests(unittest.TestCase):
    def test_ffmpeg_merge_reencodes_to_common_mp3_format(self) -> None:
        with patch("audiobooker.subprocess.run") as run_mock:
            _run_ffmpeg_concat(
                "ffmpeg",
                Path("list.txt"),
                Path("book.mp3"),
                loglevel="error",
            )

        command = run_mock.call_args.args[0]
        self.assertNotIn("copy", command)
        self.assertEqual(command[command.index("-ac") + 1], "1")
        self.assertEqual(command[command.index("-ar") + 1], "24000")
        self.assertEqual(command[command.index("-b:a") + 1], "48k")

    def test_ffmpeg_silence_command_uses_five_seconds_and_edge_format(self) -> None:
        with patch("audiobooker.subprocess.run") as run_mock:
            _run_ffmpeg_silence(
                "ffmpeg",
                Path("pause.mp3"),
                duration_ms=5000,
                loglevel="error",
            )

        command = run_mock.call_args.args[0]
        self.assertEqual(command[command.index("-t") + 1], "5.000")
        self.assertEqual(command[command.index("-ar") + 1], "24000")
        self.assertEqual(command[command.index("-b:a") + 1], "48k")
        self.assertEqual(command[-1], "pause.mp3")

    def test_silence_chunk_is_written_atomically(self) -> None:
        def fake_ffmpeg(
            _ffmpeg_bin: str,
            output_file: Path,
            _duration_ms: int,
            _loglevel: str,
        ) -> None:
            output_file.write_bytes(b"ID3" + b"\x00" * 32)

        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "audiobooker._run_ffmpeg_silence",
                new=fake_ffmpeg,
            ),
        ):
            output_file = Path(tmp_dir) / "pause.mp3"

            async def create_silence() -> None:
                await synthesize_silence_mp3(
                    duration_ms=5000,
                    file_path=output_file,
                    ffmpeg_path="ffmpeg",
                    semaphore=asyncio.Semaphore(1),
                )

            asyncio.run(create_silence())

            self.assertTrue(is_valid_audio_file(output_file, expected_ext="mp3"))
            self.assertFalse((Path(tmp_dir) / "pause.part.mp3").exists())

    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe"),
        "ffmpeg and ffprobe are required",
    )
    def test_real_silence_chunk_duration_is_five_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / "pause.mp3"

            async def create_silence() -> None:
                await synthesize_silence_mp3(
                    duration_ms=5000,
                    file_path=output_file,
                    ffmpeg_path=shutil.which("ffmpeg") or "ffmpeg",
                    semaphore=asyncio.Semaphore(1),
                )

            asyncio.run(create_silence())
            duration = float(
                subprocess.check_output(
                    [
                        shutil.which("ffprobe") or "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=noprint_wrappers=1:nokey=1",
                        str(output_file),
                    ],
                    text=True,
                ).strip()
            )

            self.assertGreaterEqual(duration, 4.9)
            self.assertLessEqual(duration, 5.2)


class CliPausePipelineTests(unittest.TestCase):
    def test_cli_routes_pause_to_silence_without_sending_tag_to_tts(self) -> None:
        speech_calls: list[str] = []
        pause_calls: list[int] = []
        routed_chunks: dict[str, tuple[str, str | int]] = {}

        async def fake_speech(
            text: str,
            file_path: Path,
            voice: str,
            rate: str,
            semaphore,
        ) -> None:
            speech_calls.append(text)
            routed_chunks[file_path.name] = ("speech", text)
            file_path.write_bytes(b"ID3" + b"\x00" * 32)

        async def fake_pause(
            duration_ms: int,
            file_path: Path,
            ffmpeg_path: str,
            semaphore,
            loglevel: str = "error",
        ) -> None:
            pause_calls.append(duration_ms)
            routed_chunks[file_path.name] = ("pause", duration_ms)
            file_path.write_bytes(b"ID3" + b"\x00" * 32)

        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "audiobooker.synthesize_chunk_edge",
                new=fake_speech,
            ),
            patch(
                "audiobooker.synthesize_silence_mp3",
                new=fake_pause,
            ),
        ):
            tmp_path = Path(tmp_dir)
            input_file = tmp_path / "book.txt"
            input_file.write_text(
                "До паузы. [[PAUSE: 5000]] После паузы.",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                output_dir=str(tmp_path / "output"),
                chunk_size=100,
                max_concurrent_tasks=1,
                skip_chunks=False,
                skip_merge=True,
                ffmpeg="ffmpeg",
                voice="ru-RU-SvetlanaNeural",
                rate="+0%",
            )

            asyncio.run(process_single_file(input_file, args))

        self.assertEqual(speech_calls, ["До паузы.", "После паузы."])
        self.assertEqual(pause_calls, [5000])
        self.assertNotIn("PAUSE", "".join(speech_calls))
        self.assertEqual(
            [routed_chunks[name] for name in sorted(routed_chunks)],
            [
                ("speech", "До паузы."),
                ("pause", 5000),
                ("speech", "После паузы."),
            ],
        )

    def test_cli_rejects_malformed_pause_before_synthesis(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch("audiobooker.synthesize_chunk_edge") as speech_mock,
            patch("audiobooker.synthesize_silence_mp3") as pause_mock,
        ):
            tmp_path = Path(tmp_dir)
            input_file = tmp_path / "book.txt"
            input_file.write_text(
                "До. [[PAUSE::5000]] После.",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                output_dir=str(tmp_path / "output"),
                chunk_size=100,
                max_concurrent_tasks=1,
                skip_chunks=False,
                skip_merge=True,
                ffmpeg="ffmpeg",
                voice="ru-RU-SvetlanaNeural",
                rate="+0%",
            )

            with self.assertRaisesRegex(ValueError, "PAUSE"):
                asyncio.run(process_single_file(input_file, args))

        speech_mock.assert_not_called()
        pause_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
