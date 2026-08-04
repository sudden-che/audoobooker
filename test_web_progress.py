import asyncio
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import web_audiobooker
from audiobooker import build_output_basename


class _DummyTask:
    def done(self) -> bool:
        return True


def _drop_task(coro):
    coro.close()
    return _DummyTask()


class WebProgressTests(unittest.TestCase):
    def setUp(self) -> None:
        web_audiobooker.WEB_JOBS.clear()
        self.client = TestClient(web_audiobooker.app)

    def tearDown(self) -> None:
        web_audiobooker.WEB_JOBS.clear()

    def test_index_contains_progress_panel(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn('id="progress-panel"', response.text)
        self.assertIn('onsubmit="submitForm(event)"', response.text)

    def test_start_creates_job_and_status_endpoint(self) -> None:
        with patch("web_audiobooker.asyncio.create_task", side_effect=_drop_task):
            response = self.client.post(
                "/start",
                data={
                    "merge_chunks": "true",
                },
                files={
                    "book_file": (
                        "book.txt",
                        "Привет, мир.".encode("utf-8"),
                        "text/plain",
                    )
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("job_id", payload)

        status_response = self.client.get(f"/status/{payload['job_id']}")
        self.assertEqual(status_response.status_code, 200)
        status_payload = status_response.json()
        self.assertEqual(status_payload["status"], "queued")
        self.assertEqual(status_payload["message"], "Задача создана")

    def test_process_uploaded_file_uses_source_filename_for_tar_and_chunks(
        self,
    ) -> None:
        async def fake_synthesize_chunk_edge(
            text: str,
            file_path: Path,
            voice: str,
            rate: str,
            semaphore,
        ) -> None:
            file_path.write_bytes(b"ID3" + b"\x00" * 32)

        source_name = "Очень длинное название книги на русском языке для проверки архива и чанков.txt"
        expected_base = build_output_basename(source_name)

        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "web_audiobooker.synthesize_chunk_edge",
                new=fake_synthesize_chunk_edge,
            ),
        ):
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "uploaded.txt"
            input_path.write_text("Первая часть.\n\nВторая часть.", encoding="utf-8")

            tar_path, result_kind = asyncio.run(
                web_audiobooker.process_uploaded_file(
                    input_file=input_path,
                    output_dir=tmp_path,
                    voice="ru-RU-SvetlanaNeural",
                    speed="+0%",
                    chunk_size=12,
                    max_concurrent_tasks=1,
                    skip_chunks=False,
                    merge_chunks=False,
                    ffmpeg_path="ffmpeg",
                    source_name=source_name,
                )
            )

            self.assertEqual(result_kind, "tar")
            self.assertEqual(tar_path.name, f"{expected_base}_parts.tar")
            self.assertLessEqual(len(expected_base), 48)
            self.assertTrue(expected_base.isascii())

            with tarfile.open(tar_path, "r") as archive:
                names = sorted(member.name for member in archive.getmembers())

            self.assertTrue(names)
            self.assertTrue(
                all(name.startswith(f"{expected_base}_chunk_") for name in names)
            )
            self.assertTrue(all(name.endswith(".mp3") for name in names))

    def test_process_uploaded_file_keeps_pause_in_order(self) -> None:
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
                "web_audiobooker.synthesize_chunk_edge",
                new=fake_speech,
            ),
            patch(
                "web_audiobooker.synthesize_silence_mp3",
                new=fake_pause,
            ),
        ):
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "uploaded.txt"
            input_path.write_text(
                "До паузы. [[PAUSE: 5000]] После паузы.",
                encoding="utf-8",
            )

            tar_path, result_kind = asyncio.run(
                web_audiobooker.process_uploaded_file(
                    input_file=input_path,
                    output_dir=tmp_path,
                    voice="ru-RU-SvetlanaNeural",
                    speed="+0%",
                    chunk_size=100,
                    max_concurrent_tasks=1,
                    skip_chunks=False,
                    merge_chunks=False,
                    ffmpeg_path="ffmpeg",
                    source_name="book.txt",
                )
            )

            with tarfile.open(tar_path, "r") as archive:
                names = sorted(member.name for member in archive.getmembers())

        self.assertEqual(result_kind, "tar")
        self.assertEqual(speech_calls, ["До паузы.", "После паузы."])
        self.assertEqual(pause_calls, [5000])
        self.assertEqual(len(names), 3)
        self.assertEqual(
            [routed_chunks[name] for name in sorted(routed_chunks)],
            [
                ("speech", "До паузы."),
                ("pause", 5000),
                ("speech", "После паузы."),
            ],
        )

    def test_process_uploaded_file_rejects_malformed_pause_before_synthesis(
        self,
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch("web_audiobooker.synthesize_chunk_edge") as speech_mock,
            patch("web_audiobooker.synthesize_silence_mp3") as pause_mock,
        ):
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "uploaded.txt"
            input_path.write_text(
                "До. [[PAUSE::5000]] После.",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "PAUSE"):
                asyncio.run(
                    web_audiobooker.process_uploaded_file(
                        input_file=input_path,
                        output_dir=tmp_path,
                        voice="ru-RU-SvetlanaNeural",
                        speed="+0%",
                        chunk_size=100,
                        max_concurrent_tasks=1,
                        skip_chunks=False,
                        merge_chunks=False,
                        ffmpeg_path="ffmpeg",
                        source_name="book.txt",
                    )
                )

        speech_mock.assert_not_called()
        pause_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
