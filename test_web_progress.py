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
                    "engine": "edge",
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

    def test_process_uploaded_file_uses_source_filename_for_tar_and_chunks(self) -> None:
        async def fake_synthesize_chunk_edge(
            text: str,
            file_path: Path,
            voice: str,
            rate: str,
            semaphore,
        ) -> None:
            file_path.write_bytes(b"ID3" + b"\x00" * 32)

        source_name = (
            "Очень длинное название книги на русском языке для проверки архива и чанков.txt"
        )
        expected_base = build_output_basename(source_name)

        with tempfile.TemporaryDirectory() as tmp_dir, patch(
            "web_audiobooker.synthesize_chunk_edge",
            new=fake_synthesize_chunk_edge,
        ):
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "uploaded.txt"
            input_path.write_text("Первая часть.\n\nВторая часть.", encoding="utf-8")

            tar_path, result_kind = asyncio.run(
                web_audiobooker.process_uploaded_file(
                    input_file=input_path,
                    output_dir=tmp_path,
                    engine="edge",
                    voice="ru-RU-SvetlanaNeural",
                    speed="+0%",
                    silero_speaker="baya",
                    sample_rate=48000,
                    put_accent=True,
                    put_yo=True,
                    device="cpu",
                    silero_model_id="v5_ru",
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


if __name__ == "__main__":
    unittest.main()
