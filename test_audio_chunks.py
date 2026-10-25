import asyncio
import argparse
import builtins
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import AsyncMock, patch

import audiobooker
from audiobooker import (
    collect_valid_audio_files,
    is_valid_audio_file,
    process_single_file,
    synthesize_chunk_edge,
)


def _write_valid_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(b"\x00\x00" * 16)


def _install_fake_edge_tts(save_impl):
    edge_module = types.ModuleType("edge_tts")
    exceptions_module = types.ModuleType("edge_tts.exceptions")

    class FakeNoAudioReceived(Exception):
        pass

    class FakeCommunicate:
        def __init__(self, text: str, voice: str, rate: str) -> None:
            self.text = text
            self.voice = voice
            self.rate = rate

        async def save(self, path: str) -> None:
            await save_impl(Path(path))

    edge_module.Communicate = FakeCommunicate
    edge_module.exceptions = exceptions_module
    exceptions_module.NoAudioReceived = FakeNoAudioReceived

    patched_modules = {
        "edge_tts": edge_module,
        "edge_tts.exceptions": exceptions_module,
    }
    return FakeNoAudioReceived, patch.dict(sys.modules, patched_modules)


class AudioChunkTests(unittest.TestCase):
    def test_collect_valid_audio_files_filters_broken_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            valid_mp3 = tmp_path / "good.mp3"
            valid_mp3.write_bytes(b"ID3" + b"\x00" * 32)

            broken_mp3 = tmp_path / "broken.mp3"
            broken_mp3.write_bytes(b"not-an-mp3")

            valid_wav = tmp_path / "good.wav"
            _write_valid_wav(valid_wav)

            actual = collect_valid_audio_files(
                [valid_mp3, broken_mp3, valid_wav],
                remove_invalid=True,
            )

            self.assertEqual(actual, [valid_mp3, valid_wav])
            self.assertFalse(broken_mp3.exists())

    def test_synthesize_chunk_edge_rejects_invalid_mp3(self) -> None:
        async def fake_save(path: Path) -> None:
            path.write_bytes(b"not-an-mp3")

        async def run_synthesis(chunk_file: Path) -> None:
            await synthesize_chunk_edge(
                text="normal text",
                file_path=chunk_file,
                voice="ru-RU-SvetlanaNeural",
                rate="+0%",
                semaphore=asyncio.Semaphore(1),
            )

        _, patched_modules = _install_fake_edge_tts(fake_save)
        with tempfile.TemporaryDirectory() as tmp_dir, patched_modules:
            tmp_path = Path(tmp_dir)
            chunk_file = tmp_path / "chunk_000000.mp3"

            with self.assertRaisesRegex(RuntimeError, "invalid MP3 chunk"):
                asyncio.run(run_synthesis(chunk_file))

            self.assertFalse(chunk_file.exists())
            self.assertFalse((tmp_path / "chunk_000000.mp3.part").exists())

    def test_synthesize_chunk_edge_fails_and_cleans_stale_file_on_no_audio(self) -> None:
        async def fake_save(path: Path) -> None:
            path.write_bytes(b"")
            raise no_audio_received()

        async def run_synthesis(chunk_file: Path) -> None:
            await synthesize_chunk_edge(
                text="normal text",
                file_path=chunk_file,
                voice="ru-RU-SvetlanaNeural",
                rate="+0%",
                semaphore=asyncio.Semaphore(1),
            )

        no_audio_received, patched_modules = _install_fake_edge_tts(fake_save)
        with tempfile.TemporaryDirectory() as tmp_dir, patched_modules:
            tmp_path = Path(tmp_dir)
            chunk_file = tmp_path / "chunk_000000.mp3"
            chunk_file.write_bytes(b"ID3" + b"\x00" * 32)

            with self.assertRaisesRegex(RuntimeError, "returned no audio"):
                asyncio.run(run_synthesis(chunk_file))

            self.assertFalse(chunk_file.exists())
            self.assertFalse((tmp_path / "chunk_000000.mp3.part").exists())

    def test_no_audio_aborts_book_before_merge(self) -> None:
        async def fake_save(_path: Path) -> None:
            raise no_audio_received()

        no_audio_received, patched_modules = _install_fake_edge_tts(fake_save)
        with tempfile.TemporaryDirectory() as tmp_dir, patched_modules, patch(
            "audiobooker.merge_audio_chunks", new_callable=AsyncMock
        ) as merge_mock:
            source = Path(tmp_dir) / "book.txt"
            source.write_text("A sentence to read.", encoding="utf-8")
            args = argparse.Namespace(
                output_dir=str(Path(tmp_dir) / "output"),
                chunk_size=100,
                max_concurrent_tasks=1,
                skip_chunks=False,
                skip_merge=False,
                ffmpeg="ffmpeg",
                voice="en-US-AriaNeural",
                rate="+0%",
            )

            with self.assertRaisesRegex(RuntimeError, "returned no audio"):
                asyncio.run(process_single_file(source, args))

            merge_mock.assert_not_awaited()

    def test_synthesize_chunk_edge_skips_non_speech_without_importing_edge(self) -> None:
        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.startswith("edge_tts"):
                raise AssertionError("edge_tts should not be imported for punctuation-only text")
            return real_import(name, globals, locals, fromlist, level)

        async def run_synthesis(chunk_file: Path) -> None:
            await synthesize_chunk_edge(
                text="!!! ??? ...",
                file_path=chunk_file,
                voice="ru-RU-SvetlanaNeural",
                rate="+0%",
                semaphore=asyncio.Semaphore(1),
            )

        with tempfile.TemporaryDirectory() as tmp_dir, patch(
            "builtins.__import__",
            side_effect=guarded_import,
        ):
            tmp_path = Path(tmp_dir)
            chunk_file = tmp_path / "chunk_000000.mp3"

            asyncio.run(run_synthesis(chunk_file))

            self.assertFalse(chunk_file.exists())
            self.assertFalse((tmp_path / "chunk_000000.mp3.part").exists())

    def test_synthesize_chunk_edge_retries_transient_network_error(self) -> None:
        attempts = {"count": 0}

        async def fake_save(path: Path) -> None:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise OSError(
                    "Cannot connect to host speech.platform.bing.com:443 "
                    "ssl:<ssl.SSLContext object at 0x0> [None]"
                )
            path.write_bytes(b"ID3" + b"\x00" * 32)

        async def run_synthesis(chunk_file: Path) -> None:
            await synthesize_chunk_edge(
                text="normal text",
                file_path=chunk_file,
                voice="ru-RU-SvetlanaNeural",
                rate="+0%",
                semaphore=asyncio.Semaphore(1),
            )

        _, patched_modules = _install_fake_edge_tts(fake_save)
        with tempfile.TemporaryDirectory() as tmp_dir, patched_modules, patch.object(
            audiobooker,
            "EDGE_TTS_MAX_RETRIES",
            new=2,
        ), patch.object(
            audiobooker,
            "EDGE_TTS_RETRY_BASE_DELAY",
            new=0.0,
        ), patch(
            "audiobooker.asyncio.sleep",
            new=AsyncMock(),
        ) as sleep_mock:
            tmp_path = Path(tmp_dir)
            chunk_file = tmp_path / "chunk_000000.mp3"

            asyncio.run(run_synthesis(chunk_file))

            self.assertEqual(attempts["count"], 2)
            self.assertTrue(is_valid_audio_file(chunk_file, expected_ext="mp3"))
            self.assertEqual(sleep_mock.await_count, 1)

    def test_synthesize_chunk_edge_raises_after_retry_exhausted(self) -> None:
        attempts = {"count": 0}

        async def fake_save(path: Path) -> None:
            attempts["count"] += 1
            raise OSError(
                "Cannot connect to host speech.platform.bing.com:443 "
                "ssl:<ssl.SSLContext object at 0x0> [None]"
            )

        async def run_synthesis(chunk_file: Path) -> None:
            await synthesize_chunk_edge(
                text="normal text",
                file_path=chunk_file,
                voice="ru-RU-SvetlanaNeural",
                rate="+0%",
                semaphore=asyncio.Semaphore(1),
            )

        _, patched_modules = _install_fake_edge_tts(fake_save)
        with tempfile.TemporaryDirectory() as tmp_dir, patched_modules, patch.object(
            audiobooker,
            "EDGE_TTS_MAX_RETRIES",
            new=2,
        ), patch.object(
            audiobooker,
            "EDGE_TTS_RETRY_BASE_DELAY",
            new=0.0,
        ), patch(
            "audiobooker.asyncio.sleep",
            new=AsyncMock(),
        ) as sleep_mock:
            tmp_path = Path(tmp_dir)
            chunk_file = tmp_path / "chunk_000000.mp3"

            with self.assertRaisesRegex(OSError, "Cannot connect to host"):
                asyncio.run(run_synthesis(chunk_file))

            self.assertEqual(attempts["count"], 3)
            self.assertFalse(chunk_file.exists())
            self.assertEqual(sleep_mock.await_count, 2)

if __name__ == "__main__":
    unittest.main()
