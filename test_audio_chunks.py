import asyncio
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from audiobooker import collect_valid_audio_files, synthesize_chunk_edge


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

        _, patched_modules = _install_fake_edge_tts(fake_save)
        with tempfile.TemporaryDirectory() as tmp_dir, patched_modules:
            tmp_path = Path(tmp_dir)
            chunk_file = tmp_path / "chunk_000000.mp3"

            with self.assertRaisesRegex(RuntimeError, "invalid MP3 chunk"):
                asyncio.run(
                    synthesize_chunk_edge(
                        text="normal text",
                        file_path=chunk_file,
                        voice="ru-RU-SvetlanaNeural",
                        rate="+0%",
                        semaphore=asyncio.Semaphore(1),
                    )
                )

            self.assertFalse(chunk_file.exists())
            self.assertFalse((tmp_path / "chunk_000000.mp3.part").exists())

    def test_synthesize_chunk_edge_cleans_stale_file_on_no_audio(self) -> None:
        async def fake_save(path: Path) -> None:
            path.write_bytes(b"")
            raise no_audio_received()

        no_audio_received, patched_modules = _install_fake_edge_tts(fake_save)
        with tempfile.TemporaryDirectory() as tmp_dir, patched_modules:
            tmp_path = Path(tmp_dir)
            chunk_file = tmp_path / "chunk_000000.mp3"
            chunk_file.write_bytes(b"ID3" + b"\x00" * 32)

            asyncio.run(
                synthesize_chunk_edge(
                    text="normal text",
                    file_path=chunk_file,
                    voice="ru-RU-SvetlanaNeural",
                    rate="+0%",
                    semaphore=asyncio.Semaphore(1),
                )
            )

            self.assertFalse(chunk_file.exists())
            self.assertFalse((tmp_path / "chunk_000000.mp3.part").exists())


if __name__ == "__main__":
    unittest.main()
