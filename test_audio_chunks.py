import asyncio
import builtins
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import audiobooker
from audiobooker import (
    collect_valid_audio_files,
    is_valid_audio_file,
    synthesize_chunk_edge,
    synthesize_chunk_silero,
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

    def test_synthesize_chunk_edge_cleans_stale_file_on_no_audio(self) -> None:
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

            asyncio.run(run_synthesis(chunk_file))

            self.assertFalse(chunk_file.exists())
            self.assertFalse((tmp_path / "chunk_000000.mp3.part").exists())

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

    def test_synthesize_chunk_silero_writes_wav_with_explicit_format(self) -> None:
        fake_numpy = types.ModuleType("numpy")
        fake_numpy.asarray = lambda audio: audio
        fake_numpy.concatenate = lambda items: sum(items, [])

        write_call: dict[str, object] = {}

        fake_soundfile = types.ModuleType("soundfile")

        def fake_write(path: str, audio, sample_rate: int, format: Optional[str] = None) -> None:
            write_call["path"] = path
            write_call["sample_rate"] = sample_rate
            write_call["format"] = format
            with wave.open(path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(b"\x00\x00" * max(1, len(audio)))

        async def run_synthesis(chunk_file: Path) -> None:
            await synthesize_chunk_silero(
                text="Обычный текст",
                file_path=chunk_file,
                language="ru",
                speaker="baya",
                sample_rate=24000,
                put_accent=True,
                put_yo=True,
                device="cpu",
                model_id="v5_ru",
                semaphore=asyncio.Semaphore(1),
            )

        fake_soundfile.write = fake_write

        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(
            sys.modules,
            {"numpy": fake_numpy, "soundfile": fake_soundfile},
        ), patch.object(audiobooker, "_silero_apply_tts", new=lambda **kwargs: [0.0, 0.0]), patch.object(
            audiobooker,
            "_silero_inited",
            new=True,
        ), patch.object(audiobooker, "init_silero", new=lambda **kwargs: None):
            tmp_path = Path(tmp_dir)
            chunk_file = tmp_path / "chunk_000000.wav"

            asyncio.run(run_synthesis(chunk_file))

            self.assertEqual(write_call["format"], "WAV")
            self.assertTrue(str(write_call["path"]).endswith("chunk_000000.wav.part"))
            self.assertTrue(is_valid_audio_file(chunk_file, expected_ext="wav"))
            self.assertFalse((tmp_path / "chunk_000000.wav.part").exists())


if __name__ == "__main__":
    unittest.main()
