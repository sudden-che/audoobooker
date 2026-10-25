import argparse
import asyncio
import sys
import tarfile
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import pytest

import audiobooker
import tg_audiobooker
import web_audiobooker


def test_pdf_text_is_read_in_page_order() -> None:
    class Page:
        def __init__(self, text: str | None) -> None:
            self.text = text

        def extract_text(self) -> str | None:
            return self.text

    class Reader:
        def __init__(self, path: str) -> None:
            assert path.endswith("book.pdf")
            self.is_encrypted = False
            self.pages = [Page("First page"), Page(None), Page("Second page")]

    fake_pypdf = types.ModuleType("pypdf")
    fake_pypdf.PdfReader = Reader
    with patch.dict(sys.modules, {"pypdf": fake_pypdf}):
        assert audiobooker.extract_pdf_text(Path("book.pdf")) == (
            "First page\n\n\n\nSecond page"
        )


def test_pdf_without_text_reports_need_for_ocr() -> None:
    class Reader:
        is_encrypted = False
        pages = [types.SimpleNamespace(extract_text=lambda: " \n ")]

        def __init__(self, _path: str) -> None:
            pass

    fake_pypdf = types.ModuleType("pypdf")
    fake_pypdf.PdfReader = Reader
    with patch.dict(sys.modules, {"pypdf": fake_pypdf}):
        with pytest.raises(ValueError, match="OCR"):
            audiobooker.extract_pdf_text(Path("scan.pdf"))


def test_cli_synthesizes_text_from_pdf() -> None:
    speech_calls: list[str] = []

    async def fake_speech(text, file_path, voice, rate, semaphore) -> None:
        speech_calls.append(text)
        file_path.write_bytes(b"ID3" + b"\x00" * 32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        source = Path(tmp_dir) / "book.pdf"
        source.write_bytes(b"%PDF-1.4")
        args = argparse.Namespace(
            output_dir=str(Path(tmp_dir) / "output"),
            chunk_size=100,
            max_concurrent_tasks=1,
            skip_chunks=False,
            skip_merge=True,
            ffmpeg="ffmpeg",
            voice="en-US-AriaNeural",
            rate="+0%",
        )
        with patch("audiobooker.extract_pdf_text", return_value="A PDF chapter."), patch(
            "audiobooker.synthesize_chunk_edge", new=fake_speech
        ):
            asyncio.run(audiobooker.process_single_file(source, args))

    assert speech_calls == ["A PDF chapter."]


def test_web_packages_audio_from_pdf() -> None:
    async def fake_speech(text, file_path, voice, rate, semaphore) -> None:
        file_path.write_bytes(b"ID3" + b"\x00" * 32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        source = Path(tmp_dir) / "uploaded.pdf"
        source.write_bytes(b"%PDF-1.4")
        with patch("web_audiobooker.extract_pdf_text", return_value="A PDF chapter."), patch(
            "web_audiobooker.synthesize_chunk_edge", new=fake_speech
        ):
            result, result_kind = asyncio.run(
                web_audiobooker.process_uploaded_file(
                    input_file=source,
                    output_dir=Path(tmp_dir),
                    voice="en-US-AriaNeural",
                    speed="+0%",
                    chunk_size=100,
                    max_concurrent_tasks=1,
                    skip_chunks=False,
                    merge_chunks=False,
                    ffmpeg_path="ffmpeg",
                    source_name="book.pdf",
                )
            )
        with tarfile.open(result) as archive:
            assert [member.name for member in archive.getmembers()] == [
                "book_chunk_000000.mp3"
            ]
        assert result_kind == "tar"


def test_telegram_downloads_and_reads_pdf() -> None:
    class TelegramFile:
        async def download_to_drive(self, path: Path) -> None:
            path.write_bytes(b"%PDF-1.4")

    class Bot:
        async def get_file(self, _file_id: str) -> TelegramFile:
            return TelegramFile()

    with tempfile.TemporaryDirectory() as tmp_dir:
        context = types.SimpleNamespace(bot=Bot())
        doc = types.SimpleNamespace(file_id="pdf-file")
        with patch("tg_audiobooker.extract_pdf_text", return_value="A PDF chapter."):
            text = asyncio.run(
                tg_audiobooker._load_document_text(
                    context=context,
                    doc=doc,
                    suffix=".pdf",
                    work_dir=Path(tmp_dir),
                )
            )

        assert text == "A PDF chapter."
