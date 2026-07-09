import tempfile
from pathlib import Path
from unittest.mock import patch

from download_delivery import (
    create_download_entry,
    cleanup_expired_downloads,
    get_download_ttl_seconds,
    has_active_downloads,
    resolve_download_token,
)


def test_create_and_resolve_download_entry() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        source = root / "book.mp3"
        source.write_bytes(b"ID3" + b"\x00" * 32)

        url, token_dir = create_download_entry(
            source,
            filename="book.mp3",
            download_dir=root / "downloads",
            ttl_seconds=900,
            public_base_url="http://example.org:43561",
        )

        token = url.rsplit("/", 1)[-1]
        metadata, file_path = resolve_download_token(
            token,
            download_dir=root / "downloads",
        )

        assert url == f"http://example.org:43561/d/{token}"
        assert token_dir.name == token
        assert metadata["filename"] == "book.mp3"
        assert metadata["media_type"] == "audio/mpeg"
        assert file_path.read_bytes() == source.read_bytes()


def test_expired_download_entry_is_removed() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        source = root / "book.mp3"
        source.write_bytes(b"ID3" + b"\x00" * 32)

        url, token_dir = create_download_entry(
            source,
            filename="book.mp3",
            download_dir=root / "downloads",
            ttl_seconds=1,
            public_base_url="http://example.org:43561",
        )
        token = url.rsplit("/", 1)[-1]

        cleanup_expired_downloads(root / "downloads", now=9999999999)

        assert not token_dir.exists()
        try:
            resolve_download_token(token, download_dir=root / "downloads")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("expired token should not resolve")


def test_download_ttl_minutes_override_seconds() -> None:
    with patch.dict(
        "os.environ",
        {
            "DOWNLOAD_TTL_SECONDS": "900",
            "DOWNLOAD_TTL_MINUTES": "2.5",
        },
        clear=False,
    ):
        assert get_download_ttl_seconds() == 150


def test_has_active_downloads_reflects_expiration() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        source = root / "book.mp3"
        source.write_bytes(b"ID3" + b"\x00" * 32)

        create_download_entry(
            source,
            filename="book.mp3",
            download_dir=root / "downloads",
            ttl_seconds=60,
            public_base_url="http://example.org:43561",
        )

        assert has_active_downloads(root / "downloads", now=1)
        assert not has_active_downloads(root / "downloads", now=9999999999)
