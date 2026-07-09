#!/usr/bin/env python3
"""Shared temporary download storage for Telegram link delivery."""

from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import tarfile
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse


TOKEN_RE = re.compile(r"^[a-f0-9]{32}$")
METADATA_FILENAME = "metadata.json"
DOWNLOAD_SERVER_LOCK = threading.Lock()
DOWNLOAD_SERVER: ThreadingHTTPServer | None = None
DOWNLOAD_SERVER_THREAD: threading.Thread | None = None
DOWNLOAD_SERVER_WATCHDOG_THREAD: threading.Thread | None = None


def get_download_dir() -> Path:
    return Path(os.environ.get("DOWNLOAD_DIR", "data/downloads"))


def get_download_ttl_seconds() -> int:
    minutes_value = os.environ.get("DOWNLOAD_TTL_MINUTES", "").strip()
    if minutes_value:
        try:
            return max(0, int(float(minutes_value) * 60))
        except ValueError:
            pass

    raw_value = os.environ.get("DOWNLOAD_TTL_SECONDS", "900")
    try:
        return max(0, int(raw_value))
    except ValueError:
        return 900


def get_public_download_base_url() -> str:
    return os.environ.get("PUBLIC_DOWNLOAD_BASE_URL", "").strip().rstrip("/")


def get_download_server_host() -> str:
    return os.environ.get("DOWNLOAD_SERVER_HOST", "0.0.0.0")


def get_download_server_port() -> int:
    raw_value = os.environ.get("DOWNLOAD_SERVER_PORT", "43561")
    try:
        return max(1, min(65535, int(raw_value)))
    except ValueError:
        return 43561


def get_download_server_idle_grace_seconds() -> int:
    raw_value = os.environ.get("DOWNLOAD_SERVER_IDLE_GRACE_SECONDS", "5")
    try:
        return max(0, int(raw_value))
    except ValueError:
        return 5


def get_telegram_safe_upload_bytes() -> int:
    raw_value = os.environ.get("TELEGRAM_SAFE_UPLOAD_BYTES", "51380224")
    try:
        return max(1, int(raw_value))
    except ValueError:
        return 51_380_224


def normalize_delivery_mode(mode: str | None = None) -> str:
    selected = (mode or os.environ.get("DELIVERY_MODE", "telegram")).strip().lower()
    if selected not in {"telegram", "send_link", "auto"}:
        return "telegram"
    return selected


def guess_media_type(path: Path) -> str:
    if path.suffix.lower() == ".mp3":
        return "audio/mpeg"
    if path.suffix.lower() == ".wav":
        return "audio/wav"
    if path.suffix.lower() == ".tar":
        return "application/x-tar"
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    value = float(size_bytes)
    for unit in ("KiB", "MiB", "GiB"):
        value /= 1024.0
        if value < 1024.0 or unit == "GiB":
            return f"{value:.1f} {unit}"
    return f"{size_bytes} B"


def _safe_name(filename: str | None, fallback: str) -> str:
    name = Path(filename or fallback).name.strip()
    return name or fallback


def _metadata_path(token_dir: Path) -> Path:
    return token_dir / METADATA_FILENAME


def cleanup_expired_downloads(
    download_dir: Path | None = None,
    *,
    now: float | None = None,
) -> None:
    root = download_dir or get_download_dir()
    if not root.exists():
        return

    current_time = time.time() if now is None else now
    for child in root.iterdir():
        if not child.is_dir() or not TOKEN_RE.fullmatch(child.name):
            continue

        metadata_file = _metadata_path(child)
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            expires_at = float(metadata.get("expires_at", 0))
        except Exception:
            expires_at = 0

        if expires_at <= current_time:
            shutil.rmtree(child, ignore_errors=True)


def has_active_downloads(
    download_dir: Path | None = None,
    *,
    now: float | None = None,
) -> bool:
    root = download_dir or get_download_dir()
    if not root.exists():
        return False

    current_time = time.time() if now is None else now
    for child in root.iterdir():
        if not child.is_dir() or not TOKEN_RE.fullmatch(child.name):
            continue

        metadata_file = _metadata_path(child)
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            expires_at = float(metadata.get("expires_at", 0))
        except Exception:
            expires_at = 0

        if expires_at > current_time:
            return True
    return False


def create_download_entry(
    source_path: Path,
    *,
    filename: str | None = None,
    media_type: str | None = None,
    download_dir: Path | None = None,
    ttl_seconds: int | None = None,
    public_base_url: str | None = None,
) -> tuple[str, Path]:
    """Copy a result file to shared storage and return public URL + token dir."""
    if not source_path.is_file():
        raise FileNotFoundError(f"Download source is not a file: {source_path}")

    base_url = (
        get_public_download_base_url()
        if public_base_url is None
        else public_base_url.strip().rstrip("/")
    )
    if not base_url:
        raise RuntimeError(
            "PUBLIC_DOWNLOAD_BASE_URL is required for DELIVERY_MODE=send_link/auto"
        )

    root = download_dir or get_download_dir()
    ttl = get_download_ttl_seconds() if ttl_seconds is None else max(0, ttl_seconds)
    created_at = int(time.time())
    expires_at = created_at + ttl

    root.mkdir(parents=True, exist_ok=True)
    cleanup_expired_downloads(root, now=created_at)

    token = uuid.uuid4().hex
    tmp_dir = root / f".{token}.tmp"
    token_dir = root / token
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=False)

    safe_filename = _safe_name(filename, source_path.name)
    stored_name = f"audio{source_path.suffix.lower() or '.bin'}"
    stored_path = tmp_dir / stored_name
    shutil.copy2(source_path, stored_path)

    metadata: dict[str, Any] = {
        "token": token,
        "filename": safe_filename,
        "stored_name": stored_name,
        "media_type": media_type or guess_media_type(source_path),
        "created_at": created_at,
        "expires_at": expires_at,
        "size_bytes": stored_path.stat().st_size,
    }
    metadata_tmp = tmp_dir / f"{METADATA_FILENAME}.tmp"
    metadata_tmp.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata_tmp.replace(_metadata_path(tmp_dir))
    tmp_dir.replace(token_dir)

    return f"{base_url}/d/{token}", token_dir


def _content_disposition(filename: str) -> str:
    safe_fallback = Path(filename).name.replace('"', "") or "download.bin"
    encoded = quote(filename, safe="")
    return f'attachment; filename="{safe_fallback}"; filename*=UTF-8\'\'{encoded}'


class _DownloadRequestHandler(BaseHTTPRequestHandler):
    server_version = "AudiobookerDownload/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_HEAD(self) -> None:
        self._serve(send_body=False)

    def do_GET(self) -> None:
        self._serve(send_body=True)

    def _send_text(self, status: HTTPStatus, text: str) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _serve(self, *, send_body: bool) -> None:
        parsed = urlparse(self.path)
        match = re.fullmatch(r"/d/([a-f0-9]{32})", parsed.path)
        if not match:
            self._send_text(HTTPStatus.NOT_FOUND, "Not found")
            return

        try:
            cleanup_expired_downloads()
            metadata, file_path = resolve_download_token(match.group(1))
        except Exception:
            self._send_text(HTTPStatus.NOT_FOUND, "Link not found or expired")
            return

        filename = str(metadata.get("filename") or file_path.name)
        media_type = str(metadata.get("media_type") or guess_media_type(file_path))
        file_size = file_path.stat().st_size

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(file_size))
        self.send_header("Content-Disposition", _content_disposition(filename))
        self.end_headers()

        if send_body:
            with file_path.open("rb") as source:
                shutil.copyfileobj(source, self.wfile)


def _download_server_watchdog(server: ThreadingHTTPServer) -> None:
    while True:
        cleanup_expired_downloads()
        if not has_active_downloads():
            grace_seconds = get_download_server_idle_grace_seconds()
            if grace_seconds:
                time.sleep(grace_seconds)
                cleanup_expired_downloads()
            if not has_active_downloads():
                server.shutdown()
                return
        time.sleep(5)


def _serve_download_server(server: ThreadingHTTPServer) -> None:
    global DOWNLOAD_SERVER, DOWNLOAD_SERVER_THREAD, DOWNLOAD_SERVER_WATCHDOG_THREAD

    try:
        server.serve_forever()
    finally:
        server.server_close()
        with DOWNLOAD_SERVER_LOCK:
            if DOWNLOAD_SERVER is server:
                DOWNLOAD_SERVER = None
                DOWNLOAD_SERVER_THREAD = None
                DOWNLOAD_SERVER_WATCHDOG_THREAD = None


def ensure_download_server_running() -> None:
    """Start the temporary download server if there is an active link."""
    global DOWNLOAD_SERVER, DOWNLOAD_SERVER_THREAD, DOWNLOAD_SERVER_WATCHDOG_THREAD

    if not has_active_downloads():
        return

    with DOWNLOAD_SERVER_LOCK:
        if DOWNLOAD_SERVER_THREAD is not None and DOWNLOAD_SERVER_THREAD.is_alive():
            return

        host = get_download_server_host()
        port = get_download_server_port()
        server = ThreadingHTTPServer((host, port), _DownloadRequestHandler)
        DOWNLOAD_SERVER = server
        DOWNLOAD_SERVER_THREAD = threading.Thread(
            target=_serve_download_server,
            args=(server,),
            name="audiobooker-download-server",
            daemon=True,
        )
        DOWNLOAD_SERVER_WATCHDOG_THREAD = threading.Thread(
            target=_download_server_watchdog,
            args=(server,),
            name="audiobooker-download-server-watchdog",
            daemon=True,
        )
        DOWNLOAD_SERVER_THREAD.start()
        DOWNLOAD_SERVER_WATCHDOG_THREAD.start()


def pack_audio_files_to_tar(audio_files: list[Path], tar_path: Path) -> Path:
    if not audio_files:
        raise ValueError("No audio files to pack")

    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as archive:
        for audio_file in audio_files:
            archive.add(audio_file, arcname=Path(audio_file.name).name)
    return tar_path


def resolve_download_token(
    token: str,
    *,
    download_dir: Path | None = None,
    now: float | None = None,
) -> tuple[dict[str, Any], Path]:
    if not TOKEN_RE.fullmatch(token):
        raise FileNotFoundError("Unknown download token")

    root = (download_dir or get_download_dir()).resolve()
    token_dir = (root / token).resolve()
    metadata_file = _metadata_path(token_dir)
    if not metadata_file.is_file():
        raise FileNotFoundError("Unknown download token")

    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    current_time = time.time() if now is None else now
    if float(metadata.get("expires_at", 0)) <= current_time:
        shutil.rmtree(token_dir, ignore_errors=True)
        raise FileNotFoundError("Download link expired")

    stored_name = Path(str(metadata.get("stored_name", ""))).name
    if not stored_name:
        raise FileNotFoundError("Download file is missing")

    file_path = (token_dir / stored_name).resolve()
    try:
        file_path.relative_to(token_dir)
    except ValueError as exc:
        raise FileNotFoundError("Invalid download metadata") from exc

    if not file_path.is_file() or file_path.is_symlink():
        raise FileNotFoundError("Download file is missing")

    return metadata, file_path
