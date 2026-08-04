#!/usr/bin/env python3
"""
Telegram-бот для генерации аудиокниг.
Принимает текстовые сообщения или файлы .txt/.fb2 и возвращает MP3.

Установка зависимостей:
    pip install python-telegram-bot

Запуск:
    BOT_TOKEN=<token> python tg_audiobooker.py
"""

from __future__ import annotations

import asyncio
import hashlib
from html.parser import HTMLParser
import logging
import os
import random
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from dotenv import load_dotenv
import telegram
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    PicklePersistence,
)

# Переменные из .env должны быть загружены до импорта audiobooker:
# его лимиты PAUSE и retry-настройки вычисляются при импорте модуля.
load_dotenv()

from tts_dependency_manager import sync_tts_dependencies_from_env  # noqa: E402
from download_delivery import (  # noqa: E402
    create_download_entry,
    ensure_download_server_running,
    format_file_size,
    get_download_ttl_seconds,
    get_telegram_safe_upload_bytes,
    guess_media_type,
    normalize_delivery_mode,
    pack_audio_files_to_tar,
)

# Импортируем движок и утилиты из основного скрипта
from audiobooker import (  # noqa: E402
    AudioSegment,
    MAX_TOTAL_PAUSE_MS,
    build_output_basename,
    collect_valid_audio_files,
    extract_fb2_text,
    merge_audio_chunks,
    prepare_audio_segments,
    synthesize_chunk_edge,
    synthesize_silence_mp3,
)

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
CPU_COUNT = os.cpu_count() or 2

# Общие параметры
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "10000"))
_mct = os.environ.get("MAX_CONCURRENT_TASKS", "").strip()
if _mct:
    MAX_CONCURRENT_TASKS = int(_mct)
else:
    MAX_CONCURRENT_TASKS = 8 if CPU_COUNT <= 2 else 24

FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")
MERGE_CHUNKS = os.environ.get("MERGE_CHUNKS", "true").lower() in ("1", "true", "yes")
DELIVERY_MODE = normalize_delivery_mode()
DOWNLOAD_TTL_SECONDS = get_download_ttl_seconds()
TELEGRAM_SAFE_UPLOAD_BYTES = get_telegram_safe_upload_bytes()

# Параметры Edge
EDGE_VOICE = os.environ.get("VOICE", "ru-RU-SvetlanaNeural")
EDGE_SPEED = os.environ.get("SPEED", "+18%")

# Максимальный размер текста
MAX_TEXT_FROM_MESSAGE = int(os.environ.get("MAX_TEXT_FROM_MESSAGE", "50000"))

# Путь к файлу базы данных/настроек
BOT_DATA_PATH = os.environ.get("BOT_DATA_PATH", "data/bot_data.pickle")
FORWARD_GROUP_DEBOUNCE_SECONDS = float(
    os.environ.get("FORWARD_GROUP_DEBOUNCE_SECONDS", "5.0")
)
MAX_CONCURRENT_REQUESTS = int(
    os.environ.get(
        "MAX_CONCURRENT_REQUESTS",
        "2" if CPU_COUNT <= 2 else "4",
    )
)
MAX_CONCURRENT_UPDATES = int(
    os.environ.get(
        "MAX_CONCURRENT_UPDATES",
        str(4 if CPU_COUNT <= 2 else max(4, MAX_CONCURRENT_REQUESTS * 2)),
    )
)

# Доступные голоса внешнего Edge TTS для рандомизации
EDGE_VOICES = [
    "ru-RU-SvetlanaNeural",
    "ru-RU-DmitryNeural",
]

HASHTAG_ONLY_TOKEN_RE = re.compile(r"^[#＃][\w-]+$", re.UNICODE)
FIRST_SENTENCE_RE = re.compile(r"(.+?(?:[.!?]+(?=\s|$)|$))", re.DOTALL)
FILENAME_SAFE_CHARS_RE = re.compile(r"[^\w\s\.\-\(\)]", re.UNICODE)
PREVIEW_MARKER_ONLY_RE = re.compile(
    r"^(?:\(?\d{1,3}[.)]\)?|[IVXLCDMivxlcdm]{1,6}[.)])$"
)
PREVIEW_LEADING_MARKER_RE = re.compile(
    r"^(?:(?:\(?\d{1,3}[.)]\)?|[IVXLCDMivxlcdm]{1,6}[.)])\s*)+"
)
PREVIEW_WORD_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*", re.UNICODE)
SUBSCRIBE_PATTERNS = re.compile(
    r"(подпишит(?:есь|е)?|подписывайт(?:есь|е)?|подписаться|"
    r"следите за нами|присоединяйтесь|читайте нас|"
    r"наш(?:а|и|е)?\s+(?:канал|группа|чат|telegram|телеграм|vk|вк)|"
    r"telegram-канал|телеграм-канал)",
    re.IGNORECASE,
)
SUBSCRIBE_HINTS = re.compile(
    r"(https?://|t\.me/|@\w{3,}|vk\.com/|telegram|телеграм|канал|группа|чат|vk|вк)",
    re.IGNORECASE,
)
SOURCE_METADATA_PREFIX_RE = re.compile(
    r"^\s*источник\s*[:\-–—]?\s*(.*)$", re.IGNORECASE
)
SOURCE_METADATA_TOKEN_RE = re.compile(
    r"^(?:#?\d+[.)]?|@\w+|t\.me/\S+|https?://\S+|[\w.-]+\.(?:com|ru|org|net|io)\S*)$",
    re.IGNORECASE,
)
SOURCE_METADATA_LEADING_RE = re.compile(
    r"^\s*источник\s*[:\-–—]?\s*"
    r"(?:#?\d+[.)]?|@\w+|t\.me/\S+|https?://\S+|[\w.-]+\.(?:com|ru|org|net|io)\S*)"
    r"(?:\s*[,;|.\-–—]\s*|\s+)?",
    re.IGNORECASE,
)
SLASH_COMMAND_RE = re.compile(r"^\s*/[A-Za-z0-9_]+(?:@[A-Za-z0-9_]+)?(?:\s|$)")
TELEGRAPH_URL_RE = re.compile(
    r"https?://(?:www\.)?(?:telegra\.ph|graph\.org)/[^\s<>\[\]\{\}\"']+",
    re.IGNORECASE,
)
TELEGRAPH_ALLOWED_HOSTS = {"telegra.ph", "graph.org"}
TELEGRAPH_TRAILING_URL_CHARS = ".,!?;:)]}>\"'»”"
TELEGRAPH_FETCH_TIMEOUT = float(os.environ.get("TELEGRAPH_FETCH_TIMEOUT", "15"))


class TelegraphHTMLParser(HTMLParser):
    """Извлекает заголовок и основной текст статьи Telegraph из HTML."""

    _PARAGRAPH_TAGS = {
        "article",
        "blockquote",
        "div",
        "figcaption",
        "h1",
        "h2",
        "h3",
        "h4",
        "header",
        "li",
        "ol",
        "p",
        "section",
        "ul",
    }
    _LINEBREAK_TAGS = {"br"}
    _IGNORE_TAGS = {"iframe", "noscript", "script", "style", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignore_depth = 0
        self._main_depth = 0
        self._article_depth = 0
        self._title_depth = 0
        self._title_parts: list[str] = []
        self._main_parts: list[str] = []
        self._article_parts: list[str] = []

    @staticmethod
    def _append_text(parts: list[str], text: str) -> None:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return
        if parts and not parts[-1].endswith((" ", "\n")):
            parts.append(" ")
        parts.append(normalized)

    @staticmethod
    def _append_break(parts: list[str]) -> None:
        if parts and not parts[-1].endswith("\n"):
            parts.append("\n")

    @staticmethod
    def _append_paragraph_break(parts: list[str]) -> None:
        if not parts:
            return
        if parts[-1].endswith("\n\n"):
            return
        if parts[-1].endswith("\n"):
            parts.append("\n")
            return
        parts.append("\n\n")

    @staticmethod
    def _finalize(parts: list[str]) -> str:
        text = "".join(parts)
        text = re.sub(r"\s+([,.:;!?])", r"\1", text)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _active_text_parts(self) -> list[list[str]]:
        active: list[list[str]] = []
        if self._main_depth > 0:
            active.append(self._main_parts)
        if self._article_depth > 0:
            active.append(self._article_parts)
        return active

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._IGNORE_TAGS:
            self._ignore_depth += 1
            return
        if self._ignore_depth:
            return

        if tag == "main":
            self._main_depth += 1
        elif tag == "article":
            self._article_depth += 1
        elif tag == "h1" and self._main_depth > 0:
            self._title_depth += 1

        if tag in self._PARAGRAPH_TAGS:
            for parts in self._active_text_parts():
                self._append_paragraph_break(parts)
            if self._title_depth > 0:
                self._append_paragraph_break(self._title_parts)
        elif tag in self._LINEBREAK_TAGS:
            for parts in self._active_text_parts():
                self._append_break(parts)
            if self._title_depth > 0:
                self._append_break(self._title_parts)

    def handle_startendtag(self, tag: str, attrs) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in self._IGNORE_TAGS:
            if self._ignore_depth > 0:
                self._ignore_depth -= 1
            return
        if self._ignore_depth:
            return

        if tag in self._PARAGRAPH_TAGS:
            for parts in self._active_text_parts():
                self._append_paragraph_break(parts)
            if self._title_depth > 0:
                self._append_paragraph_break(self._title_parts)
        elif tag in self._LINEBREAK_TAGS:
            for parts in self._active_text_parts():
                self._append_break(parts)
            if self._title_depth > 0:
                self._append_break(self._title_parts)

        if tag == "h1" and self._title_depth > 0:
            self._title_depth -= 1
        elif tag == "article" and self._article_depth > 0:
            self._article_depth -= 1
        elif tag == "main" and self._main_depth > 0:
            self._main_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._ignore_depth:
            return
        if self._title_depth > 0:
            self._append_text(self._title_parts, data)
        if self._main_depth > 0:
            self._append_text(self._main_parts, data)
        if self._article_depth > 0:
            self._append_text(self._article_parts, data)

    def get_text(self) -> str:
        title = self._finalize(self._title_parts)
        article_text = self._finalize(self._article_parts)
        main_text = self._finalize(self._main_parts)
        full_text = article_text or main_text
        if title:
            normalized_title = re.sub(r"\s+", " ", title).strip().casefold()
            normalized_body = re.sub(r"\s+", " ", full_text).strip().casefold()
            if not full_text:
                return title
            if not normalized_body.startswith(normalized_title):
                return f"{title}\n\n{full_text}"
        return full_text


DEFAULT_SETTINGS = {
    "chunk_size": CHUNK_SIZE,
    "max_concurrent_tasks": MAX_CONCURRENT_TASKS,
    "ffmpeg_path": FFMPEG_PATH,
    "merge_chunks": MERGE_CHUNKS,
    "edge_voice": EDGE_VOICE,
    "edge_speed": EDGE_SPEED,
    "random": False,
}

FORWARDED_BATCHES_KEY = "forwarded_batches"
PROCESSING_SEMAPHORE: asyncio.Semaphore | None = None
USER_PROCESSING_LOCKS: dict[int, asyncio.Lock] = {}
ForwardedItem = tuple[str, int | str, str | None]


def get_processing_semaphore() -> asyncio.Semaphore:
    """Лимитирует число одновременных запросов к синтезу."""
    global PROCESSING_SEMAPHORE
    if PROCESSING_SEMAPHORE is None:
        PROCESSING_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    return PROCESSING_SEMAPHORE


def get_user_processing_lock(user_id: int | None) -> asyncio.Lock | None:
    """Возвращает персональный lock пользователя для последовательной обработки."""
    if user_id is None:
        return None
    lock = USER_PROCESSING_LOCKS.get(user_id)
    if lock is None:
        lock = asyncio.Lock()
        USER_PROCESSING_LOCKS[user_id] = lock
    return lock


def build_forward_batch_key(
    *,
    chat_id: int | None,
    user_id: int | None,
    source_sender_id: int | str | None = None,
    media_group_id: str | None = None,
    message_thread_id: int | None = None,
) -> str:
    """Формирует ключ буфера пересланных сообщений для конкретного чата/группы."""
    source_part = (
        str(source_sender_id).replace("|", "_")
        if source_sender_id is not None
        else "unknown"
    )
    parts = [f"chat:{chat_id if chat_id is not None else 'unknown'}"]
    if message_thread_id is not None:
        parts.append(f"thread:{message_thread_id}")
    parts.append(f"user:{user_id if user_id is not None else 'unknown'}")
    parts.append(f"source:{source_part}")
    parts.append(f"group:{media_group_id or 'debounce'}")
    return "|".join(parts)


def build_forward_job_name(batch_key: str) -> str:
    """Стабильное имя джоба для конкретной группы пересылок."""
    digest = hashlib.sha1(batch_key.encode("utf-8")).hexdigest()
    return f"collector_{digest}"


def schedule_forward_collector(
    context: ContextTypes.DEFAULT_TYPE,
    *,
    batch_key: str,
    chat_id: int,
    user_id: int,
) -> bool:
    """Планирует дебаунс-джоб сборщика пересланных сообщений/файлов."""
    if not context.job_queue:
        return False

    job_name = build_forward_job_name(batch_key)
    jobs = context.job_queue.get_jobs_by_name(job_name)
    for j in jobs:
        j.schedule_removal()

    context.job_queue.run_once(
        collector_job,
        when=FORWARD_GROUP_DEBOUNCE_SECONDS,
        data={
            "batch_key": batch_key,
            "chat_id": chat_id,
            "user_id": user_id,
        },
        name=job_name,
        user_id=user_id,
        chat_id=chat_id,
    )
    return True


def ensure_forwarded_batches(user_data: dict) -> dict[str, list[ForwardedItem]]:
    """Возвращает контейнер буферов пересланных сообщений для пользователя."""
    batches = user_data.get(FORWARDED_BATCHES_KEY)
    if not isinstance(batches, dict):
        batches = {}
        user_data[FORWARDED_BATCHES_KEY] = batches
    return batches


def normalize_user_settings(settings: object) -> dict:
    """Drop legacy local-model keys and fill current Edge-only defaults."""
    normalized = DEFAULT_SETTINGS.copy()
    if isinstance(settings, dict):
        for key in DEFAULT_SETTINGS:
            if key in settings:
                normalized[key] = settings[key]
        settings.clear()
        settings.update(normalized)
        return settings
    return normalized


def get_user_settings(context: ContextTypes.DEFAULT_TYPE) -> dict:
    """Возвращает актуальные Edge-only настройки пользователя."""
    if context.user_data is None:
        return DEFAULT_SETTINGS.copy()
    if "settings" not in context.user_data:
        context.user_data["settings"] = DEFAULT_SETTINGS.copy()
    return normalize_user_settings(context.user_data["settings"])


def render_progress_bar(current: int, total: int, length: int = 15) -> str:
    """Рисует прогресс-бар из символов."""
    if total <= 0:
        return ""
    filled = int(length * current // total)
    bar = "█" * filled + "░" * (length - filled)
    percent = int(100 * current // total)
    return f"[{bar}] {percent}%"


def build_source_hashtag(source_name: str | None) -> str | None:
    """Преобразует имя источника в безопасный хештег."""
    if not source_name:
        return None
    tag = re.sub(r"[^\w]", "", source_name)
    return f"#{tag}" if tag else None


def extract_source_name(origin) -> str | None:
    """Извлекает человекочитаемое имя источника из Telegram forward origin."""
    if not origin:
        return None

    source_name = getattr(getattr(origin, "chat", None), "username", None)
    source_name = source_name or getattr(getattr(origin, "chat", None), "title", None)
    source_name = source_name or getattr(
        getattr(origin, "sender_user", None), "username", None
    )
    source_name = source_name or getattr(
        getattr(origin, "sender_user", None), "first_name", None
    )
    source_name = source_name or getattr(
        getattr(origin, "sender_chat", None), "username", None
    )
    source_name = source_name or getattr(
        getattr(origin, "sender_chat", None), "title", None
    )
    source_name = source_name or getattr(origin, "sender_user_name", None)
    return source_name


def extract_forward_sender_metadata(origin) -> tuple[int | str, str | None]:
    """Возвращает sender_id и хештег источника из Telegram forward origin."""
    sender_id: int | str = "unknown"
    if not origin:
        return sender_id, None

    s_id: int | str | None = None
    s_id = s_id or getattr(getattr(origin, "sender_user", None), "id", None)
    s_id = s_id or getattr(getattr(origin, "sender_chat", None), "id", None)
    s_id = s_id or getattr(getattr(origin, "chat", None), "id", None)
    s_id = s_id or getattr(origin, "sender_user_name", None)
    if s_id is not None:
        sender_id = s_id
    else:
        sender_id = str(origin)

    hashtag = build_source_hashtag(extract_source_name(origin))
    return sender_id, hashtag


def is_slash_command(text: str) -> bool:
    """Проверяет, выглядит ли строка как Telegram-команда вида /cmd или /cmd@bot."""
    return bool(SLASH_COMMAND_RE.match(text))


def _is_hashtag_line(line: str) -> bool:
    words = [word.strip(".,;:|•") for word in line.split()]
    if not words:
        return False
    return all(HASHTAG_ONLY_TOKEN_RE.match(word) for word in words)


def _is_subscription_fragment(fragment: str) -> bool:
    normalized = re.sub(r"\s+", " ", fragment).strip(" -–—•|")
    if not normalized or not SUBSCRIBE_PATTERNS.search(normalized):
        return False
    return len(normalized) <= 220 or bool(SUBSCRIBE_HINTS.search(normalized))


def _strip_subscription_fragments(line: str) -> str:
    fragments = re.split(r"(?<=[.!?])\s+", line.strip())
    if not fragments:
        return ""
    kept_fragments = [
        fragment for fragment in fragments if not _is_subscription_fragment(fragment)
    ]
    return " ".join(fragment.strip() for fragment in kept_fragments if fragment.strip())


def _is_source_metadata_line(line: str) -> bool:
    normalized = re.sub(r"\s+", " ", line).strip(" -–—|")
    if not normalized:
        return False

    match = SOURCE_METADATA_PREFIX_RE.match(normalized)
    if not match:
        return False

    tail = match.group(1).strip(" -–—|.,;")
    if not tail:
        return True

    tokens = [
        token.strip(".,;()[]{}") for token in tail.split() if token.strip(".,;()[]{}")
    ]
    if not tokens:
        return True

    return len(tokens) <= 4 and all(
        SOURCE_METADATA_TOKEN_RE.fullmatch(token) for token in tokens
    )


def _strip_source_metadata_prefix(line: str) -> str:
    cleaned = SOURCE_METADATA_LEADING_RE.sub("", line.strip(), count=1)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -–—|")
    return cleaned


def _normalize_preview_candidate(fragment: str) -> str:
    candidate = re.sub(r"\s+", " ", fragment).strip(" -–—|:")
    candidate = PREVIEW_LEADING_MARKER_RE.sub("", candidate).strip(" -–—|:")
    candidate = FILENAME_SAFE_CHARS_RE.sub("", candidate).strip()
    candidate = re.sub(r"\s+", " ", candidate)
    return candidate


def _is_meaningful_preview(fragment: str) -> bool:
    candidate = _normalize_preview_candidate(fragment)
    if not candidate or PREVIEW_MARKER_ONLY_RE.fullmatch(candidate):
        return False
    return len(PREVIEW_WORD_RE.findall(candidate)) >= 2


def get_text_preview(text: str, max_len: int = 80) -> str:
    """Строит имя файла по первому осмысленному предложению текста."""
    text = text.strip()
    if not text:
        return "audiobook"

    lines = [
        re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()
    ]
    while len(lines) > 1:
        leading = lines[0].strip(" -–—|:")
        if PREVIEW_MARKER_ONLY_RE.fullmatch(leading):
            lines.pop(0)
            continue
        if any(ch in leading for ch in ".!?"):
            break
        if len(leading) > 50 or len(leading.split()) > 5:
            break
        lines.pop(0)

    flattened = " ".join(lines) if lines else re.sub(r"\s+", " ", text)
    sentences = [
        fragment.strip()
        for fragment in re.split(r"(?<=[.!?])\s+", flattened)
        if fragment.strip()
    ]
    if not sentences:
        sentences = [flattened.strip()]

    preview = ""
    for sentence in sentences:
        if _is_meaningful_preview(sentence):
            preview = _normalize_preview_candidate(sentence)
            break

    if not preview:
        for sentence in sentences:
            candidate = _normalize_preview_candidate(sentence)
            if candidate and not PREVIEW_MARKER_ONLY_RE.fullmatch(candidate):
                preview = candidate
                break

    if not preview:
        match = FIRST_SENTENCE_RE.match(flattened)
        preview = _normalize_preview_candidate(
            match.group(1).strip() if match else flattened.strip()
        )

    preview = preview[:max_len].strip()
    return preview or "audiobook"


def clean_tg_post(text: str) -> str:
    """Очищает текст от мусора из новостных агрегаторов (ссылки, служебный текст)."""
    idx = text.find("Новости группируются автоматически")
    if idx != -1:
        text = text[:idx]

    text = "\n".join(line for line in text.splitlines() if not _is_hashtag_line(line))

    text = re.sub(r"[\(\[\{]\s*https?://[^\s)\]\}]+\s*[\)\]\}]", "", text)
    text = re.sub(r"https?://[^\s]+", "", text)
    text = "\n".join(
        cleaned_line
        for line in text.splitlines()
        if (
            cleaned_line := _strip_source_metadata_prefix(
                _strip_subscription_fragments(line)
            )
        )
        and not _is_source_metadata_line(cleaned_line)
    )
    text = re.sub(r"\s+,\s+", ", ", text)
    text = re.sub(r",\s*$", "", text, flags=re.MULTILINE)

    # Схлопываем лишние пустые строки
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_telegraph_url(text: str) -> str | None:
    """Возвращает первую ссылку на Telegraph из текста."""
    for match in TELEGRAPH_URL_RE.finditer(text):
        candidate = match.group(0).rstrip(TELEGRAPH_TRAILING_URL_CHARS)
        parsed = urlsplit(candidate)
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        if host in TELEGRAPH_ALLOWED_HOSTS and parsed.path and parsed.path != "/":
            return candidate
    return None


def extract_telegraph_page_text(html_content: str) -> str:
    """Извлекает человекочитаемый текст статьи Telegraph из HTML."""
    parser = TelegraphHTMLParser()
    parser.feed(html_content)
    parser.close()
    return parser.get_text().strip()


def fetch_telegraph_text(url: str) -> str:
    """Скачивает страницу Telegraph и возвращает текст статьи."""
    request = Request(
        url,
        headers={
            "User-Agent": "audiobooker/1.0",
            "Accept-Language": "ru,en;q=0.9",
        },
    )
    with urlopen(request, timeout=TELEGRAPH_FETCH_TIMEOUT) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        html_content = response.read().decode(charset, errors="replace")

    text = extract_telegraph_page_text(html_content)
    if not text:
        raise ValueError("Telegraph page is empty")
    return text


async def resolve_forwarded_text(raw_text: str) -> str:
    """Подменяет текст пересылки содержимым Telegraph, если ссылка присутствует."""
    telegraph_url = extract_telegraph_url(raw_text)
    if not telegraph_url:
        return clean_tg_post(raw_text)

    try:
        telegraph_text = await asyncio.to_thread(fetch_telegraph_text, telegraph_url)
    except Exception:
        logger.exception("Не удалось загрузить Telegraph: %s", telegraph_url)
        return clean_tg_post(raw_text)

    telegraph_text = clean_tg_post(telegraph_text)
    if telegraph_text:
        logger.info("Loaded Telegraph text from %s", telegraph_url)
        return telegraph_text

    logger.warning("Telegraph page became empty after cleaning: %s", telegraph_url)
    return clean_tg_post(raw_text)


# ============================================================

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


async def generate_audio(
    input_data: str | list[tuple[str, int | str]],
    work_dir: Path,
    settings: dict,
    name: str = "book",
    source_name: str | None = None,
    on_progress=None,
) -> Path | list[Path]:
    """
    Синтезирует текст в MP3 через внешний Edge TTS.
    input_data может быть строкой или списком (текст, sender_id).
    """
    output_basename = build_output_basename(source_name or name)
    parts_dir = work_dir / f"{output_basename}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = settings.get("chunk_size", CHUNK_SIZE)

    # Подготовка текстов и привязка голосов
    sender_voices: dict[int | str, str | None] = {}
    tasks_data: list[tuple[AudioSegment, str | None]] = []
    total_pause_ms = 0

    def append_segments(raw_text: str, assigned_voice: str | None) -> None:
        nonlocal total_pause_ms
        for segment in prepare_audio_segments(raw_text, chunk_size):
            if segment.pause_ms is not None:
                total_pause_ms += segment.pause_ms
                if total_pause_ms > MAX_TOTAL_PAUSE_MS:
                    raise ValueError(
                        "Суммарная длительность PAUSE "
                        f"{total_pause_ms} мс превышает лимит "
                        f"{MAX_TOTAL_PAUSE_MS} мс"
                    )
            tasks_data.append((segment, assigned_voice))

    if isinstance(input_data, str):
        # В режиме рандома выбираем один голос на всё сообщение
        assigned_voice = random.choice(EDGE_VOICES) if settings.get("random") else None
        append_segments(input_data, assigned_voice)
    else:
        # Список (текст, sender_id)
        for item in input_data:
            text_part, sender_id = item[0], item[1]

            # В режиме рандома закрепляем голос за отправителем
            assigned_voice = None
            if settings.get("random"):
                if sender_id not in sender_voices:
                    assigned_voice = random.choice(EDGE_VOICES)
                    sender_voices[sender_id] = assigned_voice
                else:
                    assigned_voice = sender_voices[sender_id]

            append_segments(text_part, assigned_voice)

    if not tasks_data:
        raise ValueError(
            "После очистки не осталось текста или управляющих тегов PAUSE."
        )

    max_tasks: int = settings.get("max_concurrent_tasks", MAX_CONCURRENT_TASKS)  # pyright: ignore
    semaphore = asyncio.Semaphore(max_tasks)
    ext = "mp3"
    tasks = []
    progress = {"completed": 0}
    total = len(tasks_data)

    async def _monitored_task(coro):
        await coro
        progress["completed"] += 1
        if on_progress:
            await on_progress(progress["completed"], total)

    for i, (segment, assigned_v) in enumerate(tasks_data):
        chunk_file = parts_dir / f"{output_basename}_chunk_{i:06}.{ext}"
        if segment.pause_ms is not None:
            coro = synthesize_silence_mp3(
                duration_ms=segment.pause_ms,
                file_path=chunk_file,
                ffmpeg_path=settings.get("ffmpeg_path", FFMPEG_PATH),
                semaphore=semaphore,
            )
        else:
            voice = assigned_v or settings.get("edge_voice", EDGE_VOICE)
            if voice not in EDGE_VOICES:
                fallback_voice = EDGE_VOICES[0] if EDGE_VOICES else EDGE_VOICE
                logger.warning(
                    "Unsupported Edge voice '%s', fallback to '%s'",
                    voice,
                    fallback_voice,
                )
                voice = fallback_voice
            if settings.get("random") and assigned_v is None:
                voice = random.choice(EDGE_VOICES)

            coro = synthesize_chunk_edge(
                text=segment.text or "",
                file_path=chunk_file,
                voice=voice,
                rate=settings.get("edge_speed", EDGE_SPEED),
                semaphore=semaphore,
            )
        tasks.append(asyncio.create_task(_monitored_task(coro)))

    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        logger.error(f"Error during audio generation: {e}")
        raise

    # Берем только реально валидные чанки, а битые остатки удаляем.
    expected_chunks = [
        parts_dir / f"{output_basename}_chunk_{i:06}.{ext}"
        for i in range(len(tasks_data))
    ]
    actual_chunks = collect_valid_audio_files(
        expected_chunks,
        expected_ext=ext,
        remove_invalid=True,
    )

    if not actual_chunks:
        raise ValueError(
            "После очистки не осталось озвучиваемого текста. "
            "В сообщении были только символы, ссылки или служебные фрагменты."
        )

    if settings.get("merge_chunks", MERGE_CHUNKS):
        list_file = parts_dir / "list.txt"
        with list_file.open("w", encoding="utf-8") as f:
            for part_path in actual_chunks:
                f.write(f"file '{part_path.name}'\n")

        full_file = work_dir / f"{output_basename}.{ext}"
        await merge_audio_chunks(
            ffmpeg_path=settings.get("ffmpeg_path", FFMPEG_PATH),
            list_file=list_file,
            output_file=full_file,
        )
        return full_file

    # Без склейки — возвращаем список файлов
    return actual_chunks


# ─────────────────────────── хендлеры ───────────────────────────


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    s = get_user_settings(context)
    await update.message.reply_text(
        "👋 Привет! Я конвертирую текст в аудиокнигу.\n\n"
        "Синтез выполняется внешним Edge TTS; локальные модели не устанавливаются.\n\n"
        "Отправь мне:\n"
        "• текстовое сообщение (до 50 000 символов)\n"
        "• файл .txt или .fb2\n\n"
        "Для точной паузы используй `[[PAUSE: 5000]]` — это 5 секунд.\n\n"
        "⚙️ Твои текущие настройки:\n"
        f"Голос: {s['edge_voice']}\n"
        f"Скорость: {s['edge_speed']}\n"
        f"🎲 Случайный голос: {'✅ ВКЛ' if s.get('random') else '❌ ВЫКЛ'}\n"
        f"Chunk size: {s['chunk_size']}\n\n"
        "Используй /settings для изменения параметров.",
        parse_mode="Markdown",
    )


async def cmd_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показывает меню настроек."""
    s = get_user_settings(context)
    text = (
        "⚙️ *Настройки Audiobooker*\n\n"
        f"*Edge голос:* `{s['edge_voice']}`\n"
        f"*Edge скорость:* `{s['edge_speed']}`\n"
        f"*Chunk size:* `{s['chunk_size']}`\n"
        f"*Случайный голос:* `{'ON' if s.get('random') else 'OFF'}`\n\n"
        "Выберите раздел для изменения:"
    )
    keyboard = [
        [InlineKeyboardButton("📊 Chunk Size", callback_data="set_menu_chunk")],
        [
            InlineKeyboardButton("🗣 Edge Голос", callback_data="set_menu_edge_voice"),
            InlineKeyboardButton(
                "⏩ Edge Скорость", callback_data="set_menu_edge_speed"
            ),
        ],
        [
            InlineKeyboardButton(
                f"🎲 Случайный голос: {'✅ ВКЛ' if s.get('random') else '❌ ВЫКЛ'}",
                callback_data="set_val_random_toggle",
            ),
        ],
        [InlineKeyboardButton("✅ Готово", callback_data="set_close")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    if update.callback_query:
        await update.callback_query.edit_message_text(
            text, reply_markup=reply_markup, parse_mode="Markdown"
        )
    elif update.message:
        await update.message.reply_text(
            text, reply_markup=reply_markup, parse_mode="Markdown"
        )


async def settings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка нажатий в меню настроек."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    data = query.data or ""
    s = get_user_settings(context)

    if data == "set_close":
        await query.edit_message_text("✅ Настройки сохранены.")
        return

    if data == "set_menu_chunk":
        # Простой выбор из частых вариантов
        vals = [5000, 10000, 20000, 50000]
        keyboard = [
            [InlineKeyboardButton(str(v), callback_data=f"set_val_chunk_{v}")]
            for v in vals
        ]
        keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data="set_main")])
        await query.edit_message_text(
            "Выберите размер чанка (символов):",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )
        return

    if data == "set_menu_edge_voice":
        voices = EDGE_VOICES
        keyboard = [
            [
                InlineKeyboardButton(
                    v.replace("ru-RU-", ""), callback_data=f"set_val_evoice_{v}"
                )
            ]
            for v in voices
        ]
        keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data="set_main")])
        await query.edit_message_text(
            "Выберите голос Edge:", reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return

    if data == "set_menu_edge_speed":
        speeds = ["-25%", "+0%", "+10%", "+18%", "+25%", "+50%"]
        keyboard = [
            [
                InlineKeyboardButton(speed, callback_data=f"set_val_espeed_{speed}")
                for speed in speeds[:3]
            ],
            [
                InlineKeyboardButton(speed, callback_data=f"set_val_espeed_{speed}")
                for speed in speeds[3:]
            ],
            [InlineKeyboardButton("⬅️ Назад", callback_data="set_main")],
        ]
        await query.edit_message_text(
            "Выберите скорость Edge:", reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return

    # Обработка значений
    if data == "set_main":
        await cmd_settings(update, context)
        # Изменения в context.user_data сохраняются через PicklePersistence
        return

    if data.startswith("set_val_"):
        _, _, key, val = data.split("_", 3)
        if key == "chunk":
            s["chunk_size"] = int(val)
        elif key == "evoice":
            s["edge_voice"] = val
        elif key == "espeed":
            s["edge_speed"] = val
        elif key == "random":
            if val == "toggle":
                s["random"] = not s.get("random", False)

        await cmd_settings(update, context)
        return


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    await update.message.reply_text(
        "/start    — приветствие и текущий статус\n"
        "/settings — настройка голоса, скорости и чанков\n"
        "/help     — эта справка\n\n"
        "Просто отправь текст или файл .txt/.fb2 — получишь MP3.\n"
        "`[[PAUSE: 5000]]` добавит 5 секунд тишины в указанном месте.",
        parse_mode="Markdown",
    )


async def handle_forwarded(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка пересланных сообщений — собираем их в буфер."""
    if not update.message:
        return
    raw_text = update.message.text or update.message.caption or ""
    text = await resolve_forwarded_text(raw_text)
    if is_slash_command(text):
        return
    if not text.strip():
        return  # Игнорируем пересланные вложения без текста

    # Оригинальный отправитель (для разграничения голосов в диалоге)
    sender_id, hashtag = extract_forward_sender_metadata(update.message.forward_origin)

    effective_user_id = update.effective_user.id if update.effective_user else None
    effective_chat_id = update.effective_chat.id if update.effective_chat else None

    if (
        context.user_data is None
        or effective_user_id is None
        or effective_chat_id is None
    ):
        await _process_and_reply(
            update,
            text,
            context=context,
            name=get_text_preview(text),
            caption=hashtag,
        )
        return

    batch_key = build_forward_batch_key(
        chat_id=effective_chat_id,
        user_id=effective_user_id,
        source_sender_id=sender_id,
        media_group_id=update.message.media_group_id,
        message_thread_id=update.message.message_thread_id,
    )
    forwarded_batches = ensure_forwarded_batches(context.user_data)
    forwarded_batches.setdefault(batch_key, []).append((text, sender_id, hashtag))

    if not schedule_forward_collector(
        context,
        batch_key=batch_key,
        chat_id=effective_chat_id,
        user_id=effective_user_id,
    ):
        # Если JobQueue почему-то нет (не установлены зависимости), обрабатываем сразу
        await _process_and_reply(
            update, text, context=context, name=get_text_preview(text), caption=hashtag
        )


async def collector_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Джоб для обработки собранного буфера пересланных сообщений."""
    job = context.job
    if not job or not context.application:
        return
    if not isinstance(job.data, dict):
        return
    user_id = job.data.get("user_id", getattr(job, "user_id", None))
    chat_id = job.data.get("chat_id")
    batch_key = job.data.get("batch_key")

    if user_id is None or chat_id is None or not isinstance(batch_key, str):
        return

    user_data = None
    if context.application.user_data:
        user_data = context.application.user_data.get(user_id)  # type: ignore

    if not isinstance(user_data, dict):
        return

    forwarded_batches = user_data.get(FORWARDED_BATCHES_KEY)
    if not isinstance(forwarded_batches, dict):
        return

    buffer = forwarded_batches.pop(batch_key, [])
    if not forwarded_batches:
        user_data.pop(FORWARDED_BATCHES_KEY, None)
    if not buffer:
        return

    # Формируем превью для названия
    full_text = "\n".join([b[0] for b in buffer])
    preview = get_text_preview(full_text)

    # Собираем уникальные хештеги
    hashtags = sorted(list({b[2] for b in buffer if len(b) > 2 and b[2]}))
    caption = " ".join(hashtags) if hashtags else None

    await _process_and_reply(
        None,  # update
        buffer,  # input_data (список)
        context=context,
        name=preview,
        chat_id=chat_id,  # type: ignore
        user_id=user_id,  # Передаем явно
        caption=caption,
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    text = update.message.text or ""
    text = clean_tg_post(text)
    if is_slash_command(text):
        await update.message.reply_text("Команды через / не озвучиваются.")
        return
    if not text.strip():
        await update.message.reply_text("Текст пустой, ничего не делаю.")
        return
    if len(text) > MAX_TEXT_FROM_MESSAGE:
        await update.message.reply_text(
            f"Текст слишком длинный ({len(text)} симв.). "
            f"Максимум {MAX_TEXT_FROM_MESSAGE}. Пришли файлом."
        )
        return

    await _process_and_reply(update, text, context=context, name=get_text_preview(text))


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    doc = update.message.document
    if not doc:
        return

    filename = doc.file_name or ""
    sender_id, hashtag = extract_forward_sender_metadata(update.message.forward_origin)

    suffix = Path(filename).suffix.lower()
    if suffix not in {".txt", ".fb2"}:
        await update.message.reply_text("Поддерживаются только .txt и .fb2 файлы.")
        return

    effective_user_id = update.effective_user.id if update.effective_user else None
    effective_chat_id = update.effective_chat.id if update.effective_chat else None

    if (
        update.message.forward_origin
        and context.user_data is not None
        and effective_user_id is not None
        and effective_chat_id is not None
    ):
        work_dir = Path(tempfile.gettempdir()) / f"tg_audiobooker_{uuid.uuid4().hex}"
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            text = await _load_document_text(
                context=context,
                doc=doc,
                suffix=suffix,
                work_dir=work_dir,
            )
            text = clean_tg_post(text)
            if not text.strip():
                await update.message.reply_text("Файл пустой.")
                return

            batch_key = build_forward_batch_key(
                chat_id=effective_chat_id,
                user_id=effective_user_id,
                source_sender_id=sender_id,
                media_group_id=update.message.media_group_id,
                message_thread_id=update.message.message_thread_id,
            )
            forwarded_batches = ensure_forwarded_batches(context.user_data)
            forwarded_batches.setdefault(batch_key, []).append(
                (text, sender_id, hashtag)
            )

            if not schedule_forward_collector(
                context,
                batch_key=batch_key,
                chat_id=effective_chat_id,
                user_id=effective_user_id,
            ):
                await _process_and_reply(
                    update,
                    text,
                    context=context,
                    name=Path(filename).stem or get_text_preview(text),
                    source_name=filename,
                    caption=hashtag,
                )
            return
        except Exception as e:
            logger.exception("Ошибка при обработке пересланного документа")
            await update.message.reply_text(f"❌ Ошибка: {e}")
            return
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    status_msg = await update.message.reply_text("⏳ Скачиваю файл…")

    work_dir = Path(tempfile.gettempdir()) / f"tg_audiobooker_{uuid.uuid4().hex}"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        text = await _load_document_text(
            context=context,
            doc=doc,
            suffix=suffix,
            work_dir=work_dir,
        )

        if not text.strip():
            await status_msg.edit_text("Файл пустой.")
            return

        await status_msg.edit_text(
            f"🔊 Синтезирую аудио ({len(text)} симв., ~{len(text) // CHUNK_SIZE + 1} чанков)…"
        )
        await _process_and_reply(
            update,
            text,
            context=context,
            name=Path(filename).stem or get_text_preview(text),
            source_name=filename,
            work_dir=work_dir,
            status_msg=status_msg,
            caption=hashtag,
        )

    except Exception as e:
        logger.exception("Ошибка при обработке документа")
        await status_msg.edit_text(f"❌ Ошибка: {e}")
        shutil.rmtree(work_dir, ignore_errors=True)


async def _load_document_text(
    *,
    context: ContextTypes.DEFAULT_TYPE,
    doc,
    suffix: str,
    work_dir: Path,
) -> str:
    """Скачивает Telegram-документ и извлекает из него текст."""
    tg_file = await context.bot.get_file(doc.file_id)
    local_path = work_dir / f"uploaded{suffix}"
    await tg_file.download_to_drive(local_path)

    if suffix == ".fb2":
        return extract_fb2_text(local_path)
    return local_path.read_text(encoding="utf-8")


async def _send_audio_with_retries(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int | str | None,
    file_path: Path,
    title: str | None = None,
    filename: str | None = None,
    caption: str | None = None,
    initial_timeout: int = 600,
):
    """Отправляет аудиофайл с повторными попытками при сетевых ошибках (5 попыток, таймаут x1.5)."""
    if chat_id is None:
        logger.error(f"Cannot send audio {file_path}, chat_id is None")
        return

    current_timeout = initial_timeout
    for attempt in range(1, 6):
        try:
            with file_path.open("rb") as f:
                return await context.bot.send_audio(
                    chat_id=chat_id,
                    audio=f,
                    filename=filename or file_path.name,
                    title=title or file_path.stem,
                    caption=caption,
                    read_timeout=current_timeout,
                    write_timeout=current_timeout,
                    connect_timeout=current_timeout,
                )
        except telegram.error.NetworkError as e:
            if _is_request_entity_too_large(e):
                logger.warning("Telegram rejected upload as too large: %s", e)
                raise
            if attempt == 5:
                logger.error(f"Failed to send audio after 5 attempts: {e}")
                raise
            wait_time = attempt * 2
            logger.warning(
                f"Network error on attempt {attempt}: {e}. Retrying in {wait_time}s with timeout {current_timeout}*1.5..."
            )
            current_timeout = int(current_timeout * 1.5)
            await asyncio.sleep(wait_time)


def _is_request_entity_too_large(error: BaseException) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "request entity too large",
            "entity too large",
            "payload too large",
            "file is too big",
            "file too large",
            "413",
        )
    )


def _result_paths(result: Path | list[Path]) -> list[Path]:
    return result if isinstance(result, list) else [result]


def _result_exceeds_telegram_limit(result: Path | list[Path]) -> bool:
    return any(
        path.stat().st_size > TELEGRAM_SAFE_UPLOAD_BYTES
        for path in _result_paths(result)
    )


def _largest_result_size(result: Path | list[Path]) -> int:
    sizes = [path.stat().st_size for path in _result_paths(result)]
    return max(sizes) if sizes else 0


def _prepare_download_artifact(
    result: Path | list[Path],
    work_dir: Path,
    name: str,
) -> Path:
    if isinstance(result, list):
        tar_name = f"{build_output_basename(name)}_parts.tar"
        return pack_audio_files_to_tar(result, work_dir / tar_name)
    return result


async def _edit_or_send_status(
    *,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int | str | None,
    status_msg,
    text: str,
) -> None:
    if status_msg is not None:
        await status_msg.edit_text(text)
        return
    if chat_id is not None:
        await context.bot.send_message(chat_id=chat_id, text=text)


def _link_ttl_text(ttl_seconds: int) -> str:
    if ttl_seconds <= 0:
        return "до ближайшей очистки"
    if ttl_seconds % 60 == 0:
        minutes = ttl_seconds // 60
        return f"{minutes} мин."
    return f"{ttl_seconds} сек."


async def _send_download_link(
    *,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int | str | None,
    status_msg,
    result: Path | list[Path],
    work_dir: Path,
    name: str,
    caption: str | None = None,
    reason: str | None = None,
) -> None:
    artifact = _prepare_download_artifact(result, work_dir, name)
    link, _token_dir = create_download_entry(
        artifact,
        filename=artifact.name,
        media_type=guess_media_type(artifact),
        ttl_seconds=DOWNLOAD_TTL_SECONDS,
    )
    ensure_download_server_running()

    size_text = format_file_size(artifact.stat().st_size)
    ttl_text = _link_ttl_text(DOWNLOAD_TTL_SECONDS)
    reason_text = f"\nПричина: {reason}" if reason else ""
    caption_text = f"\n\n{caption}" if caption else ""
    await _edit_or_send_status(
        context=context,
        chat_id=chat_id,
        status_msg=status_msg,
        text=(
            "🔗 Файл готов. Отправляю ссылку вместо загрузки в Telegram."
            f"{reason_text}\n"
            f"Размер: {size_text}\n"
            f"Ссылка действует: {ttl_text}\n"
            f"{link}"
            f"{caption_text}"
        ),
    )


async def _send_result_via_telegram(
    *,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int | str | None,
    result: Path | list[Path],
    name: str,
    caption: str | None = None,
) -> None:
    if isinstance(result, list):
        for chunk_path in result:
            await _send_audio_with_retries(
                context=context,
                chat_id=chat_id,
                file_path=chunk_path,
                filename=chunk_path.name,
                title=chunk_path.stem,
                caption=caption,
            )
        return

    await _send_audio_with_retries(
        context=context,
        chat_id=chat_id,
        file_path=result,
        filename=result.name,
        title=name,
        caption=caption,
    )


async def _deliver_generated_audio(
    *,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int | str | None,
    status_msg,
    result: Path | list[Path],
    work_dir: Path,
    name: str,
    caption: str | None = None,
) -> str:
    mode = DELIVERY_MODE
    if mode == "send_link":
        await _edit_or_send_status(
            context=context,
            chat_id=chat_id,
            status_msg=status_msg,
            text="🔗 Публикую временную ссылку…",
        )
        await _send_download_link(
            context=context,
            chat_id=chat_id,
            status_msg=status_msg,
            result=result,
            work_dir=work_dir,
            name=name,
            caption=caption,
            reason="выбран режим DELIVERY_MODE=send_link",
        )
        return "link"

    if mode == "auto" and _result_exceeds_telegram_limit(result):
        size_text = format_file_size(_largest_result_size(result))
        safe_limit = format_file_size(TELEGRAM_SAFE_UPLOAD_BYTES)
        await _edit_or_send_status(
            context=context,
            chat_id=chat_id,
            status_msg=status_msg,
            text="🔗 Файл больше безопасного лимита Telegram, публикую ссылку…",
        )
        await _send_download_link(
            context=context,
            chat_id=chat_id,
            status_msg=status_msg,
            result=result,
            work_dir=work_dir,
            name=name,
            caption=caption,
            reason=f"размер {size_text} больше безопасного лимита {safe_limit}",
        )
        return "link"

    try:
        await _edit_or_send_status(
            context=context,
            chat_id=chat_id,
            status_msg=status_msg,
            text=(
                f"📤 Отправляю {len(result)} файл(ов)…"
                if isinstance(result, list)
                else "📤 Отправляю файл…"
            ),
        )
        await _send_result_via_telegram(
            context=context,
            chat_id=chat_id,
            result=result,
            name=name,
            caption=caption,
        )
        return "telegram"
    except Exception as exc:
        if mode == "auto" and _is_request_entity_too_large(exc):
            await _edit_or_send_status(
                context=context,
                chat_id=chat_id,
                status_msg=status_msg,
                text="🔗 Telegram отклонил большой файл, публикую ссылку…",
            )
            await _send_download_link(
                context=context,
                chat_id=chat_id,
                status_msg=status_msg,
                result=result,
                work_dir=work_dir,
                name=name,
                caption=caption,
                reason="Telegram вернул Request Entity Too Large",
            )
            return "link"
        raise


async def _process_and_reply(
    update: Update | None,
    text: str | list[tuple[str, int | str]],
    context: ContextTypes.DEFAULT_TYPE,
    name: str = "book",
    source_name: str | None = None,
    work_dir: Path | None = None,
    status_msg=None,
    chat_id: int | None = None,
    user_id: int | None = None,
    caption: str | None = None,
) -> None:
    """Общая логика: синтез → отправка → очистка."""
    if update and update.effective_chat:
        chat_id = update.effective_chat.id
    if update and update.effective_user and user_id is None:
        user_id = update.effective_user.id

    effective_work_dir: Path
    if work_dir is None:
        effective_work_dir = (
            Path(tempfile.gettempdir()) / f"tg_audiobooker_{uuid.uuid4().hex}"
        )
        effective_work_dir.mkdir(parents=True, exist_ok=True)
    else:
        effective_work_dir = work_dir

    # Получаем настройки  (копируем, чтобы не менять глобальные во время обработки текста)
    if update:
        s = get_user_settings(context).copy()
    else:
        # Для JobQueue берем из user_data вручную
        if user_id is None and context.job:
            user_id = getattr(context.job, "user_id", None)

        if (
            user_id
            and context.application
            and context.application.user_data is not None
        ):
            all_ud = context.application.user_data
            ud = all_ud.get(user_id)  # type: ignore
            if ud is None:
                ud = {}
                try:
                    all_ud[user_id] = ud  # type: ignore
                except Exception:
                    pass

            if isinstance(ud, dict):
                if "settings" not in ud:
                    ud["settings"] = DEFAULT_SETTINGS.copy()
                s = normalize_user_settings(ud["settings"]).copy()
            else:
                s = DEFAULT_SETTINGS.copy()
        else:
            s = DEFAULT_SETTINGS.copy()

    # Проверка параметров в тексте (как для одиночного, так и для списка)
    if isinstance(text, str):
        random_match = re.search(r"random=(true|false)", text, re.IGNORECASE)
        if random_match:
            val = random_match.group(1).lower() == "true"
            # Обновляем в ПЕРСИСТЕНТНЫХ настройках пользователя
            if update:
                orig_s = get_user_settings(context)
                orig_s["random"] = val
            s["random"] = val
            text = re.sub(r"random=(true|false)", "", text, flags=re.IGNORECASE).strip()
            if not text:
                if chat_id is not None:
                    await context.bot.send_message(
                        chat_id,
                        f"✅ Случайный голос: {'ВКЛ' if val else 'ВЫКЛ'}",
                    )  # type: ignore
                return
    else:
        new_text_list = []
        found_val = None
        for item in text:
            txt, s_id = item[0], item[1]
            random_match = re.search(r"random=(true|false)", txt, re.IGNORECASE)
            if random_match:
                found_val = random_match.group(1).lower() == "true"
                txt = re.sub(
                    r"random=(true|false)", "", txt, flags=re.IGNORECASE
                ).strip()
            if txt:
                new_text_list.append((txt, s_id))

        if found_val is not None:
            if update:
                orig_s = get_user_settings(context)
                orig_s["random"] = found_val
            s["random"] = found_val
            text = new_text_list
            if not text:
                if chat_id is not None:
                    await context.bot.send_message(
                        chat_id,
                        f"✅ Случайный голос: {'ВКЛ' if found_val else 'ВЫКЛ'}",
                    )  # type: ignore
                return

    processing_semaphore = get_processing_semaphore()
    user_lock = get_user_processing_lock(user_id)
    queued = processing_semaphore.locked() or (
        user_lock.locked() if user_lock else False
    )

    if status_msg is None and chat_id is not None:
        status_msg = await context.bot.send_message(
            chat_id,  # type: ignore
            "⏳ Запрос поставлен в очередь…"
            if queued
            else "🔊 Синтезирую аудио (начало)…",
        )
    elif queued and status_msg is not None:
        try:
            await status_msg.edit_text("⏳ Запрос поставлен в очередь…")
        except Exception:
            pass

    last_update_time = 0.0
    last_percent = -1

    async def progress_callback(current, total):
        nonlocal last_update_time, last_percent
        percent = int(100 * current // total)
        now = asyncio.get_event_loop().time()
        # Обновляем не чаще раза в секунду и только если процент изменился
        if (
            now - last_update_time > 1.2 or current == total
        ) and percent != last_percent:
            bar = render_progress_bar(current, total)
            try:
                await status_msg.edit_text(
                    f"🔊 Синтезирую аудио ({current}/{total} чанков)…\n`{bar}`",
                    parse_mode="Markdown",
                )
                last_update_time = now
                last_percent = percent
            except Exception:
                pass  # Игнорируем ошибки (например, если сообщение удалено)

    async def _run_generation_and_delivery() -> None:
        if queued and status_msg is not None:
            try:
                await status_msg.edit_text("🔊 Синтезирую аудио (начало)…")
            except Exception:
                pass

        result = await generate_audio(
            text,
            effective_work_dir,
            settings=s,
            name=name,
            source_name=source_name,
            on_progress=progress_callback,
        )

        delivery_kind = await _deliver_generated_audio(
            context=context,
            chat_id=chat_id,
            status_msg=status_msg,
            result=result,
            work_dir=effective_work_dir,
            name=name,
            caption=caption,
        )
        if delivery_kind == "telegram" and status_msg is not None:
            await status_msg.delete()

    try:
        if user_lock is not None:
            async with user_lock:
                async with processing_semaphore:
                    await _run_generation_and_delivery()
        else:
            async with processing_semaphore:
                await _run_generation_and_delivery()

    except FileNotFoundError as e:
        if "ffmpeg" in str(e).lower():
            logger.error("FFmpeg not found!")
            await status_msg.edit_text(
                "❌ Ошибка: В системе не установлен `ffmpeg`.\n"
                "Он необходим для склейки чанков и конвертации в MP3.\n"
                "Установите его: `sudo apt install ffmpeg` (для Linux) или скачайте с ffmpeg.org."
            )
        else:
            logger.exception("FileNotFoundError")
            await status_msg.edit_text(f"❌ Ошибка: Файл не найден: {e}")
    except Exception as e:
        logger.exception("Ошибка при синтезе")
        error_msg = str(e)
        await status_msg.edit_text(f"❌ Ошибка синтеза: {error_msg}")
    finally:
        shutil.rmtree(effective_work_dir, ignore_errors=True)


def kill_existing_instances() -> None:
    """Завершает существующие процессы этого же скрипта, кроме текущего."""
    my_pid = os.getpid()
    try:
        # Ищем все процессы, в команде которых есть имя этого скрипта
        output = (
            subprocess.check_output(["pgrep", "-f", "tg_audiobooker.py"])
            .decode()
            .split()
        )
        for pid_str in output:
            pid = int(pid_str)
            if pid != my_pid:
                logger.info(f"Завершаю старый процесс бота (PID: {pid})...")
                try:
                    os.kill(pid, 15)  # SIGTERM
                except ProcessLookupError:
                    pass
    except subprocess.CalledProcessError:
        # pgrep возвращает 1, если ничего не найдено
        pass
    except Exception as e:
        logger.warning(f"Ошибка при поиске/завершении старых процессов: {e}")


# ─────────────────────────── main ───────────────────────────────


def main() -> None:
    # Убиваем старые копии перед стартом
    kill_existing_instances()

    if not BOT_TOKEN:
        logger.error("BOT_TOKEN is missing in environment variables!")
        raise RuntimeError(
            "Укажите BOT_TOKEN через переменную окружения:\n"
            "  export BOT_TOKEN=<ваш_токен>\n"
            "Получить токен можно у @BotFather в Telegram."
        )

    masked_token = BOT_TOKEN[:10] + "..." if len(BOT_TOKEN) > 10 else "too-short"
    logger.info(f"Starting bot with token: {masked_token}")
    logger.info(
        "Concurrency: updates=%s, requests=%s, grouping debounce=%ss",
        MAX_CONCURRENT_UPDATES,
        MAX_CONCURRENT_REQUESTS,
        FORWARD_GROUP_DEBOUNCE_SECONDS,
    )
    logger.info(
        "Delivery: mode=%s, safe_upload_bytes=%s, download_ttl_seconds=%s",
        DELIVERY_MODE,
        TELEGRAM_SAFE_UPLOAD_BYTES,
        DOWNLOAD_TTL_SECONDS,
    )

    # Настройка персистентности для сохранения настроек пользователей
    data_path = Path(BOT_DATA_PATH)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    persistence = PicklePersistence(filepath=str(data_path))

    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .persistence(persistence)
        .concurrent_updates(MAX_CONCURRENT_UPDATES)
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("settings", cmd_settings))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CallbackQueryHandler(settings_callback, pattern="^set_"))
    # Пересланные сообщения — ловим ДО остальных хендлеров;
    # берём любой пересланный контент, но обрабатываем только текст
    app.add_handler(
        MessageHandler(
            filters.FORWARDED & (filters.TEXT | filters.CAPTION) & ~filters.COMMAND,
            handle_forwarded,
        )
    )
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    logger.info("Бот запущен. Нажмите Ctrl+C для остановки.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    sync_tts_dependencies_from_env()
    main()
