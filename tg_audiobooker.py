#!/usr/bin/env python3
"""
Telegram-бот для генерации аудиокниг.
Принимает текстовые сообщения или файлы .txt/.fb2 и возвращает MP3.

Установка зависимостей:
    pip install python-telegram-bot

Запуск:
    BOT_TOKEN=<token> python tg_audiobooker.py
"""

import asyncio
import hashlib
import logging
import os
import random
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

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

from tts_dependency_manager import sync_tts_dependencies_from_env

# Импортируем движок и утилиты из основного скрипта
from audiobooker import (
    apply_audiobook_best_practices,
    build_output_basename,
    collect_valid_audio_files,
    extract_fb2_text,
    synthesize_chunk_edge,
    synthesize_chunk_silero,
    merge_audio_chunks,
    convert_to_mp3,
    split_text,
)

# Загружаем переменные из .env файла
load_dotenv()

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
CPU_COUNT = os.cpu_count() or 2

# Выбор движка: edge или silero
TTS_ENGINE = os.environ.get("TTS_ENGINE", "edge").lower()

# Общие параметры
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "10000"))
_mct = os.environ.get("MAX_CONCURRENT_TASKS", "").strip()
if _mct:
    MAX_CONCURRENT_TASKS = int(_mct)
else:
    if TTS_ENGINE == "edge":
        MAX_CONCURRENT_TASKS = 8 if CPU_COUNT <= 2 else 24
    else:
        MAX_CONCURRENT_TASKS = 1 if CPU_COUNT <= 2 else CPU_COUNT

FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")
MERGE_CHUNKS = os.environ.get("MERGE_CHUNKS", "true").lower() in ("1", "true", "yes")

# Параметры Edge
EDGE_VOICE = os.environ.get("VOICE", "ru-RU-SvetlanaNeural")
EDGE_SPEED = os.environ.get("SPEED", "+18%")

# Параметры Silero
SILERO_LANGUAGE = os.environ.get("SILERO_LANGUAGE", "ru")
SILERO_SPEAKER = os.environ.get("SILERO_SPEAKER", "baya")
SILERO_SAMPLE_RATE = int(os.environ.get("SILERO_SAMPLE_RATE", "48000"))
SILERO_PUT_ACCENT = os.environ.get("SILERO_PUT_ACCENT", "true").lower() == "true"
SILERO_PUT_YO = os.environ.get("SILERO_PUT_YO", "true").lower() == "true"
DEVICE = os.environ.get("DEVICE", "cpu")
SILERO_MODEL_ID = os.environ.get("SILERO_MODEL_ID", "v5_ru")

# Максимальный размер текста
MAX_TEXT_FROM_MESSAGE = int(os.environ.get("MAX_TEXT_FROM_MESSAGE", "50000"))

# Путь к файлу базы данных/настроек
BOT_DATA_PATH = os.environ.get("BOT_DATA_PATH", "data/bot_data.pickle")
FORWARD_GROUP_DEBOUNCE_SECONDS = float(
    os.environ.get("FORWARD_GROUP_DEBOUNCE_SECONDS", "1.5")
)
MAX_CONCURRENT_REQUESTS = int(
    os.environ.get(
        "MAX_CONCURRENT_REQUESTS",
        "2" if CPU_COUNT <= 2 else ("4" if TTS_ENGINE == "edge" else "2"),
    )
)
MAX_CONCURRENT_UPDATES = int(
    os.environ.get(
        "MAX_CONCURRENT_UPDATES",
        str(4 if CPU_COUNT <= 2 else max(4, MAX_CONCURRENT_REQUESTS * 2)),
    )
)

# Списки доступных голосов и дикторов для рандомизации
EDGE_VOICES = [
    "ru-RU-SvetlanaNeural",
    "ru-RU-DmitryNeural",
]
SILERO_SPEAKERS = ["aidar", "baya", "kseniya", "xenia", "eugene"]
SILERO_MODEL_IDS = ["v5_ru", "v4_ru", "v3_1_ru"]

HASHTAG_ONLY_TOKEN_RE = re.compile(r"^[#＃][\w-]+$", re.UNICODE)
FIRST_SENTENCE_RE = re.compile(r"(.+?(?:[.!?]+(?=\s|$)|$))", re.DOTALL)
FILENAME_SAFE_CHARS_RE = re.compile(r'[^\w\s\.\-\(\)]', re.UNICODE)
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
SOURCE_METADATA_PREFIX_RE = re.compile(r"^\s*источник\s*[:\-–—]?\s*(.*)$", re.IGNORECASE)
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
SLASH_COMMAND_RE = re.compile(
    r"^\s*/[A-Za-z0-9_]+(?:@[A-Za-z0-9_]+)?(?:\s|$)"
)


def get_silero_model_major(model_id: str) -> int:
    """Возвращает major-версию модели Silero из строки вида v5_ru."""
    match = re.match(r"v(\d+)", model_id.lower())
    return int(match.group(1)) if match else 0


SILERO_RANDOM_MODEL_IDS = [
    model_id for model_id in SILERO_MODEL_IDS if get_silero_model_major(model_id) >= 5
]

DEFAULT_SETTINGS = {
    "engine": TTS_ENGINE,
    "chunk_size": CHUNK_SIZE,
    "max_concurrent_tasks": MAX_CONCURRENT_TASKS,
    "ffmpeg_path": FFMPEG_PATH,
    "merge_chunks": MERGE_CHUNKS,
    "edge_voice": EDGE_VOICE,
    "edge_speed": EDGE_SPEED,
    "silero_language": SILERO_LANGUAGE,
    "silero_speaker": SILERO_SPEAKER,
    "silero_sample_rate": SILERO_SAMPLE_RATE,
    "silero_put_accent": SILERO_PUT_ACCENT,
    "silero_put_yo": SILERO_PUT_YO,
    "silero_model_id": SILERO_MODEL_ID,
    "device": DEVICE,
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
    media_group_id: str | None = None,
    message_thread_id: int | None = None,
) -> str:
    """Формирует ключ буфера пересланных сообщений для конкретного чата/группы."""
    parts = [f"chat:{chat_id if chat_id is not None else 'unknown'}"]
    if message_thread_id is not None:
        parts.append(f"thread:{message_thread_id}")
    parts.append(f"user:{user_id if user_id is not None else 'unknown'}")
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


def get_user_settings(context: ContextTypes.DEFAULT_TYPE) -> dict:
    """Возвращает настройки пользователя или дефолтные."""
    if context.user_data is None:
        return DEFAULT_SETTINGS.copy()
    if "settings" not in context.user_data:
        context.user_data["settings"] = DEFAULT_SETTINGS.copy()
    return context.user_data["settings"]


def render_progress_bar(current: int, total: int, length: int = 15) -> str:
    """Рисует прогресс-бар из символов."""
    if total <= 0:
        return ""
    filled = int(length * current // total)
    bar = "█" * filled + "░" * (length - filled)
    percent = int(100 * current // total)
    return f"[{bar}] {percent}%"


def choose_random_silero_model_id() -> str:
    """Выбирает случайную Silero-модель только из ветки v5+."""
    if SILERO_RANDOM_MODEL_IDS:
        return random.choice(SILERO_RANDOM_MODEL_IDS)

    if get_silero_model_major(SILERO_MODEL_ID) >= 5:
        return SILERO_MODEL_ID

    return "v5_ru"


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
    kept_fragments = [fragment for fragment in fragments if not _is_subscription_fragment(fragment)]
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

    tokens = [token.strip(".,;()[]{}") for token in tail.split() if token.strip(".,;()[]{}")]
    if not tokens:
        return True

    return len(tokens) <= 4 and all(SOURCE_METADATA_TOKEN_RE.fullmatch(token) for token in tokens)


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

    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
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
        if (cleaned_line := _strip_source_metadata_prefix(_strip_subscription_fragments(line)))
        and not _is_source_metadata_line(cleaned_line)
    )
    text = re.sub(r"\s+,\s+", ", ", text)
    text = re.sub(r",\s*$", "", text, flags=re.MULTILINE)

    # Схлопываем лишние пустые строки
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


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
    Синтезирует текст в MP3.
    input_data может быть строкой или списком (текст, sender_id).
    """
    output_basename = build_output_basename(source_name or name)
    parts_dir = work_dir / f"{output_basename}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = settings.get("chunk_size", CHUNK_SIZE)
    engine = settings.get("engine", TTS_ENGINE)
    silero_model_id = settings.get("silero_model_id", SILERO_MODEL_ID)

    # Если включен режим рандома - выбираем случайно для этой конкретной задачи
    if settings.get("random"):
        engine = random.choice(["edge", "silero"])
        logger.info(f"Random mode: Picked engine {engine}")

    # Silero имеет жесткие ограничения на длину текста (обычно 800-1000 символов).
    # Если выбран Silero, принудительно ограничиваем размер чанка.
    if engine == "silero" and chunk_size > 800:
        logger.warning(
            f"Chunk size {chunk_size} is too large for Silero. Capping at 800."
        )
        chunk_size = 800

    # В режиме рандома для Silero используем только модели версии v5+
    if settings.get("random") and engine == "silero":
        silero_model_id = choose_random_silero_model_id()

    # Подготовка текстов и привязка голосов
    sender_voices: dict[int | str, str | None] = {}

    tasks_data = []  # list [(text, voice_or_speaker)]

    if isinstance(input_data, str):
        input_data = apply_audiobook_best_practices(
            input_data,
            lang=settings.get("silero_language", SILERO_LANGUAGE),
        )
        # В режиме рандома выбираем один голос на всё сообщение
        assigned_voice = None
        if settings.get("random"):
            if engine == "edge":
                assigned_voice = random.choice(EDGE_VOICES)
            else:
                assigned_voice = random.choice(SILERO_SPEAKERS)

        chunks = split_text(input_data, chunk_size)
        for c in chunks:
            tasks_data.append((c, assigned_voice))
    else:
        # Список (текст, sender_id)
        for item in input_data:
            text_part, sender_id = item[0], item[1]
            text_part = apply_audiobook_best_practices(
                text_part,
                lang=settings.get("silero_language", SILERO_LANGUAGE),
            )
            p_chunks = split_text(text_part, chunk_size)

            # В режиме рандома закрепляем голос за отправителем
            assigned_voice = None
            if settings.get("random"):
                if sender_id not in sender_voices:
                    if engine == "edge":
                        assigned_voice = random.choice(EDGE_VOICES)
                    else:
                        assigned_voice = random.choice(SILERO_SPEAKERS)
                    sender_voices[sender_id] = assigned_voice
                else:
                    assigned_voice = sender_voices[sender_id]

            for pc in p_chunks:
                tasks_data.append((pc, assigned_voice))

    max_tasks: int = settings.get("max_concurrent_tasks", MAX_CONCURRENT_TASKS)  # pyright: ignore
    semaphore = asyncio.Semaphore(max_tasks)
    ext = "mp3" if engine == "edge" else "wav"
    tasks = []
    progress = {"completed": 0}
    total = len(tasks_data)

    async def _monitored_task(coro):
        await coro
        progress["completed"] += 1
        if on_progress:
            await on_progress(progress["completed"], total)

    for i, (chunk, assigned_v) in enumerate(tasks_data):
        chunk_file = parts_dir / f"{output_basename}_chunk_{i:06}.{ext}"
        if engine == "edge":
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
                text=chunk,
                file_path=chunk_file,
                voice=voice,
                rate=settings.get("edge_speed", EDGE_SPEED),
                semaphore=semaphore,
            )
        else:
            speaker = assigned_v or settings.get("silero_speaker", SILERO_SPEAKER)
            if settings.get("random") and assigned_v is None:
                speaker = random.choice(SILERO_SPEAKERS)

            coro = synthesize_chunk_silero(
                text=chunk,
                file_path=chunk_file,
                language=settings.get("silero_language", SILERO_LANGUAGE),
                speaker=speaker,
                sample_rate=settings.get("silero_sample_rate", SILERO_SAMPLE_RATE),
                put_accent=settings.get("silero_put_accent", SILERO_PUT_ACCENT),
                put_yo=settings.get("silero_put_yo", SILERO_PUT_YO),
                device=settings.get("device", DEVICE),
                model_id=silero_model_id,
                semaphore=semaphore,
            )
        tasks.append(asyncio.create_task(_monitored_task(coro)))

    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        logger.error(f"Error during audio generation: {e}")
        # Если это Silero и ошибка была фатальной, прокидываем дальше
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

        if ext == "wav":
            mp3_file = work_dir / f"{output_basename}.mp3"
            try:
                await convert_to_mp3(
                    ffmpeg_path=settings.get("ffmpeg_path", FFMPEG_PATH),
                    input_audio=full_file,
                    output_mp3=mp3_file,
                )
                return mp3_file
            except Exception:
                # Если конвертация не удалась, возвращаем WAV
                return full_file

        return full_file

    # Без склейки — возвращаем список файлов
    return actual_chunks


# ─────────────────────────── хендлеры ───────────────────────────


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    s = get_user_settings(context)
    engine = s["engine"]
    engine_info = (
        f"Движок: {engine}\n"
        f"Голос/Диктор: {s['edge_voice'] if engine == 'edge' else s['silero_speaker']}\n"
        f"Скорость: {s['edge_speed'] if engine == 'edge' else 'N/A'}\n"
        f"Model: {s['silero_model_id'] if engine == 'silero' else 'N/A'}\n"
        f"🎲 Рандом-мод: {'✅ ВКЛ' if s.get('random') else '❌ ВЫКЛ'}"
    )
    await update.message.reply_text(
        "👋 Привет! Я конвертирую текст в аудиокнигу.\n\n"
        "Отправь мне:\n"
        "• текстовое сообщение (до 50 000 символов)\n"
        "• файл .txt или .fb2\n\n"
        f"⚙️ Твои текущие настройки:\n{engine_info}\n"
        f"Chunk size: {s['chunk_size']}\n\n"
        "Используй /settings для изменения параметров."
    )


async def cmd_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показывает меню настроек."""
    s = get_user_settings(context)
    text = (
        "⚙️ *Настройки Audiobooker*\n\n"
        f"*Движок:* `{s['engine']}`\n"
        f"*Edge голос:* `{s['edge_voice']}`\n"
        f"*Edge скорость:* `{s['edge_speed']}`\n"
        f"*Silero диктор:* `{s['silero_speaker']}`\n"
        f"*Silero модель:* `{s['silero_model_id']}`\n"
        f"*Chunk size:* `{s['chunk_size']}`\n"
        f"*Random Mode:* `{'ON' if s.get('random') else 'OFF'}`\n\n"
        "Выберите раздел для изменения:"
    )
    keyboard = [
        [
            InlineKeyboardButton("🚀 Движок", callback_data="set_menu_engine"),
            InlineKeyboardButton("📊 Chunk Size", callback_data="set_menu_chunk"),
        ],
        [
            InlineKeyboardButton("🗣 Edge Голос", callback_data="set_menu_edge_voice"),
            InlineKeyboardButton(
                "⏩ Edge Скорость", callback_data="set_menu_edge_speed"
            ),
        ],
        [
            InlineKeyboardButton(
                "🎙 Silero Диктор", callback_data="set_menu_silero_speaker"
            ),
            InlineKeyboardButton(
                "📦 Silero Модель", callback_data="set_menu_silero_model"
            ),
        ],
        [
            InlineKeyboardButton(
                f"🎲 Рандом: {'✅ ВКЛ' if s.get('random') else '❌ ВЫКЛ'}",
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

    # Главные разделы
    if data == "set_menu_engine":
        keyboard = [
            [
                InlineKeyboardButton(
                    "Edge (Online)", callback_data="set_val_engine_edge"
                ),
                InlineKeyboardButton(
                    "Silero (Local)", callback_data="set_val_engine_silero"
                ),
            ],
            [InlineKeyboardButton("⬅️ Назад", callback_data="set_main")],
        ]
        await query.edit_message_text(
            "Выберите движок:", reply_markup=InlineKeyboardMarkup(keyboard)
        )
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
        voices = [
            "ru-RU-SvetlanaNeural",
            "ru-RU-DmitryNeural",
            "ru-RU-ArtemNeural",
            "ru-RU-SaniyaNeural",
        ]
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

    if data == "set_menu_silero_speaker":
        speakers = ["aidar", "baya", "kseniya", "xenia", "eugene"]
        keyboard = [
            [InlineKeyboardButton(sp, callback_data=f"set_val_sspeak_{sp}")]
            for sp in speakers
        ]
        keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data="set_main")])
        await query.edit_message_text(
            "Выберите диктора Silero:", reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return

    if data == "set_menu_silero_model":
        keyboard = [
            [InlineKeyboardButton(m, callback_data=f"set_val_smodel_{m}")]
            for m in SILERO_MODEL_IDS
        ]
        keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data="set_main")])
        await query.edit_message_text(
            "Выберите версию модели Silero:",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )
        return

    # Обработка значений
    if data == "set_main":
        await cmd_settings(update, context)
        # Изменения в context.user_data сохраняются через PicklePersistence
        return

    if data.startswith("set_val_"):
        _, _, key, val = data.split("_", 3)
        if key == "engine":
            s["engine"] = val
            # Adjust max_concurrent_tasks if it's default and engine changes
            # 40 for edge, cpu_count for silero
            if (
                s["max_concurrent_tasks"] in (2, 40)
                or s["max_concurrent_tasks"] == os.cpu_count()
            ):
                s["max_concurrent_tasks"] = (
                    40 if val == "edge" else (os.cpu_count() or 2)
                )
        elif key == "chunk":
            s["chunk_size"] = int(val)
        elif key == "evoice":
            s["edge_voice"] = val
        elif key == "espeed":
            s["edge_speed"] = val
        elif key == "sspeak":
            s["silero_speaker"] = val
        elif key == "smodel":
            s["silero_model_id"] = val
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
        "/settings — настройка движка, голоса, скорости и чанков\n"
        "/help     — эта справка\n\n"
        "Просто отправь текст или файл .txt/.fb2 — получишь MP3."
    )


async def handle_forwarded(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка пересланных сообщений — собираем их в буфер."""
    if not update.message:
        return
    text = update.message.text or update.message.caption or ""
    text = clean_tg_post(text)
    if is_slash_command(text):
        return
    if not text.strip():
        return  # Игнорируем пересланные вложения без текста

    # Оригинальный отправитель (для разграничения голосов в диалоге)
    sender_id, hashtag = extract_forward_sender_metadata(update.message.forward_origin)

    effective_user_id = update.effective_user.id if update.effective_user else None
    effective_chat_id = update.effective_chat.id if update.effective_chat else None

    if context.user_data is None or effective_user_id is None or effective_chat_id is None:
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

        text = apply_audiobook_best_practices(text, lang=SILERO_LANGUAGE)
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
            if attempt == 5:
                logger.error(f"Failed to send audio after 5 attempts: {e}")
                raise
            wait_time = attempt * 2
            logger.warning(
                f"Network error on attempt {attempt}: {e}. Retrying in {wait_time}s with timeout {current_timeout}*1.5..."
            )
            current_timeout = int(current_timeout * 1.5)
            await asyncio.sleep(wait_time)


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
                s = ud["settings"].copy()
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
                        f"✅ Режим Random Mode установлен в: {'ВКЛ' if val else 'ВЫКЛ'}",
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
                        f"✅ Режим Random Mode установлен в: {'ВКЛ' if found_val else 'ВЫКЛ'}",
                    )  # type: ignore
                return

    processing_semaphore = get_processing_semaphore()
    user_lock = get_user_processing_lock(user_id)
    queued = processing_semaphore.locked() or (user_lock.locked() if user_lock else False)

    if status_msg is None and chat_id is not None:
        status_msg = await context.bot.send_message(
            chat_id,  # type: ignore
            "⏳ Запрос поставлен в очередь…" if queued else "🔊 Синтезирую аудио (начало)…",
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

    try:
        if user_lock is not None:
            async with user_lock:
                async with processing_semaphore:
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

                    if isinstance(result, list):
                        await status_msg.edit_text(f"📤 Отправляю {len(result)} файл(ов)…")
                        for chunk_path in result:
                            await _send_audio_with_retries(
                                context=context,
                                chat_id=chat_id,  # type: ignore
                                file_path=chunk_path,
                                filename=chunk_path.name,
                                title=chunk_path.stem,
                                caption=caption,
                            )
                    else:
                        await status_msg.edit_text("📤 Отправляю файл…")
                        await _send_audio_with_retries(
                            context=context,
                            chat_id=chat_id,  # type: ignore
                            file_path=result,
                            filename=result.name,
                            title=name,
                            caption=caption,
                        )

                    await status_msg.delete()
        else:
            async with processing_semaphore:
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

                if isinstance(result, list):
                    await status_msg.edit_text(f"📤 Отправляю {len(result)} файл(ов)…")
                    for chunk_path in result:
                        await _send_audio_with_retries(
                            context=context,
                            chat_id=chat_id,  # type: ignore
                            file_path=chunk_path,
                            filename=chunk_path.name,
                            title=chunk_path.stem,
                            caption=caption,
                        )
                else:
                    await status_msg.edit_text("📤 Отправляю файл…")
                    await _send_audio_with_retries(
                        context=context,
                        chat_id=chat_id,  # type: ignore
                        file_path=result,
                        filename=result.name,
                        title=name,
                        caption=caption,
                    )

                await status_msg.delete()

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
        if "Input text is too long" in error_msg:
            error_msg = (
                "Текст слишком длинный для Silero. Попробуйте уменьшить chunk_size."
            )
        elif "CUDA out of memory" in error_msg:
            error_msg = "Недостаточно видеопамяти (CUDA OOM). Переключитесь на CPU или уменьшите количество потоков."

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
