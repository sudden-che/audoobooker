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
import logging
import os
import shutil
import tarfile
import tempfile
import uuid
from pathlib import Path

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# Импортируем движок и утилиты из основного скрипта
from audiobooker import (
    clean_text,
    extract_fb2_text,
    synthesize_chunk_edge,
    synthesize_chunk_silero,
    merge_audio_chunks,
    convert_to_mp3,
)

# ============================================================
# НАСТРОЙКИ — задаются через переменные окружения
# ============================================================
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")

# Выбор движка: edge или silero
TTS_ENGINE = os.environ.get("TTS_ENGINE", "edge").lower()

# Общие параметры
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "10000"))
_mct = os.environ.get("MAX_CONCURRENT_TASKS", "").strip()
if _mct:
    MAX_CONCURRENT_TASKS = int(_mct)
else:
    MAX_CONCURRENT_TASKS = 40 if TTS_ENGINE == "edge" else 2

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
# ============================================================

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


async def generate_audio(text: str, work_dir: Path, name: str = "book") -> Path:
    """Синтезирует текст в MP3 (или TAR с чанками) и возвращает путь к результату."""
    parts_dir = work_dir / f"{name}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    chunks = [text[i : i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

    ext = "mp3" if TTS_ENGINE == "edge" else "wav"
    tasks = []

    for i, chunk in enumerate(chunks):
        chunk_file = parts_dir / f"{name}_chunk_{i:06}.{ext}"
        if TTS_ENGINE == "edge":
            tasks.append(
                asyncio.create_task(
                    synthesize_chunk_edge(
                        text=chunk,
                        file_path=chunk_file,
                        voice=EDGE_VOICE,
                        rate=EDGE_SPEED,
                        semaphore=semaphore,
                    )
                )
            )
        else:
            tasks.append(
                asyncio.create_task(
                    synthesize_chunk_silero(
                        text=chunk,
                        file_path=chunk_file,
                        language=SILERO_LANGUAGE,
                        speaker=SILERO_SPEAKER,
                        sample_rate=SILERO_SAMPLE_RATE,
                        put_accent=SILERO_PUT_ACCENT,
                        put_yo=SILERO_PUT_YO,
                        device=DEVICE,
                        model_id=SILERO_MODEL_ID,
                        semaphore=semaphore,
                    )
                )
            )

    await asyncio.gather(*tasks)

    if MERGE_CHUNKS:
        list_file = parts_dir / "list.txt"
        with list_file.open("w", encoding="utf-8") as f:
            for i in range(len(chunks)):
                part_path = (parts_dir / f"{name}_chunk_{i:06}.{ext}").resolve()
                f.write(f"file '{part_path.as_posix()}'\n")

        full_file = work_dir / f"full_{name}.{ext}"
        await merge_audio_chunks(
            ffmpeg_path=FFMPEG_PATH,
            list_file=list_file,
            output_file=full_file,
        )

        if ext == "wav":
            mp3_file = work_dir / f"full_{name}.mp3"
            try:
                await convert_to_mp3(
                    ffmpeg_path=FFMPEG_PATH,
                    input_audio=full_file,
                    output_mp3=mp3_file,
                )
                return mp3_file
            except Exception:
                # Если конвертация не удалась, возвращаем WAV
                return full_file

        return full_file

    # Без склейки — упаковываем в TAR
    tar_path = work_dir / f"{name}_parts.tar"
    with tarfile.open(tar_path, "w") as tar:
        for audio_file in sorted(parts_dir.glob(f"*.{ext}")):
            tar.add(audio_file, arcname=audio_file.name)
    return tar_path


# ─────────────────────────── хендлеры ───────────────────────────


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    engine_info = (
        f"Движок: {TTS_ENGINE}\n"
        f"Голос/Диктор: {EDGE_VOICE if TTS_ENGINE == 'edge' else SILERO_SPEAKER}\n"
        f"Скорость: {EDGE_SPEED if TTS_ENGINE == 'edge' else 'N/A'}\n"
        f"Sample Rate: {SILERO_SAMPLE_RATE if TTS_ENGINE == 'silero' else 'N/A'}"
    )
    await update.message.reply_text(
        "👋 Привет! Я конвертирую текст в аудиокнигу.\n\n"
        "Отправь мне:\n"
        "• текстовое сообщение (до 50 000 символов)\n"
        "• файл .txt или .fb2\n\n"
        f"{engine_info}\n"
        f"Chunk: {CHUNK_SIZE}"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "/start — приветствие\n"
        "/help  — эта справка\n\n"
        "Просто отправь текст или файл .txt/.fb2 — получишь MP3."
    )


async def handle_forwarded(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка пересланных сообщений — извлекаем только текст, вложения игнорируем."""
    # Берём текст из text (обычное сообщение) или caption (фото/видео/документ с подписью)
    text = update.message.text or update.message.caption or ""
    if not text.strip():
        await update.message.reply_text(
            "В пересланном сообщении нет текста (только вложения?). Нечего озвучивать."
        )
        return
    if len(text) > MAX_TEXT_FROM_MESSAGE:
        await update.message.reply_text(
            f"Текст слишком длинный ({len(text)} симв.). "
            f"Максимум {MAX_TEXT_FROM_MESSAGE}. Пришли файлом."
        )
        return

    await _process_and_reply(update, text, name="forwarded")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text or ""
    if not text.strip():
        await update.message.reply_text("Текст пустой, ничего не делаю.")
        return
    if len(text) > MAX_TEXT_FROM_MESSAGE:
        await update.message.reply_text(
            f"Текст слишком длинный ({len(text)} симв.). "
            f"Максимум {MAX_TEXT_FROM_MESSAGE}. Пришли файлом."
        )
        return

    await _process_and_reply(update, text, name="message")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    doc = update.message.document
    if not doc:
        return

    filename = doc.file_name or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in {".txt", ".fb2"}:
        await update.message.reply_text("Поддерживаются только .txt и .fb2 файлы.")
        return

    status_msg = await update.message.reply_text("⏳ Скачиваю файл…")

    work_dir = Path(tempfile.gettempdir()) / f"tg_audiobooker_{uuid.uuid4().hex}"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        tg_file = await context.bot.get_file(doc.file_id)
        local_path = work_dir / f"uploaded{suffix}"
        await tg_file.download_to_drive(local_path)

        if suffix == ".fb2":
            text = extract_fb2_text(local_path)
        else:
            text = local_path.read_text(encoding="utf-8")

        text = clean_text(text)
        if not text.strip():
            await status_msg.edit_text("Файл пустой.")
            return

        await status_msg.edit_text(
            f"🔊 Синтезирую аудио ({len(text)} симв., ~{len(text) // CHUNK_SIZE + 1} чанков)…"
        )
        await _process_and_reply(
            update,
            text,
            name=Path(filename).stem,
            work_dir=work_dir,
            status_msg=status_msg,
        )

    except Exception as e:
        logger.exception("Ошибка при обработке документа")
        await status_msg.edit_text(f"❌ Ошибка: {e}")
        shutil.rmtree(work_dir, ignore_errors=True)


async def _process_and_reply(
    update: Update,
    text: str,
    name: str = "book",
    work_dir: Path | None = None,
    status_msg=None,
) -> None:
    """Общая логика: синтез → отправка → очистка."""
    effective_work_dir: Path
    if work_dir is None:
        effective_work_dir = Path(tempfile.gettempdir()) / f"tg_audiobooker_{uuid.uuid4().hex}"
        effective_work_dir.mkdir(parents=True, exist_ok=True)
    else:
        effective_work_dir = work_dir

    if status_msg is None:
        status_msg = await update.message.reply_text(
            f"🔊 Синтезирую аудио ({len(text)} симв., ~{len(text) // CHUNK_SIZE + 1} чанков)…"
        )

    try:
        result_path = await generate_audio(text, effective_work_dir, name=name)

        await status_msg.edit_text("📤 Отправляю файл…")

        if result_path.suffix in {".mp3", ".wav"}:
            with result_path.open("rb") as f:
                await update.message.reply_audio(
                    audio=f,
                    filename=result_path.name,
                    title=name,
                )
        else:
            with result_path.open("rb") as f:
                await update.message.reply_document(
                    document=f, filename=result_path.name
                )

        await status_msg.delete()

    except Exception as e:
        logger.exception("Ошибка при синтезе")
        await status_msg.edit_text(f"❌ Ошибка синтеза: {e}")
    finally:
        shutil.rmtree(effective_work_dir, ignore_errors=True)


# ─────────────────────────── main ───────────────────────────────


def main() -> None:
    if not BOT_TOKEN:
        raise RuntimeError(
            "Укажите BOT_TOKEN через переменную окружения:\n"
            "  export BOT_TOKEN=<ваш_токен>\n"
            "Получить токен можно у @BotFather в Telegram."
        )

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    # Пересланные сообщения — ловим ДО остальных хендлеров;
    # берём любой пересланный контент, но обрабатываем только текст
    app.add_handler(MessageHandler(filters.FORWARDED & ~filters.COMMAND, handle_forwarded))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    logger.info("Бот запущен. Нажмите Ctrl+C для остановки.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
