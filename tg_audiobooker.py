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
import re
import shutil
import subprocess
import tarfile
import tempfile
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from edge_tts import Communicate
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ============================================================
# НАСТРОЙКИ — задаются через переменные окружения
# ============================================================
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")

# Параметры синтеза
VOICE = os.environ.get("VOICE", "ru-RU-SvetlanaNeural")
SPEED = os.environ.get("SPEED", "+18%")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "10000"))
MAX_CONCURRENT_TASKS = int(os.environ.get("MAX_CONCURRENT_TASKS", "40"))
FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")
MERGE_CHUNKS = os.environ.get("MERGE_CHUNKS", "true").lower() in ("1", "true", "yes")

# Максимальный размер текста, принятого прямо из сообщения (символов)
MAX_TEXT_FROM_MESSAGE = int(os.environ.get("MAX_TEXT_FROM_MESSAGE", "50000"))
# ============================================================

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ─────────────────────────── утилиты ────────────────────────────

def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("«", '"').replace("»", '"')
    text = "".join(c for c in text if c.isprintable() or c in "\n\t")
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_fb2_text(fb2_path: Path) -> str:
    tree = ET.parse(fb2_path)
    root = tree.getroot()
    ns = {"fb": "http://www.gribuser.ru/xml/fictionbook/2.0"}
    paragraphs = root.findall(".//fb:body//fb:p", ns)

    def p_text(p_el):
        return "".join(p_el.itertext()).strip()

    lines = [p_text(p) for p in paragraphs if p_text(p)]
    return clean_text("\n\n".join(lines))


async def synthesize_chunk(
    chunk_text: str,
    file_path: Path,
    voice: str,
    speed: str,
    semaphore: asyncio.Semaphore,
) -> None:
    async with semaphore:
        communicate = Communicate(text=chunk_text, voice=voice, rate=speed)
        await communicate.save(str(file_path))


async def generate_audio(text: str, work_dir: Path, name: str = "book") -> Path:
    """Синтезирует текст в MP3 (или TAR с чанками) и возвращает путь к результату."""
    parts_dir = work_dir / f"{name}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    chunks = [text[i: i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

    tasks = [
        asyncio.create_task(
            synthesize_chunk(
                chunk_text=chunk,
                file_path=parts_dir / f"{name}_chunk_{i:06}.mp3",
                voice=VOICE,
                speed=SPEED,
                semaphore=semaphore,
            )
        )
        for i, chunk in enumerate(chunks)
    ]
    await asyncio.gather(*tasks)

    if MERGE_CHUNKS:
        ffmpeg_bin = (
            shutil.which(FFMPEG_PATH)
            if Path(FFMPEG_PATH).name == FFMPEG_PATH
            else FFMPEG_PATH
        )
        if not ffmpeg_bin:
            raise RuntimeError(f"ffmpeg не найден: {FFMPEG_PATH}")

        list_file = parts_dir / "list.txt"
        with list_file.open("w", encoding="utf-8") as f:
            for i in range(len(chunks)):
                part_path = (parts_dir / f"{name}_chunk_{i:06}.mp3").resolve()
                f.write(f"file '{part_path.as_posix()}'\n")

        full_file = work_dir / f"full_{name}.mp3"
        subprocess.run(
            [
                ffmpeg_bin,
                "-f", "concat",
                "-safe", "0",
                "-i", str(list_file),
                "-c", "copy",
                "-loglevel", "error",
                str(full_file),
            ],
            check=True,
            capture_output=True,
        )
        return full_file

    # Без склейки — упаковываем в TAR
    tar_path = work_dir / f"{name}_parts.tar"
    with tarfile.open(tar_path, "w") as tar:
        for mp3 in sorted(parts_dir.glob("*.mp3")):
            tar.add(mp3, arcname=mp3.name)
    return tar_path


# ─────────────────────────── хендлеры ───────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 Привет! Я конвертирую текст в аудиокнигу.\n\n"
        "Отправь мне:\n"
        "• текстовое сообщение (до 50 000 символов)\n"
        "• файл .txt или .fb2\n\n"
        f"Голос: {VOICE}, скорость: {SPEED}, chunk: {CHUNK_SIZE}"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "/start — приветствие\n"
        "/help  — эта справка\n\n"
        "Просто отправь текст или файл .txt/.fb2 — получишь MP3."
    )


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
            f"🔊 Синтезирую аудио ({len(text)} симв., ~{len(text)//CHUNK_SIZE+1} чанков)…"
        )
        await _process_and_reply(update, text, name=Path(filename).stem, work_dir=work_dir, status_msg=status_msg)

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
    own_dir = work_dir is None
    if own_dir:
        work_dir = Path(tempfile.gettempdir()) / f"tg_audiobooker_{uuid.uuid4().hex}"
        work_dir.mkdir(parents=True, exist_ok=True)

    if status_msg is None:
        status_msg = await update.message.reply_text(
            f"🔊 Синтезирую аудио ({len(text)} симв., ~{len(text)//CHUNK_SIZE+1} чанков)…"
        )

    try:
        result_path = await generate_audio(text, work_dir, name=name)

        await status_msg.edit_text("📤 Отправляю файл…")

        if result_path.suffix == ".mp3":
            with result_path.open("rb") as f:
                await update.message.reply_audio(
                    audio=f,
                    filename=result_path.name,
                    title=name,
                )
        else:
            with result_path.open("rb") as f:
                await update.message.reply_document(document=f, filename=result_path.name)

        await status_msg.delete()

    except Exception as e:
        logger.exception("Ошибка при синтезе")
        await status_msg.edit_text(f"❌ Ошибка синтеза: {e}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


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
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    logger.info("Бот запущен. Нажмите Ctrl+C для остановки.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
