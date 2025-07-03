import asyncio
from edge_tts import Communicate
from pathlib import Path
import os
import subprocess

# === Установка флагов ===
SKIP_CHUNKS = True     # True → не перезаписывать существующие чанки
SKIP_MERGE = False     # True → не выполнять объединение

# === Указание пути к ffmpeg (если не в PATH) ===
FFMPEG_PATH = r"ffmpeg.exe"  # или путь к ffmpeg.exe

# === Настройки ===
INPUT_FILE = "Vafin_Govorit-Vafin.IOENYQ.559764.txt"
OUTPUT_DIR = Path(INPUT_FILE).stem
VOICE = "ru-RU-SvetlanaNeural"
CHUNK_SIZE = 5000
MAX_CONCURRENT_TASKS = 20

# === Подготовка ===




Path(OUTPUT_DIR).mkdir(exist_ok=True)
text = Path(INPUT_FILE).read_text(encoding="utf-8")
chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

async def synthesize_chunk(text, file_path):
    if SKIP_CHUNKS and Path(file_path).exists():
        print(f"[~] Пропущено (уже существует): {file_path}")
        return

    async with semaphore:
        communicate = Communicate(text=text, voice=VOICE, rate="+12%")
        await communicate.save(file_path)
        print(f"[+] Сохранено: {file_path}")

async def main():
    print(f"[=] Генерация {len(chunks)} фрагментов параллельно...")

    tasks = []
    for i, chunk in enumerate(chunks):
        file_path = Path(OUTPUT_DIR) / f"{OUTPUT_DIR}_chunk_{i:03}.mp3"
        tasks.append(synthesize_chunk(chunk, str(file_path)))

    await asyncio.gather(*tasks)

    if SKIP_MERGE:
        print("[~] Объединение отключено (SKIP_MERGE = True)")
        return

    # === Подготовка списка для склейки ===
    print("[=] Подготовка list.txt для ffmpeg ...")
    list_file = Path(OUTPUT_DIR) / "list.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for i in range(len(chunks)):
            part_path = (Path(OUTPUT_DIR) / f"{OUTPUT_DIR}_chunk_{i:03}.mp3").resolve()
            f.write(f"file '{part_path.as_posix()}'\n")

    # === Склейка без перекодирования ===
    output_file = f"{OUTPUT_DIR}.mp3"
    print("[=] Склейка через ffmpeg без перекодирования ...")
    subprocess.run([
        FFMPEG_PATH,
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        output_file
    ], check=True)

    print(f"✅ Готово: {output_file}")

# === Запуск ===
asyncio.run(main())
